'''
Author: Sudharsansai, UCLA
Model Definitions for Abstractive Text Summarization
'''


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np

import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from dataloader import *


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    
    def __init__(self, vocab_size, emb_size, hidden_size, batch_size, variable_lengths = True):
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size)
        self.hidden = self.init_hidden()
    
    def forward(self, inputs, input_lengths):
        # inputs should be of shape (seq_len, batch_size)
        embedded = self.embedding(inputs)
        
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        
        output, hidden = self.gru(embedded, self.hidden)
        
        assert hidden.size()[0] == 1
        assert hidden.size()[1] == self.batch_size
        assert hidden.size()[2] == self.hidden_size
        
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
            
        # output is of shape (max_seq_len, batch_size, hidden_size)
        return output, hidden, input_lengths
    
    def init_hidden(self):
        return Variable(torch.zeros(1, self.batch_size, self.hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)



class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, batch_size, output_size, dropout_p=0.1, max_len=20):
        super(AttnDecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_len

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.gru = nn.GRU(self.emb_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.attn_hidden_redn = nn.Linear(self.hidden_size+self.emb_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs, encoder_lengths):
        # encoder_lengths has shape (batch_size)
        # encoder_outputs has shape (encoder_seq_len, batch_size, hidden_size)
        
        # embedded has shape (1, batch_size, emb_size)
        embedded = self.embedding(inputs)
        #embedded = self.dropout(embedded)
        
        attn_hidden = torch.cat((embedded, hidden), dim=2)
        assert attn_hidden.size()[0] == 1
        assert attn_hidden.size()[1] == self.batch_size
        assert attn_hidden.size()[2] == self.hidden_size + self.emb_size
        attn_hidden = self.attn_hidden_redn(attn_hidden)
        # now the attn_hidden has shape (1, batch_size, hidden_size)
        attn_hidden = attn_hidden.transpose(0,1)
        # now the attn_hidden has shape (batch_size, 1, hidden_size)
        encoder_outputs = encoder_outputs.transpose(0,1).transpose(1,2)
        # encoder_outputs has shape (batch_size, hidden_size, encoder_seq_len)
        assert encoder_outputs.size()[0] == self.batch_size
        assert encoder_outputs.size()[1] == self.hidden_size
        attn_weights = torch.bmm(attn_hidden, encoder_outputs)
        # attn_weights should have shape (batch_size, 1, encoder_seq_len)
        #print('Attn_weights Shape: ', attn_weights.shape)
        #attn_weights = attn_weights.transpose(0,1)
        
        # masked attention
        for i in range(attn_weights.shape[0]):
            for j in range(attn_weights.shape[2]):
                if j>=encoder_lengths[i]:
                    attn_weights.data[i][0][j] = float('-inf')
        
        attn_weights = F.softmax(attn_weights, dim=2)
        # attn_weights.shape is (batch_size, 1, encoder_seq_len)
        
        context_vectors = encoder_outputs.transpose(1,2)
        # context vectors is encoder_outputs with shape (batch_size, encoder_seq_len, hidden_size)
        attn_context = torch.bmm(attn_weights, context_vectors)
        
        # attention_context should have shape (batch_size, 1, hidden_size)
        assert attn_context.size()[0] == self.batch_size
        assert attn_context.size()[1] == 1
        assert attn_context.size()[2] == self.hidden_size
        
        attn_context = attn_context.transpose(0,1)
        # now attn_context has shape (1, batch_size, hidden_size)
        #print('Attention Context Shape: ', attn_context.shape)
        #print('Hidden Shape: ',hidden.shape)
        new_hidden = torch.cat((attn_context, hidden), dim=2)
        # new_hidden has shape (1, batch_size, 2*hidden_size)
        
        new_hidden = self.attn_combine(new_hidden)
        
        decoder_output, hidden = self.gru(embedded, new_hidden)
        output = self.out(hidden)
        output = F.log_softmax(output, dim=2)
        output = output.view(self.batch_size, self.output_size)
        #print('Output of Decoder shape: ', output.shape)
        #print('Hidden state of Decoder shape: ', hidden.shape)
        # output is of shape (batch_size, output_size)
        # hidden is of shape (batch_size, hidden_size)
        return output, hidden
    
        '''
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
        '''

    def initHidden(self):
        return Variable(torch.zeros(1, self.batch_size, self.hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)