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
UNK_token = 3

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
        #self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attn_bilinear_encoder = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_bilinear_decoder = nn.Linear(self.hidden_size, self.hidden_size)
        #self.attn_hidden_redn = nn.Linear(self.hidden_size+self.emb_size, self.hidden_size)
        #self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.pointer_prob_mat = nn.Linear(3*self.hidden_size, 1)
        self.out = nn.Linear(3*self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_hidden, encoder_lengths, previous_weights, decoder_hidden, encoder_inputs, time_step=0):
    	# hidden has shape (1,batch_size, hidden_size)
        # encoder_lengths has shape (batch_size)
        # encoder_hidden has shape (encoder_seq_len, batch_size, hidden_size)
        # inputs has shape (1, batch_size)
        # decoder_hidden has shape (decoder_seq_len, batch_size, hidden_size)

        embedded =self.embedding(inputs)
        # embedded_normalized = torch.norm(embedded, p=2, dim=2, keepdim=True).detach()
        # embedded = embedded.div(embedded_normalized)


        decoder_output, h_t_d = self.gru(embedded, hidden)
        # decoder_output has shape (1, batch_size, hidden_size)
        # h_t_d has shape (1, batch_size, hidden_size)

        encoder_hidden = encoder_hidden.transpose(0,1).transpose(1,2)
        # encoder_hidden has shape (batch_size, hidden_size, encoder_seq_len)
        
        transformed_hidden_encoder = self.attn_bilinear_encoder(h_t_d).transpose(0,1)
        # transformed_hidden_encoder has shape (batch_size, 1, hidden_size)

        e_t = torch.bmm(transformed_hidden_encoder, encoder_hidden)
        # e_t has shape (batch_size, 1,  encoder_seq_len)

        # masked attention
        mask = torch.zeros(e_t.shape[0],1,e_t.shape[2]).type(torch.cuda.ByteTensor)
        for i in range(mask.shape[0]):
            if(mask.shape[2]<encoder_lengths[i]):
                mask[i][0][encoder_lengths[i]:] = 1

        e_t.data.masked_fill_(mask, -float('inf'))

        # Normalizing e_t
        #e_t_normalized = torch.norm(e_t, p=2, dim=2, keepdim=True).detach()
        #e_t = e_t.div(e_t_normalized)
        #e_t = self.dropout(e_t)

        #e_prime_t = torch.exp(e_t)
        # e_prime_t has shape (batch_size, 1, encoder_seq_len)

        '''
        if(time_step!=0):
        	assert e_t.size()==previous_weights.size()
        	e_prime_t = torch.div(e_prime_t, previous_weights)

        previous_weights = torch.add(previous_weights, 1, e_prime_t)
        '''

        alpha_t_e = F.softmax(e_t, dim=2)
        # alpha_t_e has shape (batch_size, 1, encoder_seq_len)

        encoder_hidden = encoder_hidden.transpose(1,2)
        # encoder_hidden has shape(batch_size, encoder_seq_len, hidden_size)

        c_t_e = torch.bmm(alpha_t_e, encoder_hidden)
        # context_vectors has shape (batch_size, 1, hidden_size)
        
        h_t_d = h_t_d.transpose(0,1)
        # h-t_d has shape (batch_size, 1, hidden_size)

        # encoder-decoder attention with weight penalty ##############



        #####################################################

        # intra-decoder attention ################

        transformed_hidden_decoder = self.attn_bilinear_decoder(h_t_d)
        # transformer_hidden_decoder has shape (batch_size, 1, hidden_size)

        decoder_hidden = decoder_hidden.transpose(0,1).transpose(1,2)
        # decoder_hidden has shape (batch_size, hidden_size, decoder_seq_len)

        e_t_d = torch.bmm(transformed_hidden_decoder, decoder_hidden)
        # e_t_d has shape (batch_size, 1, decoder_seq_len)

        # Normalizing e_t_d
        #e_t_d_normalized = torch.norm(e_t_d, p=2, dim=2, keepdim=True).detach()
        #e_t_d = e_t_d.div(e_t_d_normalized)
        # e_t_d has shape (batch_size, 1, decoder_seq_len)

        alpha_t_d = F.softmax(e_t_d, dim=2)
        # alpha_t_d has shape (batch_size, 1, decoder_seq_len)

        decoder_hidden = decoder_hidden.transpose(1,2)
        # decoder_hidden has shape (batch_size, decoder_seq_len, hidden_size)
        
        c_t_d = torch.bmm(alpha_t_d, decoder_hidden)
        # c_t_d has shape (batch_size, 1, hidden_size)


        ##########################################

        new_hidden = torch.cat((c_t_e, h_t_d, c_t_d), dim=2)
        # new_hidden has shape (batch_size, 1, 3*hidden_size)

        if((new_hidden!=new_hidden).any()):
            print('new_hidden has nans')

        gen_output = self.out(new_hidden)
        # output has shape (batch_size, 1, output_size)

        # Normalizing output
        #output_normalized = torch.norm(output, p=2, dim=2, keepdim=True).detach()
        #output = output.div(output_normalized)
        '''
        for i in range(output.shape[0]):
            for j in range(output.shape[2]):
                print(output[i][0][j].data)
        '''
        gen_output = F.log_softmax(gen_output, dim=2).view(self.batch_size, self.output_size)
        # output has shape (batch_size, output_size)


        ###### Pointer Network #############################

        pointer_prob = self.pointer_prob_mat(new_hidden).squeeze(2)
        # pointer_prob has shape (batch_size, 1)

        encoder_inputs = encoder_inputs.transpose(0,1).detach()
        # encoder_inputs is of shape (batch_size, encoder_seq_len)

        '''
        pointer_mask = torch.zeros (self.batch_size , encoder_inputs.shape[1], self.output_size).type(torch.cuda.FloatTensor)
        # pointer_mask is of shape (batch_size, encoder_seq_len, output_size)
        print('Pointer_mask shape: ',pointer_mask.shape)
        for i in range(pointer_mask.shape[0]):
            for j in range(pointer_mask.shape[1]):
                if(j<encoder_lengths[i]):
                    pointer_mask[i][j][encoder_inputs.data[i][j]]=1
        pointer_mask = Variable(pointer_mask, requires_grad=False)
        # pointer_mask has shape (batch_size, encoder_seq_len, output_size)
        '''

        alpha_t_e = alpha_t_e.transpose(1,2).squeeze(2)
        # alpha_t_e has shape (batch_size, encoder_seq_len)

        #final_output = (1-pointer_prob)*gen_output
        # final_output has shape (batch_size, output_size)

        alpha_t_e = pointer_prob*alpha_t_e

        pointer_mask = Variable(torch.zeros(self.batch_size, self.output_size).type(torch.cuda.FloatTensor))

        for i in range(encoder_inputs.shape[0]):
            for j in range(encoder_inputs.shape[1]):
                pointer_mask.data[i][encoder_inputs.data[i][j]] += alpha_t_e.data[i][j]

        #pointer_mask = Variable(pointer_mask, requires_grad=False) 

        #pointer_mask = torch.bmm(alpha_t_e.unsqueeze(0), pointer_mask.unsqueeze(0)).squeeze(0)

        #pointer_mask = pointer_mask * alpha_t_e

        final_output = (1-pointer_prob)*gen_output + pointer_mask

        #pointer_mask = torch.mul(alpha_t_e, pointer_mask)
        # pointer_output has shape (batch_size, encoder_seq_len, output_size)

        #pointer_mask = torch.sum(pointer_mask, dim=1)
        # pointer_output has shape (batch_size, output_size)

        ##############################################################
        
        #print('Pointer_prob shape: ',pointer_prob.shape)
        #pointer_prob = F.sigmoid(pointer_prob)
        # pointer_prob has shape (batch_size, 1, 1)

        #pointer_prob = pointer_prob.squeeze(2)
        # pointer_prob has shape (batch_size, 1)

        #final_output = (1-pointer_prob)*gen_output + pointer_prob*pointer_mask

        assert final_output.shape[0] == self.batch_size
        assert final_output.shape[1] == self.output_size

        h_t_d = h_t_d.transpose(0,1)
        # h_t_d has shape (1, batch_size, hidden_size)

        return final_output, h_t_d, previous_weights

    def initHidden(self):
        return Variable(torch.zeros(1, self.batch_size, self.hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)