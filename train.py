'''
Author: Sudharsansai, UCLA
Trainer for Abstractive Text Summarization
'''


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import DRASmodels
import dataloader
import time
import math

SOS_token = 0
EOS_token = 1
UNK_token = 3

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

# Training the model

teacher_forcing_ratio = 1
hidden_size = 100
emb_size = 300
batch_size = 4
lr = 3e-4
plot_every = 100
print_every = 10
n_epochs = 1
path = "../data/cnndm/"
src_filename = "trunc_val.src.txt"
tgt_filename = "trunc_val.tgt.txt"

input_lang, output_lang, pairs, val_pairs, test_pairs = dataloader.fetch_data(path, src_filename, tgt_filename, num_val=500, num_test=500)

src_vocab_size = input_lang.n_words
tgt_vocab_size = output_lang.n_words


def train_seq2seq_attention(input_tensor, target_tensor, encoder, input_in_target_lang, decoder, encoder_optimizer, decoder_optimizer, 
                         criterion, lengths_input, lengths_target, max_length):

    input_tensor = Variable(input_tensor, requires_grad=False)
    target_tensor = Variable(target_tensor, requires_grad=False)
    input_in_target_lang = Variable(input_in_target_lang, requires_grad=False)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    loss = 0

    encoder_outputs, encoder_hidden, encoder_lengths = encoder(input_tensor, lengths_input)
 	
    for i in range(encoder_outputs.shape[0]):
        for j in range(encoder_outputs.shape[1]):
            if(i>encoder_lengths[j]):
                encoder_outputs[i,j,:] = 0

    decoder_input = torch.cuda.LongTensor([[SOS_token]*batch_size])
    decoder_input = Variable(decoder_input.view(1, batch_size), requires_grad=False)
    decoder_hidden = Variable(torch.zeros(1, batch_size, hidden_size).type(torch.cuda.FloatTensor), requires_grad=False)
    
    h_t_d = encoder_hidden
    previous_weights = Variable(torch.zeros(encoder_outputs.shape[1], 1, encoder_outputs.shape[0]).type(torch.cuda.FloatTensor), requires_grad=False)
    #previous_weights = Variable(previous_weights)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, h_t_d, previous_weights = decoder(decoder_input, h_t_d, encoder_outputs, encoder_lengths, previous_weights, 
                                                                    decoder_hidden, input_in_target_lang, di)
            if(di==0):
                decoder_hidden = h_t_d
            else:
                decoder_hidden = torch.cat((decoder_hidden, h_t_d), dim=0)
            #decoder_output = decoder_output.view(1, batch_size, -1)
            loss += criterion(decoder_output.view(batch_size, -1), target_tensor[di][:].view(batch_size))
            decoder_input = target_tensor[di][:].view(1, batch_size)  # Teacher forcing
            decoder_input = decoder_input.type(torch.cuda.LongTensor)

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, h_t_d, previous_weights = decoder(decoder_input, decoder_hidden, encoder_outputs, encoder_lengths, previous_weights, 
                                                                    decoder_hidden, input_in_target_lang, di)
            if(di==0):
                decoder_hidden = h_t_d
            else:
                decoder_hidden = torch.cat((decoder_hidden, h_t_d), dim=0)
            #decoder_output = decoder_output.view(1, batch_size, -1)
            topv, topi = decoder_output.topk(1, dim=2)
            decoder_input = topi.view(1, self.batch_size).detach()  # detach from history as input
            decoder_input = decoder_input.type(torch.cuda.LongTensor)
            loss += criterion(decoder_output.view(batch_size, -1), target_tensor[di][:].view(batch_size))

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.detach() / target_length



def trainIters(encoder, decoder, n_epochs, print_every=100, plot_every=100, learning_rate=3e-4, batch_size=batch_size):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [dataloader.tensorsFromPair(input_lang, output_lang, pairs[i])
                      for i in range(len(pairs))]
    #print('Training Pairs 1 Shape: ',training_pairs[0][0].shape)
    criterion = nn.NLLLoss().cuda()
    for num_epoch in range(n_epochs):
        start = time.time()
        print('Epoch Number: ', num_epoch, ' --------------------------------------------')
        permutation = np.random.permutation(len(pairs))
        num_batches = len(pairs)//batch_size
        print_loss_total = 0
        plot_loss_total = 0
        for batch_num in range(num_batches):
            print('Batch Number: ', batch_num, ' #####################################')
            cur_batch = list(permutation[batch_num*batch_size:(batch_num+1)*batch_size])
            input_tensor, target_tensor, lengths_input, lengths_target = dataloader.make_batch([training_pairs[i] for i in cur_batch])

            input_in_target_lang = torch.zeros(input_tensor.shape).type(torch.cuda.LongTensor)

            for i in range(input_tensor.shape[0]):
                for j in range(input_tensor.shape[1]):
                    input_in_target_lang[i][j] = output_lang.word2index.get(input_lang.index2word[input_tensor[i][j]],3)

            loss = train_seq2seq_attention(input_tensor, target_tensor, encoder, input_in_target_lang, 
                         decoder, encoder_optimizer, decoder_optimizer, criterion, lengths_input, lengths_target, max_length=120)
            print_loss_total += loss
            plot_loss_total += loss

            if batch_num % print_every == 0:
                if(batch_num==0):
                    print_loss_avg = print_loss_total 
                else:
                    print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (dataloader.timeSince(start, (batch_num+1)*1.0 / num_batches),
                                             batch_num, batch_num / num_batches * 100, print_loss_avg))

            if batch_num % plot_every == 0:
                if(batch_num==0):
                    plot_loss_avg = plot_loss_total
                else:
                    plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    #dataloader.showPlot(plot_losses)


# In[83]:


encoder = DRASmodels.EncoderRNN(vocab_size=src_vocab_size, emb_size=emb_size, hidden_size=hidden_size, batch_size=batch_size, variable_lengths = True)

encoder = encoder.cuda()

decoder = DRASmodels.AttnDecoderRNN(src_vocab_size, emb_size, hidden_size, 
                     batch_size, output_size=tgt_vocab_size, dropout_p=0.15, max_len = 100)
decoder = decoder.cuda()

trainIters(encoder, decoder, n_epochs=n_epochs, print_every=print_every, plot_every=plot_every, learning_rate=lr, batch_size=batch_size)
#trainIters(encoder, decoder, 75000, print_every=5000)
