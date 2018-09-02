
'''
Author: Sudharsansai, UCLA
A simple sequence to sequence model with attention for Abstractive Text Summarization
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


# In[5]:


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# In[8]:


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[9]:


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# In[10]:


def readLangs(path, lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(path+'%s-%s.txt' % (lang1, lang2), encoding='utf-8').        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

# In[75]:


def prepareData(path, lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(path, lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

path = "./data/"
input_lang, output_lang, pairs = prepareData(path, 'eng', 'fra', True)
print(random.choice(pairs))
np.random.shuffle(pairs)
testing_pairs = pairs[len(pairs)-100:]
pairs = pairs[:len(pairs)-100]


# In[76]:


'''
This is a helper function to print time elapsed and estimated time remaining given the current time and progress %.
'''

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[77]:


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
        return Variable(torch.zeros(1, self.batch_size, self.hidden_size), requires_grad=False)


# In[78]:


class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, batch_size, output_size, dropout_p=0.1, max_len=20):
        super(AttnDecoderRNN, self).__init__()

        # seq_len is the maximum sequence length of the encoder outputs
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
        
        #self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs, encoder_lengths):
        # encoder_lengths has shape (batch_size)
        # encoder_outputs has shape (encoder_seq_len, batch_size, hidden_size)
        
        # embedded has shape (1, batch_size, emb_size)
        embedded = self.embedding(inputs).view(1, self.batch_size, -1)
        embedded = self.dropout(embedded)
        
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
        assert attn_context.size()[0] == batch_size
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
        return torch.zeros(1, self.batch_size, self.hidden_size, device=device)


# In[79]:


# preparing training data
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.LongTensor(indexes).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# In[80]:


# Training the model

teacher_forcing_ratio = 1



def train_seq2seq_attention(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, 
                         criterion, lengths_input, lengths_target, max_length):

    input_tensor = Variable(input_tensor, requires_grad=False)
    target_tensor = Variable(target_tensor, requires_grad=False)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    loss = 0

    encoder_outputs, encoder_hidden, encoder_lengths = encoder(input_tensor, lengths_input)
 
    decoder_input = torch.LongTensor([[SOS_token]*batch_size])
    decoder_input = Variable(decoder_input.view(1, batch_size), requires_grad=False)
    
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, encoder_lengths)
            #decoder_output = decoder_output.view(1, batch_size, -1)
            loss += criterion(decoder_output.view(batch_size, -1), target_tensor[di][:])
            decoder_input = target_tensor[di][:].view(1, batch_size)  # Teacher forcing
            decoder_input = decoder_input.type(torch.LongTensor)

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, encoder_lengths)
            #decoder_output = decoder_output.view(1, batch_size, -1)
            topv, topi = decoder_output.topk(1, dim=2)
            decoder_input = topi.view(1, self.batch_size).detach()  # detach from history as input
            decoder_input = decoder_input.type(torch.LongTensor)
            loss += criterion(decoder_output.view(batch_size, -1), target_tensor[di][:])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


# In[81]:


# takes a list of pairs of sentences (where words are indexed) and converts them to a batch of size (max_seq_len, batch_size)
# and also returns the sequence lengths, along with padding them by pad=1
def make_batch(training_pairs):
    lengths1 = []
    lengths2 = []
    for i in training_pairs:
        lengths1.append(len(i[0]))
        lengths2.append(len(i[1]))
    lengths1 = np.array(lengths1)
    lengths2 = np.array(lengths2)
    max_len1 = max(lengths1)
    max_len2 = max(lengths2)
    # b is indices of lengths sorted in descending order
    b = lengths1.argsort()[::-1][:len(lengths1)]
    inp = torch.ones(int(max_len1), int(len(training_pairs))).type(torch.LongTensor)
    targets = torch.ones(int(max_len2), int(len(training_pairs))).type(torch.LongTensor)
    ret_lengths1 = []
    ret_lengths2 = []
    for i in range(len(b)):
        for j in range(len(training_pairs[b[i]][0])):
            inp[j][i] = training_pairs[b[i]][0][j][0]
        for j in range(len(training_pairs[b[i]][1])):
            targets[j][i] = training_pairs[b[i]][1][j][0]
        ret_lengths1.append(len(training_pairs[b[i]][0]))
        ret_lengths2.append(len(training_pairs[b[i]][1]))
        
    # lengths2 wont be sorted, lengths1 will be sorted in descending order
    return inp, targets, ret_lengths1, ret_lengths2
        


# In[82]:



def trainIters(encoder, decoder, n_epochs, print_every=100, plot_every=100, learning_rate=3e-4, batch_size=16):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(pairs[i])
                      for i in range(len(pairs))]
    #print('Training Pairs 1 Shape: ',training_pairs[0][0].shape)
    criterion = nn.NLLLoss()
    for num_epoch in range(n_epochs):
        start = time.time()
        print('Epoch Number: ', num_epoch, ' --------------------------------------------')
        permutation = np.random.permutation(len(pairs))
        num_batches = len(pairs)//batch_size
        print_loss_total = 0
        plot_loss_total = 0
        for batch_num in range(num_batches):
            cur_batch = list(permutation[batch_num*batch_size:(batch_num+1)*batch_size])
            input_tensor, target_tensor, lengths_input, lengths_target =                             make_batch([training_pairs[i] for i in cur_batch])

            loss = train_seq2seq_attention(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, lengths_input, lengths_target, max_length=20)
            print_loss_total += loss
            plot_loss_total += loss

            if batch_num % print_every == 0:
                if(batch_num==0):
                    print_loss_avg = print_loss_total 
                else:
                    print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, (batch_num+1)*1.0 / num_batches),
                                             batch_num, batch_num / num_batches * 100, print_loss_avg))

            if batch_num % plot_every == 0:
                if(batch_num==0):
                    plot_loss_avg = plot_loss_total
                else:
                    plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


# In[83]:


hidden_size = 256
emb_size = 100
batch_size = 16
src_vocab_size = input_lang.n_words
tgt_vocab_size = output_lang.n_words
lr = 3e-4
plot_every = 100
print_every = 200
n_epochs = 100

encoder = EncoderRNN(vocab_size=src_vocab_size, emb_size=emb_size, hidden_size=hidden_size, batch_size=batch_size, variable_lengths = True)

#  vocab_size, emb_size, hidden_size, batch_size, output_size, dropout_p=0.1, max_len=20
decoder = AttnDecoderRNN(src_vocab_size, emb_size, hidden_size, 
                     batch_size, output_size=tgt_vocab_size, dropout_p=0.15, max_len = 20)

trainIters(encoder, decoder, n_epochs=n_epochs, print_every=print_every, plot_every=plot_every, learning_rate=lr, batch_size=batch_size)
#trainIters(encoder, decoder, 75000, print_every=5000)


# In[84]:


def evaluate(encoder, decoder, input_tensor, input_lengths, max_length):
    # sentence pairs are of shape (max_len, batch_size)
    
    input_tensor = Variable(input_tensor, requires_grad=False)
    encoder_outputs, encoder_hidden, encoder_lengths = encoder(input_tensor, input_lengths)

    #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    decoder_input = torch.LongTensor([[SOS_token]*batch_size])  # SOS
    decoder_input = Variable(decoder_input.view(1, batch_size), requires_grad=False)
    
    decoder_hidden = encoder_hidden
    
    decoded_words = []
    
    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, encoder_lengths)
        decoder_output = decoder_output.view(1, batch_size, tgt_vocab_size)
        topv, topi = decoder_output.topk(1, dim=2)
        decoder_input = topi.view(1, batch_size).detach()
        decoder_input = decoder_input.type(torch.LongTensor)
        decoded_words.append([output_lang.index2word[index] for index in decoder_input.data[0][:]])

    input_words = [[input_lang.index2word[index] for index in input_tensor.data[i][:]] 
                   for i in range(input_tensor.data.shape[0])]

    return input_words, decoded_words


# In[85]:


def evaluateRandomly(encoder, decoder, starting_point):
    training_pairs = [tensorsFromPair(testing_pairs[i])
                      for i in range(starting_point, starting_point+batch_size)]
    pair = training_pairs
    input_tensor, target_tensor, lengths_input, lengths_target = make_batch(pair)
    input_words, output_words = evaluate(encoder, decoder, input_tensor, lengths_input, max_length=target_tensor.shape[0])
    
    targets = []
    decoded_words = [[output_lang.index2word[index] for index in target_tensor[i][:]] for i in range(target_tensor.shape[0])]
    
    words1 = [[] for _ in range(len(input_words[0]))]
    words2 = [[] for _ in range(len(output_words[0]))]
    words3 = [[] for _ in range(len(decoded_words[0]))]

    for i in range(len(input_words)):
        for j in range(len(input_words[i])):
            words1[j].append(input_words[i][j])

    for i in range(len(output_words)):
        for j in range(len(output_words[i])):
            words2[j].append(output_words[i][j])
            
    for i in range(len(decoded_words)):
        for j in range(len(decoded_words[i])):
            words3[j].append(decoded_words[i][j])    
    
    # words1: input words
    # words2: predicted words
    # words3: actual target words
    return words1, words2, words3


# In[86]:


# evaluate

for i in range(0, len(testing_pairs)-batch_size, batch_size):

	input_words, output_words, targets = evaluateRandomly(encoder, decoder, starting_point = i)

	assert len(input_words)==len(output_words)

	for i in range(len(input_words)):
	    print('Source Language: ', input_words[i],' -> Target Language: ', output_words[i], targets[i])

