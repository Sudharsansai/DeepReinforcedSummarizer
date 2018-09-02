'''
Author: Sudharsansai, UCLA
Dataloader Helper functions for Abstractive Text Summarizer
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


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 3: "UNK"}
        self.n_words = 3  # Count SOS, EOS and UNK

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

def preprocess(inp):
	return inp

def readData(path, src, tgt):
    print("Reading lines...")

    src_lines = open(path+src, encoding="utf-8").read().strip().split('\n')
    tgt_lines = open(path+tgt, encoding="utf-8").read().strip().split('\n')

    assert len(src_lines)==len(tgt_lines)
    pairs = [(preprocess(src_lines[i]), preprocess(tgt_lines[i])) for i in range(len(src_lines))]

    input_lang = Lang(src)
    output_lang = Lang(tgt)

    return input_lang, output_lang, pairs


def prepareData(path, src, tgt, reverse=False):
    input_lang, output_lang, pairs = readData(path, src, tgt)
    print("Read %s sentence pairs" % len(pairs))
    #print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, ' : ',input_lang.n_words)
    print(output_lang.name, ' : ',output_lang.n_words)
    return input_lang, output_lang, pairs


def fetch_data(path, src, tgt, num_val, num_test):
	'''
	Inputs:
		path: path name
		src: source filename
		tgt: target filename
		num_val: number of validation examples
		num_test: number of testing examples
	Outputs:
		Input Lang: a Lang object
		Output Lang: "
		pairs: training pairs
		val_pairs: validation pairs
		test_pairs: testing_pairs
	'''
	input_lang, output_lang, pairs = prepareData(path, src, tgt, True)
	assert len(pairs)>(num_val+num_test)
	print('A random example...')
	print(random.choice(pairs))
	np.random.shuffle(pairs)
	test_pairs = pairs[len(pairs)-num_test:]
	pairs = pairs[:len(pairs)-num_test]
	val_pairs = pairs[len(pairs)-num_val:]
	pairs = pairs[:len(pairs)-num_val]
	return input_lang, output_lang, pairs, val_pairs, test_pairs

'''
This is a helper function to print time elapsed and estimated time remaining given the current time and progress %.
'''

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

# preparing training data
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.cuda.LongTensor(indexes).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    #input_in_target_tensor = tensorFromSentence(output_lang, pair[0])
    return (input_tensor, target_tensor)


# takes a list of pairs of sentences (where words are indexed) and converts them to a batch of size (max_seq_len, batch_size)
# and also returns the sequence lengths, along with padding them by pad=1 (which is EOS)
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
    inp = torch.ones(int(max_len1), int(len(training_pairs))).type(torch.cuda.LongTensor)
    #input_in_target_lang = torch.ones(int(max_len1), int(len(training_pairs))).type(torch.cuda.LongTensor)
    targets = torch.ones(int(max_len2), int(len(training_pairs))).type(torch.cuda.LongTensor)
    ret_lengths1 = []
    ret_lengths2 = []
    for i in range(len(b)):
        for j in range(len(training_pairs[b[i]][0])):
            inp[j][i] = training_pairs[b[i]][0][j][0]
            #input_in_target_lang[j][i] = training_pairs[b[i]][2][j][0]
        for j in range(len(training_pairs[b[i]][1])):
            targets[j][i] = training_pairs[b[i]][1][j][0]
        ret_lengths1.append(len(training_pairs[b[i]][0]))
        ret_lengths2.append(len(training_pairs[b[i]][1]))
        
    # lengths2 wont be sorted, lengths1 will be sorted in descending order
    return inp, targets, ret_lengths1, ret_lengths2