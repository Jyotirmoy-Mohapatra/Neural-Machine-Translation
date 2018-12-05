from io import open
import random
import torch
from params import *
from torch.utils.data import Dataset
import numpy as np
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 60
"""

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UNK"}
        self.n_words = 3  # Count SOS and EOS

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

    def indexesFromSentence(self, sentence):
        return [(self.word2index[word] if word in self.word2index else UNK_token) for word in sentence.split(' ')]


    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def readLangs(lang1, lang2, dataset, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines1 = open(scratch+'iwslt-vi-en/train.tok.%s' % (lang1), encoding='utf-8').\
        read().strip().split('\n')
    lines2 = open(scratch+'iwslt-vi-en/train.tok.%s' % (lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[l1, l2] for l1,l2 in zip(lines1,lines2)]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    maxLen = 0
    maxsent = []
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        #maxLen = max(maxLen,len(pair[0].split(' ')))
        #if maxLen < len(pair[0].split(' ')):
        #    maxLen = len(pair[0].split(' '))
        #    maxsent = [pair[0],pair[1]]
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    #print("MAX_LENGTH = ",maxLen)
    #print(maxsent)
    return input_lang, output_lang, pairs

#create a function that reads the dev set and generates vi-en sentence pairs and filters them
def dataforEval(dataset,lang1,lang2):
    print("Reading lines...")

    # Read the file and split into lines
    lines1 = open(scratch+'iwslt-vi-en/'+dataset+'.tok.%s' % (lang1), encoding='utf-8').\
        read().strip().split('\n')
    lines2 = open(scratch+'iwslt-vi-en/'+dataset+'.tok.%s' % (lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    epairs = [[l1, l2] for l1,l2 in zip(lines1,lines2)]
    epairs = filterPairs(epairs)
    #source = []
    ref = []

    for i in range(len(epairs)):
        #source.append(pairs[i][0])
        ref.append(epairs[i][1])
    return epairs, [ref]

input_lang, output_lang, pairs = prepareData('vi', 'en', False)
print(random.choice(pairs))

def tensorsFromPair(pair):
    return (input_lang.tensorFromSentence(pair[0]), output_lang.tensorFromSentence(pair[1]))



def nmt_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0,MAX_LENGTH-datum[1])), 
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    #print(label_list[:10])
    print(label_list[:5])
    print(data_list[:5])
    data_list = torch.tensor(data_list)
    length_list = torch.LongTensor(length_list)
    #label_list = torch.tensor(label_list)
    return [data_list, length_list, label_list]

class NMTDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, target_list):
        """
        @param data_list: list of source language tokens 
        @param target_list: list of targets language tokens
        """
        self.data_list = data_list
        self.target_list = target_list
        #print(len(self.data_list),len(self.target_list))
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        token_idx = self.data_list[key][:MAX_LENGTH]
        #print (token_idx)
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]

input_indices = []
target_indices = []

for pair in pairs:
    pair_t = tensorsFromPair(pair)
    input_indices.append(pair_t[0])
    target_indices.append(pair_t[1])

#print(input_indices[:5])
train_dataset = NMTDataset(input_indices, target_indices)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=nmt_collate_func,
                                           shuffle=True)
