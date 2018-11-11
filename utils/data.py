from io import open
import random
import torch
from params import *
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

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')]


    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def readLangs(lang1, lang2, reverse=False):
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


input_lang, output_lang, pairs = prepareData('vi', 'en', False)
print(random.choice(pairs))


def tensorsFromPair(pair):
    return (input_lang.tensorFromSentence(pair[0]), output_lang.tensorFromSentence(pair[1]))