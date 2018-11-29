from torchtext import data, datasets

def tokenize_vi(text):
        return text.split(" ")

def tokenize_en(text):
    return text.split(" ")

def tokenize_zh(text):
    return text.split(" ")

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(tokenize=tokenize_vi, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

MAX_LEN = 100
train, val, test = datasets.TranslationDataset.splits(
    exts=('.vi', '.en'), fields=(SRC, TGT), path='Neural-Machine-Translation/iwslt-vi-en-processed/',train='train.tok',validation='dev.tok',test='test.tok',
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
        len(vars(x)['trg']) <= MAX_LEN,)
MIN_FREQ = 2
count=0
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
