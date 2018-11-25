import os
import torch
import argparse
from utils.train import *
from Decoder.DecoderRNN import *
from Encoder.encoderRNN import *
from Transformer import *
from Decoder.AttnDecoderRNN import *
#from utils.Data_Loader_Transformer import *
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#hidden_size = 256

#input_lang, output_lang, pairs = prepareData('vi', 'en', False)
#print(random.choice(pairs))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch SNLI')
parser.add_argument('--output', type=str, default='', metavar='P',
                    help="path to store saved models")
args = parser.parse_args()
"""
if len(args.output)>0:
    try:
        os.mkdir(scratch+args.output)
    except OSError:
        print("File Creation Error")
    else:
        print("File Creation Success!")
"""

#devices = [0, 1, 2, 3]
pad_idx = TGT.vocab.stoi["<blank>"]
model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
if torch.cuda.is_available():
	model.to(device)
	criterion.to(device)
BATCH_SIZE = 12000
train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0, \
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), \
                            batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0, \
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), \
                            batch_size_fn=batch_size_fn, train=False)
#model_par = nn.DataParallel(model, device_ids=devices)

trainItersTransformer(args, model, train_iter, valid_iter, criterion, pad_idx)


#encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
#.to(device)

#decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)
#decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
#attn_decoder1 = BahdanauAttnDecoderRNN(hidden_size, output_lang.n_words, n_layers=1, dropout_p=0.1).to(device)

##UNCOMMENT TO TRAIN THE MODEL
#trainIters(args, encoder1, decoder1, no_of_iterations, print_every=5000)

#encoder1.load_state_dict(torch.load("encoder.pth"))
#attn_decoder1.load_state_dict(torch.load("attn_decoder.pth"))



#evaluateRandomly(encoder1, decoder1)
