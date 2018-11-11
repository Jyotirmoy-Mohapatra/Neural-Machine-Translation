import os
import torch
import argparse
from utils.train import *
from Decoder.DecoderRNN import *
from Encoder.encoderRNN import *
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#hidden_size = 256

#input_lang, output_lang, pairs = prepareData('vi', 'en', False)
#print(random.choice(pairs))
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

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
#attn_decoder1 = BahdanauAttnDecoderRNN(hidden_size, output_lang.n_words, n_layers=1, dropout_p=0.1).to(device)

##UNCOMMENT TO TRAIN THE MODEL
trainIters(args, encoder1, decoder1, no_of_iterations, print_every=5000)

#encoder1.load_state_dict(torch.load("encoder.pth"))
#attn_decoder1.load_state_dict(torch.load("attn_decoder.pth"))



#evaluateRandomly(encoder1, decoder1)