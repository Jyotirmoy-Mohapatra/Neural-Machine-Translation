import torch

from utils.eval import *
from Decoder.DecoderRNN import *
from Encoder.encoderRNN import *
from Encoder.encoderCNN import *
from Decoder.AttnDecoderRNN import *
from Decoder.Attention import *
import argparse
from sacreBLEU.sacrebleu import *
from utils.Data_Loader_Torchtext import *

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#hidden_size = 256

#input_lang, output_lang, pairs = prepareData('vi', 'en', False)
#print(random.choice(pairs))

parser = argparse.ArgumentParser(description='PyTorch SNLI')
parser.add_argument('--output', type=str, default='', metavar='P',
                    help="path to store saved models")
args = parser.parse_args()

print(len(SRC.vocab))
encoder1 = EncoderRNN(len(SRC.vocab), hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, len(TGT.vocab)).to(device)
#encoder1 = EncoderCNN(len(SRC.vocab), hidden_size).to(device)
#attn = Attention(hidden_size,hidden_size)
#decoder = AttnDecoderRNN(hidden_size, len(TGT.vocab),attn).to(device)

#decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
#attn_decoder1 = BahdanauAttnDecoderRNN(hidden_size, output_lang.n_words, n_layers=1, dropout_p=0.1).to(device)

encoder1.load_state_dict(torch.load(scratch+args.output+"encoder.pth"))
decoder1.load_state_dict(torch.load(scratch+args.output+"decoder.pth"))

#evalpairs, ref = dataforEval('dev','vi','en')

#evaluateRandomly(evalpairs,encoder1, decoder1)
evaluateDataset(encoder1, decoder1)
