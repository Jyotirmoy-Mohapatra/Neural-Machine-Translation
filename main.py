import torch

from utils.train import *
from utils.eval import *
from Decoder.DecoderRNN import *
from Encoder.encoderRNN import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 256

#input_lang, output_lang, pairs = prepareData('vi', 'en', False)
#print(random.choice(pairs))

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
#attn_decoder1 = BahdanauAttnDecoderRNN(hidden_size, output_lang.n_words, n_layers=1, dropout_p=0.1).to(device)

##UNCOMMENT TO TRAIN THE MODEL
trainIters(encoder1, decoder1, 75000, print_every=5)

#encoder1.load_state_dict(torch.load("encoder.pth"))
#attn_decoder1.load_state_dict(torch.load("attn_decoder.pth"))



evaluateRandomly(encoder1, decoder1)