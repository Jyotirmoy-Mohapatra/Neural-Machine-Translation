import torch

from utils.eval import *
from Decoder.DecoderRNN import *
from Encoder.encoderRNN import *

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#hidden_size = 256

#input_lang, output_lang, pairs = prepareData('vi', 'en', False)
#print(random.choice(pairs))

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
#attn_decoder1 = BahdanauAttnDecoderRNN(hidden_size, output_lang.n_words, n_layers=1, dropout_p=0.1).to(device)

encoder1.load_state_dict(torch.load("encoder.pth"))
decoder1.load_state_dict(torch.load("decoder.pth"))

evaluateRandomly(encoder1, decoder1)