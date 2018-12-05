import os
import argparse
from utils.train import *
from Decoder.DecoderRNN import *
from Encoder.encoderRNN import *
from Transformer import *
from Decoder.AttnDecoderRNN import *
from utils.Data_Loader_Torchtext import *
from torchtext.data import Field, BucketIterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
parser = argparse.ArgumentParser(description='PyTorch NMT')
parser.add_argument('--output', type=str, default='', metavar='P',
                    help="path to store saved models")
parser.add_argument('--model', type=str, default='encoder', metavar='M',
                    help="which model to run 1.encoder 2.attention 3.selfattention")
args = parser.parse_args()

pad_idx = TGT.vocab.stoi["<blank>"]
<<<<<<< HEAD
if args.model == "selfattention":
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    model.to(device)
    criterion.to(device)
    BATCH_SIZE = 2500
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0, \
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), \
                                batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0, \
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), \
                                batch_size_fn=batch_size_fn, train=False)
#model_par = nn.DataParallel(model, device_ids=devices)

    trainItersTransformer(args, model, train_iter, valid_iter, criterion, pad_idx,n_iters=100,plot_every=1,save_every=10,warmup=20)
else:
=======
model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
model.to(device)
criterion.to(device)
BATCH_SIZE = 2500
train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0, \
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), \
                            batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0, \
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), \
                            batch_size_fn=batch_size_fn, train=False)
#model_par = nn.DataParallel(model, device_ids=devices)

trainItersTransformer(args, model, train_iter, valid_iter, criterion, pad_idx,n_iters=100,plot_every=1,save_every=10,warmup=20)
>>>>>>> a229383c05cbe3c598247404824fbc5bbdc96995

    encoder = EncoderRNN(len(SRC.vocab), hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, len(TGT.vocab)).to(device)

<<<<<<< HEAD
    if args.model == "attention":
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
    
    train_iter, valid_iter, test_iterator = BucketIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=device)
    ##UNCOMMENT TO TRAIN THE MODEL
    trainIters(args, train_iter, valid_iter, encoder, decoder, print_every=10)
=======
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
>>>>>>> a229383c05cbe3c598247404824fbc5bbdc96995
