import os
import argparse
from utils.train import *
from Decoder.DecoderRNN import *
from Encoder.encoderRNN import *
from Transformer import *
from Decoder.AttnDecoderRNN import *
from Decoder.Attention import *
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
    print(device)
    encoder = EncoderRNN(len(SRC.vocab), hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, len(TGT.vocab)).to(device)
    if args.model == "cnn":
        encoder = EncoderCNN(len(SRC.vocab), hidden_size).to(device)
    if args.model == "attention":
        #encoder = EncoderRNN(len(SRC.vocab), hidden_size,bidirectional=True).to(device)
        attn = Attention(hidden_size,hidden_size)
        decoder = AttnDecoderRNN(hidden_size, len(TGT.vocab),attn).to(device)
    train_iter, valid_iter, test_iterator = BucketIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device=device)
    print(len(train_iter))
    print(len(valid_iter))
    print(len(test_iterator))
    ##UNCOMMENT TO TRAIN THE MODEL
    trainIters(args, train_iter, valid_iter, encoder, decoder, print_every=10)
