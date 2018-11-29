import os
import torch
import argparse
#from utils.train import *
from Decoder.DecoderRNN import *
from Encoder.encoderRNN import *
from Transformer import *
from Decoder.AttnDecoderRNN import *


device = torch.device("cuda")
print(device)
parser = argparse.ArgumentParser(description='Transformer Generate')

parser.add_argument('--output', type=str, default='', metavar='P',
                    help="filename to store translated text.")
parser.add_argument('--model', type=str, default='', metavar='P',
                    help="model to generate for.")

args = parser.parse_args()



def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

BATCH_SIZE = 2500
test_iter = MyIterator(test, batch_size=BATCH_SIZE, device=0, \
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), \
                            batch_size_fn=batch_size_fn, train=False)
#test_iter.SRC=test_iter.SRC.to(device)
#test_iter.TRG=test_iter.TRG.to(device)
#test_iter.src=test_iter.src.to(device)

model = make_model(len(SRC.vocab), len(TGT.vocab), N=1)
model.to(device)
model.load_state_dict(torch.load(args.model))
f_out=open(args.output,'w')
for i, batch in enumerate(test_iter):
    src = batch.src.to(device).transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, 
                        max_len=100, start_symbol=TGT.vocab.stoi["<s>"])
    batch.trg=batch.trg.to(device)
    translated_line="Translation: "
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        translated_line+=sym+" "
    f_out.write(translated_line)
    
    actual_line="\nTarget: "
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        actual_line+=sym+" "
    f_out.write(actual_line+"\n\n")
f_out.close()


