import os
import torch
import argparse
from utils.train import *
from Decoder.DecoderRNN import *
from Encoder.encoderRNN import *
from Transformer import *
from Decoder.AttnDecoderRNN import *

parser.add_argument('--output', type=str, default='', metavar='P',
                    help="filename to store translated text.")
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

BATCH_SIZE = 12000
test_iter = MyIterator(test, batch_size=BATCH_SIZE, device=0, \
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), \
                            batch_size_fn=batch_size_fn, train=False)
f_out=open(args.output,'w')
for i, batch in enumerate(test_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, 
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])

    translated_line="Translation:\t"
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        translated_line+=sym+" "
    f_out.write(translated_line)
    actual_line="\nTarget:\t"
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        actual_line+=sym+" "
    f_out.write(actual_line)
f_out.close()