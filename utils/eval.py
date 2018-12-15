import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sacreBLEU.sacrebleu import *

#from utils.plot import *
#from utils.data import *
from params import*
#from utils.beam_search import *
from torchtext.data import Field, BucketIterator
from utils.Data_Loader_Torchtext import *
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    """
    Function that generate translation.
    First, feed the source sentence into the encoder and obtain the hidden states from encoder.
    Secondly, feed the hidden states into the decoder and unfold the outputs from the decoder.
    Lastly, for each outputs from the decoder, collect the corresponding words in the target language's vocabulary.
    And collect the attention for each output words.
    @param encoder: the encoder network
    @param decoder: the decoder network
    @param sentence: string, a sentence in source language to be translated
    @param max_length: the max # of words that the decoder can return
    @output decoded_words: a list of words in target language
    @output decoder_attentions: a list of vector, each of which sums up to 1.0
    """    
    # process input sentence
    with torch.no_grad():
        input_tensor = input_lang.tensorFromSentence(sentence)
        input_length = input_tensor.size()[0]
        # encode the source lanugage
        encoder_hidden = encoder.initHidden(1)
        
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        """for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        """
        input_tensor = input_tensor.view(-1,1)
        #print(input_tensor.shape)
        encoder_outputs, encoder_hidden = encoder(input_tensor,encoder_hidden)
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        print(decoder_input.shape)
        # decode the context vector
        decoder_hidden = encoder_hidden # decoder starts from the last encoding sentence
        print(decoder_hidden.shape)
        # output of this function
        decoded_words = []
        #decoder_attentions = torch.zeros(max_length, max_length)
        print(encoder_outputs.shape)
        #gen_words = beamsearch(decoder, decoder_hidden,encoder_outputs,beam_width=1, clip_len=15)
        
        for di in range(max_length):
            # for each time step, the decoder network takes two inputs: previous outputs and the previous hidden states
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            #decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            print("words: ",decoded_words)
            decoder_input = topi.squeeze().detach()

        #return decoded_words, decoder_attentions[:di + 1]
        
        return gen_words[0],decoded_words


def evaluateRandomly(evalpairs, encoder, decoder, n=10):
    """
    Randomly select a English sentence from the dataset and try to produce its French translation.
    Note that you need a correct implementation of evaluate() in order to make this function work.
    """    
    """for i in range(n):
        pair = random.choice(evalpairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, greed = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        greed_o = ' '.join(greed[:-1])
        bleu_score=sentence_bleu(output_sentence,pair[1])
        print('<', output_sentence)
        print('@', greed_o)
        print("BLEU Score:",bleu_score)
        print('')
        """
    train_iter, valid_iter, test_iterator = BucketIterator.splits(
        (train, val, test), batch_size=BATCH_SIZE, device=device)
    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            src = batch.src
            trg = batch.trg
            inp =""
            for w in src[:,0]:
                inp += SRC.vocab.itos[w.item()]
                inp += " "
            print(inp)

            op=""
            for w in trg[1:-1,0]:
                op += TGT.vocab.itos[w.item()]
                op += " "
            print(op)
            encoder_hidden = encoder.initHidden(src.size(1))
            max_len = src.shape[0]
            batch_size = trg.shape[1]
            trg_vocab_size = len(TGT.vocab)
            outputs = torch.zeros(max_len, batch_size, trg_vocab_size).cuda()
            cell, hidden = encoder(src, encoder_hidden)
            decoder_input = trg[0,:].view(1,-1)
            decoder_hidden = encoder_hidden
            res=""
            for t in range(1, max_len):    
                output, hidden, attn = decoder(decoder_input, hidden, cell)
                outputs[t] = output
                #teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.max(1)[1].view(1,-1)
                res+=" "+TGT.vocab.itos[top1[0][0]]
                decoder_input = top1
            print(res)
 

def evaluateDataset(encoder, decoder):
    """
    Randomly select a English sentence from the dataset and try to produce its French translation.
    Note that you need a correct implementation of evaluate() in order to make this function work.
    """    
    res = []
    ref = []
    """
    for i in range(len(evalpairs)):
        pair = evalpairs[i]
        #print('>', pair[0])
        #print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words[:-1])
        res.append(output_sentence)
    """
    train_iter, valid_iter, test_iterator = BucketIterator.splits(
        (train, val, test), batch_size=BATCH_SIZE, device=device)
    print(len(train_iter))
    print(len(valid_iter))
    print(len(test_iterator))
    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            src = batch.src
            trg = batch.trg
            #print(src.size(1))
            for col in range(trg.size(1)):
                op=""
                for w in trg[1:-1,col]:
                    op += TGT.vocab.itos[w.item()]
                    op += " "
                ref.append(op)
            encoder_hidden = encoder.initHidden(src.size(1))
            max_len = src.shape[0]
            batch_size = trg.shape[1]
            trg_vocab_size = len(TGT.vocab)
            cell, hidden = encoder(src, encoder_hidden)
            #hidden = encoder(src)
            #cell = torch.randn(4).to(device)
            decoder_input = trg[0,:].view(1,-1)
            #decoder_hidden = encoder_hidden
            res_sen=['']*batch_size
            cvec = hidden
            for t in range(1, max_len):    
                output, hidden, attn = decoder(decoder_input, hidden, cvec)
                #print(output[:,:20])
                top1 = output.max(1)[1].view(1,-1)
                #print(TGT.vocab.itos[top1[0][0]])
                for i in range(top1.size(1)):
                    res_sen[i] += " "+TGT.vocab.itos[top1[0][i]]
                decoder_input = top1
            res+=res_sen
    print(ref[:15])
    print(res[:15])
    """print(len(ref))"""
    print(len(res))
    bleu_score = corpus_bleu(res,[ref])
    print("BLEU: ", bleu_score)
