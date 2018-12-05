import time
import math
import random
from utils.plot import *
#from utils.data import *
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from Transformer import *
#from utils.Data_Loader_Transformer import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")


#teacher_forcing_ratio = 0.5
def run_epoch(args,data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        batch.src=batch.src.to(device)
        batch.trg=batch.trg.to(device)
        batch.trg_mask=batch.trg_mask.to(device)
        batch.src_mask=batch.src_mask.to(device)
        batch.trg_y=batch.trg_y.float().to(device)
        batch.ntokens=batch.ntokens.float().to(device)
        #print(time.time()-start_time)
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        start_time=time.time()
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden(input_tensor.size(1))

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    max_len = target_tensor.shape[0]
    batch_size = target_tensor.shape[1]
    trg_vocab_size = len(TGT.vocab)
    
    loss = 0
    
    cell, hidden = encoder(input_tensor, encoder_hidden)
    outputs = torch.zeros(max_len, batch_size, trg_vocab_size).cuda()
    decoder_input = target_tensor[0,:].view(1,-1)
    decoder_hidden = encoder_hidden
    for t in range(1, max_len):    
        output, hidden, attn = decoder(decoder_input, hidden, cell)
        outputs[t] = output
        teacher_force = random.random() < teacher_forcing_ratio
        top1 = output.max(1)[1].view(1,-1)
        decoder_input = (target_tensor[t].view(1,-1) if teacher_force else top1)
 
    
    loss = criterion(outputs[1:].view(-1, outputs.shape[2]), target_tensor[1:].view(-1))
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def evaluateModel(encoder, decoder,iterator, criterion):
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg
            encoder_hidden = encoder.initHidden(src.size(1))
            #output = model(src, trg, 0) #turn off teacher forcing
            batch_size = trg.shape[1]
            max_len = trg.shape[0]
            trg_vocab_size = len(TGT.vocab)
        
            #tensor to store decoder outputs
            outputs = torch.zeros(max_len, batch_size, trg_vocab_size).cuda()
        
            #last hidden state of the encoder is used as the initial hidden state of the decoder
            cell, hidden = encoder(src, encoder_hidden)
            decoder_input = trg[0,:].view(1,-1)
            decoder_hidden = encoder_hidden
            #first input to the decoder is the <sos> tokens
         
            for t in range(1, max_len):
            
                output, hidden, attn = decoder(decoder_input, hidden, cell)
                outputs[t] = output
                top1 = output.max(1)[1].view(1,-1)
                decoder_input = top1

            loss = criterion(outputs[1:].view(-1, outputs.shape[2]), trg[1:].view(-1))

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def trainIters(args, train_iter, valid_iter, encoder, decoder, print_every=10, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    best_loss = float('inf')
    #print(len(train_iter))
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    #training_pairs = [tensorsFromPair(random.choice(pairs))
    #                  for i in range(n_iters)]
    criterion = nn.NLLLoss()
    for iter in range(1, n_iters + 1):
        #training_pair = training_pairs[iter - 1]
        #input_tensor = training_pair[0]
        #target_tensor = training_pair[1]
        train_loss = 0
        #print("Epoch: ",iter)
        for i, batch in enumerate(train_iter):
            input_tensor = batch.src
            target_tensor = batch.trg
            loss = train(input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
            train_loss += loss

        train_loss = train_loss/len(train_iter)
        valid_loss = evaluateModel(encoder, decoder, valid_iter, criterion)

        print(f'| Epoch: {iter+0:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
        """
        if iter % print_every == 0:
            torch.save(encoder.state_dict(), scratch+args.output+"encoder.pth")
            torch.save(decoder.state_dict(), scratch+args.output+"decoder.pth")
            print_loss_avg = epoch_loss / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        """
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(encoder.state_dict(), scratch+args.output+"encoder.pth")
            torch.save(decoder.state_dict(), scratch+args.output+"decoder.pth")
        #if iter % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

    showPlot(args, plot_losses)

def trainItersTransformer(args,model,train_iter,valid_iter,criterion, pad_idx, n_iters=1,plot_every=1,save_every=10,warmup=2000):
    plot_losses = 0
    train_plot_losses=0
    plot_loss_list=[]
    train_plot_loss_list=[]
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, warmup,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(1,n_iters+1):
        print("\nEpoch number:",epoch)
        #start_time_train=time.time()
        model.train()
        train_loss=run_epoch(args,(rebatch(pad_idx, b) for b in train_iter), 
                  model, 
                  SimpleLossCompute(model.generator, criterion, opt=model_opt))
        #print("Time to train:",epoch,time.time()-start_time_train)
        #start_time_val=time.time()

        model.eval()
        loss = run_epoch(args,(rebatch(pad_idx, b) for b in valid_iter), 
                          model, 
                          SimpleLossCompute(model.generator, criterion, opt=None))
        #print("Time to validate:",epoch,time.time()-start_time_val)
        plot_losses+=loss
        train_plot_losses+=train_loss
        if epoch % save_every==0:
            print("... Saving model.")
            torch.save(model.state_dict(), scratch+args.output+str(epoch)+"_transformer_model.pth")
        if epoch % plot_every==0:
            plot_loss_avg=plot_losses/plot_every
            train_plot_loss_avg=train_plot_losses/plot_every
            plot_loss_list.append(plot_loss_avg)
            train_plot_loss_list.append(train_plot_loss_avg)
            plot_losses=0
            train_plot_losses=0
    showPlot(args, plot_loss_list)
    showPlot(args,train_plot_loss_list,train=True)
