import time
import math
from utils.plot import *
#from utils.data import *
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from Transformer import *
#from utils.Data_Loader_Transformer import *




#teacher_forcing_ratio = 0.5
def run_epoch(args,data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
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
    encoder_hidden = encoder.initHidden(1)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        #print("\n", encoder_output.size(), encoder_outputs.size(), "\n")
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    encoder_hidden=torch.sum(encoder_hidden,dim=0,keepdim=True)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

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



def trainIters(args, input_lang, output_lang, pairs, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.001, transformer=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every


    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs),input_lang,output_lang)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            torch.save(encoder.state_dict(), scratch+args.output+"encoder.pth")
            torch.save(decoder.state_dict(), scratch+args.output+"decoder.pth")
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(args, plot_losses)

def trainItersTransformer(args,model,train_iter,valid_iter,criterion, pad_idx, n_iters=75000,plot_every=100,save_every=15000):
    plot_losses = 0
    train_plot_losses=0
    plot_loss_list=[]
    train_plot_loss_list=[]
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(1,n_iters+1):
        model.train()
        train_loss=run_epoch(args,(rebatch(pad_idx, b) for b in train_iter), 
                  model, 
                  SimpleLossCompute(model.generator, criterion, opt=model_opt))
        model.eval()
        loss = run_epoch(args,(rebatch(pad_idx, b) for b in valid_iter), 
                          model, 
                          SimpleLossCompute(model.generator, criterion, opt=None))
        plot_losses+=loss
        train_plot_losses+=train_loss
        if epoch % save_every==0:
            print("\n\nEpoch number:",epoch,"\n... Saving model.")
            torch.save(model.state_dict(), scratch+args.output+epoch+"_transformer_model.pth")
        if epoch % plot_every==0:
            plot_loss_avg=plot_losses/plot_every
            train_plot_loss_avg=train_plot_losses/plot_every
            plot_loss_list.append(plot_loss_avg)
            train_plot_loss_list.append(train_plot_loss_avg)
            plot_losses=0
            train_plot_losses=0
    showPlot(args, plot_loss_list)
    showPlot(args,train_plot_loss_list,train=True)