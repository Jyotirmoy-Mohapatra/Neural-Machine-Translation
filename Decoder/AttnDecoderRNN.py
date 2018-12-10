import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from params import *

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, attention, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        #self.max_length = max_length
        self.attention = attention
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        #self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        #self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size*2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size*3, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        #print("input: ",input.shape)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        #print("embedded: ",embedded.shape)
        #print("hidden: ",hidden.shape)
        a = self.attention(hidden.squeeze(0), encoder_outputs)

        a = a.unsqueeze(1)
        #print("a: ", a.shape)
        encoder_outputs=encoder_outputs.permute(1,0,2)
        #print("encoder_outputs: ",encoder_outputs.shape)

        context = torch.bmm(a, encoder_outputs)

        context = context.permute(1, 0, 2)
        #print("context: ", context.shape)

        rnn_input = torch.cat((embedded, context), dim=2)
        #print("rnn_input: ",rnn_input.shape)

        output, hidden = self.gru(rnn_input, hidden)
        #print("output: ",output.shape)
        #print("hidden: ",hidden.shape)
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        context = context.squeeze(0)
        
        output = self.softmax(self.out(torch.cat((output, context, embedded), dim=1)))
        
        #output = [bsz, output dim]
        
        return output, hidden,a
        """
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), 1)), dim=1)
        encoder_outputs=encoder_outputs.permute(1,0,2)
        print("a: ",attn_weights.unsqueeze(1).shape)
        print("enc outputs: ",encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1)  ,
                                 encoder_outputs)

        attn_applied = attn_applied.permute(1,0,2)
        #print("Context: ",attn_applied.shape)
        output = torch.cat((embedded, attn_applied), 2)
        #print("gru input: ",output.shape)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        #print(output.squeeze(0).shape)
        output, hidden = self.gru(output.squeeze(0), hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
        """
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)