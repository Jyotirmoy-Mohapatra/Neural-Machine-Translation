import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from params import *
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, bidirectional=False,num_layers=1):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.bi = bidirectional
		self.num_layers=num_layers
		self.embedding = nn.Embedding(output_size	, hidden_size)
		if self.bi==True:
			self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers, bidirectional=True)
		else:
			self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers)
		if self.bi:
			self.out = nn.Linear(hidden_size*2, output_size)
		else:
			self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden, weights):
		#print(input.shape)
		batch_size = input.size(0)
		#reset hidden state here
		if torch.cuda.is_available():
			self.hidden = self.initHidden(batch_size).cuda()
		else:
			self.hidden = self.initHidden(batch_size)
		#print(input.size())
		output = self.embedding(input)
		output = F.relu(output)
		output, self.hidden = self.gru(output, hidden)
		#print("\n",output.size(),"\n")
		#print(self.out(output[0]).size())
		output = self.softmax(self.out(output[0]))
		return output, self.hidden, weights

	def initHidden(self,batch_size):
		if self.bi==True:
			return torch.zeros(2*self.num_layers, batch_size, self.hidden_size, device=device)
		else:
			return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

