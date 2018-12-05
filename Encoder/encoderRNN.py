import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from params import *
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, bidirectional=False,num_layers=1):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.bi=bidirectional
		self.num_layers=num_layers
		if self.bi==True:
			self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers, bidirectional=True)
		else:
			self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers)
		
    
	def forward(self, X, hidden):
		#print(X.shape)
		batch_size = X.size(1)
		#reset hidden state here
		if torch.cuda.is_available():
			self.hidden = self.initHidden(batch_size).cuda()
		else:
			self.hidden = self.initHidden(batch_size)

		#print(hidden.shape)
		embedded = self.embedding(X)
		output = embedded
		output, self.hidden = self.gru(output, hidden)
		if self.bi:
			output=(output[:,:,:self.hidden_size]+output[:,:,:self.hidden_size])
			#self.hidden=torch.sum(self.hidden,dim=0,keepdim=True)
		#print(output.size(),self.hidden.size())
		return output, self.hidden

	def initHidden(self, batch_size):
		if self.bi==True:
			return torch.zeros(2*self.num_layers, batch_size, self.hidden_size, device=device)
		else:
			return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

