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
		batch_size = 1
		#reset hidden state here
		if torch.cuda.is_available():
			self.hidden = self.initHidden(batch_size).cuda()
		else:
			self.hidden = self.initHidden(batch_size)

		embedded = self.embedding(X).view(1, 1, -1)
		output = embedded
		output, self.hidden = self.gru(output, hidden)
		return output, self.hidden

	def initHidden(self, batch_size):
		if self.bi==True:
			return torch.zeros(self.num_layers*2, batch_size, self.hidden_size, device=device)
		else:
			return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

