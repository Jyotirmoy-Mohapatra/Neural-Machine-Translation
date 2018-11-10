import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, bidirectional=False,num_layers=1):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(output_size, hidden_size)
		if self.bi==True:
			self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers, bidirectional=True)
		else:
			self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		batch_size, seq_len = X.size()
		#reset hidden state here
		if torch.cuda.is_available():
			self.hidden = self.init_hidden(batch_size).cuda()
		else:
			self.hidden = self.init_hidden(batch_size)
		output = self.embedding(input).view(1, 1, -1)
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self):
		if self.bi==True:
			return torch.zeros(self.num_layers*2, batch_size, self.hidden_size, device=device)
		else:
			return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

