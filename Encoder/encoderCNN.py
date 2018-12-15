import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from params import *

class EncoderCNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderCNN, self).__init__()
		self.hidden_size=hidden_size
		self.embedding = nn.Embedding(input_size, 2*hidden_size)
		self.conv1 = nn.Conv1d(2*hidden_size, hidden_size, kernel_size=3, padding=1)
		self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
	
	def forward(self,x):
		batch_size,seq_len=x.size()
		#print(x.shape)
		x=self.embedding(x)
		#print(x.shape)
		hidden = self.conv1(x.transpose(1,2)).transpose(1,2)
		#print(hidden.shape)
		hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
		#print(hidden.shape)
		hidden = self.conv2(hidden.transpose(1,2)).transpose(1,2)
		hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
		#print(hidden.shape)
		hidden=torch.sum(hidden,dim=0,keepdim=True)
		#print(hidden.shape)
		return hidden


