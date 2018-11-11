import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 5

hidden_size = 256

no_of_iterations = 10000

teacher_forcing_ratio = 0.5

scratch = "/home/jyotirmoy/Desktop/NLP Project/"

home = ""