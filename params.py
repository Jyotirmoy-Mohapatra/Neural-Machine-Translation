import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 100

hidden_size = 256

no_of_iterations = 20

teacher_forcing_ratio = 0.5

scratch = ""

home = ""
