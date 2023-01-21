import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
batch_size = 4 # number of independent sequences to run in parallel
block_size = 8 # max context lenght of the sequences
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'