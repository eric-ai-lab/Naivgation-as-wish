import torch
import torch.nn as nn
from torch import Tensor

class SimplePrompt(nn.Module):
    def __init__(self, channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.global_emb, gain = 1)

    def add(self, x: Tensor):
        return x + self.global_emb

    def mul(self, x: Tensor):
        return x * self.global_emb