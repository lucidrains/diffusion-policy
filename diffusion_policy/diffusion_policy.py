import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, reduce, repeat

class DiffusionPolicy(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
