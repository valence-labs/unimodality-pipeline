
import torch
from torch import nn
from abc import abstractmethod
from typing import Tuple

class BaseDoubleHeadModel(nn.Module):
    def __init__(
        self, 
        **kwargs,
        ):
        super().__init__()
        

    @abstractmethod
    def encode_mol(
        self,
        t: torch.Tensor
    ):
        pass
    
    @abstractmethod
    def encode_img(
        self,
        t: torch.Tensor
    ):
        pass
    
    @abstractmethod
    def forward(
        self, 
        x: Tuple[torch.Tensor, torch.Tensor]
        ):
        pass
