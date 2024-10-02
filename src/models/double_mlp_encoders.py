import torch
from torch import nn
from copy import copy
from Typing import List, Tuple
from .base_double_head_model import BaseModel
from ..tools.constants import ACTIVATIONS

class MLP(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_dims: List[int],
        activations: List[str], 
        output_size: int,
        output_activation: str,
        ):
        super().__init__()
        self.activations = copy(activations)
        if len(activations) != len(hidden_dims):
            if len(activations) > 1:
                raise ValueError(f"Length of activations should be equal to length of hidden dimensions or 1")
            self.activations = self.activations * len(hidden_dims)
        
        self.dimensions = [input_size] + hidden_dims + [output_size]
        self.layers = nn.Sequential()
        for i in range(1, len(self.dimensions)):
            self.layers.append(nn.Sequential(nn.Linear(self.dimensions[i-1], self.dimensions[i]), ACTIVATIONS[self.activations[i-1]]))
        self.output_activation = output_activation
        if self.output_activation is not None:
            self.layers.append(ACTIVATIONS[self.output_activation])

    def forward(self, x):
        return self.layers(x)



class DoubleMLPEncoders(BaseModel):
    def __init__(
        self, 
        molecule_encoder: MLP, 
        image_encoder: MLP,
        ):
        super().__init__()
        self.molecule_encoder = molecule_encoder
        self.image_encoder = image_encoder
        
    def encode_mol(
        self,
        t: torch.Tensor
    ):
        return self.molecule_encoder(t)
    
    def encode_img(
        self,
        t: torch.Tensor
    ):
        return self.image_encoder(t)
    
    def forward(
        self, 
        x: Tuple[torch.Tensor, torch.Tensor]
        ):
        mol = self.molecule_encoder(x[0])
        img = self.image_encoder(x[1])
        return mol, img
