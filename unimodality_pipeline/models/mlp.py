from torch import nn
from copy import copy
import logging
from typing import List
from ..tools.constants import ACTIVATIONS


### Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )
logger = logging.getLogger(__name__)

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
        self.dimensions = [input_size] + hidden_dims + [output_size]
        
        if len(activations) != len(self.dimensions) - 1:
            if len(activations) > 1:
                raise ValueError(f"Length of activations should be equal to length of hidden dimensions or 1")
            self.activations = self.activations * (len(self.dimensions) - 1)
        self.layers = nn.Sequential()
        for i in range(1, len(self.dimensions) - 1):
            self.layers.append(nn.Sequential(nn.Linear(self.dimensions[i-1], self.dimensions[i]), ACTIVATIONS[self.activations[i-1]]))
        self.layers.append(nn.Sequential(nn.Linear(self.dimensions[len(self.dimensions) - 2], self.dimensions[len(self.dimensions) - 1])))
        self.output_activation = output_activation
        if self.output_activation is not None and self.output_activation.lower() != 'linear':
            self.layers.append(ACTIVATIONS[self.output_activation])

    def forward(self, x):
        return self.layers(x)



