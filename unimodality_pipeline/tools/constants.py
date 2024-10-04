import torch.nn as nn

AGGREGATIONS = {"mean", "sum"}

ACTIVATIONS_KEYS = {"relu", "gelu", "selu", "softmax", "sigmoid" , "tanh", "linear"}

ACTIVATIONS = {
    "relu": nn.ReLU(), 
    "gelu": nn.GELU(), 
    "selu": nn.SELU(), 
    "softmax": nn.Softmax(), 
    "sigmoid": nn.Sigmoid(), 
    "tanh": nn.Tanh()
    }

TEST_EXPERIMENTS = [41188, 45477, 46079, 46047, 35078, 44194, 44193, 46226, 59644, 59632, 47344, 55500]

