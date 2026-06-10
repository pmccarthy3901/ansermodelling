import torch
import torch.nn as nn 

class FFNN(nn.Module):
    '''
    Feed Forward Neural Network


    Parameters
    ---------- 
    input_dim : int 
        Input array dimension (8 for Anser)
    output_dim : int 
        Output array dimension (5 for Anser)
    hidden_dims : list[int]
        List of hidden dimensions (default [64])
    dropout : float
        Dropout percentage 
    '''

    def __init__(self,
                 input_dim : int = 8,
                 output_dim : int = 5,
                 hidden_dims : list[int] = [64],
                 dropout : float = 0.0):
        
        super().__init__() 

        layer_sizes = [input_dim] + hidden_dims
        layers: list[nn.Module] = []

        for in_dim,out_dim in zip(layer_sizes[:-1],layer_sizes[1:]):
            layers.append(nn.Linear(in_dim,out_dim))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.network = nn.Sequential(*layers)


    def forward(self,x):
        return self.network(x)

