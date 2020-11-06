from genos import recursive_instantiate
import torch
from torch import nn

class King:
    def __init__(self, name:str, queen:str, allegiance:str):
        self.name = name
        self.queen = queen
        self.allegiance = allegiance

    def __repr__(self):
        return f"Name:{self.name}\nQueen:{self.queen}\nAllegiance:{self.allegiance}"
        
    def print_name(self):
        print(self.name)



ned = {
    "cls": "examples.example.King",
    "params":{
        "name": "Eddard Stark",
        "queen": "Catelyn Stark",
        "allegiance": "Robert Baratheon"
    }
}

obj = recursive_instantiate(ned)
print(obj)
# Name:Eddard Stark
# Queen:Catelyn Stark
# Allegiance:Robert Baratheon

obj.print_name()
# Eddard Stark

#################################################################################


class House:
    def __init__(self, king:King, home:str, sigil:str):
        self.king = king
        self.home = home
        self.sigil = sigil

    def __repr__(self):
        return f"King:{self.king.name}\nHome:{self.home}\nSigil:{self.sigil}"


stark = {
    "cls": "examples.example.House",
    "params": {
        "king":{
            "cls": "examples.example.King",
            "params":{
                "name": "Eddard Stark",
                "queen": "Catelyn Stark",
                "allegiance": "Robert Baratheon"
                }
        },
        "home":"Winterfell",
        "sigil":"Direwolf"
    }
}

obj = recursive_instantiate(stark)
print(obj)
# output
# King:Eddard Stark
# Home:Winterfell
# Sigil:Direwolf'

#################################################################################

# Deep Learning Example using PyTorch

class ActivationLayer(nn.Module):
    '''
    Gives two choices for activation function: ReLU or Tanh.
    Introduces non-linearity in neural-networks. Usually applied 
    after an affine transformation.
    '''
    def __init__(self, activation:str):
        super().__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
    
    def forward(self, x):
        return self.activation(x)
    

class AffineLayer(nn.Module):
    '''
    Performs an affine transformation on the input tensor.
    For an input tensor "x", the output is W.x + b, where W 
    is a trainable weight matrix and b is the bias vector.
    '''
    def __init__(self, in_features:int, out_features:int, activation: ActivationLayer):
        super().__init__()
        self.transform = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation
    
    def forward(self, x):
        return self.activation(self.transform(x))
        
class LSTMLayer(nn.Module):
    '''
    A wrapper over LSTM layer.
    '''
    def __init__(self, input_size, hidden_size, batch_first, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.dropout(output)        
        


class CustomModel(nn.Module):
    
    def __init__(self, affine_layer:AffineLayer, lstm_layer:LSTMLayer):
        super().__init__()
        self.affine_layer = affine_layer
        self.lstm_layer = lstm_layer
    
    def forward(self, x):
        return self.affine_layer(self.lstm_layer(x))


custom_obj = \
{
    "cls": "examples.example.CustomModel",
    "params": {
        "affine_layer": {
            "cls": "examples.example.AffineLayer",
            "params": {
                "in_features": 256,
                "out_features": 256,
                "activation": {
                    "cls": "examples.example.ActivationLayer",
                    "params": {
                        "activation": "relu"
                    }
                }
            },
            "cls": "sexamples.example.LSTMLayer",
            "params":{
                "input_size": 256,
                "hidden_size":256,
                "batch_first":True,
            }
        }
        
    }
}


model = recursive_instantiate(custom_obj)
x = torch.randn(32, 100, 256)
out = model(x)
print(out.shape)
# [32, 100, 256]

#################################################################################
