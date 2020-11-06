[![CircleCI](https://circleci.com/gh/Neural-Space/genos.svg?style=svg&circle-token=3658f580f8183f441023a1a4234716410bd74338)](https://app.circleci.com/pipelines/github/Neural-Space)

# Genos
Instantiate objects and call functions using dictionary configs in Python using Genos.

# Install

## Pip
```bash
pip install genos
```

## Poetry
```bash
poetry add genos
```
# Dev Setup

## Prerequisites

- Python >=3.5
- Tested on Mac 10.15.6 Catalina, Ubuntu 18.04

## installation
```shell script
# clone the repo
$ git clone https://github.com/Neural-Space/genos.git
# Install system-level dependencies
$ make install-system-deps
$ # Install environment level dependencies
make install-deps
```

### Testing and Code formating

```
# run the tests to make sure everything works
make unit-test
# check coverage of the code
make test-coverage
```

# Contribution guide
Read contrib guide [here](https://github.com/Neural-Space/genos/blob/%232-advanced-docs/CONTRIBUTING.md).

# Basic Usage

The following examples will show how this library can make your life easier. First, let's consider a basic example where we simply instantiate a single class. 

```python
class King:
    def __init__(self, name:str, queen:str, allegiance:str):
        self.name = name
        self.queen = queen
        self.allegiance = allegiance

    def __repr__(self):
        return f"Name:{self.name}\nQueen:{self.queen}\nAllegiance:{self.allegiance}"
        
    def print_name(self):
        print(self.name)
```
We need to pass 3 parameters to instantiate this class. Note that these classes are located in the `/examples/example.py` file and this will change according to your folder-structure. So, let's say we wish to instantiate a `King` object for Eddard Stark because, afterall, _Winter is coming._
```python
from genos import recursive_instantiate

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
```

Well, this seemed quite simple. But rarely are things so simple in life. Consider another class that takes an instance of `King` as a parameter.

```python
class House:
    def __init__(self, king:King, home:str, sigil:str):
        self.king = king
        self.home = home
        self.sigil = sigil

    def __repr__(self):
        return f"King:{self.king.name}\nHome:{self.home}\nSigil:{self.sigil}"

```
This is where recursive instantiation comes into action. To initialize an object for this class, we can very easily create a nested dictionary and pass it to our magic method. Of course, we'll be instantiating an object for House Stark.
```python
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
# Sigil:Direwolf
```
# Advanced 

### ML example
Such workflows where we need to instantiate multiple classes recursively is more evident in Deep Learning and related fields. We created this tool to make things easier for us. The following example shows a scenario where you need different components/modules to create your own custom neural network for some specific task. The individual classes are merely wrappers around `PyTorch` functions. Let's get started.

```python
from torch import nn
import torch

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
```
The three classes above will now be used to create a custom neural network. Note carefully that in order to instantiate an `AffineLayer`, we need to pass an object of `ActivationLayer`. The `CustomModel` will comprise of two components: `AffineLayer` and `LSTMLayer`.

```python

class CustomModel(nn.Module):
    
    def __init__(self, affine_layer:AffineLayer, lstm_layer:LSTMLayer):
        super().__init__()
        self.affine_layer = affine_layer
        self.lstm_layer = lstm_layer
    
    def forward(self, x):
        return self.affine_layer(self.lstm_layer(x))
```
The instantiation of this class using genos will be as follows.
```python
from genos import recursive_instantiate

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
```


