<p align="center"><img src="https://docs.google.com/drawings/d/e/2PACX-1vQ65pCWYymvOtQXCUSsWfq0xeaE6fFmQ-_QT003eZRbeLiwKoE7xvDe6fCeuBx_ha7aCjpN3mu_WLl9/pub?w=1536&h=480" alt="logo" width="70%" /></p>  
<p align="center">
  <a href="https://app.circleci.com/pipelines/github/Neural-Space">
    <img src="https://circleci.com/gh/Neural-Space/genos.svg?style=shield&circle-token=3658f580f8183f441023a1a4234716410bd74338" alt="CircleCI" />
  </a>
  <a href="https://lgtm.com/projects/g/Neural-Space/genos/alerts/">
    <img src="https://img.shields.io/lgtm/alerts/g/Neural-Space/genos.svg?logo=lgtm&logoWidth=18" alt="Total alerts" />
  </a>
  <a href="https://lgtm.com/projects/g/Neural-Space/genos/context:python">
    <img src="https://img.shields.io/lgtm/grade/python/g/Neural-Space/genos.svg?logo=lgtm&logoWidth=18" alt="Language grade: Python" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="mit" />
  </a>
</p>

--------------------------------------



Instantiate objects and call functions using dictionary configs in Python using Genos. 
This package was originally developed to help python developers in making configurable software components. 

While [Hydra](https://github.com/facebookresearch/hydra) lets you instantiate objects and functions, it doesn't support recursive instantiation. 
Plus, Hydra is mostly used for config management. 
So, we decided to build Genos by referring to Hydra and added the functionality of recursive instantiation. E.g.,

**Install Genos**
```bash
pip install genos
```

Instantiate a Python Class using Genos

```python
from genos import recursive_instantiate

ned = {
    "cls": "genos.examples.King",
    "params":{
        "name": "Eddard Stark",
        "queen": "Catelyn Stark",
        "allegiance": "Robert Baratheon"
    }
}

obj = recursive_instantiate(ned)
print(obj)
```   

# Contributors

[![](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/images/0)](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/links/0)[![](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/images/1)](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/links/1)[![](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/images/2)](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/links/2)[![](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/images/3)](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/links/3)[![](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/images/4)](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/links/4)[![](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/images/5)](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/links/5)[![](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/images/6)](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/links/6)[![](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/images/7)](https://sourcerer.io/fame/kushalj001/Neural-Space/genos/links/7)

# Install

### Pip
```bash
pip install genos
```

### Poetry
```bash
poetry add genos
```

# Basic Usage

The following examples will show how this library can make your life easier by letting you instantiate python objects from dictionaries.
First, let's consider a basic example where we simply instantiate a single class. 

### Single Class Instantiation
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

We need to pass 3 parameters to instantiate this class. 
Note that these classes are located in the `genos.examples.*` subpackage. 
So, let's say we wish to instantiate a `King` object for Eddard Stark because, afterall, _Winter is coming._

```python
from genos import recursive_instantiate

ned = {
    "cls": "genos.examples.King",
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
```

### Recursive Class Instantiation

Well, this seemed quite simple. 
But rarely are things so simple in life. 
Consider another class that takes an instance of `King` as a parameter.

```python
class House:
    def __init__(self, king:King, home:str, sigil:str):
        self.king = king
        self.home = home
        self.sigil = sigil

    def __repr__(self):
        return f"King:{self.king.name}\nHome:{self.home}\nSigil:{self.sigil}"

```

This is where recursive instantiation comes into action. 
To initialize an object for this class, we can very easily create a nested dictionary and pass it to our magic method. 
Of course, we'll be instantiating an object for House Stark.
```python
from genos import recursive_instantiate

stark = {
    "cls": "genos.examples.House",
    "params": {
        "king":{
            "cls": "genos.examples.King",
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

### Instantiation Using Positional Arguments  
The examples shown above always use keyword arguments to instantiate the classes. But we can also choose to simply pass in the positional arguments as shown below.  

```python
from genos import recursive_instantiate

stark = {
    "cls": "genos.examples.House",
    "args": [
        {
            "cls": "genos.examples.King",
            "params":{
                "name": "Eddard Stark",
                "queen": "Catelyn Stark",
                "allegiance": "Robert Baratheon"
                }
        },
        "Winterfell",
        "Direwolf"
    ]
}

obj = recursive_instantiate(stark)
print(obj)
# output
# King:Eddard Stark
# Home:Winterfell
# Sigil:Direwolf
```

### Instantiation Using Positional and Keyword Arguments
The following example makes use of both positional and keyword arguments together to instantiate the `House` class. We do not pass the keyword for the `king` parameter but we do so for the following parameters:`home` and `sigil`.
```python
from genos import recursive_instantiate

stark = {
    "cls": "genos.examples.House",
    "args": [
        {
            "cls": "genos.examples.King",
            "params":{
                "name": "Eddard Stark",
                "queen": "Catelyn Stark",
                "allegiance": "Robert Baratheon"
                }
        }
    ],
    "params": {
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

### Call A Function
Just like we classes, we can also instantiate functions by calling `recursive_instantiate`. The following example shows how we can instantiate and call a simple `multiply` function using `genos`.
```python
from genos import recursive_instantiate

function_call = {
    "cls": "genos.examples.multiply",
    "args": [12, 1.3]
}

result = recursive_instantiate(function_call)
print(result)
# output
# 15.600000000000001
```

# Advanced Usage

### Deep Learning Example using PyTorch

For running the following examples you will need to install `Pytorch`.

```shell script
pip install torch
```

Such workflows where we need to instantiate multiple classes recursively is more evident in Deep Learning and related fields. 
NeuralSpace has been actively working in this space, building tools for Natural Language Processing (NLP). We have created this tool to make things easier for us. 
The following example shows a scenario where you need different components/modules to create your own custom neural network for some specific task. The individual classes are merely wrappers around `PyTorch` functions. Let's get started.

The following example classes can be found in `genos.examples.complex_examples.py`.

```python
from torch import nn

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
The three classes above will now be used to create a custom neural network. 
Note carefully that in order to instantiate an `AffineLayer`, we need to pass an object of `ActivationLayer`. 
The `CustomModel` will comprise of two components: `AffineLayer` and `LSTMLayer`.

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
    "cls": "genos.examples.CustomModel",
    "params": {
        "affine_layer": {
            {
                "cls": "examples.example.AffineLayer",
                "params": {
                    "in_features": 256,
                    "out_features": 256,
                    "activation": {
                        "cls": "genos.examples.ActivationLayer",
                        "params": {
                            "activation": "relu"
                        }
                    }
                }
            },
            {
                "cls": "genos.examples.LSTMLayer",
                "params":{
                    "input_size": 256,
                    "hidden_size":256,
                    "batch_first":True,
                }
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

### Get Class Reference from Class Path
If you do not wish to use `genos` for instantiating your functions or classes, you can still use it to find and load different classes within your project structure. The following example shows the usage of `get_class()` function from `genos` to locate the `King` class.
```python
from genos import get_class

class_path = "genos.examples.King"
class_reference = get_class(class_path)

eddard_stark = class_reference(name="Eddard Stark", queen="Catelyn Stark", 
                               allegiance="Robert Baratheon")
print(eddard_stark)
# Name:Eddard Stark
# Queen:Catelyn Stark
# Allegiance:Robert Baratheon
```


### Get Function Reference from Class Path
Similar to the `get_class()` method above, you can also use `get_method()` function from `genos` to find functions in your project structure and instantiate them normally.
```python
from genos import get_method

method_path = "genos.examples.multiply"
method_reference = get_method(method_path)

result = method_reference(2, 3.5)
print(result)
# 7.0
```


# Dev Setup

## Prerequisites

- Python >=3.7, <4
- Tested on Mac 10.15.6 Catalina, Ubuntu 18.04


## Install Bleeding Edge Version 

```shell script
# clone the repo
$ git clone https://github.com/Neural-Space/genos.git
# Install system-level dependencies
$ make install-system-deps
 # Install environment level dependencies
$ make install-deps
```

### Testing and Code formatting

```shell script
# run the tests to make sure everything works
$ make unit-test

# check coverage of the code
$ make test-coverage
```

# Contribution guide
Read the contribution guideline over [here](https://github.com/Neural-Space/genos/blob/main/CONTRIBUTING.md).

# Attribution
Icons made by <a href="https://www.flaticon.com/authors/skyclick" title="Skyclick">Skyclick</a> from <a href="https://www.flaticon.com/" title="Flaticon"> www.flaticon.com</a>
