"""
Authors:
 - Ayushman Dash <ayushman@neuralspace.ai>
 - Kushal Jain <kushal@neuralspace.ai>
"""

from torch import nn


# Deep Learning Example using PyTorch
class ActivationLayer(nn.Module):
    """
    Gives two choices for activation function: ReLU or Tanh.
    Introduces non-linearity in neural-networks. Usually applied
    after an affine transformation.
    """

    def __init__(self, activation: str):
        super().__init__()

        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(x)


class AffineLayer(nn.Module):
    """
    Performs an affine transformation on the input tensor.
    For an input tensor "x", the output is W.x + b, where W
    is a trainable weight matrix and b is the bias vector.
    """

    def __init__(
        self, in_features: int, out_features: int, activation: ActivationLayer
    ):
        super().__init__()
        self.transform = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.transform(x))


class LSTMLayer(nn.Module):
    """
    A wrapper over LSTM layer.
    """

    def __init__(self, input_size, hidden_size, batch_first, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.dropout(output)


class CustomModel(nn.Module):
    def __init__(self, affine_layer: AffineLayer, lstm_layer: LSTMLayer):
        super().__init__()
        self.affine_layer = affine_layer
        self.lstm_layer = lstm_layer

    def forward(self, x):
        return self.affine_layer(self.lstm_layer(x))
