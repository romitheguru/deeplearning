import torch
import torch.nn as nn


class PyTorchNN(nn.Module):
    def __init__(self, layers):
        super(PyTorchNN, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
