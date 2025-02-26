import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeLayer(nn.Module):
    def __init__(self, num_prototypes, prototype_dim):
        super(PrototypeLayer, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, prototype_dim))

    def forward(self, x):
        distances = torch.cdist(x, self.prototypes)
        similarities = -distances
        return similarities
