# models/proto_capsule_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lbp_layer import LBPLayer
from .kmeans_routing import KMeansRouting
from .capsule_routing import SigmoidRouting
from .prototype_layer import PrototypeLayer

class ProtoCapsuleNet(nn.Module):
    def __init__(self, num_classes, num_prototypes):
        super(ProtoCapsuleNet, self).__init__()
        self.lbp_layer = LBPLayer(num_channels=1)
        self.primary_capsules = nn.Conv2d(1, 32, kernel_size=9, stride=2)  # Feature capsules
        self.kmeans_routing = KMeansRouting(num_capsules=num_classes)
        self.sigmoid_routing = SigmoidRouting(num_iterations=3)
        self.prototype_layer = PrototypeLayer(num_prototypes, 16)  # Prototype size 16

    def forward(self, x):
        x = self.lbp_layer(x)  # LBP Feature Extraction
        x = self.primary_capsules(x)
        x = x.view(x.shape[0], -1, 16)  # Reshape for capsules
        x = self.kmeans_routing(x)  # Apply K-Means Routing
        x = self.sigmoid_routing(x)  # Apply Sigmoid Normalization
        similarities = self.prototype_layer(x)  # Compute ProtoPNet similarities
        return similarities
