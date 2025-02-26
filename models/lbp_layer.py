# models/lbp_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class LBPLayer(nn.Module):
    def __init__(self, num_channels=1, radius=1, num_points=8):
        super(LBPLayer, self).__init__()
        self.radius = radius
        self.num_points = num_points
        self.num_channels = num_channels

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        lbp_features = torch.zeros((batch_size, channels, height, width), device=x.device)

        for i in range(batch_size):
            for c in range(channels):
                img = x[i, c].cpu().numpy() * 255  # Convert to numpy
                img = img.astype(np.uint8)
                lbp = cv2.LBP(img, self.num_points, self.radius, method='uniform')
                lbp_features[i, c] = torch.tensor(lbp, device=x.device).float() / 255.0  # Normalize

        return lbp_features
