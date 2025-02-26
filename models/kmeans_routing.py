# models/kmeans_routing.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class KMeansRouting(nn.Module):
    def __init__(self, num_capsules, num_iterations=3):
        super(KMeansRouting, self).__init__()
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules

    def forward(self, u_hat):
        batch_size, num_route_nodes, num_capsules, out_dim = u_hat.shape
        centroids = u_hat.mean(dim=1, keepdim=True)  # Initialize centroids

        for _ in range(self.num_iterations):
            distances = torch.norm(u_hat - centroids, dim=-1, keepdim=True)
            cluster_assignments = F.softmax(-distances, dim=2)  # Assign based on similarity
            centroids = (cluster_assignments * u_hat).sum(dim=1, keepdim=True) / cluster_assignments.sum(dim=1, keepdim=True)

        return centroids.squeeze(1)
