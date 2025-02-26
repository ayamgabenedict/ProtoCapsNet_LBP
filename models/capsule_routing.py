# models/capsule_routing.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidRouting(nn.Module):
    def __init__(self, num_iterations=3):
        super(SigmoidRouting, self).__init__()
        self.num_iterations = num_iterations

    def forward(self, u_hat):
        batch_size, num_route_nodes, num_capsules, out_dim = u_hat.shape
        logits = torch.zeros(batch_size, num_route_nodes, num_capsules, 1, device=u_hat.device)

        for _ in range(self.num_iterations):
            coupling_coeffs = torch.sigmoid(logits)  # Sigmoid instead of SoftMax
            s_j = (coupling_coeffs * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if _ < self.num_iterations - 1:
                agreement = (u_hat * v_j).sum(dim=-1, keepdim=True)
                logits = logits + agreement

        return v_j.squeeze(1)

    @staticmethod
    def squash(s_j):
        norm = torch.norm(s_j, dim=-1, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (s_j / (norm + 1e-8))
