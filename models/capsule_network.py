import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, routing_iters=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes
        self.routing_iters = routing_iters

        self.W = nn.Parameter(torch.randn(1, num_route_nodes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = torch.zeros(1, self.num_route_nodes, self.num_capsules, 1).to(x.device)
        for iteration in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < self.routing_iters - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), v_j).squeeze(4)
                b_ij = b_ij + a_ij

        return v_j.squeeze(1)

    @staticmethod
    def squash(s_j):
        s_j_norm = torch.norm(s_j, dim=-1, keepdim=True)
        return (s_j_norm**2 / (1 + s_j_norm**2)) * (s_j / s_j_norm)

class CapsuleNetwork(nn.Module):
    def __init__(self, image_channels, conv_out_channels, num_primary_units, primary_unit_size, num_classes, output_unit_size, routing_iters=3):
        super(CapsuleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, conv_out_channels, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_primary_units, num_route_nodes=0, in_channels=conv_out_channels, out_channels=primary_unit_size)
        self.digit_capsules = CapsuleLayer(num_classes, num_route_nodes=num_primary_units, in_channels=primary_unit_size, out_channels=output_unit_size, routing_iters=routing_iters)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return x
