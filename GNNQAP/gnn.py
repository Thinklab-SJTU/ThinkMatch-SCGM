import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features):
        super(GNNLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_nfeat = out_node_features
        self.out_efeat = out_edge_features

        self.n2e_func = nn.Sequential(nn.Linear(self.in_nfeat, self.in_efeat), nn.ReLU())  # node to adjacent edges
        self.e2e_func = nn.Sequential(nn.Linear(self.in_efeat * 2, self.out_efeat), nn.ReLU())  # self-update in edge

        self.e2n_func = nn.Sequential(nn.Linear(self.out_efeat, self.in_nfeat), nn.ReLU())  # edge to adjacent node
        self.n2n_func = nn.Sequential(nn.Linear(self.in_nfeat * 2, self.out_nfeat), nn.ReLU())  # self-update in node


    def forward(self, A, W, x, norm=True):
        """
        :param A: adjacent matrix in 0/1 (b x n x n)
        :param W: edge feature tensor (b x n x n x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        x1 = self.n2e_func(x)
        W1 = torch.mul(A.unsqueeze(-1), x1.unsqueeze(1))
        W2 = torch.cat((W, W1), dim=-1)
        W_new = self.e2e_func(W2)

        if norm is True:
            A = F.normalize(A, p=1, dim=2)

        x2 = torch.sum(torch.mul(A.unsqueeze(-1), self.e2n_func(W_new)), dim=2)
        x3 = torch.cat((x, x2), dim=-1)
        x_new = self.n2n_func(x3)

        return W_new, x_new