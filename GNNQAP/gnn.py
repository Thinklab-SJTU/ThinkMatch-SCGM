import torch
import torch.nn as nn
import torch.nn.functional as F

from GMN.bi_stochastic import BiStochastic
from GMN.voting_layer import Voting

from utils.sparse import to_sparse

from collections import Iterable

'''
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
'''

class GNNLayer(nn.Module):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features,
                 sk_channel=False, sk_iter=20, voting_alpha=20):
        super(GNNLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        if sk_channel:
            assert out_node_features == out_edge_features + 1
            self.out_nfeat = out_node_features - 1
            self.sk = BiStochastic(sk_iter)
            self.classifier = nn.Linear(self.out_efeat, 1)
            self.voting_layer = Voting(voting_alpha)
        else:
            assert out_node_features == out_edge_features
            self.out_nfeat = out_node_features
            self.sk = self.classifier = self.voting_layer = None

        self.e_func = nn.Sequential(
            nn.Linear(self.in_efeat + self.in_nfeat, self.out_efeat),
            nn.ReLU(),
            nn.Linear(self.out_efeat, self.out_efeat),
            nn.ReLU()
        )

        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            nn.ReLU()
        )

    def forward(self, A, W, x, n1=None, n2=None, norm=True):
        """
        :param A: adjacent matrix in 0/1 (b x n x n)
        :param W: edge feature tensor (b x n x n x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        W1 = torch.mul(A.unsqueeze(-1), x.unsqueeze(1))
        W2 = torch.cat((W, W1), dim=-1)
        W_new = self.e_func(W2)

        if norm is True:
            A = F.normalize(A, p=1, dim=2)

        #x2 = torch.sum(torch.mul(A.unsqueeze(-1), W_new), dim=2)
        #x3 = torch.cat((x, x2), dim=-1)
        #x_new = self.n_func(x3)
        #W_norm = F.normalize(W_new, p=1, dim=2)
        x1 = self.n_func(x)
        x2 = torch.sum(torch.mul(torch.mul(A.unsqueeze(-1), W_new), x1.unsqueeze(1)), dim=2)
        #x_new += self.n_self_func(x)

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x3 = self.classifier(x2)
            x4 = self.voting_layer(x3.view(x.shape[0], n2.max(), n1.max()).transpose(1, 2), n1, n2)
            x5 = self.sk(x4, n1, n2, dummy_row=True).transpose(2, 1).contiguous()

            x_new = torch.cat((x2, x5.view(x.shape[0], n1.max() * n2.max(), -1)), dim=-1)
        else:
            x_new = x2

        return W_new, x_new


class HyperGNNLayer(nn.Module):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features, eps=0.0001,
                 sk_channel=False, sk_iter=20, voting_alpha=20):
        super(HyperGNNLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.eps = eps
        if sk_channel:
            assert out_node_features == out_edge_features + 1
            self.out_nfeat = out_node_features - 1
            self.sk = BiStochastic(sk_iter)
            self.classifier = nn.Linear(self.out_efeat, 1)
            self.voting_layer = Voting(voting_alpha)
        else:
            assert out_node_features == out_edge_features
            self.out_nfeat = out_node_features
            self.sk = self.classifier = self.voting_layer = None

        self.e_func = nn.Sequential(
            nn.Linear(self.in_efeat + self.in_nfeat, self.out_efeat),
            nn.ReLU(),
            nn.Linear(self.out_efeat, self.out_efeat),
            nn.ReLU()
        )

        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            nn.ReLU()
        )

    def forward(self, A, W, x, n1=None, n2=None, norm=True):
        """wrapper function of forward (support dense/sparse)"""
        if not isinstance(A, Iterable):
            A = [A]
            W = [W]

        W_new = []
        for _A, _W in zip(A, W):
            if type(_W) is tuple or (type(_W) is torch.Tensor and _W.is_sparse):
                _W_new, _x = self.forward_sparse(_A, _W, x, norm=False)
            else:
                _W_new, _x = self.forward_dense(_A, _W, x, norm)
            try:
                x2 += _x
            except NameError:
                x2 = _x
            W_new.append(_W_new)

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x3 = self.classifier(x2)
            x4 = self.voting_layer(x3.view(x.shape[0], n2.max(), n1.max()).transpose(1, 2), n1, n2)
            x5 = self.sk(x4, n1, n2, dummy_row=True).transpose(2, 1).contiguous()
            x_new = torch.cat((x2, x5.view(x.shape[0], n1.max() * n2.max(), -1)), dim=-1)
        else:
            x_new = x2

        return W_new, x_new

    def forward_sparse(self, A, W, x, norm=True):
        """
        :param A: adjacent tensor in 0/1 (b x {n x ... x n})
        :param W: edge feature tensor (b x {n x ... x n} x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        order = len(A.shape) - 1

        if type(W) is tuple:
            W_ind, W_val = W
        elif type(W) is torch.Tensor and W.is_sparse:
            W_ind = W._indices()
            W_val = W._values()
        else:
            raise ValueError('Unknown datatype {}'.format(type(W)))

        x_agg = torch.zeros(W_val.shape[0], self.in_nfeat, device=W_val.device)
        for i in range(order - 1):
            x_agg += x[W_ind[0, :], W_ind[2 + i, :]]
        W_new_val = self.e_func(
            #torch.cat((W_val, torch.zeros(1, 1, device=W_val.device).expand(W_val.shape[0], self.in_nfeat)), dim=-1)
            torch.cat((W_val, x_agg / (order - 1)), dim=-1)

        )
        if norm is True:
            A_sum = torch.sum(A, dim=tuple(range(2, order + 1)), keepdim=True) + self.eps
            A = A / A_sum.expand_as(A)
            assert torch.sum(torch.isnan(A)) == 0

        if not A.is_sparse:
            A = to_sparse(A)

        x1 = self.n_func(x)

        #x_new = torch.mul(A.unsqueeze(-1), W_new)
        tp_val = torch.mul(A._values().unsqueeze(-1), W_new_val)
        for i in range(order - 1):
            #x1_shape = [x1.shape[0]] + [1] * (order - 1 - i) + list(x1.shape[1:])
            #x_new = torch.sum(torch.mul(x_new, x1.view(*x1_shape)), dim=-2)
            tp_val = x1[W_ind[0, :], W_ind[-1-i, :], :] * tp_val

        assert torch.all(W_ind == A._indices())

        x_new = torch.zeros_like(x1)
        x_new.index_put_((W_ind[0, :], W_ind[1, :]), tp_val, True)

        #x_new += self.n_self_func(x)

        return (W_ind, W_new_val), x_new


    def forward_dense(self, A, W, x, norm=True):
        """
        :param A: adjacent tensor in 0/1 (b x {n x ... x n})
        :param W: edge feature tensor (b x {n x ... x n} x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        order = len(A.shape) - 1

        for i in range(order - 1):
            x_newshape = [x.shape[0]] + [1] * order + [x.shape[2]]
            x_newshape[-2 - i] = x.shape[1]
            if i == 0:
                W1 = torch.mul(A.unsqueeze(-1), x.view(x_newshape))
            else:
                W1 += torch.mul(A.unsqueeze(-1), x.view(x_newshape))
        #W_new = self.e_func(torch.cat((W, W1), dim=-1))
        W_new = self.e_func(torch.cat((W, torch.zeros_like(W1)), dim=-1))

        if norm is True:
            A_sum = torch.sum(A, dim=tuple(range(2, order + 1)), keepdim=True)
            A = A / A_sum.expand_as(A)
            A[torch.isnan(A)] = 0

        x1 = self.n_func(x)

        x_new = torch.mul(A.unsqueeze(-1), W_new)
        for i in range(order - 1):
            x1_shape = [x1.shape[0]] + [1] * (order - 1 - i) + list(x1.shape[1:])
            x_new = torch.sum(torch.mul(x_new, x1.view(*x1_shape)), dim=-2)
        #x_new += self.n_self_func(x)

        return W_new, x_new
