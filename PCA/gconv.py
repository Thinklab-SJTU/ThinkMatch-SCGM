import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    J = W_size[-1]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output

class Gconv(nn.Module):
    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)
        #self.a_bn = nn.BatchNorm1d(self.num_outputs)
        #self.u_bn = nn.BatchNorm1d(self.num_outputs)


    def forward(self, A, x, norm=True):
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)

        ax = self.a_fc(x)
        #ax = self.a_bn(ax.view(-1, self.num_outputs))
        #ax = ax.view(*x_size[:-1], self.num_outputs)

        ux = self.u_fc(x)
        #ux = self.u_bn(ux.view(-1, self.num_outputs))
        #ux = ux.view(*x_size[:-1], self.num_outputs)

        x = torch.bmm(A, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)

        return x


class GconvEdge(nn.Module):
    def __init__(self, in_features, out_features, in_edges, out_edges=None):
        super(GconvEdge, self).__init__()
        if out_edges is None:
            out_edges = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.out_edges = out_edges
        # self.node_fc = nn.Linear(in_features, out_features // self.out_edges)
        self.node_fc = nn.Linear(in_features, out_features)
        self.node_sfc = nn.Linear(in_features, out_features)
        self.edge_fc = nn.Linear(in_edges, self.out_edges)

    def forward(self, A, emb_node, emb_edge, mode=1):
        """
        A: connectivity matrix {0,1}^(n*n)
        emb_node: node embedding n*d
        emb_edge: edge embedding n*n*d
        """
        # pdb.set_trace()
        # mode = 1

        if mode == 1:
            node_x = self.node_fc(emb_node)
            node_sx = self.node_sfc(emb_node)
            edge_x = self.edge_fc(emb_edge)

            A = A.unsqueeze(-1)
            # pdb.set_trace()
            A = torch.mul(A.expand_as(edge_x), edge_x)

            node_x = torch.matmul(A.transpose(2, 3).transpose(1, 2),
                                  node_x.unsqueeze(2).transpose(2, 3).transpose(1, 2))
            # node_x = [torch.sum(dsnorm(A[:, :, :, f:f+1]) *  node_x.unsqueeze(1), dim=2)
            #      for f in range(A.shape[-1])]
            # node_x = torch.cat(node_x, dim=-1)
            node_x = node_x.squeeze(-1).transpose(1, 2)
            node_x = F.relu(node_x) + F.relu(node_sx)
            edge_x = F.relu(edge_x)

            return node_x, edge_x
        elif mode == 2:
            # with DPP
            node_x = self.node_fc(emb_node)
            node_sx = self.node_sfc(emb_node)
            edge_x = self.edge_fc(emb_edge)

            d_x = node_x.unsqueeze(1) - node_x.unsqueeze(2)
            d_x = torch.sum(d_x ** 2, dim=3, keepdim=False)
            d_x = torch.exp(-d_x)

            A = A.unsqueeze(-1)
            # pdb.set_trace()
            A = torch.mul(A.expand_as(edge_x), edge_x)

            node_x = torch.matmul(A.transpose(2, 3).transpose(1, 2),
                                  node_x.unsqueeze(2).transpose(2, 3).transpose(1, 2))
            node_x = node_x.squeeze(-1).transpose(1, 2)
            node_x = F.relu(node_x) + F.relu(node_sx)
            edge_x = F.relu(edge_x)
            return node_x, edge_x, d_x


'''
class Gconv(nn.Module):
    def __init__(self, feature_maps, J):
        super(Gconv, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_outputs = feature_maps[2]
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fc2 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.u_fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.u_fc2 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        x0 = input[1]
        x = gmul(input) # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x1 = F.relu(self.fc1(x)) + F.relu(self.u_fc1(x0)) # has size (bs, N, num_outputs)
        x2 = self.fc2(x) + self.u_fc2(x0)
        x = torch.cat((x1, x2), dim=-1)
        x = self.bn(x.view(-1, self.num_outputs))
        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x

class GNN(nn.Module):
    def __init__(self, in_features, num_features, num_layers, J):
        super(GNN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [in_features, num_features]
        #self.featuremap_mi = [num_features, num_features]
        #self.featuremap_end = [num_features, num_features]
        self.layer0 = Gconv(*self.featuremap_in)
        #for i in range(num_layers):
        #    module = Gconv(*self.featuremap_mi)
        #    self.add_module('layer{}'.format(i + 1), module)
        #self.layerlast = Gconv_last(self.featuremap_end, J)

    def forward(self, A, x0):
        cur = self.layer0(A, x0)
        #for i in range(self.num_layers):
        #    cur = self._modules['layer{}'.format(i+1)](cur)
        #out = self.layerlast(cur)
        out = cur
        return out[1]
'''

class Siamese_Gconv(nn.Module):
    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, g2):
        emb1 = self.gconv(*g1)
        emb2 = self.gconv(*g2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2

class Siamese_GconvEdge(nn.Module):
    def __init__(self, in_features, num_features, in_edges, out_edges=None):
        super(Siamese_GconvEdge, self).__init__()
        self.in_feature = in_features
        self.gconv1 = GconvEdge(in_features, num_features, in_edges, out_edges)
        self.gconv2 = GconvEdge(in_features, num_features, in_edges, out_edges)

    def forward(self, g1, g2):
        # pdb.set_trace()
        emb1, emb_edge1 = self.gconv1(*g1)
        emb2, emb_edge2 = self.gconv2(*g2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2, emb_edge1, emb_edge2
