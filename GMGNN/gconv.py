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


    def forward(self, A, x):
        x_size = x.size()
        A_norm = F.normalize(A, p=1, dim=-1)

        ax = self.a_fc(x)
        #ax = self.a_bn(ax.view(-1, self.num_outputs))
        #ax = ax.view(*x_size[:-1], self.num_outputs)

        ux = self.u_fc(x)
        #ux = self.u_bn(ux.view(-1, self.num_outputs))
        #ux = ux.view(*x_size[:-1], self.num_outputs)

        x = torch.bmm(A_norm, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)

        return A, x

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
'''

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

class Siamese_GNN(nn.Module):
    def __init__(self, in_features, num_features, num_layers, J):
        super(Siamese_GNN, self).__init__()
        self.gnn = GNN(in_features, num_features, num_layers, J)

    def forward(self, g1, g2):
        emb1 = self.gnn(*g1)
        emb2 = self.gnn(*g2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2

if __name__ == '__main__':
    # test modules
    bs =  4
    num_features = 10
    num_layers = 5
    N = 8
    x = torch.ones((bs, N, num_features))
    W1 = torch.eye(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    W2 = torch.ones(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    J = 2
    W = torch.cat((W1, W2), 3)
    input = [Variable(W), Variable(x)]
    ######################### test gmul ##############################
    # feature_maps = [num_features, num_features, num_features]
    # out = gmul(input)
    # print(out[0, :, num_features:])
    ######################### test gconv ##############################
    # feature_maps = [num_features, num_features, num_features]
    # gconv = Gconv(feature_maps, J)
    # _, out = gconv(input)
    # print(out.size())
    ######################### test gnn ##############################
    # x = torch.ones((bs, N, 1))
    # input = [Variable(W), Variable(x)]
    # gnn = GNN(num_features, num_layers, J)
    # out = gnn(input)
    # print(out.size())
    ######################### test siamese gnn ##############################
    x = torch.ones((bs, N, 1))
    input1 = [Variable(W), Variable(x)]
    input2 = [Variable(W.clone()), Variable(x.clone())]
    siamese_gnn = Siamese_GNN(num_features, num_layers, J)
    out = siamese_gnn(input1, input2)
    print(out.size())