import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math


class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    Me = X * Lambda * Y^T
    Mp = Ux * Uy^T
    Parameter: scale of weight d
    Input: edgewise (pairwise) feature X, Y
           pointwise (unary) feature Ux, Uy
    Output: edgewise affinity matrix Me
            pointwise affinity matrix Mp
    Weight: weight matrix Lambda = [[Lambda1, Lambda2],
                                    [Lambda2, Lambda1]]
            where Lambda1, Lambda2 > 0
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.lambda1 = Parameter(Tensor(self.d, self.d))
        self.lambda2 = Parameter(Tensor(self.d, self.d))
        self.relu = nn.ReLU()  # problem: if weight<0, then always grad=0. So this parameter is never updated!
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.lambda1.size(1) * 2)
        self.lambda1.data.uniform_(-stdv, stdv)
        self.lambda2.data.uniform_(-stdv, stdv)
        self.lambda1.data += torch.eye(self.d)
        self.lambda2.data += torch.eye(self.d)

    def forward(self, X, Y, Ux, Uy):
        assert X.shape[1] == Y.shape[1] == 2 * self.d
        assert Ux.shape[1] == Uy.shape[1] == self.d
        lambda1 = self.relu(self.lambda1)
        lambda2 = self.relu(self.lambda2)
        weight = torch.cat((torch.cat((lambda1, lambda2)),
                            torch.cat((lambda2, lambda1))), 1)
        Me = torch.matmul(X.transpose(1, 2), weight)
        Me = torch.matmul(Me, Y)
        Mp = torch.matmul(Ux.transpose(1, 2), Uy)

        return Me, Mp


if __name__ == '__main__':
    import torch.optim as optim

    X = torch.randn((2, 50, 5))
    Y = torch.randn((2, 50, 10))
    U_src = torch.randn((2, 25, 3))
    U_tgt = torch.randn((2, 25, 6))
    Me_gt = torch.ones((2, 5, 10))
    Mp_gt = torch.ones((2, 3, 6))

    criterion = nn.MSELoss()

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.affine = Affinity(25)
            self.lin_x = nn.Linear(50 * 5, 50 * 5, bias=True)
            self.lin_y = nn.Linear(50 * 10, 50 * 10, bias=True)
            self.lin_us = nn.Linear(25 * 3, 25 * 3, bias=True)
            self.lin_ut = nn.Linear(25 * 6, 25 * 6, bias=True)

        def forward(self, X, Y, Us, Ut):
            X = self.lin_x(X.reshape((2, 50 * 5)))
            Y = self.lin_y(Y.reshape((2, 50 * 10)))
            Us = self.lin_us(Us.reshape((2, 25 * 3)))
            Ut = self.lin_ut(Ut.reshape((2, 25 * 6)))
            X = X.reshape((2, 50, 5))
            Y = Y.reshape((2, 50, 10))
            Us = Us.reshape((2, 25, 3))
            Ut = Ut.reshape((2, 25, 6))
            return self.affine(X, Y, Us, Ut)

    model = Net()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for iter in range(100):
        optimizer.zero_grad()
        print('iter {} / {}'.format(iter, 100))
        with torch.set_grad_enabled(True):
            Me, Mp = model(X, Y, U_src, U_tgt)
            loss_e = criterion(Me, Me_gt)
            loss_p = criterion(Mp, Mp_gt)
            loss_e.backward()
            loss_p.backward()
            optimizer.step()

        print('loss_e={:.4f}'.format(loss_e))
        print('loss_p={:.4f}'.format(loss_p))
