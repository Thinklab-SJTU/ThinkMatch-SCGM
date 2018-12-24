import torch
import torch.nn as nn


class BiStochastic(nn.Module):
    """
    BiStochastic Layer turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, epsilon=1e-4):
        super(BiStochastic, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def forward(self, s):
        batch_size = s.shape[0]
        nonzero_mask = (s != 0).to(s.dtype)

        for i in range(self.max_iter):
            if i % 2 == 0:
                # column norm
                ones = torch.ones(batch_size, s.shape[1], s.shape[1], device=s.device)
                tmp = 1 / torch.bmm(ones, s)
                s = s * tmp * nonzero_mask
            else:
                # row norm
                ones = torch.ones(batch_size, s.shape[2], s.shape[2], device=s.device)
                tmp = 1 / (torch.bmm(s, ones) + self.epsilon)
                s = tmp * s * nonzero_mask

        return s


if __name__ == '__main__':
    bs = BiStochastic(max_iter=8, epsilon=1e-4)
    inp = torch.tensor([[[1., 0, 1.],
                         [1., 0, 3.],
                         [2., 0, 1.],
                         [4., 0, 2.]]], requires_grad=True)
    outp = bs(inp, (3, 4))

    print(outp)
    l = torch.sum(outp)
    l.backward()
    print(inp.grad * 1e10)

    outp2 = torch.tensor([[0.1, 0.1, 1],
                          [2, 3, 4.]], requires_grad=True)

    l = torch.sum(outp2)
    l.backward()
    print(outp2.grad)
