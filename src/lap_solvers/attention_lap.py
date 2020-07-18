import torch
import torch.nn as nn


class AttentionLAP(nn.Module):
    """
    Attention-based LAP, given an arbitrary matrix, attention-based LAP outputs a permutation matrix in a greedy way.
    Parameter:
    Input: input matrix s
    Output: bi-stochastic matrix s / perm-matrix p
    """
    def __init__(self, alpha=1.):
        super(AttentionLAP, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=0)

    def forward(self, s, nrows=None, ncols=None, dtype=torch.float32):
        batch_size = s.shape[0]

        assert s.shape[2] >= s.shape[1]

        col_slice = []
        for b in range(batch_size):
            col_slice.append(list(range(0, ncols[b] if ncols is not None else s.shape[1])))

        new_bs = torch.zeros_like(s)
        #new_p = torch.zeros_like(s)

        for r in range(s.shape[1]):
            for b in range(batch_size):
                if nrows is None or r < nrows[b]:
                    new_bs[b, r, col_slice[b]] = self.softmax(s[b, r, col_slice[b]] * self.alpha)
                    max_idx = col_slice[b][torch.argmax(new_bs[b, r, col_slice[b]])]

                    #new_p[b, r, max_idx] = 1
                    col_slice[b].remove(max_idx)

        return new_bs#, new_p
