import torch
import torch.nn as nn


class PlaneStochastic(nn.Module):
    """
    multi-dimensional Sinkhorn turns the input tensor into a plane-stochastic tensor.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input tensor t
    Output: bi-stochastic matrix t
    """
    def __init__(self, max_iter=10, tau=1., epsilon=1e-4, log_forward=True):
        super(PlaneStochastic, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        self.log_forward = log_forward

    def forward(self, t, ns_list=None, dummy_row=False, dtype=torch.float32):
        batch_size = t.shape[0]
        num_order = len(t.shape) - 1

        # log_t = log(exp(t / self.tau))
        log_t = t / self.tau

        #if dummy_row:
        #    dummy_shape = list(s.shape)
        #    dummy_shape[1] = s.shape[2] - s.shape[1]
        #    ori_nrows = nrows
        #    nrows = ncols
        #    s = torch.cat((s, torch.full(dummy_shape, -float('inf')).to(s.device)), dim=1)
        #    for b in range(batch_size):
        #        s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100

        ret_log_t = torch.full_like(t, -float('inf'))

        for b in range(batch_size):
            tensor_slices = [b]
            if ns_list is not None:
                for o in range(num_order):
                    tensor_slices.append(slice(0, ns_list[o][b]))
            else:
                for o in range(num_order):
                    tensor_slices.append(slice(0, t.shape[o+1]))
            b_log_t = log_t[tensor_slices]

            for i in range(self.max_iter):
                for o in range(num_order):
                    dim_list = list(range(num_order))
                    dim_list.pop(o)
                    log_sum = torch.logsumexp(b_log_t, dim=dim_list, keepdim=True)
                    b_log_t = b_log_t - log_sum

            ret_log_t[tensor_slices] = b_log_t

        #if dummy_row and dummy_shape[1] > 0:
        #    ret_log_s = ret_log_s[:, :-dummy_shape[1]]
        #    for b in range(batch_size):
        #        ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

        return torch.exp(ret_log_t)
