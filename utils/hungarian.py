import torch
import scipy.optimize as opt
import numpy as np


def hungarian(s: torch.Tensor):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :return: optimal permutation matrix
    """
    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().numpy() * -1
    for b in range(batch_num):
        row, col = opt.linear_sum_assignment(perm_mat[b])
        perm_mat[b] = np.zeros_like(perm_mat[b])
        perm_mat[b, row, col] = 1
    perm_mat = torch.from_numpy(perm_mat).to(device)

    return perm_mat
