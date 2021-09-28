import torch
import numpy as np
# Gurobi must be installed from https://www.gurobi.com/
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import coo_matrix
from itertools import product
from src.utils.pad_tensor import pad_tensor

def gurobi_qap(K, n1, n2, max=True, time_limit=None, Y=None, log_file=[]):
    """
    wrapper function for Gurobi QAP solver to support batched operation
    :param K: (b x n1n2 x n1n2) QAP cost matrix / affinity matrix
    :param n1: (b) number of nodes in graph 1
    :param n2: (b) number of nodes in graph 2
    :param max: if max=True, then the objective is maximized. Else it is minimized.
    :param time_limit: Time limit in seconds. None for infinite time.
    :param Y: (b x n1 x n2) init solution value or None
    :return: solution
    """
    batch_size = K.shape[0]
    res = [
        gurobi_qap_solver(
            K[b],
            n1[b].item(),
            n2[b].item(),
            max,
            time_limit,
            Y[b] if Y is not None else None,
            log_file[b]
        ).to(dtype=K.dtype) for b in range(batch_size)
    ]
    res = pad_tensor(res)
    return torch.stack(res)


def gurobi_qap_solver(K, n1, n2, max=True, time_limit=None, Y=None, log_file=''):
    """
    Gurobi QAP solver
    """
    try:
        device = K.device
        K = K.cpu().numpy()

        m = gp.Model('qap')
        m.setParam('LogToConsole', False)
        m.setParam('LogFile', log_file)

        if time_limit is not None:
            m.setParam('TimeLimit', time_limit)

        x = m.addMVar(n1 * n2, vtype=GRB.BINARY, name='x')
        m.setObjective(x @ K @ x, GRB.MAXIMIZE if max else GRB.MINIMIZE)

        col_starts = (np.arange(n2) * n1).tolist()
        rows = list(range(n1))

        c1_x = np.zeros(n1 * n2, dtype=int)
        c1_y = np.zeros(n1 * n2, dtype=int)
        c2_x = np.zeros(n1 * n2, dtype=int)
        c2_y = np.zeros(n1 * n2, dtype=int)
        for idx, (i, j) in enumerate(product(col_starts, rows)):
            c1_x[idx] = j
            c1_y[idx] = i + j
            c2_x[idx] = i // n1
            c2_y[idx] = i + j
        c1 = coo_matrix((np.ones(n1 * n2, dtype=int), (c1_x, c1_y)))
        c2 = coo_matrix((np.ones(n1 * n2, dtype=int), (c2_x, c2_y)))
        m.addConstr(c1 @ x == 1, 'c1')
        m.addConstr(c2 @ x <= 1, 'c2')
        if Y is not None:
            for y, v in zip(Y, m.getVars()):
                v.start = y
            m.update()

        m.optimize()

        res = [v.X for v in m.getVars()]
        res = np.array(res).astype(int).reshape(n2, n1).transpose()
        return torch.from_numpy(res).to(device)
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))


if __name__ == '__main__':
    from torch import Tensor
    def kronecker_torch(t1: Tensor, t2: Tensor):
        batch_num = t1.shape[0]
        t1dim1, t1dim2 = t1.shape[1], t1.shape[2]
        t2dim1, t2dim2 = t2.shape[1], t2.shape[2]
        t1 = t1.reshape(batch_num, -1, 1)
        t2 = t2.reshape(batch_num, 1, -1)
        tt = torch.bmm(t1, t2)
        tt = tt.reshape(batch_num, t1dim1, t1dim2, t2dim1, t2dim2)
        tt = tt.permute([0, 1, 3, 2, 4])
        tt = tt.reshape(batch_num, t1dim1 * t2dim1, t1dim2 * t2dim2)
        return tt

    n1 = 10
    n2 = 15

    A1 = (torch.randn(n1, n1))
    #A1 = torch.randint(0, 2, (n1, n1), dtype=torch.float)
    randperm = torch.randperm(n1)
    gt = torch.zeros(n1, n2)
    gt[randperm, torch.arange(n1)] = 1
    A2 = torch.chain_matmul(gt.transpose(0, 1), A1, gt)
    #A2 = A2 + torch.randn_like(A2) / 100
    K = kronecker_torch(A2.unsqueeze(0), A1.unsqueeze(0)).squeeze(0)
    res = gurobi_qap_solver(K, n1, n2).to(dtype=torch.float)
    print(res)
    print(gt)
    print(torch.norm(res - gt))
