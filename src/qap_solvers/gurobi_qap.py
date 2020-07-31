import torch
import numpy as np
# Gurobi must be installed from https://www.gurobi.com/
from gurobipy import *


def gurobi_qap(K, n1, n2, Y=None):
    m = Model('qap')
    m.setParam('OutputFlag', False)

    x = m.addVars(n1 * n2, vtype=GRB.BINARY, name='x')
    obj = quicksum(quicksum(x[i] * x[j] * K[i, j] for j in range(n1 * n2)) for i in range(n1 * n2))
    m.setObjective(obj, GRB.MAXIMIZE)

    row_starts = (np.arange(n1) * n2).tolist()
    cols = list(range(n2))

    m.addConstrs((quicksum(x[i + j] for i in row_starts) <= 1
                  for j in cols), 'c0')
    m.addConstrs((quicksum(x[i + j] for j in cols) <= 1
                  for i in row_starts), 'c1')
    if Y is not None:
        for y, v in zip(Y, m.getVars()):
            v.start = y
        m.update()

    m.optimize()

    res = [v.X for v in m.getVars()]
    res = np.array(res).astype(int).reshape(n2, n1).transpose()
    return torch.from_array(res).to(K.device)


if __name__ == '__main__':
    n1 = 15
    n2 = 15
    #K = torch.randn(n1 * n2, n1 * n2)

    A1 = (torch.randn(n1, n1))
    randperm = torch.randperm(n1)
    gt = torch.zeros(n1, n2)
    gt[randperm, torch.arange(n1)] = 1
    A2 = torch.chain_matmul(gt.transpose(0, 1), A1, gt)
    K = kronecker_torch(A2.unsqueeze(0), A1.unsqueeze(0)).squeeze(0)
    res = gurobi_qap(K, n1, n2)
    print(res)
