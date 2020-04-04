import torch
import torch.nn as nn
from utils.hungarian import hungarian
from GMN.bi_stochastic import BiStochastic
from itertools import product, permutations
from HiPPI.spectral_clustering import spectral_clustering
from utils.fgm import kronecker_torch
from GMN.rrwm import RRWM


class HiPPI(nn.Module):
    """
    HiPPI solver for multiple graph matching: Higher-order Projected Power Iteration in ICCV 2019

    This operation does not support batched input, and all input tensors should not have the first batch dimension.

    Parameter: maximum iteration max_iter
               sinkhorn iteration sk_iter
               sinkhorn regularization sk_tau
    Input: multi-graph similarity matrix W
           initial multi-matching matrix U0
           number of nodes in each graph ms
           size of universe d
           (optional) projector to doubly-stochastic matrix (sinkhorn) or permutation matrix (hungarian)
    Output: multi-matching matrix U
    """
    def __init__(self, max_iter=50, sk_iter=20, sk_tau=1/200.):
        super(HiPPI, self).__init__()
        self.max_iter = max_iter
        self.sinkhorn = BiStochastic(max_iter=sk_iter, tau=sk_tau)
        self.hungarian = hungarian

    def forward(self, W, U0, ms, d, projector='sinkhorn'):
        num_graphs = ms.shape[0]

        U = U0
        for i in range(self.max_iter):
            lastU = U
            WU = torch.mm(W, U) #/ num_graphs
            V = torch.chain_matmul(WU, U.t(), WU) #/ num_graphs ** 2

            #V_median = torch.median(torch.flatten(V, start_dim=-2), dim=-1).values
            #V_var, V_mean = torch.var_mean(torch.flatten(V, start_dim=-2), dim=-1)
            #V = V - V_mean
            #V = V / torch.sqrt(V_var)

            #V = V / V_median

            U = []
            m_start = 0
            m_indices = torch.cumsum(ms, dim=0)
            for m_end in m_indices:
                if projector == 'sinkhorn':
                    U.append(self.sinkhorn(V[m_start:m_end, :d], dummy_row=True))
                elif projector == 'hungarian':
                    U.append(self.hungarian(V[m_start:m_end, :d]))
                else:
                    raise NameError('Unknown projector {}.'.format(projector))
                m_start = m_end
            U = torch.cat(U, dim=0)

            #print('iter={}, diff={}, var={}, vmean={}, vvar={}'.format(i, torch.norm(U-lastU), torch.var(torch.sum(U, dim=0)), V_mean, V_var))

            if torch.norm(U - lastU) < 1e-5:
                print(i)
                break

        return U

class IGMGM(nn.Module):
    """
    Iterative Graduated Multi-Graph Matching solver.

    This operation does not support batched input, and all input tensors should not have the first batch dimension.

    Parameter: maximum iteration max_iter
               sinkhorn iteration sk_iter
               initial sinkhorn regularization sk_tau0
               sinkhorn regularization decaying factor sk_beta
               minimum tau value min_tau
               convergence tolerance conv_tal
    Input: multi-graph similarity matrix W
           initial multi-matching matrix U0
           number of nodes in each graph ms
           size of universe n_univ
           (optional) projector to doubly-stochastic matrix (sinkhorn) or permutation matrix (hungarian)
    Output: multi-matching matrix U
    """
    def __init__(self, max_iter=200, sk_iter=20, sk_tau0=0.5, sk_beta=0.5, converge_tol=1e-5, min_tau=1e-2):
        super(IGMGM, self).__init__()
        self.max_iter = max_iter
        self.sk_iter = sk_iter
        self.sk_tau0 = sk_tau0
        self.sk_beta = sk_beta
        self.converge_tol = converge_tol
        self.min_tau = min_tau

    def forward(self, A, W, U0, ms, n_univ, Alpha, Adjs, num_clusters=2):
        return self.forwarddd(A, W, U0, ms, n_univ, Alpha, Adjs, num_clusters)
        #return self.forward_grad_match_cluster(A, W, U0, ms, n_univ, num_clusters)
        #cluster_v, cluster_M = spectral_clustering(Alpha, num_clusters, return_dist=True)
        #cluster_M = (cluster_v.unsqueeze(0) == cluster_v.unsqueeze(1)).to(dtype=Alpha.dtype)
        beta = 0.
        #cluster_M = (1 - beta) * cluster_M + beta
        #cluster_M = torch.cat((torch.cat((torch.ones(8,8), torch.full((8,8),beta))),
        #                       torch.cat((torch.full((8,8),beta), torch.ones(8,8)))),dim=1)
        cluster_M = torch.ones(8,8)
        U = self.forward_backup(A, W, U0, ms, n_univ, cluster_M)

        cluster_v = torch.zeros(1,1)
        return U, cluster_v

    def forwarddd(self, A, W, U0, ms, n_univ, Alpha, Adjs, num_clusters=2):
        num_graphs = ms.shape[0]
        U = U0
        m_indices = torch.cumsum(ms, dim=0)
        #cluster_v = spectral_clustering(Alpha, num_clusters)
        #cluster_M = (cluster_v.unsqueeze(0) == cluster_v.unsqueeze(1)).to(dtype=Alpha.dtype)
        #beta = .8
        #cluster_M = (1 - beta) * cluster_M + beta
        cluster_M = torch.ones(num_graphs, num_graphs, device=A.device)
        beta=0.
        #cluster_M = torch.cat((torch.cat((torch.ones(8, 8), torch.full((8, 8), beta))),
        #                       torch.cat((torch.full((8, 8), beta), torch.ones(8, 8)))), dim=1)

        # matching
        U = self.forward_backup(A, W, U, ms, n_univ, cluster_M)
        for i in range(5):
            lastU = U

            # clustering
            def get_alpha(scale=10):
                Alpha = torch.zeros(num_graphs, num_graphs, device=A.device)
                for idx1, idx2 in product(range(num_graphs), repeat=2):
                    if idx1 == idx2:
                        continue
                    start_x = m_indices[idx1 - 1] if idx1 != 0 else 0
                    end_x = m_indices[idx1]
                    start_y = m_indices[idx2 - 1] if idx2 != 0 else 0
                    end_y = m_indices[idx2]
                    A_i = A[start_x:end_x, start_x:end_x]
                    A_j = A[start_y:end_y, start_y:end_y]
                    #Adj_i = Adjs['{}'.format(idx1)][:(end_x-start_x), :(end_x-start_x)]
                    #Adj_j = Adjs['{}'.format(idx2)][:(end_y-start_y), :(end_y-start_y)]
                    W_ij = W[start_x:end_x, start_y:end_y]
                    U_i = U[start_x:end_x, :]
                    U_j = U[start_y:end_y, :]
                    X_ij = torch.mm(U_i, U_j.t())
                    #Alpha_ij = torch.sum(W_ij * X_ij)
                    #Alpha_ij = torch.exp(-torch.norm(torch.chain_matmul(X_ij.t(), A_i, X_ij) - A_j) / scale)
                    Alpha_ij = torch.sum(W_ij * X_ij) \
                               + torch.exp(-torch.norm(torch.chain_matmul(X_ij.t(), A_i, X_ij) - A_j) / scale)
                    #K = kronecker_torch(A_j.unsqueeze(0), A_i.unsqueeze(0)).squeeze(0) + torch.diagflat(W_ij.t())
                    #v = RRWM()(K.unsqueeze(0), num_src=torch.tensor([A_i.shape[0]]), ns_src=torch.tensor([A_i.shape[0]]), ns_tgt=torch.tensor([A_j.shape[0]]))
                    #S = v.view(v.shape[0], A_i.shape[0], -1).transpose(1, 2).squeeze(0)
                    #X_ij = hungarian(S)
                    #x = X.t().reshape(-1, 1)
                    #Alpha_ij = torch.chain_matmul(x.t(), K, x).item()
                    #X_ij = hungarian(W_ij)
                    #Alpha_ij = torch.sum(X * W_ij)
                    #Alpha_ij = torch.sum(W_ij * X_ij) \
                    #           + torch.exp(-torch.norm(torch.chain_matmul(X_ij.t(), A_i, X_ij) - A_j))
                    Alpha[idx1, idx2] = Alpha_ij
                return Alpha
            Alpha = get_alpha(1)
            Alpha_D = torch.diagflat(torch.sum(Alpha, dim=-1))
            Alpha_L = Alpha_D - Alpha
            cluster_v = spectral_clustering(Alpha, num_clusters, normalized=True)
            if torch.sum(cluster_v) != 8:
                print('!')
            cluster_M = (cluster_v.unsqueeze(0) == cluster_v.unsqueeze(1)).to(dtype=Alpha.dtype)
            beta = 0.
            cluster_M = (1 - beta) * cluster_M + beta
            #cluster_M /= 10

            #for j in range(9):
            #    _cluster_v = spectral_clustering(Alpha, num_clusters)
            #    _cluster_M = (_cluster_v.unsqueeze(0) == _cluster_v.unsqueeze(1)).to(dtype=Alpha.dtype)
            #    _cluster_M = (1 - beta) * _cluster_M + beta
            #    if torch.sum(_cluster_M * Alpha_L) < torch.sum(cluster_M * Alpha_L):
            #        cluster_v = _cluster_v
            #        cluster_M = _cluster_M

            # matching
            U = self.forward_backup(A, W, U, ms, n_univ, cluster_M, projector='hungarian' if i != 0 else 'sinkhorn')

            print(torch.norm(lastU - U))

            if torch.norm(lastU - U) < 1e-5:
                break

        return U, cluster_v

    def forward_grad_match_cluster(self, A, W, U0, ms, n_univ, num_clusters=2):
        num_graphs = ms.shape[0]
        U = U0
        m_indices = torch.cumsum(ms, dim=0)

        cluster_M = torch.ones(num_graphs, num_graphs, device=A.device)

        projector = 'sinkhorn'
        stage = 'matching'
        sinkhorn_tau = self.sk_tau0
        beta = 1.
        for i in range(self.max_iter):
            lastU = U
            V = torch.zeros_like(U)

            for idx1, idx2 in product(range(num_graphs), repeat=2):
                start_x = m_indices[idx1 - 1] if idx1 != 0 else 0
                end_x = m_indices[idx1]
                start_y = m_indices[idx2 - 1] if idx2 != 0 else 0
                end_y = m_indices[idx2]
                A_i = A[start_x:end_x, start_x:end_x]
                A_j = A[start_y:end_y, start_y:end_y]
                W_ij = W[start_x:end_x, start_y:end_y]
                U_i = U[start_x:end_x, :]
                U_j = U[start_y:end_y, :]
                V_i_update = torch.mm(W_ij, U_j) + torch.chain_matmul(A_i, U_i, U_j.t(), A_j, U_j)
                V[start_x:end_x, :] += V_i_update * cluster_M[idx1, idx2]
            U_list = []
            m_start = 0
            for m_end in m_indices:
                if projector == 'hungarian':
                    U_list.append(hungarian(V[m_start:m_end, :n_univ]))
                elif projector == 'sinkhorn':
                    U_list.append(BiStochastic(max_iter=self.sk_iter, tau=sinkhorn_tau)(V[m_start:m_end, :n_univ], dummy_row=True))
                else:
                    raise NameError('Unknown projecter name: {}'.format(projector))
                m_start = m_end
            U = torch.cat(U_list, dim=0)
            if torch.norm(U - lastU) < self.converge_tol:
                #if stage == 'matching':
                    stage = 'clustering'
                    # clustering
                    Alpha = torch.zeros(num_graphs, num_graphs, device=A.device)
                    for idx1, idx2 in product(range(num_graphs), repeat=2):
                        if idx1 == idx2:
                            continue
                        start_x = m_indices[idx1 - 1] if idx1 != 0 else 0
                        end_x = m_indices[idx1]
                        start_y = m_indices[idx2 - 1] if idx2 != 0 else 0
                        end_y = m_indices[idx2]
                        A_i = A[start_x:end_x, start_x:end_x]
                        A_j = A[start_y:end_y, start_y:end_y]
                        W_ij = W[start_x:end_x, start_y:end_y]
                        U_i = U[start_x:end_x, :]
                        U_j = U[start_y:end_y, :]
                        X_ij = torch.mm(U_i, U_j.t())
                        Alpha_ij = torch.sum(W_ij * X_ij) \
                               + torch.exp(-torch.norm(torch.chain_matmul(X_ij.t(), A_i, X_ij) - A_j))
                        Alpha[idx1, idx2] = Alpha_ij
                    cluster_v = spectral_clustering(Alpha, num_clusters)
                    if torch.sum(cluster_v) != 8:
                        print('!')
                    cluster_M = (cluster_v.unsqueeze(0) == cluster_v.unsqueeze(1)).to(dtype=Alpha.dtype)
                    beta = 0.
                    cluster_M = (1 - beta) * cluster_M + beta
                #else:
                    if projector == 'hungarian':
                        #beta -= 0.1
                        #print(i)
                        break
                    elif sinkhorn_tau > self.min_tau:
                        #print(i, sinkhorn_tau)
                        sinkhorn_tau *= self.sk_beta
                        #beta -= 0.1
                    else:
                        #print(i, sinkhorn_tau)
                        projector = 'hungarian'
                    stage = 'matching'


            if i + 1 == self.max_iter:
                U_list = [hungarian(_) for _ in U_list]
                U = torch.cat(U_list, dim=0)

        return U, cluster_v

    def forward_backup(self, A, W, U0, ms, n_univ, cluster_M, projector = 'sinkhorn'):
        num_graphs = ms.shape[0]
        U = U0
        m_indices = torch.cumsum(ms, dim=0)


        sinkhorn_tau = self.sk_tau0
        #beta = 0.9
        for i in range(self.max_iter):
            lastU = U
            V = torch.zeros_like(U)

            '''
            Alpha = torch.zeros(num_graphs, num_graphs, device=A.device)
            for idx1, idx2 in product(range(num_graphs), repeat=2):
                start_x = m_indices[idx1 - 1] if idx1 != 0 else 0
                end_x = m_indices[idx1]
                start_y = m_indices[idx2 - 1] if idx2 != 0 else 0
                end_y = m_indices[idx2]
                A_i = A[start_x:end_x, start_x:end_x]
                A_j = A[start_y:end_y, start_y:end_y]
                W_ij = W[start_x:end_x, start_y:end_y]
                U_i = U[start_x:end_x, :]
                U_j = U[start_y:end_y, :]
                Alpha_ij = torch.sum(U_i * torch.mm(W_ij, U_j)) \
                           + torch.sum(U_i * torch.chain_matmul(A_i, U_i, U_j.t(), A_j, U_j))
                Alpha[idx1, idx2] = Alpha_ij
            cluster_v = spectral_clustering(Alpha, num_clusters)
            cluster_M = (cluster_v.unsqueeze(0) == cluster_v.unsqueeze(1)).to(dtype=Alpha.dtype)
            #beta = sinkhorn_tau * 2
            cluster_M = (1 - beta) * cluster_M + beta
            '''

            #for idx1, idx2 in product(range(num_graphs), repeat=2):
            for idx1, idx2 in product(range(num_graphs), repeat=2):
                if idx1 == idx2:
                    continue
                start_x = m_indices[idx1 - 1] if idx1 != 0 else 0
                end_x = m_indices[idx1]
                start_y = m_indices[idx2 - 1] if idx2 != 0 else 0
                end_y = m_indices[idx2]
                A_i = A[start_x:end_x, start_x:end_x]
                A_j = A[start_y:end_y, start_y:end_y]
                W_ij = W[start_x:end_x, start_y:end_y]
                #K_ij = K['{},{}'.format(idx1, idx2)]
                #K_ij = kronecker_torch(A_j.unsqueeze(0), A_i.unsqueeze(0)).squeeze(0)
                U_i = U[start_x:end_x, :]
                U_j = U[start_y:end_y, :]
                #X_ij = torch.mm(U_i, U_j.t())
                #x_ij = X_ij.t().reshape(-1, 1)
                #v_ij = torch.mm(K_ij, x_ij)
                #V_ij = v_ij.reshape(end_y - start_y, -1).t()
                #V_i_update = torch.mm(V_ij, U_j) + torch.mm(W_ij, U_j)
                V_i_update = torch.mm(W_ij, U_j) + torch.chain_matmul(A_i, U_i, U_j.t(), A_j, U_j)
                V[start_x:end_x, :] += V_i_update * cluster_M[idx1, idx2]
            U_list = []
            m_start = 0
            for m_end in m_indices:
                if projector == 'hungarian':
                    U_list.append(hungarian(V[m_start:m_end, :n_univ]))
                elif projector == 'sinkhorn':
                    U_list.append(BiStochastic(max_iter=self.sk_iter, tau=sinkhorn_tau)(V[m_start:m_end, :n_univ], dummy_row=True))
                else:
                    raise NameError('Unknown projecter name: {}'.format(projector))
                m_start = m_end
            U = torch.cat(U_list, dim=0)
            if torch.norm(U - lastU) < self.converge_tol:
                if projector == 'hungarian':
                    #beta -= 0.1
                    #print(i)
                    break
                elif sinkhorn_tau > self.min_tau:
                    #print(i, sinkhorn_tau)
                    sinkhorn_tau *= self.sk_beta
                    #beta -= 0.1
                else:
                    #print(i, sinkhorn_tau)
                    projector = 'hungarian'
            if i + 1 == self.max_iter:
                U_list = [hungarian(_) for _ in U_list]
                U = torch.cat(U_list, dim=0)

        return U #, cluster_v