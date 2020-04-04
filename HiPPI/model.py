import torch
import torch.nn as nn
import torch.nn.functional as functional

from GMN.bi_stochastic import BiStochastic as Sinkhorn
from GMN.displacement_layer import Displacement
from utils.build_graphs import reshape_edge_feature
from utils.feature_align import feature_align
from utils.fgm import construct_m
from NGM.geo_edge_feature import geo_edge_feature
from PCA.affinity_layer import AffinityInp
from GMN.affinity_layer import InnerpAffinity as QuadInnerpAffinity
from GMN.affinity_layer import GaussianAffinity as QuadGaussianAffinity
from HiPPI.hippi import HiPPI, IGMGM
from utils.hungarian import hungarian
from utils.fgm import kronecker_torch
from HiPPI.spectral_clustering import spectral_clustering

from itertools import combinations, product, chain
import numpy as np

from utils.config import cfg

import GMN.backbone
CNN = eval('GMN.backbone.{}'.format(cfg.BACKBONE))


def pad_tensor(inp):
    assert type(inp[0]) == torch.Tensor
    it = iter(inp)
    t = next(it)
    max_shape = list(t.shape)
    while True:
        try:
            t = next(it)
            for i in range(len(max_shape)):
                max_shape[i] = int(max(max_shape[i], t.shape[i]))
        except StopIteration:
            break
    max_shape = np.array(max_shape)

    padded_ts = []
    for t in inp:
        pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
        pad_pattern[::-2] = max_shape - np.array(t.shape)
        pad_pattern = tuple(pad_pattern.tolist())
        padded_ts.append(functional.pad(t, pad_pattern, 'constant', 0))

    return padded_ts


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.affinity_layer = AffinityInp(cfg.IGMGM.FEATURE_CHANNEL)
        self.tau = cfg.IGMGM.SK_TAU
        self.bi_stochastic = Sinkhorn(max_iter=cfg.IGMGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.IGMGM.SK_EPSILON)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.IGMGM.FEATURE_CHANNEL * 2, alpha=cfg.IGMGM.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        #self.hippi = HiPPI(sk_iter=41, sk_tau=1./20)
        self.univ_factor = torch.tensor(cfg.IGMGM.UNIV_FACTOR)
        self.igmgm = IGMGM(
            max_iter=cfg.IGMGM.MAX_ITER,
            sk_iter=cfg.IGMGM.SK_ITER_NUM, sk_tau0=cfg.IGMGM.INIT_TAU, sk_beta=cfg.IGMGM.BETA,
            converge_tol=cfg.IGMGM.CONVERGE_TOL, min_tau=cfg.IGMGM.MIN_TAU
        )

        if cfg.NGM.EDGE_FEATURE == 'cat':
            self.quad_affinity_layer = QuadInnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            self.quad_affinity_layer = QuadGaussianAffinity(1, sigma=1.)

        self.graph_learning_layer = nn.Sequential(
            nn.Linear(cfg.IGMGM.FEATURE_CHANNEL, cfg.IGMGM.FEATURE_CHANNEL, bias=False),
            nn.ReLU(),
            #nn.Linear(cfg.IGMGM.FEATURE_CHANNEL, cfg.IGMGM.FEATURE_CHANNEL)
        )

        #self.bi_stochastic2 = BiStochastic(max_iter=cfg.NGM.BS_ITER_NUM, epsilon=cfg.NGM.BS_EPSILON, tau=1/20.)

    def forward(self, data, Ps, Gs, Hs, Gs_ref, Hs_ref, KGs, KHs, ns, type='img', pretrain_backbone=False, **kwargs):
        batch_size = data[0].shape[0]
        num_graphs = len(data)
        device = data[0].device

        # extract graph feature
        if type == 'img' or type == 'image':
            data_cat = torch.cat(data, dim=0)
            P_cat = torch.cat(pad_tensor(Ps), dim=0)
            n_cat = torch.cat(ns, dim=0)
            node = self.node_layers(data_cat)
            edge = self.edge_layers(node)
            U = feature_align(node, P_cat, n_cat, cfg.PAIR.RESCALE)
            F = feature_align(edge, P_cat, n_cat, cfg.PAIR.RESCALE)
            feats = torch.cat((U, F), dim=1)
            feats = self.l2norm(feats)
            feats = torch.split(feats, batch_size, dim=0)
        elif type == 'feat' or type == 'feature':
            feats = data
        else:
            raise ValueError('unknown type string {}'.format(type))

        # store features and variables in feat_list
        feat_list = []
        global_feat = []
        iterator = zip(feats, Ps, Gs, Hs, Gs_ref, Hs_ref, ns)
        ms = torch.zeros(batch_size, num_graphs, dtype=torch.int)
        for idx, (feat, P, G, H, G_ref, H_ref, n) in enumerate(iterator):
            global_feat.append(functional.max_pool1d(feat, kernel_size=feat.shape[-1]).squeeze(-1))
            feat_list.append((idx, feat, P, G, H, G_ref, H_ref, n))
            ms[:, idx] = n
        msmax = torch.max(ms, dim=1).values
        mscum = torch.cumsum(ms, dim=1)
        mssum = mscum[:, -1]
        global_feat = torch.stack(global_feat, dim=1)

        # compute multi-adjacency matrix A
        adj_scaling_factor = 1.
        A = [torch.zeros(m.item(), m.item(), device=device) for m in mssum]
        Adj_s = {}
        for idx, feat, P, G, H, G_ref, H_ref, n in feat_list:
            edge_lens = torch.sqrt(torch.sum((P.unsqueeze(1) - P.unsqueeze(2)) ** 2, dim=-1))
            median_lens = torch.median(torch.flatten(edge_lens, start_dim=-2), dim=-1).values
            median_lens = median_lens.unsqueeze(-1).unsqueeze(-1)
            A_ii = torch.exp(- edge_lens ** 2 / median_lens ** 2 / adj_scaling_factor)# todo

            Adj_s['{}'.format(idx)] = torch.bmm(G, H.transpose(1,2))

            # graph learning
            #edge_feat = self.graph_learning_layer(feat.transpose(1, 2))
            #A_ii *= torch.bmm(edge_feat, edge_feat.transpose(1, 2))
            for b in range(batch_size):
                start_idx = mscum[b, idx] - n[b]
                end_idx = mscum[b, idx]
                A[b][start_idx:end_idx, start_idx:end_idx] += A_ii[b, :n[b], :n[b]] - torch.eye(n[b], device=device)
                #As[b]['{}'.format(idx)] = A_ii[b, :n[b], :n[b]]

        Alpha = torch.bmm(global_feat, global_feat.transpose(1, 2))
        #Beta_v = torch.stack([spectral_clustering(Alpha[b], 3) for b in range(batch_size)])
        #Beta = Alpha #(Beta_v.unsqueeze(1) == Beta_v.unsqueeze(2)).to(dtype=Alpha.dtype)
        #Beta_v = Beta[:, 1, :]

        #Alpha_D = torch.stack([torch.diagflat(torch.sum(Alpha[b], dim=-1)) for b in range(batch_size)])
        #Alpha_e, Alpha_v = torch.symeig(Alpha_D - Alpha, eigenvectors=True)
        #topargs = torch.argsort(torch.abs(Alpha_e), descending=True)[-2]
        #Alpha_v = Alpha_v[:, :, 1:2]
        #Alpha_v_indices = torch.topk(Alpha_v, k=num_graphs // 2, dim=-1).indices
        #Beta_v = torch.zeros(batch_size, num_graphs, 1, device=device)
        #for b in range(batch_size):
        #    Beta_v[b, Alpha_v_indices[b]] = 1
        #Beta_v = (Alpha_v > 0).to(dtype=Alpha_v.dtype)
        #Beta = torch.bmm(Beta_v, Beta_v.transpose(1, 2))
        #Beta_v = (Beta_v - 1) * -1
        #Beta += torch.bmm(Beta_v, Beta_v.transpose(1, 2))
        #Beta = Beta * (1 - beta) + beta

        # compute similarity matrix W
        W = [torch.eye(m.item(), device=device) for m in mssum]
        Wds = [torch.eye(m.item(), device=device) for m in mssum]
        P = [torch.eye(m.item(), device=device) for m in mssum]
        for src, tgt in combinations(feat_list, 2):
            src_idx, src_feat, P_src, G_src, H_src, _, __, n_src = src
            tgt_idx, tgt_feat, P_tgt, _, __, G_tgt, H_tgt, n_tgt = tgt
            W_ij = self.affinity_layer(src_feat.transpose(1, 2), tgt_feat.transpose(1, 2))
            #W_ij = torch.exp(10. * (W_ij - 1)) #/ torch.exp(torch.tensor(20.).to(W_ij.device))
            for b in range(batch_size):
                start_x = mscum[b, src_idx] - n_src[b]
                end_x = mscum[b, src_idx]
                start_y = mscum[b, tgt_idx] - n_tgt[b]
                end_y = mscum[b, tgt_idx]
                W_ijb = W_ij[b, :n_src[b], :n_tgt[b]]
                if end_y - start_y >= end_x - start_x:
                    W_ij_ds = self.bi_stochastic(W_ijb, dummy_row=True)
                else:
                    W_ij_ds = self.bi_stochastic(W_ijb.t(), dummy_row=True).t()
                W[b][start_x:end_x, start_y:end_y] += W_ijb
                W[b][start_y:end_y, start_x:end_x] += W_ijb.t()
                Wds[b][start_x:end_x, start_y:end_y] += W_ij_ds
                Wds[b][start_y:end_y, start_x:end_x] += W_ij_ds.t()

                #Ws[b]['{},{}'.format(src_idx, tgt_idx)] = W_ij_ds
                #Ws[b]['{},{}'.format(tgt_idx, src_idx)] = W_ij_ds.t()

                P_ij = hungarian(W_ij[b, :n_src[b], :n_tgt[b]])
                P[b][start_x:end_x, start_y:end_y] += P_ij
                P[b][start_y:end_y, start_x:end_x] += P_ij.t()

        '''
        # compute affinity matrix M
        Ms = {}
        for src, tgt in product(feat_list, repeat=2):
            src_idx, src_feat, P_src, G_src, H_src, _, __, n_src = src
            tgt_idx, tgt_feat, P_tgt, _, __, G_tgt, H_tgt, n_tgt = tgt
            U_src = src_feat[:, :src_feat.shape[1] // 2, :n_src]
            F_src = src_feat[:, src_feat.shape[1] // 2:, :n_src]
            U_tgt = tgt_feat[:, :tgt_feat.shape[1] // 2, :n_tgt]
            F_tgt = tgt_feat[:, tgt_feat.shape[1] // 2:, :n_tgt]

            if cfg.NGM.EDGE_FEATURE == 'cat':
                X = reshape_edge_feature(F_src, G_src, H_src)
                Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)
            elif cfg.NGM.EDGE_FEATURE == 'geo':
                X = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
                Y = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]
            else:
                raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))

            K_G = KGs['{},{}'.format(src_idx, tgt_idx)]
            K_H = KHs['{},{}'.format(src_idx, tgt_idx)]
            Me, Mp = self.quad_affinity_layer(X, Y, U_src, U_tgt)
            Ms['{},{}'.format(src_idx, tgt_idx)] = construct_m(Me, torch.zeros_like(Mp), K_G, K_H)
            #for b in range(batch_size):
            #    start_x = mscum[b, src_idx] - n_src[b]
            #    end_x = mscum[b, src_idx]
            #    start_y = mscum[b, tgt_idx] - n_tgt[b]
            #    end_y = mscum[b, tgt_idx]
            #    A1 = A[b][start_x:end_x, start_x:end_x]
            #    A2 = A[b][start_y:end_y, start_y:end_y]
            #    Ms['{},{}'.format(src_idx, tgt_idx)] = kronecker_torch(A2.unsqueeze(0), A1.unsqueeze(0))
        '''
        '''
        # return pairwise matching
        if pretrain_backbone:
            pred_s = []
            indices = []
            for idx1, idx2 in combinations(range(num_graphs), 2):
                s = []
                for b in range(batch_size):
                    start_x = mscum[b, idx1-1] if idx1 != 0 else 0
                    end_x = mscum[b, idx1]
                    start_y = mscum[b, idx2-1] if idx2 != 0 else 0
                    end_y = mscum[b, idx2]
                    if end_y - start_y >= end_x - start_x:
                        s.append(self.bi_stochastic(W[b][start_x:end_x, start_y:end_y], dummy_row=True))
                    else:
                        s.append(self.bi_stochastic(W[b][start_y:end_y, start_x:end_x], dummy_row=True).t())
                pred_s.append(torch.stack(pad_tensor(s), dim=0))
                indices.append((idx1, idx2))
            return pred_s, indices
        '''

        # compute fused similarity W_bar
        W_bar = []
        for A_, W_ in zip(A, Wds):
            W_bar.append(torch.chain_matmul(W_.t(), A_, W_))# / num_graphs ** 2)
            #W_bar.append(A_)

        '''
        U0 = []
        # compute initial matching U0
        for b in range(batch_size):
            #e, v = torch.symeig(W[b], eigenvectors=True)
            e, v = torch.symeig(P[b], eigenvectors=True)
            topargs = torch.argsort(torch.abs(e), descending=True)[:torch.ceil(msmax[b] * self.univ_factor).to(dtype=torch.int64)] # set universe size here
            v = v[:, topargs]
            U0_b = []
            for i in range(num_graphs):
                if i == 0:
                    start_idx = 0
                else:
                    start_idx = mscum[b, i-1]
                end_idx = mscum[b, i]
                U0_b.append(self.bi_stochastic(v[start_idx:end_idx, :], dummy_row=True))
                #U0_b.append(hungarian(v[start_idx:end_idx, :]))
            U0.append(torch.cat(U0_b, dim=0))
        '''

        # HiPPI
        U = [[] for _ in range(num_graphs)]
        cluster_v = []
        for b in range(batch_size):
            #U_b = self.hippi(W_bar[b], U0[b], ms[b], msmax[b], projector='hungarian')
                                                                                 # set universe size here
            univ_size = torch.ceil(msmax[b] * self.univ_factor).to(dtype=torch.int64)
            #Ms_b = {k: Ms[k][b] for k in Ms}
            Adjs_b = {k: Adj_s[k][b] for k in Adj_s}
            U_b, cluster_v_b = self.igmgm(A[b], Wds[b], torch.rand(torch.sum(ms[b]), univ_size, device=device), ms[b], univ_size, Alpha[b], Adjs_b)
            #U_b, Beta = hippi(A[b], W[b], (Beta * (1 - beta) + beta)[b], U0b, ms[b], torch.ceil(msmax[b] * self.univ_factor).to(dtype=torch.int64))
            #U_b = U0[b]

            #print('-' * 10)
            #U_bref = self.hippi(W_bar[b], U0[b], ms[b], msmax[b], projector='hungarian')
            #diff = 0
            cluster_v.append(cluster_v_b)
            for i in range(num_graphs):
                if i == 0:
                    start_idx = 0
                else:
                    start_idx = mscum[b, i-1]
                end_idx = mscum[b, i]
                U[b].append(U_b[start_idx:end_idx, :])
                #U[b].append(U0[b][start_idx:end_idx, :])
                #diff += torch.norm(hungarian(U_b[start_idx:end_idx, :]) - U_bref[start_idx:end_idx, :]) ** 2

            #print('sk vs hung diff={}'.format(diff))
        cluster_v = torch.stack(cluster_v)

        # collect results
        pred_s = []
        pred_s_discrete = []
        indices = []
        for idx1, idx2 in chain(combinations(range(num_graphs//2), 2), combinations(range(num_graphs//2, num_graphs), 2)):
        #for idx1, idx2 in combinations(range(num_graphs), 2):
            s = []
            for b in range(batch_size):
                start_x = mscum[b, idx1 - 1] if idx1 != 0 else 0
                end_x = mscum[b, idx1]
                start_y = mscum[b, idx2 - 1] if idx2 != 0 else 0
                end_y = mscum[b, idx2]
                if end_y - start_y >= end_x - start_x:
                    s.append(Wds[b][start_x:end_x, start_y:end_y])
                else:
                    s.append(Wds[b][start_y:end_y, start_x:end_x].t())
            pred_s.append(torch.stack(pad_tensor(s), dim=0))
            #s = [hungarian(_) for _ in s]
            s = []
            for b in range(batch_size):
                #s.append(self.bi_stochastic2(torch.mm(U[b][idx1], U[b][idx2].t())))
                s.append(torch.mm(U[b][idx1], U[b][idx2].t()))
            pred_s_discrete.append(torch.stack(pad_tensor(s), dim=0))
            indices.append((idx1, idx2))

        return pred_s, indices, pred_s_discrete, cluster_v #Beta[0, :].unsqueeze(0)


def lawlers_iter(Ms, U0, ms, d):
    U = U0
    for i in range(100):
        lastU = U
        V = torch.zeros_like(U)
        for idx1, idx2 in product(range(num_graphs), repeat=2):
            start1 = mscum[b, idx1 - 1] if idx1 != 0 else 0
            end1 = mscum[b, idx1]
            start2 = mscum[b, idx2 - 1] if idx2 != 0 else 0
            end2 = mscum[b, idx2]
            M_ij = Ms['{},{}'.format(idx1, idx2)][b]
            U_i = U[start1:end1, :]
            U_j = U[start2:end2, :]
            X_ij = torch.mm(U_i, U_j.t())
            x_ij = X_ij.t().reshape(-1, 1)
            v_ij = torch.mm(M_ij, x_ij)
            V_ij = v_ij.reshape(end2 - start2, -1).t()
            V_i = torch.mm(V_ij, U_j)
            V[start1:end1, :] += V_i
        U = []
        m_start = 0
        m_indices = torch.cumsum(ms, dim=0)
        for m_end in m_indices:
            U.append(hungarian(V[m_start:m_end, :d]))
            m_start = m_end
        U = torch.cat(U, dim=0)
        if torch.norm(U - lastU) < 1e-5:
            break
    return U
