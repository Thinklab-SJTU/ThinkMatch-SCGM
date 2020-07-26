import torch
import torch.nn as nn
import torch.nn.functional as functional

import numpy as np

from src.lap_solvers.sinkhorn import Sinkhorn as Sinkhorn
from models.GMN.displacement_layer import Displacement
from src.feature_align import feature_align
from models.PCA.affinity_layer import AffinityInp
from models.GMN.affinity_layer import InnerpAffinity as QuadInnerpAffinity
from models.GMN.affinity_layer import GaussianAffinity as QuadGaussianAffinity
from src.gconv import Siamese_Gconv
from models.GANN.ga_mgmc import GA_MGMC
from src.lap_solvers.hungarian import hungarian
from src.utils.pad_tensor import pad_tensor

from itertools import combinations, product, chain

from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.affinity_layer = AffinityInp(cfg.GANN.FEATURE_CHANNEL)
        self.tau = cfg.GANN.SK_TAU
        self.bi_stochastic = Sinkhorn(max_iter=cfg.GANN.SK_ITER_NUM,
                                      tau=self.tau, epsilon=cfg.GANN.SK_EPSILON, batched_operation=False)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.GANN.FEATURE_CHANNEL * 2, alpha=cfg.GANN.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        #self.hippi = GANN(sk_iter=41, sk_tau=1./20)
        self.univ_size = torch.tensor(cfg.GANN.UNIV_SIZE)
        self.quad_weight = cfg.GANN.QUAD_WEIGHT
        self.cluster_quad_weight = cfg.GANN.CLUSTER_QUAD_WEIGHT
        self.gamgm = GA_MGMC(
            max_iter=cfg.GANN.MAX_ITER,
            sk_iter=cfg.GANN.SK_ITER_NUM, sk_tau0=cfg.GANN.INIT_TAU, sk_gamma=cfg.GANN.GAMMA,
            cluster_beta=cfg.GANN.BETA,
            converge_tol=cfg.GANN.CONVERGE_TOL, min_tau=cfg.GANN.MIN_TAU, projector0=cfg.GANN.PROJECTOR
        )

        if cfg.NGM.EDGE_FEATURE == 'cat':
            self.quad_affinity_layer = QuadInnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            self.quad_affinity_layer = QuadGaussianAffinity(1, sigma=1.)

        self.graph_learning_layer = nn.Sequential(
            nn.Linear(cfg.GANN.FEATURE_CHANNEL, cfg.GANN.FEATURE_CHANNEL, bias=False),
            nn.ReLU(),
            #nn.Linear(cfg.GANN.FEATURE_CHANNEL, cfg.GANN.FEATURE_CHANNEL)
        )

        self.gnn_layer = 3
        GNN_FEAT = 2048
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.GANN.FEATURE_CHANNEL, GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(GNN_FEAT, GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        #self.bi_stochastic2 = BiStochastic(max_iter=cfg.NGM.BS_ITER_NUM, epsilon=cfg.NGM.BS_EPSILON, tau=1/20.)
    def forward(self, data, Ps, Gs, Hs, Gs_ref, Hs_ref, KGs, KHs, ns, gt_cls=None, type='img', pretrain_backbone=False, num_clusters=2, return_cluster=False, **kwargs):
        """
        wrapper function of forward pass, ground truth class information is input to output intra-class matching results
        """
        assert num_clusters == 1 or data[0].shape[0] == 1, "Only batch size == 1 is allowed given multiple clusters"
        U, cluster_v, Wds, mscum = \
            self.real_forward(data, Ps, Gs, Hs, Gs_ref, Hs_ref, KGs, KHs, ns, type, pretrain_backbone, num_clusters)
        if gt_cls is not None:
            cls_indicator = []
            for b in range(len(gt_cls[0])):
                cls_indicator.append([])
                for i in range(len(gt_cls)):
                    cls_indicator[b].append(gt_cls[i][b])
        else:
            cls_indicator = cluster_v.cpu().numpy().tolist()
        pred_s, indices, pred_s_discrete = \
            self.collect_in_class_matching_wrapper(U, Wds, mscum, cls_indicator[0])
        if return_cluster:
            return pred_s, indices, pred_s_discrete, cluster_v
        else:
            return pred_s, indices, pred_s_discrete


    def real_forward(self, data, Ps, Gs, Hs, Gs_ref, Hs_ref, KGs, KHs, ns, type='img', pretrain_backbone=False, num_clusters=2, **kwargs):
        """
        the real forward function.
        :return U: stacked multi-matching matrix
        :return cluster_v: clustering indicator vector
        :return Wds: doubly-stochastic pairwise matching results
        :return mscum: cumsum of number of nodes in graphs
        """
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
            feats[torch.isnan(feats)] = 0.

            # gnn
            #feats = feats.transpose(1, 2)
            #G_cat = torch.cat(pad_tensor(Gs), dim=0)
            #H_cat = torch.cat(pad_tensor(Hs), dim=0)
            #A_cat = torch.bmm(G_cat, H_cat.transpose(1, 2))
            #for i in range(self.gnn_layer):
            #    gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            #    feats = gnn_layer([A_cat, feats])
            #feats = feats.transpose(1, 2)

            feats = torch.split(feats, batch_size, dim=0)
        elif type == 'feat' or type == 'feature':
            feats = data
        else:
            raise ValueError('unknown type string {}'.format(type))

        # store features and variables in feat_list
        feat_list = []
        global_feat = []
        iterator = zip(feats, Ps, Gs, Hs, Gs_ref, Hs_ref, ns)
        ms = torch.zeros(batch_size, num_graphs, dtype=torch.int, device=device)
        for idx, (feat, P, G, H, G_ref, H_ref, n) in enumerate(iterator):
            global_feat.append(functional.max_pool1d(feat, kernel_size=feat.shape[-1]).squeeze(-1))
            feat_list.append((idx, feat, P, G, H, G_ref, H_ref, n))
            ms[:, idx] = n
        msmax = torch.max(ms, dim=1).values
        mscum = torch.cumsum(ms, dim=1)
        mssum = mscum[:, -1]
        global_feat = torch.stack(global_feat, dim=1)

        # compute multi-adjacency matrix A
        A = [torch.zeros(m.item(), m.item(), device=device) for m in mssum]
        #Adj_s = {}
        for idx, feat, P, G, H, G_ref, H_ref, n in feat_list:
            edge_lens = torch.sqrt(torch.sum((P.unsqueeze(1) - P.unsqueeze(2)) ** 2, dim=-1)) * torch.bmm(G, H.transpose(1, 2))
            median_lens = torch.median(torch.flatten(edge_lens, start_dim=-2), dim=-1).values
            median_lens = median_lens.unsqueeze(-1).unsqueeze(-1)
            A_ii = torch.exp(- edge_lens ** 2 / median_lens ** 2 / cfg.GANN.SCALE_FACTOR)
            diag_A_ii = torch.diagonal(A_ii, dim1=-2, dim2=-1)
            diag_A_ii[:] = 0
            #A_ii = edge_lens ** 2 / median_lens ** 2 / cfg.GANN.SCALE_FACTOR #todo

            #Adj_s['{}'.format(idx)] = torch.bmm(G, H.transpose(1,2))

            # graph learning
            #edge_feat = self.graph_learning_layer(feat.transpose(1, 2))
            #A_ii *= torch.bmm(edge_feat, edge_feat.transpose(1, 2))
            for b in range(batch_size):
                start_idx = mscum[b, idx] - n[b]
                end_idx = mscum[b, idx]
                A[b][start_idx:end_idx, start_idx:end_idx] += A_ii[b, :n[b], :n[b]]
        # compute similarity matrix W
        W = [torch.zeros(m.item(), m.item(), device=device) for m in mssum]
        Wds = [torch.zeros(m.item(), m.item(), device=device) for m in mssum]
        #P = [torch.zeros(m.item(), m.item(), device=device) for m in mssum]
        for src, tgt in product(feat_list, repeat=2):
            src_idx, src_feat, P_src, G_src, H_src, _, __, n_src = src
            tgt_idx, tgt_feat, P_tgt, _, __, G_tgt, H_tgt, n_tgt = tgt
            if src_idx < tgt_idx:
                continue
            W_ij = self.affinity_layer(src_feat.transpose(1, 2), tgt_feat.transpose(1, 2))
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
                Wds[b][start_x:end_x, start_y:end_y] += W_ij_ds
                if src_idx != tgt_idx:
                    W[b][start_y:end_y, start_x:end_x] += W_ijb.t()
                    Wds[b][start_y:end_y, start_x:end_x] += W_ij_ds.t()

                #P_ij = hungarian(W_ij[b, :n_src[b], :n_tgt[b]])
                #P[b][start_x:end_x, start_y:end_y] += P_ij
                #if src_idx != tgt_idx:
                #    P[b][start_y:end_y, start_x:end_x] += P_ij.t()

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
            topargs = torch.argsort(torch.abs(e), descending=True)[:self.univ_size]
            v = v[:, topargs]
            U0_b = []
            for i in range(num_graphs):
                if i == 0:
                    start_idx = 0
                else:
                    start_idx = mscum[b, i-1]
                end_idx = mscum[b, i]
                #U0_b.append(self.bi_stochastic(v[start_idx:end_idx, :], dummy_row=True))
                U0_b.append(hungarian(v[start_idx:end_idx, :]))
            U0.append(torch.cat(U0_b, dim=0))
        '''

        # GANN
        U = [[] for _ in range(batch_size)]
        #U = [] # todo
        cluster_v = []
        for b in range(batch_size):
            #U0_b = torch.rand(torch.sum(ms[b]), self.univ_size, device=device) / self.univ_size.to(dtype=torch.float) * 2
            U0_b = torch.full((torch.sum(ms[b]), self.univ_size), 1 / self.univ_size.to(dtype=torch.float), device=device)
            U0_b += torch.randn_like(U0_b) / 1000
            #U0_b = U0[b]
            U_b, cluster_v_b = self.gamgm(A[b], Wds[b], U0_b, ms[b], self.univ_size, self.quad_weight, self.cluster_quad_weight, num_clusters)
            #U_b, Beta = hippi(A[b], W[b], (Beta * (1 - beta) + beta)[b], U0b, ms[b], torch.ceil(msmax[b] * self.univ_factor).to(dtype=torch.int64))
            #U_b = U0[b]

            cluster_v.append(cluster_v_b)
            #cluster_v = cluster_v_b # todo
            for i in range(num_graphs):
                if i == 0:
                    start_idx = 0
                else:
                    start_idx = mscum[b, i-1]
                end_idx = mscum[b, i]
                U[b].append(U_b[start_idx:end_idx, :])
                #for j, Ubx in enumerate(U_b):  #todo
                #    if i == 0:
                #        U.append([])
                #    U[j].append(Ubx[start_idx:end_idx, :])
        cluster_v = torch.stack(cluster_v)

        return U, cluster_v, Wds, mscum

    @staticmethod
    def collect_in_class_matching_wrapper(U, Wds, mscum, gt_cls):
        """
        :param U: Stacked matching-to-universe matrix
        :param Wds: pairwise matching result in doubly-stochastic matrix
        :param mscum: cumsum of number of nodes in graphs
        :param gt_cls: ground truth classes
        """
        batch_size = len(U)
        # collect results
        pred_s = []
        pred_s_discrete = []
        indices = []
        unique_gt_cls = set(gt_cls)

        intra_class_iterator = []
        for cls in unique_gt_cls:
            idx_range = np.where(np.array(gt_cls) == cls)[0]
            intra_class_iterator.append(combinations(idx_range, 2))
        intra_class_iterator = chain(*intra_class_iterator)

        for idx1, idx2 in intra_class_iterator:
            s = []
            for b in range(batch_size):
                #b = 0 # todo
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

        return pred_s, indices, pred_s_discrete


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
