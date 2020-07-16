import torch
import torch.nn as nn
import torch.nn.functional as functional

from lib.bi_stochastic import BiStochastic
from models.GMN.voting_layer import Voting
from models.GMN.displacement_layer import Displacement
from lib.build_graphs import reshape_edge_feature
from lib.feature_align import feature_align
from lib.factorize_graph_matching import construct_m
from models.NGM.gnn import GNNLayer
from models.NGM.geo_edge_feature import geo_edge_feature
from models.GMN.affinity_layer import InnerpAffinity, GaussianAffinity

from itertools import combinations
import numpy as np


from lib.utils.config import cfg

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
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        if cfg.NGM.EDGE_FEATURE == 'cat':
            self.affinity_layer = InnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            self.affinity_layer = GaussianAffinity(1, cfg.NGM.GAUSSIAN_SIGMA)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))
        self.tau = 1 / cfg.NGM.VOTING_ALPHA #nn.Parameter(torch.Tensor([1 / cfg.NGM.VOTING_ALPHA]))
        self.bi_stochastic = BiStochastic(max_iter=cfg.NGM.BS_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.NGM.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.NGM.FEATURE_CHANNEL * 2, alpha=cfg.NGM.FEATURE_CHANNEL * 2, beta=0.5, k=0)

        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            #self.register_parameter('alpha_{}'.format(i), nn.Parameter(torch.Tensor([cfg.NGM.VOTING_ALPHA / (2 ** (self.gnn_layer - i - 1))])))
            #alpha = getattr(self, 'alpha_{}'.format(i))
            alpha = cfg.NGM.VOTING_ALPHA
            if i == 0:
                #gnn_layer = Gconv(1, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha, edge_emb=cfg.NGM.EDGE_EMB)
                #gnn_layer = HyperConvLayer(1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            else:
                #gnn_layer = Gconv(cfg.NGM.GNN_FEAT, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha, edge_emb=cfg.NGM.EDGE_EMB)
                #gnn_layer = HyperConvLayer(cfg.NGM.GNN_FEAT[i-1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i-1],
                #                           cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + (1 if cfg.NGM.SK_EMB else 0), 1)

        #if pretrained:
        #    load_model(self, 'output/ngm_em_synthetic/params/params_0020.pt')

        self.bi_stochastic2 = BiStochastic(max_iter=cfg.NGM.BS_ITER_NUM, epsilon=cfg.NGM.BS_EPSILON, tau=1/2.)

    def forward(self, data, Ps, Gs, Hs, Gs_ref, Hs_ref, KGs, KHs, ns, type='img', **kwargs):
        batch_size = data[0].shape[0]
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

        # extract reference graph feature
        feat_list = []
        joint_indices = [0]
        iterator = zip(feats, Ps, Gs, Hs, Gs_ref, Hs_ref, ns)
        for idx, (feat, P, G, H, G_ref, H_ref, n) in enumerate(iterator):
            feat_list.append(
                (idx,
                 feat,
                 P, G, H, G_ref, H_ref, n
                 )
            )
            joint_indices.append(joint_indices[-1] + P.shape[1])

        joint_S = torch.zeros(batch_size, joint_indices[-1], joint_indices[-1], device=device)
        joint_S_diag = torch.diagonal(joint_S, dim1=1, dim2=2)
        joint_S_diag += 1

        pred_s = []
        indices = []
        for src, tgt in combinations(feat_list, 2):
            # pca forward
            src_idx, src_feat, P_src, G_src, H_src, _, __, n_src = src
            tgt_idx, tgt_feat, P_tgt, _, __, G_tgt, H_tgt, n_tgt = tgt
            K_G = KGs['{},{}'.format(src_idx, tgt_idx)]
            K_H = KHs['{},{}'.format(src_idx, tgt_idx)]
            s = self.__ngm_forward(src_feat, tgt_feat, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, K_G, K_H, n_src, n_tgt)

            if src_idx > tgt_idx:
                joint_S[:, joint_indices[tgt_idx]:joint_indices[tgt_idx+1], joint_indices[src_idx]:joint_indices[src_idx+1]] += s.transpose(1, 2)
            else:
                joint_S[:, joint_indices[src_idx]:joint_indices[src_idx+1], joint_indices[tgt_idx]:joint_indices[tgt_idx+1]] += s

        matching_s = []
        for b in range(batch_size):
            e, v = torch.symeig(joint_S[b], eigenvectors=True)
            topargs = torch.argsort(torch.abs(e), descending=True)[:joint_indices[1]]
            diff = e[topargs[:-1]] - e[topargs[1:]]
            if torch.min(torch.abs(diff)) > 1e-4:
                matching_s.append(len(data) * torch.mm(v[:, topargs], v[:, topargs].transpose(0, 1)))
            else:
                matching_s.append(joint_S[b])

        matching_s = torch.stack(matching_s, dim=0)

        for idx1, idx2 in combinations(range(len(data)), 2):
            s = matching_s[:, joint_indices[idx1]:joint_indices[idx1+1], joint_indices[idx2]:joint_indices[idx2+1]]
            s = self.bi_stochastic2(s)
            #s = torch.clamp(s, 0, 1)
            pred_s.append(s)
            indices.append((idx1, idx2))

            #s = joint_S[:, joint_indices[idx1]:joint_indices[idx1+1], joint_indices[idx2]:joint_indices[idx2+1]]
            #pred_s.append(s)
            #indices.append((idx1, idx2))

        return pred_s, indices

    def __ngm_forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, K_G, K_H, ns_src, ns_tgt):
        U_src = src[:, :src.shape[1] // 2, :]
        F_src = src[:, src.shape[1] // 2:, :]
        U_tgt = tgt[:, :tgt.shape[1] // 2, :]
        F_tgt = tgt[:, tgt.shape[1] // 2:, :]

        if cfg.NGM.EDGE_FEATURE == 'cat':
            X = reshape_edge_feature(F_src, G_src, H_src)
            Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            X = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
            Y = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))

        #K_G = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(G_tgt.cpu().numpy(), G_src.cpu().numpy())]
        #K_H = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(H_tgt.cpu().numpy(), H_src.cpu().numpy())]
        #K_G = CSRMatrix3d(K_G).to(src.device)
        #K_H = CSRMatrix3d(K_H).transpose().to(src.device)

        # affinity layer
        Me, Mp = self.affinity_layer(X, Y, U_src, U_tgt)

        M = construct_m(Me, torch.zeros_like(Mp), K_G, K_H)

        A = (M > 0).to(M.dtype)

        if cfg.NGM.FIRST_ORDER:
            emb = Mp.transpose(1, 2).contiguous().view(Mp.shape[0], -1, 1)
        else:
            emb = torch.ones(M.shape[0], M.shape[1], 1, device=M.device)

        emb_M = M.unsqueeze(-1)

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_M, emb = gnn_layer(A, emb_M, emb, ns_src, ns_tgt) #, norm=False)

        v = self.classifier(emb)
        s = v.view(v.shape[0], U_tgt.shape[2], -1).transpose(1, 2)

        ss = self.bi_stochastic(s, ns_src, ns_tgt, dummy_row=True)

        return ss
