import torch
import torch.nn as nn

from GMN.bi_stochastic import BiStochastic
from GMN.voting_layer import Voting
from GMN.displacement_layer import Displacement
from utils.build_graphs import reshape_edge_feature
from utils.feature_align import feature_align
from utils.fgm import construct_m
from GMGNN.gconv import Gconv
from GNNQAP.gnn import GNNLayer
from GNNQAP.geo_edge_feature import geo_edge_feature
from GMN.affinity_layer import InnerpAffinity, GaussianAffinity

from utils.config import cfg

import GMN.backbone
CNN = eval('GMN.backbone.{}'.format(cfg.BACKBONE))


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        if cfg.GNNQAP.EDGE_FEATURE == 'cat':
            self.affinity_layer = InnerpAffinity(cfg.GNNQAP.FEATURE_CHANNEL)
        elif cfg.GNNQAP.EDGE_FEATURE == 'geo':
            self.affinity_layer = GaussianAffinity(1, cfg.GNNQAP.GAUSSIAN_SIGMA)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.GNNQAP.EDGE_FEATURE))
        self.bi_stochastic = BiStochastic(max_iter=cfg.GNNQAP.BS_ITER_NUM, epsilon=cfg.GNNQAP.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.GNNQAP.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.GNNQAP.FEATURE_CHANNEL * 2, alpha=cfg.GNNQAP.FEATURE_CHANNEL * 2, beta=0.5, k=0)

        self.gnn_layer = cfg.GNNQAP.GNN_LAYER
        for i in range(self.gnn_layer):
            #self.register_parameter('alpha_{}'.format(i), nn.Parameter(torch.Tensor([cfg.GNNQAP.VOTING_ALPHA / (2 ** (self.gnn_layer - i - 1))])))
            #alpha = getattr(self, 'alpha_{}'.format(i))
            alpha = cfg.GNNQAP.VOTING_ALPHA
            if i == 0:
                #gnn_layer = Gconv(1, cfg.GNNQAP.GNN_FEAT)
                gnn_layer = GNNLayer(1, 1, cfg.GNNQAP.GNN_FEAT + (1 if cfg.GNNQAP.SK_EMB else 0), cfg.GNNQAP.GNN_FEAT,
                                     sk_channel=cfg.GNNQAP.SK_EMB, voting_alpha=alpha)
            else:
                #gnn_layer = Gconv(cfg.GNNQAP.GNN_FEAT, cfg.GNNQAP.GNN_FEAT)
                gnn_layer = GNNLayer(cfg.GNNQAP.GNN_FEAT + (1 if cfg.GNNQAP.SK_EMB else 0), cfg.GNNQAP.GNN_FEAT,
                                     cfg.GNNQAP.GNN_FEAT + (1 if cfg.GNNQAP.SK_EMB else 0), cfg.GNNQAP.GNN_FEAT,
                                     sk_channel=cfg.GNNQAP.SK_EMB, voting_alpha=alpha)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.GNNQAP.GNN_FEAT + (1 if cfg.GNNQAP.SK_EMB else 0), 1)

    def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H, type='img'):
        if type == 'img' or type == 'image':
            # extract feature
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)

            # arrange features
            U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
            F_src = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
        elif type == 'feat' or type == 'feature':
            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('unknown type string {}'.format(type))

        if cfg.GNNQAP.EDGE_FEATURE == 'cat':
            X = reshape_edge_feature(F_src, G_src, H_src)
            Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)
        elif cfg.GNNQAP.EDGE_FEATURE == 'geo':
            X = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
            Y = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.GNNQAP.EDGE_FEATURE))

        # affinity layer
        #Me1, Mp1 = self.affinity_layer1(X1, Y1, U_src, U_tgt)
        Me, Mp = self.affinity_layer(X, Y, U_src, U_tgt)

        #M = construct_m(Me1, torch.zeros_like(Mp1), K_G, K_H)
        M = construct_m(Me, torch.zeros_like(Mp), K_G, K_H)

        A = (M > 0).to(M.dtype)

        #d = torch.sum(M, dim=-1)
        #d_max = torch.max(d, dim=-1)[0]
        #M_prime = torch.zeros_like(M)
        #for b in range(M.shape[0]):
        #    M_prime[b] = M[b] / d_max[b]
        #M = M_prime

        if cfg.GNNQAP.FIRST_ORDER:
            emb = Mp.transpose(1, 2).contiguous().view(Mp.shape[0], -1, 1)
        else:
            emb = torch.ones(M.shape[0], M.shape[1], 1, device=M.device)

        M = M.unsqueeze(-1)

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            M, emb = gnn_layer(A, M, emb, ns_src, ns_tgt) #, norm=False)
            #emb = gnn_layer(M, emb)

        v = self.classifier(emb)
        s = v.view(v.shape[0], P_tgt.shape[1], -1).transpose(1, 2)

        ss = self.voting_layer(s, ns_src, ns_tgt)
        ss = self.bi_stochastic(ss, ns_src, ns_tgt)#, dummy_row=True)

        d, _ = self.displacement_layer(ss, P_src, P_tgt)

        if cfg.GNNQAP.OUTP_SCORE:
            return s, ss, d
        else:
            return ss, d