import torch
import torch.nn as nn

from GMN.backbone import VGG16

from GMN.bi_stochastic import BiStochastic
from GMN.voting_layer import Voting
from GMN.displacement_layer import Displacement
from utils.build_graphs import reshape_edge_feature
from utils.feature_align import feature_align
from utils.fgm import construct_m
from GMGNN.gconv import Gconv
from GNNQAP.gnn import GNNLayer
from GMN.affinity_layer import Affinity

from utils.config import cfg


class Net(VGG16):
    def __init__(self):
        super(Net, self).__init__()
        self.affinity_layer = Affinity(cfg.GNNQAP.FEATURE_CHANNEL)
        self.bi_stochastic = BiStochastic(max_iter=cfg.GNNQAP.BS_ITER_NUM, epsilon=cfg.GNNQAP.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.GNNQAP.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.GNNQAP.FEATURE_CHANNEL * 2, alpha=cfg.GNNQAP.FEATURE_CHANNEL * 2, beta=0.5, k=0)

        self.gnn_layer = cfg.GNNQAP.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                #gnn_layer = Gconv(1, cfg.GNNQAP.GNN_FEAT)
                gnn_layer = GNNLayer(1, 1, cfg.GNNQAP.GNN_FEAT, cfg.GNNQAP.GNN_FEAT)
            else:
                #gnn_layer = Gconv(cfg.GNNQAP.GNN_FEAT, cfg.GNNQAP.GNN_FEAT)
                gnn_layer = GNNLayer(cfg.GNNQAP.GNN_FEAT, cfg.GNNQAP.GNN_FEAT, cfg.GNNQAP.GNN_FEAT, cfg.GNNQAP.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.GNNQAP.GNN_FEAT, 1)

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

        X = reshape_edge_feature(F_src, G_src, H_src)
        Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)

        # affinity layer
        Me, Mp = self.affinity_layer(X, Y, U_src, U_tgt)

        M = construct_m(Me, torch.zeros_like(Mp), K_G, K_H)

        A = (M > 0).to(M.dtype)

        #d = torch.sum(M, dim=-1)
        #d_max = torch.max(d, dim=-1)[0]
        #M_prime = torch.zeros_like(M)
        #for b in range(M.shape[0]):
        #    M_prime[b] = M[b] / d_max[b]
        #M = M_prime

        emb = Mp.transpose(1, 2).contiguous().view(Mp.shape[0], -1, 1)
        M = M.unsqueeze(-1)

        # emb = torch.ones(M.shape[0], M.shape[1], 1, device=M.device)

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            M, emb = gnn_layer(A, M, emb) #, norm=False)
            #print("min={:.2f}; max={:.2f}; >0={:.2f}".format(torch.min(emb), torch.max(emb), torch.sum(emb > 0).to(torch.float) / emb.shape[0] / emb.shape[1] / emb.shape[2]))

        #print('-' * 20)

        v = self.classifier(emb)
        s = v.view(v.shape[0], P_tgt.shape[1], -1).transpose(1, 2)

        s = self.voting_layer(s, ns_src, ns_tgt)
        s = self.bi_stochastic(s, ns_src, ns_tgt)

        d, _ = self.displacement_layer(s, P_src, P_tgt)
        return s, d
