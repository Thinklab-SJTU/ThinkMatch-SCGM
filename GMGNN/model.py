import torch
import torch.nn as nn
from torchvision import models

from GMN.backbone import VGG16

from GMN.power_iteration import PowerIteration
from GMN.bi_stochastic import BiStochastic
from GMN.voting_layer import Voting
from GMN.displacement_layer import Displacement
from utils.build_graphs import build_graphs, reshape_edge_feature
from utils.feature_align import feature_align
from utils.fgm import construct_m
from GMGNN.gconv import Siamese_GNN
from GMGNN.affinity_layer import Affinity

from utils.config import cfg


class Net(VGG16):
    def __init__(self):
        super(Net, self).__init__()
        self.bi_stochastic = BiStochastic(max_iter=cfg.GMN.BS_ITER_NUM, epsilon=cfg.GMN.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.GMN.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.GMN.FEATURE_CHANNEL * 2, alpha=cfg.GMN.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        #self.siamese_gnn = Siamese_GNN(cfg.GMGNN.GNN_FEAT, cfg.GMGNN.GNN_LAYER - 1, 1)
        self.gnn_layer = cfg.GMGNN.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_GNN(cfg.GMN.FEATURE_CHANNEL * 2, cfg.GMGNN.GNN_FEAT, 0, 1)
            else:
                gnn_layer = Siamese_GNN(cfg.GMGNN.GNN_FEAT, cfg.GMGNN.GNN_FEAT, 0, 1)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.GMGNN.GNN_FEAT))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.GMGNN.GNN_FEAT * 2, cfg.GMGNN.GNN_FEAT))
        #self.affine_func = Affinity(cfg.GMGNN.GNN_FEAT)

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

        # adjacency matrices
        A_src = torch.bmm(G_src, H_src.transpose(1, 2))
        A_tgt = torch.bmm(G_tgt, H_tgt.transpose(1, 2))

        emb1, emb2 = torch.cat((U_src, F_src), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt), dim=1).transpose(1, 2)
        ss = []
        '''
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
            affinity = getattr(self, 'affinity_{}'.format(i))
            s = affinity(emb1, emb2)
            #s = torch.bmm(emb1, emb2.transpose(1, 2))
            s = self.voting_layer(s, ns_src, ns_tgt)
            s = self.bi_stochastic(s, ns_src, ns_tgt)
            ss.append(s)

            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
        '''
        s = None
        for x in range(6):
            for i in range(self.gnn_layer):
                if i == 0:
                    if s is None:
                        gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                        emb1_0, emb2_0 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                        s = torch.zeros(emb1.shape[0], emb1.shape[1], emb2.shape[1]).cuda()

                    cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                    emb1 = cross_graph(torch.cat((emb1_0, torch.bmm(s, emb2_0)), dim=-1))
                    emb2 = cross_graph(torch.cat((emb2_0, torch.bmm(s.transpose(1, 2), emb1_0)), dim=-1))
                if i==1:
                    gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                    emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                    affinity = getattr(self, 'affinity_{}'.format(i))
                    s = affinity(emb1, emb2)
                    s = self.voting_layer(s, ns_src, ns_tgt)
                    s = self.bi_stochastic(s, ns_src, ns_tgt)
                    ss.append(s)

        d, _ = self.displacement_layer(s, P_src, P_tgt)
        return ss, d
