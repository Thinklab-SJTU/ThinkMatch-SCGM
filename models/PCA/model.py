import torch
import torch.nn as nn

from src.lap_solvers.sinkhorn import Sinkhorn
from models.GMN.displacement_layer import Displacement
from src.feature_align import feature_align
from src.gconv import Siamese_Gconv
from models.PCA.affinity_layer import Affinity

from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.sinkhorn = Sinkhorn(max_iter=cfg.PCA.BS_ITER_NUM, epsilon=cfg.PCA.BS_EPSILON, tau=cfg.PCA.SK_TAU)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.PCA.FEATURE_CHANNEL * 2, alpha=cfg.PCA.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.gnn_layer = cfg.PCA.GNN_LAYER
        #self.pointer_net = PointerNet(cfg.PCA.GNN_FEAT, cfg.PCA.GNN_FEAT // 2, alpha=cfg.PCA.VOTING_ALPHA)
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PCA.FEATURE_CHANNEL * 2, cfg.PCA.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PCA.GNN_FEAT, cfg.PCA.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.PCA.GNN_FEAT))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.PCA.GNN_FEAT * 2, cfg.PCA.GNN_FEAT))
        self.cross_iter = cfg.PCA.CROSS_ITER
        self.cross_iter_num = cfg.PCA.CROSS_ITER_NUM
        #self.rrwm = RRWM()
        #self.affinity_layer = InnerpAffinity(cfg.PCA.FEATURE_CHANNEL)

    def reload_backbone(self):
        self.node_layers, self.edge_layers = self.get_backbone(True)

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

        if not self.cross_iter:
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)
                #s = torch.bmm(emb1, emb2.transpose(1, 2))
                s = self.bi_stochastic(s, ns_src, ns_tgt, dummy_row=True)
                ss.append(s)

                if i == self.gnn_layer - 2:
                    cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                    new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                    new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                    emb1 = new_emb1
                    emb2 = new_emb2
        else:
            s = None
            for x in range(self.cross_iter_num):
                for i in range(self.gnn_layer):
                    if i == 0:
                        if s is None:
                            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                            emb1_0, emb2_0 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                            s = torch.zeros(emb1.shape[0], emb1.shape[1], emb2.shape[1]).cuda()
                            #affinity = getattr(self, 'affinity_{}'.format(i))
                            #s = affinity(emb1_0, emb2_0)
                            #s = self.voting_layer(s, ns_src, ns_tgt)
                            #s = self.bi_stochastic(s, ns_src, ns_tgt, dummy_row=True)

                        cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                        emb1 = cross_graph(torch.cat((emb1_0, torch.bmm(s, emb2_0)), dim=-1))
                        emb2 = cross_graph(torch.cat((emb2_0, torch.bmm(s.transpose(1, 2), emb1_0)), dim=-1))
                    else:
                        gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                        emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                        affinity = getattr(self, 'affinity_{}'.format(i))
                        s = affinity(emb1, emb2)
                        s = self.bi_stochastic(s, ns_src, ns_tgt, dummy_row=True)
                        ss.append(s)

        d, _ = self.displacement_layer(s, P_src, P_tgt)
        return s, d
