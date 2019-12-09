import torch
import torch.nn as nn
from torchvision import models

from GMN.affinity_layer import InnerpAffinity as Affinity
#from GMN.affinity_layer import GaussianAffinity
from GMN.power_iteration import PowerIteration
from GMN.rrwm import RRWM
from GMN.bi_stochastic import BiStochastic
from GMN.voting_layer import Voting
from GMN.displacement_layer import Displacement
from utils.build_graphs import build_graphs, reshape_edge_feature
from NGM.geo_edge_feature import geo_edge_feature
from utils.feature_align import feature_align
from utils.fgm import construct_m

from utils.config import cfg

import GMN.backbone
CNN = eval('GMN.backbone.{}'.format(cfg.BACKBONE))


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.affinity_layer = Affinity(cfg.GMN.FEATURE_CHANNEL)
        #self.affinity_layer = GaussianAffinity(1, 5.e-7)
        if cfg.GMN.GM_SOLVER == 'SM':
            self.gm_solver = PowerIteration(max_iter=cfg.GMN.PI_ITER_NUM, stop_thresh=cfg.GMN.PI_STOP_THRESH)
        elif cfg.GMN.GM_SOLVER == 'RRWM':
            self.gm_solver = RRWM()
        self.bi_stochastic = BiStochastic(max_iter=cfg.GMN.BS_ITER_NUM, epsilon=cfg.GMN.BS_EPSILON, log_forward=False)
        self.voting_layer = Voting(alpha=cfg.GMN.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.GMN.FEATURE_CHANNEL * 2, alpha=cfg.GMN.FEATURE_CHANNEL * 2, beta=0.5, k=0)

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
            # feature pooling for target. Since they are arranged in grids, this can be done more efficiently
            #ap = nn.AvgPool2d(kernel_size=2, stride=2)
            #U_tgt = ap(tgt_node)
            #U_tgt = U_tgt.view(-1,  # batch size
            #                   cfg.GMN.FEATURE_CHANNEL, cfg.PAIR.CANDIDATE_LENGTH)
            #F_tgt = tgt_edge.view(-1,  # batch size
            #                      cfg.GMN.FEATURE_CHANNEL, cfg.PAIR.CANDIDATE_LENGTH)
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
        #X = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
        #Y = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]

        # affinity layer
        Me, Mp = self.affinity_layer(X, Y, U_src, U_tgt)

        M = construct_m(Me, Mp, K_G, K_H)

        v = self.gm_solver(M, num_src=P_src.shape[1], ns_src=ns_src, ns_tgt=ns_tgt)
        s = v.view(v.shape[0], P_tgt.shape[1], -1).transpose(1, 2)

        s = self.voting_layer(s, ns_src, ns_tgt)
        s = self.bi_stochastic(s, ns_src, ns_tgt)

        d, _ = self.displacement_layer(s, P_src, P_tgt)
        return s, d, M
