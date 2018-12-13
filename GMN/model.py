import torch
import torch.nn as nn
from torchvision import models

from GMN.affinity_layer import Affinity
from GMN.power_iteration import PowerIteration
from GMN.bi_stochastic import BiStochastic
from GMN.voting_layer import Voting
from GMN.displacement_layer import Displacement
from utils.build_graphs import build_graphs, reshape_edge_feature
from utils.feature_align import feature_align
from utils.fgm import construct_m

from utils.config import cfg


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.node_layers, self.edge_layers = self.get_backbone()
        self.affinity_layer = Affinity(cfg.GMN.FEATURE_CHANNEL)
        self.power_iteration = PowerIteration(max_iter=cfg.GMN.PI_ITER_NUM, stop_thresh=cfg.GMN.PI_STOP_THRESH)
        self.bi_stochastic = BiStochastic(max_iter=cfg.GMN.BS_ITER_NUM, epsilon=cfg.GMN.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.GMN.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.GMN.FEATURE_CHANNEL * 2, alpha=cfg.GMN.FEATURE_CHANNEL * 2, beta=0.5, k=0)

    def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H,
                summary_writer=None):

        # extract feature
        src_node = self.node_layers(src)
        src_edge = self.edge_layers(src_node)
        tgt_node = self.node_layers(tgt)
        tgt_edge = self.edge_layers(tgt_node)
        #print('extract feature')

        # feature normalization
        src_node = self.l2norm(src_node)
        src_edge = self.l2norm(src_edge)
        tgt_node = self.l2norm(tgt_node)
        tgt_edge = self.l2norm(tgt_edge)
        #print('feature norm')

        # arrange features
        U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
        F_src = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)
        # feature pooling for target. Since they are arranged in grids, this can be done more efficiently
        ap = nn.AvgPool2d(kernel_size=2, stride=2)
        U_tgt = ap(tgt_node)
        U_tgt = U_tgt.view(-1,  # batch size
                           cfg.GMN.FEATURE_CHANNEL, cfg.PAIR.CANDIDATE_LENGTH)
        F_tgt = tgt_edge.view(-1,  # batch size
                              cfg.GMN.FEATURE_CHANNEL, cfg.PAIR.CANDIDATE_LENGTH)
        #print('feature pooling')

        X = reshape_edge_feature(F_src, G_src, H_src)
        Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)
        #print('reshape XY')

        #import matplotlib.pyplot as plt
        #import numpy as np

        # affinity layer
        Me, Mp = self.affinity_layer(X, Y, U_src, U_tgt)
        #print('Me', Me)
        #print('Mp', Mp)

        #print('affinity')

        M = construct_m(Me, Mp, K_G, K_H)
        #print('construct M')
        '''
        for i in range(4):
            fig = plt.figure()
            sc = plt.imshow(M.cpu().detach().numpy()[i])
            sc.set_cmap('OrRd')
            plt.colorbar()
            fig.savefig('M{}.png'.format(i))
            summary_writer.add_figure('M{}'.format(i), fig)
        '''

        v = self.power_iteration(M)
        #print('v', v)
        s = self.bi_stochastic(v, (-1, cfg.PAIR.CANDIDATE_LENGTH))
        #print('s', s)
        s = self.voting_layer(s, ns_src)
        d, _ = self.displacement_layer(s, P_src, P_tgt)
        return s, d

    @staticmethod
    def get_backbone():
        """
        Get pretrained VGG16 models for feature extraction.
        :return: feature sequence
        """
        model = models.vgg16_bn(pretrained=True)

        conv_layers = nn.Sequential(
            *list(model.features.children()))

        conv_list = node_list = edge_list = []

        # get the output of relu4_2(node features) and relu5_1(edge features)
        cnt_m, cnt_r = 1, 0
        for layer, module in enumerate(conv_layers):
            if isinstance(module, nn.Conv2d):
                cnt_r += 1
            if isinstance(module, nn.MaxPool2d):
                cnt_r = 0
                cnt_m += 1
            conv_list += [module]

            if cnt_m == 4 and cnt_r == 3 and isinstance(module, nn.Conv2d):
                node_list = conv_list
                conv_list = []
            elif cnt_m == 5 and cnt_r == 2 and isinstance(module, nn.Conv2d):
                edge_list = conv_list
                break

        assert len(node_list) > 0 and len(edge_list) > 0

        # Set the layers as a nn.Sequential module
        node_layers = nn.Sequential(*node_list)
        edge_layers = nn.Sequential(*edge_list)
        
        return node_layers, edge_layers
