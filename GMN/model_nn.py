import torch
import torch.nn as nn
from utils.feature_align import feature_align
from utils.config import cfg
from GMN.model import Net as GMN_NET


class Net(GMN_NET):
    def __init__(self):
        super(Net, self).__init__()
        self.affinity_layer = None
        self.power_iteration = None

    def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H,
                summary_writer=None):
        # feature extraction
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
        ap = nn.AvgPool2d(kernel_size=2, stride=2)
        U_tgt = ap(tgt_node)
        U_tgt = U_tgt.view(-1,  # batch size
                           cfg.GMN.FEATURE_CHANNEL, cfg.PAIR.CANDIDATE_LENGTH)
        F_tgt = tgt_edge.view(-1,  # batch size
                              cfg.GMN.FEATURE_CHANNEL, cfg.PAIR.CANDIDATE_LENGTH)

        X = torch.cat((U_src, F_src), dim=1)
        Y = torch.cat((U_tgt, F_tgt), dim=1)

        s = torch.matmul(X.transpose(1, 2), Y)
        s = self.bi_stochastic(s)
        s = self.voting_layer(s, ns_src)

        d, _ = self.displacement_layer(s, P_src, P_tgt)
        return s, d
