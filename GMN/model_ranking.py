import torch
import torch.nn as nn
from utils.feature_align import feature_align
from utils.config import cfg

from GMN.backbone import VGG16

from GMN.bi_stochastic import BiStochastic
from GMN.voting_layer import Voting
from GMN.displacement_layer import Displacement


class Net(VGG16):
    def __init__(self):
        super(Net, self).__init__()
        self.affinity_layer = None
        self.bi_stochastic = BiStochastic(max_iter=cfg.GMN.BS_ITER_NUM, epsilon=cfg.GMN.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.GMN.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.GMN.FEATURE_CHANNEL * 2, alpha=cfg.GMN.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.fc = nn.Sequential(
            nn.Linear(cfg.GMN.FEATURE_CHANNEL * 16, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 16 * 16)
        )

    def forward(self, src, P_src, ns_src, summary_writer=None):
        # feature extraction
        src_node = self.node_layers(src)
        src_edge = self.edge_layers(src_node)

        # feature normalization
        src_node = self.l2norm(src_node)
        src_edge = self.l2norm(src_edge)

        # arrange features
        U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
        F_src = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)

        X = torch.cat((U_src, F_src), dim=1)


        s = self.fc(X)

        s = self.bi_stochastic(s)
        s = self.voting_layer(s, ns_src)

        return s, None
