import torch
import torch.nn as nn
from utils.feature_align import feature_align
from utils.config import cfg

from GMN.backbone import VGG16

from GMN.bi_stochastic import BiStochastic
from GMN.voting_layer import Voting
from GMN.displacement_layer import Displacement


class Net(VGG16):
    def __init__(self, classes, kpt_lens):
        super(Net, self).__init__()
        self.affinity_layer = None
        self.bi_stochastic = BiStochastic(max_iter=cfg.GMN.BS_ITER_NUM, epsilon=cfg.GMN.BS_EPSILON)
        self.voting_layer = Voting(alpha=cfg.GMN.VOTING_ALPHA)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.GMN.FEATURE_CHANNEL * 2, alpha=cfg.GMN.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        for cls, len in zip(classes, kpt_lens):
            fc = nn.Sequential(
                nn.Linear(cfg.GMN.FEATURE_CHANNEL * len, 4096),
                nn.ReLU(),
                nn.Linear(4096, len * len)
            )
            setattr(self, 'fc_{}'.format(cls), fc)

    def forward(self, src, P_src, ns_src, cls, summary_writer=None):
        # feature extraction
        src_node = self.node_layers(src)
        src_edge = self.edge_layers(src_node)

        # feature normalization
        src_node = self.l2norm(src_node)
        src_edge = self.l2norm(src_edge)

        # arrange features
        #U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
        X = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)

        #X = torch.cat((U_src, F_src), dim=1)

        # Randomly permute features
        fc = getattr(self, 'fc_{}'.format(cls))
        kp_num_gt = fc[0].in_features // X.shape[1]
        X_pad = torch.zeros(X.shape[0], X.shape[1], kp_num_gt, device=src.device)
        X_pad[:, :, :X.shape[2]] += X
        tmp_perm = torch.randperm(kp_num_gt, device=src.device)
        tmp_perm_mat = torch.zeros(kp_num_gt, kp_num_gt, device=src.device)
        tmp_perm_mat[tmp_perm, range(kp_num_gt)] = 1

        X_pad_permed = torch.matmul(X_pad, tmp_perm_mat)

        s = fc(X_pad_permed.view(X.shape[0], -1)).view(-1, kp_num_gt, kp_num_gt)

        s = torch.matmul(s, tmp_perm_mat.t())[:, :X.shape[2], :]

        new_s = torch.zeros_like(s)
        for i, b in enumerate(range(X.shape[0])):
            new_s[b, :ns_src[i], :] = s[b, :ns_src[i], :]

        new_s = self.bi_stochastic(new_s)
        new_s = self.voting_layer(new_s, ns_src)

        return new_s, None
