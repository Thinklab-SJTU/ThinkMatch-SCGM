import torch
import torch.nn as nn

from src.lap_solvers.sinkhorn import Sinkhorn, GumbelSinkhorn
from models.GMN.voting_layer import Voting
from models.GMN.displacement_layer import Displacement
from src.build_graphs import reshape_edge_feature
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_m
from models.NGM.gnn import GNNLayer
from models.NGM.geo_edge_feature import geo_edge_feature
from models.GMN.affinity_layer import InnerpAffinity, GaussianAffinity
from src.lap_solvers.hungarian import hungarian
import math
from itertools import combinations


from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        if cfg.NGM.EDGE_FEATURE == 'cat':
            self.affinity_layer = InnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            self.affinity_layer = GaussianAffinity(1, cfg.NGM.GAUSSIAN_SIGMA)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))
        self.tau = 1 / cfg.NGM.VOTING_ALPHA #nn.Parameter(torch.Tensor([1 / cfg.NGM.VOTING_ALPHA]))
        self.bi_stochastic = Sinkhorn(max_iter=cfg.NGM.BS_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.BS_EPSILON)
        self.bi_stochastic_g = GumbelSinkhorn(max_iter=cfg.NGM.BS_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.BS_EPSILON)
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
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=alpha, edge_emb=cfg.NGM.EDGE_EMB)
                #gnn_layer = HyperConvLayer(1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            else:
                #gnn_layer = Gconv(cfg.NGM.GNN_FEAT, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=alpha, edge_emb=cfg.NGM.EDGE_EMB)
                #gnn_layer = HyperConvLayer(cfg.NGM.GNN_FEAT[i-1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i-1],
                #                           cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + (1 if cfg.NGM.SK_EMB else 0), 1)

        actor_inp_dim = (cfg.NGM.GNN_FEAT[-1] + (1 if cfg.NGM.SK_EMB else 0)) * 4

        self.action_layer = nn.Sequential(
            nn.Linear(actor_inp_dim, actor_inp_dim//2),
            nn.ReLU(),
            nn.Linear(actor_inp_dim//2, 1)
        )
        self.softmax = nn.Softmax(dim=-1)

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
        elif type == 'affmat':
            pass
        else:
            raise ValueError('unknown type string {}'.format(type))

        if type != 'affmat':
            tgt_len = P_tgt.shape[1]
            if cfg.NGM.EDGE_FEATURE == 'cat':
                X = reshape_edge_feature(F_src, G_src, H_src)
                Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)
            elif cfg.NGM.EDGE_FEATURE == 'geo':
                X = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
                Y = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]
            else:
                raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))

            # affinity layer
            Me, Mp = self.affinity_layer(X, Y, U_src, U_tgt)

            M = construct_m(Me, torch.zeros_like(Mp), K_G, K_H)

            A = (M > 0).to(M.dtype)

            if cfg.NGM.FIRST_ORDER:
                emb = Mp.transpose(1, 2).contiguous().view(Mp.shape[0], -1, 1)
            else:
                emb = torch.ones(M.shape[0], M.shape[1], 1, device=M.device)
        else:
            tgt_len = int(math.sqrt(src.shape[2]))
            M = src
            A = (M > 0).to(M.dtype)
            emb = torch.ones(M.shape[0], M.shape[1], 1, device=M.device)

        emb_M = M.unsqueeze(-1)

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_M, emb = gnn_layer(A, emb_M, emb, ns_src, ns_tgt) #, norm=False)

        # compute initial matching by NGM embeddings
        v = self.classifier(emb)
        s = v.view(v.shape[0], tgt_len, -1).transpose(1, 2)
        matchings = [hungarian(s, ns_src, ns_tgt)]

        # RL trajectory
        trajectory_len = 10
        for i in range(trajectory_len):
            cur_matching = matchings[-1]
            # build all actions
            act_feats = []
            act_rows = []
            for r1, r2 in combinations(range(cur_matching.shape[1]), 2):
                cur_matching_bool = cur_matching.to(dtype=torch.bool)
                swap_index = torch.LongTensor(range(cur_matching.shape[1]))
                swap_index[r1] = r2
                swap_index[r2] = r1
                swap_matching_bool = cur_matching[:, swap_index, :].to(dtype=torch.bool)
                interest_mask = torch.logical_xor(cur_matching_bool, swap_matching_bool)#, out=torch.empty_like(cur_matching))
                elem1_mask = (interest_mask * cur_matching_bool).transpose(1, 2).reshape(v.shape[0], -1)
                elem0_mask = (interest_mask * swap_matching_bool).transpose(1, 2).reshape(v.shape[0], -1)
                elem1_emb = torch.masked_select(emb, elem1_mask.unsqueeze(-1)).reshape(v.shape[0], -1)
                elem0_emb = torch.masked_select(emb, elem0_mask.unsqueeze(-1)).reshape(v.shape[0], -1)
                act_feat = torch.cat((elem1_emb, elem0_emb), dim=-1)
                act_feats.append(act_feat)
                act_rows.append((r1, r2))
            act_feats = torch.stack(act_feats, dim=1)
            act_scores = self.action_layer(act_feats).squeeze(-1)
            act_selection = self.softmax(act_scores)

        if type != 'affmat':
            d, _ = self.displacement_layer(ss, P_src, P_tgt)
        else:
            d = None

        return ss, d, M
