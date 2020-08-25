import torch
import torch.nn as nn

from src.lap_solvers.sinkhorn import Sinkhorn, GumbelSinkhorn
from models.GMN.displacement_layer import Displacement
from src.build_graphs import reshape_edge_feature
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_m
from models.NGM.gnn import GNNLayer
from models.NGM.geo_edge_feature import geo_edge_feature
from models.GMN.affinity_layer import InnerpAffinity, GaussianAffinity
from src.evaluation_metric import objective_score
from src.lap_solvers.hungarian import hungarian
import math
from src.utils.gpu_memory import free_gpu_memory

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
        self.tau = cfg.NGM.SK_TAU
        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.BS_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.BS_EPSILON)
        self.gumbel_sinkhorn = GumbelSinkhorn(max_iter=cfg.NGM.BS_ITER_NUM, tau=self.tau * 20, epsilon=cfg.NGM.BS_EPSILON)
        #self.rrwm = PowerIteration()
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.NGM.FEATURE_CHANNEL * 2, alpha=cfg.NGM.FEATURE_CHANNEL * 2, beta=0.5, k=0)

        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            #self.register_parameter('alpha_{}'.format(i), nn.Parameter(torch.Tensor([cfg.NGM.VOTING_ALPHA / (2 ** (self.gnn_layer - i - 1))])))
            #alpha = getattr(self, 'alpha_{}'.format(i))
            tau = cfg.NGM.SK_TAU
            if i == 0:
                #gnn_layer = Gconv(1, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                #gnn_layer = HyperConvLayer(1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            else:
                #gnn_layer = Gconv(cfg.NGM.GNN_FEAT, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                #gnn_layer = HyperConvLayer(cfg.NGM.GNN_FEAT[i-1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i-1],
                #                           cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + (1 if cfg.NGM.SK_EMB else 0), 1)

    def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H, type='img', **kwargs):
        batch_size = src.shape[0]
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

            K = construct_m(Me, torch.zeros_like(Mp), K_G, K_H)

            A = (K > 0).to(K.dtype)

            if cfg.NGM.FIRST_ORDER:
                emb = Mp.transpose(1, 2).contiguous().view(Mp.shape[0], -1, 1)
            else:
                emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)
        else:
            tgt_len = int(math.sqrt(src.shape[2]))
            K = src
            dmax = (torch.max(torch.sum(K, dim=2, keepdim=True), dim=1, keepdim=True).values + 1e-5)
            K = K / dmax * 1000
            A = (K > 0).to(K.dtype)
            emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

        emb_K = K.unsqueeze(-1)

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_K, emb = gnn_layer(A, emb_K, emb, ns_src, ns_tgt) #, norm=False)

        v = self.classifier(emb)
        s = v.view(v.shape[0], tgt_len, -1).transpose(1, 2)

        if self.training or cfg.NGM.GUMBEL_SK <= 0:
            ss = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
        else:
            #ss = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
            #opt_obj_score = objective_score(hungarian(ss, ns_src, ns_tgt), K, ns_src)

            gumbel_sample_num = cfg.NGM.GUMBEL_SK
            ss_gumbel = self.gumbel_sinkhorn(s, ns_src, ns_tgt, sample_num=gumbel_sample_num, dummy_row=True)
            
            #for ri in range(0, gumbel_sample_num):
            #    min_obj_score = torch.stack((opt_obj_score, objective_score(hungarian(ss_gumbel[ri::gumbel_sample_num], ns_src, ns_tgt), K, ns_src))).max(dim=0)
            #    opt_obj_score = min_obj_score.values
            #    ss = min_obj_score.indices.unsqueeze(-1).unsqueeze(-1) * ss_gumbel[ri::gumbel_sample_num] + (1 - min_obj_score.indices).unsqueeze(-1).unsqueeze(-1) * ss

            repeat = lambda x, rep_num=gumbel_sample_num: torch.repeat_interleave(x, rep_num, dim=0)
            ss_gumbel = hungarian(ss_gumbel, repeat(ns_src), repeat(ns_tgt))
            ss_gumbel = ss_gumbel.reshape(batch_size, gumbel_sample_num, ss_gumbel.shape[-2], ss_gumbel.shape[-1])

            if ss_gumbel.device.type == 'cuda':
                dev_idx = ss_gumbel.device.index
                free_mem = free_gpu_memory(dev_idx) - 500 * 1024 ** 2
                K_mem_size = K.element_size() * K.nelement()
                max_repeats = free_mem // K_mem_size
            else:
                max_repeats = gumbel_sample_num

            obj_score = []
            for idx in range(0, gumbel_sample_num, max_repeats):
                if idx + max_repeats > gumbel_sample_num:
                    rep_num = gumbel_sample_num - idx
                else:
                    rep_num = max_repeats
                obj_score.append(
                    objective_score(
                        ss_gumbel[:, idx:(idx+rep_num), :, :].reshape(-1, ss_gumbel.shape[-2], ss_gumbel.shape[-1]),
                        repeat(K, rep_num)
                    ).reshape(batch_size, -1)
                )
            obj_score = torch.cat(obj_score, dim=1)
            #obj_score = objective_score(ss_gumbel, repeat(K)).reshape(s.shape[0], gumbel_sample_num)
            min_obj_score = obj_score.min(dim=1)
            ss = ss_gumbel[torch.arange(batch_size), min_obj_score.indices.cpu(), :, :]

            #opt_obj_score = objective_score(perm_mat, ori_affmtx, n1_gt)

        if type != 'affmat':
            d, _ = self.displacement_layer(ss, P_src, P_tgt)
        else:
            d = None

        return ss, d, K
