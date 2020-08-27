import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lap_solvers.sinkhorn import Sinkhorn
from models.GMN.displacement_layer import Displacement
from src.build_graphs import reshape_edge_feature
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_m
from models.NGM.gnn import HyperGNNLayer
from models.NGM.geo_edge_feature import geo_edge_feature
from models.GMN.affinity_layer import GaussianAffinity, InnerpAffinity
#from NGM.rrwhm import RRWHM

from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.geo_affinity_layer = GaussianAffinity(1, cfg.NGM.GAUSSIAN_SIGMA)
        self.feat_affinity_layer = InnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        self.feat_affinity_layer3 = InnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        self.tau = 1 / cfg.NGM.VOTING_ALPHA
        self.bi_stochastic = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.NGM.FEATURE_CHANNEL * 2, alpha=cfg.NGM.FEATURE_CHANNEL * 2, beta=0.5, k=0)

        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            #self.register_parameter('alpha_{}'.format(i), nn.Parameter(torch.Tensor([cfg.NGM.VOTING_ALPHA])))
            #alpha = getattr(self, 'alpha_{}'.format(i))
            alpha = cfg.NGM.VOTING_ALPHA
            if i == 0:
                gnn_layer = HyperGNNLayer(
                    1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                    sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha
                )
            else:
                gnn_layer = HyperGNNLayer(
                    cfg.NGM.GNN_FEAT[i - 1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i - 1],
                    cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                    sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha
                )
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + (1 if cfg.NGM.SK_EMB else 0), 1)

        self.weight2 = nn.Parameter(torch.tensor(cfg.NGM.WEIGHT2)).detach()
        self.weight3 = nn.Parameter(torch.tensor(cfg.NGM.WEIGHT3)).detach()

        #self.rrwhm = RRWHM()

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
        dx = geo_edge_feature(P_src, G_src, H_src)[:, :1, :]
        dy = geo_edge_feature(P_tgt, G_tgt, H_tgt)[:, :1, :]

        # affinity layer
        if cfg.NGM.EDGE_FEATURE == 'cat':
            Me, Mp = self.feat_affinity_layer(X, Y, U_src, U_tgt)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            Me, Mp = self.geo_affinity_layer(dx, dy, U_src, U_tgt)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))


        M = construct_m(Me, torch.zeros_like(Mp), K_G, K_H)
        A = (M > 0).to(M.dtype)

        # build 3-order affinity tensor
        hshape = list(A.shape) + [A.shape[-1]]
        order3A = A.unsqueeze(1).expand(hshape) * A.unsqueeze(2).expand(hshape) * A.unsqueeze(3).expand(hshape)
        hyperA = order3A

        #hyperA = A.unsqueeze(1).expand(hshape) + A.unsqueeze(2).expand(hshape) + A.unsqueeze(3).expand(hshape)
        #hyperA_sum = torch.sum(hyperA, dim=tuple(range(2, 4)), keepdim=True)
        #hyperA = hyperA / hyperA_sum.expand_as(hyperA)
        #hyperA[torch.isnan(hyperA)] = 0
        #hyperA = hyperA * order3A

        if cfg.NGM.ORDER3_FEATURE == 'cat':
            Me_3, _ = self.feat_affinity_layer3(X, Y, torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), w1=0.5, w2=1)
            M_3 = construct_m(Me_3, torch.zeros_like(Mp), K_G, K_H)
            #M_3 = construct_m(Me_3, torch.zeros_like(Mp), K_G[0], K_H[0], K_G[1], K_H[1])
            hyperM = (M_3.unsqueeze(1).expand(hshape) + M_3.unsqueeze(2).expand(hshape) + M_3.unsqueeze(3).expand(hshape)) * F.relu(self.weight3)
            #hyperA = (order3A > 0).to(hyperM.dtype)
        elif cfg.NGM.ORDER3_FEATURE == 'geo':
            Me_d, _ = self.geo_affinity_layer(dx, dy, torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))

            m_d_src = construct_m(dx.squeeze().unsqueeze(-1).expand_as(Me_d), torch.zeros_like(Mp), K_G, K_H).cpu()
            m_d_tgt = construct_m(dy.squeeze().unsqueeze(-2).expand_as(Me_d), torch.zeros_like(Mp), K_G, K_H).cpu()
            order3A = order3A.cpu()
            #m_d_src = construct_m(dx.squeeze().unsqueeze(-1).expand_as(Me_d), torch.zeros_like(Mp), K_G[0], K_H[0], K_G[1], K_H[1])
            #m_d_tgt = construct_m(dy.squeeze().unsqueeze(-2).expand_as(Me_d), torch.zeros_like(Mp), K_G[0], K_H[0], K_G[1], K_H[1])

            cum_sin = torch.zeros_like(order3A)
            for i in range(3):
                def calc_sin(t):
                    a = t.unsqueeze(i % 3 + 1).expand(hshape)
                    b = t.unsqueeze((i + 1) % 3 + 1).expand(hshape)
                    c = t.unsqueeze((i + 2) % 3 + 1).expand(hshape)
                    cos = torch.clamp((a.pow(2) + b.pow(2) - c.pow(2)) / (2 * a * b + 1e-15), -1, 1)
                    #cos[torch.isnan(cos)] = 0
                    cos *= order3A
                    sin = torch.sqrt(1 - cos.pow(2)) * order3A
                    assert torch.sum(torch.isnan(sin)) == 0
                    return sin
                sin_src = calc_sin(m_d_src)
                sin_tgt = calc_sin(m_d_tgt)
                cum_sin += torch.abs(sin_src - sin_tgt)

            hyperM = torch.exp(- 1 / cfg.NGM.SIGMA3 * cum_sin) * order3A
            hyperM = hyperM.cuda()
            order3A = order3A.cuda()
        elif cfg.NGM.ORDER3_FEATURE == 'none':
            hyperM = torch.zeros_like(hyperA)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.ORDER3_FEATURE))

        # encode second order M to hyper tensor
        #hyperM_diag = torch.diagonal(hyperM, dim1=1, dim2=2)
        #hyperM_diag += M * F.relu(self.weight2)
        #hyperM_diag = torch.diagonal(hyperM, dim1=2, dim2=3)
        #hyperM_diag += M * F.relu(self.weight2)
        #hyperM_diag = torch.diagonal(hyperM, dim1=1, dim2=3)
        #hyperM_diag += M * F.relu(self.weight2)
        #hyperA_diag = torch.diagonal(hyperA, dim1=1, dim2=2)
        #hyperA_diag += A
        #hyperA_diag = torch.diagonal(hyperA, dim1=2, dim2=3)
        #hyperA_diag += A
        #hyperA_diag = torch.diagonal(hyperA, dim1=1, dim2=3)
        #hyperA_diag += A

        # RRWHM
        #v = self.rrwhm(hyperM, num_src=P_src.shape[1], ns_src=ns_src, ns_tgt=ns_tgt)
        #s = v.view(v.shape[0], P_tgt.shape[1], -1).transpose(1, 2)
        #ss = self.sinkhorn(s, ns_src, ns_tgt)
        #d, _ = self.displacement_layer(ss, P_src, P_tgt)
        #return ss, d

        hyperA = hyperA.cpu()
        hyperA_sum = torch.sum(hyperA, dim=tuple(range(2, 3 + 1)), keepdim=True) + 1e-10
        hyperA = hyperA / hyperA_sum
        hyperA = hyperA.to_sparse().coalesce().cuda()

        #hyperM2 = (M.unsqueeze(1).expand(hshape) + M.unsqueeze(2).expand(hshape) + M.unsqueeze(3).expand(hshape)) * order3A
        #hyperM += hyperM2 * F.relu(self.weight2)

        hyperM = hyperM.sparse_mask(hyperA)
        #hyperM = hyperM.to_sparse(len(hyperM.shape) - 1)
        #torch.cuda.empty_cache()
        hyperM = (hyperM._indices(), hyperM._values().unsqueeze(-1))

        if cfg.NGM.FIRST_ORDER:
            emb = Mp.transpose(1, 2).contiguous().view(Mp.shape[0], -1, 1)
        else:
            emb = torch.ones(M.shape[0], M.shape[1], 1, device=M.device)

        #I = Mp.transpose(1, 2).contiguous().view(Mp.shape[0], -1, 1)
        #A1 = (I > 0).to(I.dtype).squeeze(-1)

        #M = M * F.relu(self.weight2)
        A_sum = torch.sum(A, dim=2, keepdim=True) + 1e-10
        A = A / A_sum
        M = M.unsqueeze(-1)

        #pack_M = [I, M, hyperM]
        #pack_A = [A1, A, hyperA]
        pack_M = [M, hyperM]
        pack_A = [A, hyperA]
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            #M, emb2 = gnn_layer(A, M, emb)
            #hyperM, emb3 = gnn_layer(hyperA, hyperM, emb)
            #emb = emb2 + emb3
            #hyperM, emb = gnn_layer(hyperA, hyperM, emb)
            #M, emb = gnn_layer(A, M, emb)
            pack_M, emb = gnn_layer(pack_A, pack_M, emb, ns_src, ns_tgt, norm=False)

        v = self.classifier(emb)
        s = v.view(v.shape[0], P_tgt.shape[1], -1).transpose(1, 2)

        ss = self.bi_stochastic(s, ns_src, ns_tgt)

        d, _ = self.displacement_layer(ss, P_src, P_tgt)

        return ss, d
