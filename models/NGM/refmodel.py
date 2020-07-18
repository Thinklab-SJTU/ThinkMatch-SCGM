import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.lap_solvers.sinkhorn import Sinkhorn
from models.GMN.voting_layer import Voting
from models.GMN.displacement_layer import Displacement
from src.build_graphs import reshape_edge_feature
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_m
from models.NGM.gnn import GNNLayer
from models.GMN.affinity_layer import InnerpAffinity, GaussianAffinity
import math

from src.utils.config import cfg

CNN = eval('src.backbone.{}'.format(cfg.BACKBONE))


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        if cfg.NGM.EDGE_FEATURE == 'cat':
            self.affinity_layer = InnerpAffinity(cfg.NGM.FEATURE_CHANNEL)
        elif cfg.NGM.EDGE_FEATURE == 'geo':
            self.affinity_layer = GaussianAffinity(1, cfg.NGM.GAUSSIAN_SIGMA)
        else:
            raise ValueError('Unknown edge feature type {}'.format(cfg.NGM.EDGE_FEATURE))
        self.bi_stochastic = Sinkhorn(max_iter=cfg.NGM.BS_ITER_NUM, epsilon=cfg.NGM.BS_EPSILON)
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
                                     sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            else:
                #gnn_layer = Gconv(cfg.NGM.GNN_FEAT, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + (1 if cfg.NGM.SK_EMB else 0), 1)

    def forward(self, src, ref_model, P_src, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H, type='img'):
        if type == 'img' or type == 'image':
            # extract feature
            src_node = self.node_layers(src)
            src_edge = self.edge_layers(src_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)

            # arrange features
            U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
            F_src = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)
        elif type == 'feat' or type == 'feature':
            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
        else:
            raise ValueError('unknown type string {}'.format(type))

        U_tgt, F_tgt = ref_model

        if cfg.NGM.EDGE_FEATURE == 'cat':
            X = reshape_edge_feature(F_src, G_src, H_src)
            Y = reshape_edge_feature(F_tgt, G_tgt, H_tgt)
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

        emb_M = M.unsqueeze(-1)

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_M, emb = gnn_layer(A, emb_M, emb, ns_src, ns_tgt) #, norm=False)

        v = self.classifier(emb)
        s = v.view(v.shape[0], -1, P_src.shape[1]).transpose(1, 2)

        ss = self.voting_layer(s, ns_src, ns_tgt)
        ss = self.bi_stochastic(ss, ns_src, ns_tgt, dummy_row=True)

        d = 0

        if cfg.NGM.OUTP_SCORE:
            return s, ss, d
        else:
            return ss, d


class RefGraph(nn.Module):
    def __init__(self, cls_num_nodes, node_dim, edge_dim):
        """
        :param cls_num_nodes: dictionary {cls_name: num_nodes}
        :param node_dim: node feature dimension size
        :param edge_dim: edge feature dimension size
        """
        super(RefGraph, self).__init__()
        self.classes = []
        for cls in cls_num_nodes:
            num_node = cls_num_nodes[cls]
            node_feat = torch.empty(node_dim, num_node)
            edge_feat = torch.empty(edge_dim, num_node)
            node_feat.uniform_(- 1 / math.sqrt(node_dim), 1 / math.sqrt(node_dim))
            edge_feat.uniform_(- 1 / math.sqrt(edge_dim), 1 / math.sqrt(edge_dim))
            adjacency = torch.empty(num_node, num_node)
            self.register_parameter('{}_node'.format(cls), nn.Parameter(node_feat))
            self.register_parameter('{}_edge'.format(cls), nn.Parameter(edge_feat))
            self.register_parameter('{}_adj'.format(cls), nn.Parameter(adjacency))
            self.classes.append(cls)

    def get_ref(self, cls_list):
        node_feat = pad_stack([getattr(self, '{}_node'.format(cls)) for cls in cls_list])
        edge_feat = pad_stack([getattr(self, '{}_edge'.format(cls)) for cls in cls_list])
        adjacency = pad_stack([getattr(self, '{}_adj'.format(cls)) for cls in cls_list])
        return node_feat, edge_feat, adjacency

    def forward(self):
        print('Warning: forward function of RefGraph is not implemented and it should not be called.')
        pass


def pad_stack(data: list):
    """
    Pad data to same length and stack them.
    :param data: data dict
    :return: mini-batch
    """
    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor or type(inp[0]) == nn.Parameter
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            # pad_pattern = torch.from_numpy(np.asfortranarray(pad_pattern))
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor or type(inp[0]) == nn.Parameter:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    return stack(data)
