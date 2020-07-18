import torch
import torch.nn as nn
from itertools import combinations

from src.lap_solvers.sinkhorn import Sinkhorn
from models.GMN.displacement_layer import Displacement
from src.feature_align import feature_align
from src.gconv import Siamese_Gconv
from models.PCA.affinity_layer import Affinity
from src.utils.model_sl import load_model

from src.utils.config import cfg

CNN = eval('src.backbone.{}'.format(cfg.BACKBONE))


class Net(CNN):
    def __init__(self, pretrained_pca=True):
        super(Net, self).__init__()
        # PCA layers
        self.bi_stochastic = Sinkhorn(max_iter=cfg.PCA.BS_ITER_NUM, epsilon=cfg.PCA.BS_EPSILON, tau=1 / cfg.PCA.VOTING_ALPHA)
        self.bi_stochastic2 = Sinkhorn(max_iter=cfg.PCA.BS_ITER_NUM, epsilon=cfg.PCA.BS_EPSILON, tau=1 / 2.)
        self.displacement_layer = Displacement()
        self.l2norm = nn.LocalResponseNorm(cfg.PCA.FEATURE_CHANNEL * 2, alpha=cfg.PCA.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.gnn_layer = cfg.PCA.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PCA.FEATURE_CHANNEL * 2, cfg.PCA.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PCA.GNN_FEAT, cfg.PCA.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.PCA.GNN_FEAT))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.PCA.GNN_FEAT * 2, cfg.PCA.GNN_FEAT))

        if pretrained_pca:
            #load_model(self, 'output/vgg16_pca_voc/params/params_0008.pt.out.5858')
            load_model(self, 'output/vgg16_pca_voc/params/params_0003.pt.4willow')

    def forward(self, data, Ps, Gs, Hs, ns, type='img', **kwargs):
        batch_size = data[0].shape[0]
        device = data[0].device

        # extract reference graph feature
        feat_list = []
        joint_indices = [0]
        iterator = zip(data, Ps, Gs, Hs, ns)
        for idx, (dat, P, G, H, n) in enumerate(iterator):
            if type == 'img' or type == 'image':
                node = self.node_layers(dat)
                edge = self.edge_layers(node)
                node = self.l2norm(node)
                edge = self.l2norm(edge)
                U = feature_align(node, P, n, cfg.PAIR.RESCALE)
                F = feature_align(edge, P, n, cfg.PAIR.RESCALE)
                feat = torch.cat((U, F), dim=1)
            elif type == 'feat' or type == 'feature':
                feat = dat
            else:
                raise ValueError('unknown type string {}'.format(type))
            A = torch.bmm(G, H.transpose(1, 2))
            feat_list.append((idx, feat, A, n))
            joint_indices.append(joint_indices[-1] + P.shape[1])

        joint_S = torch.zeros(batch_size, joint_indices[-1], joint_indices[-1], device=device)
        joint_S_diag = torch.diagonal(joint_S, dim1=1, dim2=2)
        joint_S_diag += 1

        pred_s = []
        indices = []
        for src, tgt in combinations(feat_list, 2):
            # pca forward
            src_idx, src_feat, A_src, n_src = src
            tgt_idx, tgt_feat, A_tgt, n_tgt = tgt
            s = self.__pca_forward(src_feat, tgt_feat, A_src, A_tgt, n_src, n_tgt)

            if src_idx > tgt_idx:
                joint_S[:, joint_indices[tgt_idx]:joint_indices[tgt_idx+1], joint_indices[src_idx]:joint_indices[src_idx+1]] += s.transpose(1, 2)
            else:
                joint_S[:, joint_indices[src_idx]:joint_indices[src_idx+1], joint_indices[tgt_idx]:joint_indices[tgt_idx+1]] += s

        matching_s = []
        for b in range(batch_size):
            e, v = torch.symeig(joint_S[b], eigenvectors=True)
            sort_idx = torch.argsort(torch.abs(e), descending=True, dim=-1)[:joint_indices[1]]
            matching_s.append(len(data) * torch.mm(v[:, sort_idx], v[:, sort_idx].transpose(0, 1)))

        matching_s = torch.stack(matching_s, dim=0)

        for idx1, idx2 in combinations(range(len(data)), 2):
            if idx1 == 0 or idx2 == 0:
                continue
            s = matching_s[:, joint_indices[idx1]:joint_indices[idx1 + 1], joint_indices[idx2]:joint_indices[idx2 + 1]]
            s = self.bi_stochastic2(s)
            pred_s.append(s)
            indices.append((idx1, idx2))

        return pred_s, indices

    def __pca_forward(self, src, tgt, A_src, A_tgt, ns_src, ns_tgt):
        emb1, emb2 = src.transpose(1, 2), tgt.transpose(1, 2)
        ss = []

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
            affinity = getattr(self, 'affinity_{}'.format(i))
            s = affinity(emb1, emb2)
            s = self.bi_stochastic(s, ns_src, ns_tgt, dummy_row=True)
            ss.append(s)

            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                emb1 = new_emb1
                emb2 = new_emb2

        return ss[-1]
