import torch
import torch.nn as nn

from src.lap_solvers.sinkhorn import Sinkhorn
from models.GMN.displacement_layer import Displacement
from src.feature_align import feature_align
from src.gconv import Siamese_Gconv
from models.PCA.affinity_layer import Affinity
from src.utils.model_sl import load_model

from src.utils.config import cfg

from src.backbone import *
CNN = eval(cfg.BACKBONE)


class Net(CNN):
    def __init__(self, pretrained_pca=True):
        super(Net, self).__init__()
        # PCA layers
        self.bi_stochastic = Sinkhorn(max_iter=cfg.PCA.BS_ITER_NUM, epsilon=cfg.PCA.BS_EPSILON, tau=1 / cfg.PCA.VOTING_ALPHA)
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
            load_model(self, 'output/vgg16_pca_voc/params/params_0005.pt.6540')

        self.fr_feat = nn.Parameter(torch.Tensor([0.8]))

        # RNN layers
        #self.forget_gate_feat = nn.Sequential(
        #    nn.Linear(cfg.PCA.FEATURE_CHANNEL * 4, cfg.PCA.FEATURE_CHANNEL * 2), nn.ReLU(),
        #    nn.Linear(cfg.PCA.FEATURE_CHANNEL * 2, 1), nn.Sigmoid()
        #)
        #self.forget_gate_adj = nn.Sequential(
        #    nn.Linear(cfg.PCA.FEATURE_CHANNEL * 4, cfg.PCA.FEATURE_CHANNEL * 2), nn.ReLU(),
        #    nn.Linear(cfg.PCA.FEATURE_CHANNEL * 2, 1), nn.Sigmoid()
        #)
        #ref_feat_dim = cfg.PCA.FEATURE_CHANNEL * 2
        #self.init_ref_feat = torch.zeros(ref_feat_dim, 1)
        #nn.Parameter(
        #    torch.empty(1, ref_feat_dim, 1).uniform_(- 1 / math.sqrt(ref_feat_dim), 1 / math.sqrt(ref_feat_dim))
        #)

    def forward(self, data, Ps, Gs, Hs, ns, iter_times=2, type='img'):
        #device = data[0].device
        #batch_size = Ps[0].shape[0]
        #max_num_nodes = Ps[0].shape[1]
        #ref_feat = torch.zeros(batch_size, self.init_ref_feat.shape[0], max_num_nodes).to(device)
        #ref_adj = torch.zeros(batch_size, max_num_nodes, max_num_nodes).to(device)
        #for b in range(batch_size):
        #    ref_feat[b, :, :ns[0][b]] = self.init_ref_feat.expand_as(ref_feat[b, :, :ns[0][b]])
        #    ref_adj[b, :ns[0][b], :ns[0][b]] = 1.

        # extract reference graph feature
        iterator = zip(data, Ps, Gs, Hs, ns)
        tgt, P_tgt, G_tgt, H_tgt, ns_tgt = next(iterator)
        if type == 'img' or type == 'image':
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
            tgt_feat = torch.cat((U_tgt, F_tgt), dim=1)
        elif type == 'feat' or type == 'feature':
            tgt_feat = tgt
        else:
            raise ValueError('unknown type string {}'.format(type))

        A_tgt = torch.bmm(G_tgt, H_tgt.transpose(1, 2))

        pred_s = []
        indices = []
        for i in range(iter_times):
            iterator = enumerate(zip(data, Ps, Gs, Hs, ns))
            _ = next(iterator)
            for idx, (src, P_src, G_src, H_src, ns_src) in iterator:
                # pca forward
                s, src_feat = self.__pca_forward(src, tgt_feat, P_src, G_src, H_src, A_tgt, ns_src, ns_tgt, type)

                # compute forget rate
                #fr_feat = self.forget_gate_feat(torch.cat((src_feat, tgt_feat), dim=1).transpose(1, 2)).transpose(1, 2)
                #fr_adj = self.forget_gate_adj(torch.cat((src_feat.max(-1)[0], ref_feat.max(-1)[0]), dim=-1)).unsqueeze(-1)
                fr_feat = self.fr_feat ** (i + 1)
                #s_hung = s * hungarian(s, ns_src, ns_tgt)
                s_sum = torch.sum(s, dim=1, keepdim=True)
                #fr_feat = fr_feat * s_sum

                #s_hungarian = hungarian(s, ns_src, ns_src)
                tgt_feat = fr_feat * torch.matmul(src_feat, s) \
                           + (1 - fr_feat * s_sum) * tgt_feat
                #ref_adj = fr_adj * torch.matmul(s_hungarian.transpose(1, 2), torch.matmul(src_adj, s_hungarian)) \
                #          + (1 - fr_adj) * ref_adj
                pred_s.append(s)
                indices.append(idx)

        return pred_s, indices

    def __pca_forward(self, src, tgt, P_src, G_src, H_src, A_tgt, ns_src, ns_tgt, type='img'):
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

        U_tgt = tgt[:, :tgt.shape[1] // 2, :]
        F_tgt = tgt[:, tgt.shape[1] // 2:, :]

        # adjacency matrix
        A_src = torch.bmm(G_src, H_src.transpose(1, 2))

        emb1, emb2 = torch.cat((U_src, F_src), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt), dim=1).transpose(1, 2)
        ss = []

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
            affinity = getattr(self, 'affinity_{}'.format(i))
            s = affinity(emb1, emb2)
            #s = torch.bmm(emb1, emb2.transpose(1, 2))
            s = self.bi_stochastic(s, ns_src, ns_tgt, dummy_row=True)
            ss.append(s)

            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                emb1 = new_emb1
                emb2 = new_emb2
        '''
        s = None
        for x in range(6):
            for i in range(self.gnn_layer):
                if i == 0:
                    if s is None:
                        gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                        emb1_0, emb2_0 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                        s = torch.zeros(emb1.shape[0], emb1.shape[1], emb2.shape[1]).cuda()

                    cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                    emb1 = cross_graph(torch.cat((emb1_0, torch.bmm(s, emb2_0)), dim=-1))
                    emb2 = cross_graph(torch.cat((emb2_0, torch.bmm(s.transpose(1, 2), emb1_0)), dim=-1))
                if i==1:
                    gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                    emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                    affinity = getattr(self, 'affinity_{}'.format(i))
                    s = affinity(emb1, emb2)
                s = self.bi_stochastic(s, ns_src, ns_tgt, dummy_row=True)
                    ss.append(s)
        '''

        return ss[-1], torch.cat((U_src, F_src), dim=1)
