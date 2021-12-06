import torch
import itertools

from models.BBGM.affinity_layer import InnerProductWithWeightsAffinity
from models.BBGM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat
from src.utils.pad_tensor import pad_tensor
from models.NGM.gnn import GNNLayer
from src.lap_solvers.sinkhorn import Sinkhorn
from src.lap_solvers.hungarian import hungarian

from src.utils.config import cfg
from src.utils.c_loss import simclr_loss
import copy
from src.backbone import *
CNN = eval(cfg.BACKBONE)


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.NGM.FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = cfg.NGM.FEATURE_CHANNEL * 2
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)

        self.rescale = cfg.PROBLEM.RESCALE
        self.mlp = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.ReLU(),
            nn.Linear(1024, 256, bias=False)
        )
        self.tau = cfg.NGM.SK_TAU
        self.mgm_tau = cfg.NGM.MGM_SK_TAU
        self.univ_size = cfg.NGM.UNIV_SIZE

        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)
        self.sinkhorn_mgm = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, epsilon=cfg.NGM.SK_EPSILON, tau=self.mgm_tau)
        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            tau = cfg.NGM.SK_TAU
            if i == 0:
                gnn_layer = GNNLayer(1, 1,
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
            else:
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, 1)

    def forward(
        self,
        data_dict,
    ):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        batch_size = data_dict['batch_size']
        num_graphs = len(images)

        if cfg.PROBLEM.TYPE == '2GM' and 'gt_perm_mat' in data_dict:
            gt_perm_mats = [data_dict['gt_perm_mat']]
        elif cfg.PROBLEM.TYPE == 'MGM' and 'gt_perm_mat' in data_dict:
            perm_mat_list = data_dict['gt_perm_mat']
            gt_perm_mats = [torch.bmm(pm_src, pm_tgt.transpose(1, 2)) for pm_src, pm_tgt in lexico_iter(perm_mat_list)]
        else:
            raise ValueError('Ground truth information is required during training.')

        global_list = []
        orig_graph_list = []

        old_shape = data_dict['gt_perm_mat'][0].shape
        max_shape = max(data_dict['gt_perm_mat'][0].shape)
        perm_old = torch.zeros([len(data_dict['gt_perm_mat']), max_shape, max_shape],
                               device=data_dict['gt_perm_mat'].device)
        perm_old[:, 0:old_shape[0], 0:old_shape[1]] = data_dict['gt_perm_mat']
        # perm_list = []
        node_feature_cl = []
        idx = 0
        perms = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            idx += 1
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
            node_features = torch.cat((U, F), dim=1)

            if cfg.PROBLEM.SSL:
                n_p_c = n_p.cpu().detach().numpy()

                if cfg.SSL.C_LOSS:
                    node_features_format = torch.zeros([perm_old.shape[0], perm_old.shape[1], node_features.shape[1]],
                                                       device=perm_old.device)  # B x N x D
                    pre = 0
                    for i in range(len(n_p_c)):
                        node_features_format[i][0: n_p_c[i]] = node_features[pre: pre + n_p_c[i]]
                        pre += int(n_p_c[i])
                    node_feature_cl.append(node_features_format)

                if idx == 2:
                    pre = 0
                    rate = cfg.SSL.MIX_RATE
                    for i in range(len(n_p_c)):
                        perm = torch.randperm(int(n_p_c[i]))
                        perms.append(perm)
                        if not cfg.SSL.MIX_DETACH:
                            node_features[pre: pre + n_p_c[i]] = node_features[pre: pre + n_p_c[i]] * (
                                        1 - rate) + rate * \
                                                                 node_features[pre + perm]
                        else:
                            node_features[pre: pre + n_p_c[i]] = node_features[pre: pre + n_p_c[i]] * (
                                        1 - rate) + rate * \
                                                                 node_features[pre + perm].detach()

                        pre += int(n_p_c[i])

            # perm_list.append(perms)
            graph.x = node_features

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        if cfg.PROBLEM.SSL and cfg.SSL.C_LOSS:
            z1 = node_feature_cl[0]
            z2 = node_feature_cl[1]
            z2_ = torch.bmm(perm_old, z2)
            z1_cross = torch.zeros(z1.shape, device=z1.device)
            z2_cross = torch.zeros(z1.shape, device=z1.device)
            for i in range(len(z2_)):
                non_zeros = z2_[i].sum(axis=1).nonzero()[:, 0]
                z2_cross[i][0: len(non_zeros)] = z2_[i][non_zeros]
                z1_cross[i][0: len(non_zeros)] = z1[i][non_zeros]
            c_loss = simclr_loss(torch.nn.functional.normalize(self.mlp(z1_cross), dim=-1),
                                 torch.nn.functional.normalize(self.mlp(z2_cross), dim=-1))
            data_dict['c_loss'] = c_loss

        if cfg.PROBLEM.SSL and not cfg.SSL.MIX_DETACH:
            perm_new = torch.zeros(perm_old.shape).to(perm_old.device)
            for i in range(len(perm_old)):
                perm = perms[i]
                for j in range(len(perm)):
                    perm_new[i][j, perm[j]] = 1
            perm_new_ = torch.bmm(perm_old, perm_new)[:, 0: old_shape[0], 0: old_shape[1]]
            data_dict['gt_perm_mat_old'] = copy.deepcopy(data_dict['gt_perm_mat'])
            data_dict['gt_perm_mat_new'] = perm_new_

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights,
                                 use_global=cfg.SSL.USE_GLOBAL)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]
        # unary_affs_list = unary_affs_list_[:, 0]
        # x_ = unary_affs_list_[:, 1]
        # y_ = unary_affs_list_[:, 2]

        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights,
                               use_global=cfg.SSL.USE_GLOBAL)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]

        s_list, mgm_s_list, x_list, mgm_x_list, indices = [], [], [], [], []

        for unary_affs, quadratic_affs, (idx1, idx2) in zip(unary_affs_list, quadratic_affs_list, lexico_iter(range(num_graphs))):
            kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict['KGHs']['{},{}'.format(idx1, idx2)]
            Kp = torch.stack(pad_tensor(unary_affs), dim=0)
            Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
            K = construct_aff_mat(Ke, Kp, kro_G, kro_H)
            if num_graphs == 2: data_dict['aff_mat'] = K

            if cfg.NGM.FIRST_ORDER:
                emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
            else:
                emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

            if cfg.NGM.POSITIVE_EDGES:
                A = (K > 0).to(K.dtype)
            else:
                A = (K != 0).to(K.dtype)

            emb_K = K.unsqueeze(-1)

            # NGM qap solver
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb_K, emb = gnn_layer(A, emb_K, emb, n_points[idx1], n_points[idx2])

            v = self.classifier(emb)
            s = v.view(v.shape[0], points[idx2].shape[1], -1).transpose(1, 2)

            ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
            x = hungarian(ss, n_points[idx1], n_points[idx2])
            s_list.append(ss)
            x_list.append(x)
            indices.append((idx1, idx2))

        if num_graphs > 2:
            joint_indices = torch.cat((torch.cumsum(torch.stack([torch.max(np) for np in n_points]), dim=0), torch.zeros((1,), dtype=torch.long, device=K.device)))
            joint_S = torch.zeros(batch_size, torch.max(joint_indices), torch.max(joint_indices), device=K.device)
            for idx in range(num_graphs):
                for b in range(batch_size):
                    start = joint_indices[idx-1]
                    joint_S[b, start:start+n_points[idx][b], start:start+n_points[idx][b]] += torch.eye(n_points[idx][b], device=K.device)

            for (idx1, idx2), s in zip(indices, s_list):
                if idx1 > idx2:
                    joint_S[:, joint_indices[idx2-1]:joint_indices[idx2], joint_indices[idx1-1]:joint_indices[idx1]] += s.transpose(1, 2)
                else:
                    joint_S[:, joint_indices[idx1-1]:joint_indices[idx1], joint_indices[idx2-1]:joint_indices[idx2]] += s

            matching_s = []
            for b in range(batch_size):
                e, v = torch.symeig(joint_S[b], eigenvectors=True)
                diff = e[-self.univ_size:-1] - e[-self.univ_size+1:]
                if self.training and torch.min(torch.abs(diff)) <= 1e-4:
                    matching_s.append(joint_S[b])
                else:
                    matching_s.append(num_graphs * torch.mm(v[:, -self.univ_size:], v[:, -self.univ_size:].transpose(0, 1)))

            matching_s = torch.stack(matching_s, dim=0)

            for idx1, idx2 in indices:
                s = matching_s[:, joint_indices[idx1-1]:joint_indices[idx1], joint_indices[idx2-1]:joint_indices[idx2]]
                s = self.sinkhorn_mgm(torch.log(torch.relu(s)), n_points[idx1], n_points[idx2]) # only perform row/col norm, do not perform exp
                x = hungarian(s, n_points[idx1], n_points[idx2])

                mgm_s_list.append(s)
                mgm_x_list.append(x)

        if cfg.PROBLEM.TYPE == '2GM':
            data_dict.update({
                'ds_mat': s_list[0],
                'perm_mat': x_list[0]
            })
        elif cfg.PROBLEM.TYPE == 'MGM':
            data_dict.update({
                'ds_mat_list': mgm_s_list,
                'perm_mat_list': mgm_x_list,
                'graph_indices': indices,
                'gt_perm_mat_list': gt_perm_mats
            })

        return data_dict
