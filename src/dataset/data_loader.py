import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from src.build_graphs import build_graphs
from src.factorize_graph_matching import kronecker_sparse, kronecker_torch
from src.sparse_torch import CSRMatrix3d
from src.dataset import *

from src.utils.config import cfg

from itertools import combinations


class GMDataset(Dataset):
    def __init__(self, name, length, pad=16, cls=None, problem='pairwise', **args):
        self.name = name
        self.ds = eval(self.name)(**args)
        self.length = length  # NOTE images pairs are sampled randomly, so there is no exact definition of dataset size
                              # length here represents the iterations between two checkpoints
        self.obj_size = self.ds.obj_resize
        self.classes = self.ds.classes
        self.cls = None if cls == 'none' else cls
        self.problem_type = problem

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.problem_type == 'pairwise':
            return self.get_pair(idx, self.cls)
        elif self.problem_type == 'multi':
            return self.get_multi(idx, self.cls)
        else:
            raise NameError("Unknown problem type: {}".format(self.problem_type))

    def get_pair(self, idx, cls):
        #anno_pair, perm_mat = self.ds.get_pair(self.cls if self.cls is not None else
        #                                       (idx % (cfg.BATCH_SIZE * len(self.classes))) // cfg.BATCH_SIZE)
        try:
            anno_pair, perm_mat = self.ds.get_pair(cls, tgt_outlier=cfg.PAIR.REF_OUTLIER, src_outlier=cfg.PAIR.GT_OUTLIER)
        except TypeError:
            anno_pair, perm_mat = self.ds.get_pair(cls)
        if perm_mat.shape[0] <= 2 or perm_mat.size >= cfg.PAIR.MAX_PROB_SIZE > 0:
            return self.__getitem__(idx)

        cls = [anno['cls'] for anno in anno_pair]
        P1_gt = [(kp['x'], kp['y']) for kp in anno_pair[0]['keypoints']]
        P2_gt = [(kp['x'], kp['y']) for kp in anno_pair[1]['keypoints']]

        n1_gt, n2_gt = len(P1_gt), len(P2_gt)

        P1_gt = np.array(P1_gt)
        P2_gt = np.array(P2_gt)

        #P1 = P2 = make_grids((0, 0), cfg.PAIR.RESCALE, cfg.PAIR.CANDIDATE_SHAPE)
        #n1 = n2 = P1.shape[0]
        G1_gt, H1_gt, e1_gt = build_graphs(P1_gt, n1_gt, stg=cfg.PAIR.GT_GRAPH_CONSTRUCT, sym=cfg.PAIR.SYM_ADJACENCY)
        if cfg.PAIR.REF_GRAPH_CONSTRUCT == 'same':
            G2_gt = perm_mat.transpose().dot(G1_gt)
            H2_gt = perm_mat.transpose().dot(H1_gt)
            e2_gt= e1_gt
        else:
            G2_gt, H2_gt, e2_gt = build_graphs(P2_gt, n2_gt, stg=cfg.PAIR.REF_GRAPH_CONSTRUCT, sym=cfg.PAIR.SYM_ADJACENCY)


        #G2_gt = np.dot(perm_mat.transpose(), G1_gt)
        #H2_gt = np.dot(perm_mat.transpose(), H1_gt)
        #e2_gt = e1_gt
        #G1_gt, H1_gt, e1_gt = build_graphs(P1_gt, n1_gt, stg='fc')
        #G2_gt, H2_gt, e2_gt = build_graphs(P2_gt, n2_gt, stg='fc')
        #G1,    H1   , e1    = build_graphs(P1,    n1,    stg='fc')
        #G2,    H2   , e2    = build_graphs(P2,    n2,    stg='fc')

        ret_dict = {'Ps': [torch.Tensor(x) for x in [P1_gt, P2_gt]], # P1, P2]],
                    'ns': [torch.tensor(x) for x in [n1_gt, n2_gt]], # n1, n2]],
                    'es': [torch.tensor(x) for x in [e1_gt, e2_gt]], # e1, e2]],
                    'gt_perm_mat': perm_mat,
                    'Gs': [torch.Tensor(x) for x in [G1_gt, G2_gt]], # G1, G2]],
                    'Hs': [torch.Tensor(x) for x in [H1_gt, H2_gt]]} #H1, H2]]}

        imgs = [anno['image'] for anno in anno_pair]
        if imgs[0] is not None:
            trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
                    ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs
        elif 'feat' in anno_pair[0]['keypoints'][0]:
            feat1 = np.stack([kp['feat'] for kp in anno_pair[0]['keypoints']], axis=-1)
            feat2 = np.stack([kp['feat'] for kp in anno_pair[1]['keypoints']], axis=-1)
            ret_dict['features'] = [torch.Tensor(x) for x in [feat1, feat2]]

        return ret_dict

    def get_multi(self, idx, cls):
        if (self.ds.sets == 'test' and cfg.PAIR.TEST_ALL_GRAPHS) or (self.ds.sets == 'train' and cfg.PAIR.TRAIN_ALL_GRAPHS):
            num_graphs = self.ds.len(cls)
            anno_list, perm_mat_list = self.ds.get_multi(cls, num=num_graphs)
        else:
            anno_list, perm_mat_list = self.ds.get_multi(cls, num=cfg.PAIR.NUM_GRAPHS) # num_graphs)
        assert isinstance(perm_mat_list, list)
        refetch = False
        for pm in perm_mat_list:
            if pm.shape[0] <= 2 or pm.shape[1] <= 2 or pm.size >= cfg.PAIR.MAX_PROB_SIZE > 0:
                refetch = True
                break
        if refetch:
            return self.__getitem__(idx)

        cls = [anno['cls'] for anno in anno_list]
        Ps_gt = [[(kp['x'], kp['y']) for kp in anno_dict['keypoints']] for anno_dict in anno_list]

        ns_gt = [len(P_gt) for P_gt in Ps_gt]

        Ps_gt = [np.array(P_gt) for P_gt in Ps_gt]

        # P1 = P2 = make_grids((0, 0), cfg.PAIR.RESCALE, cfg.PAIR.CANDIDATE_SHAPE)
        # n1 = n2 = P1.shape[0]
        Gs_gt = []
        Hs_gt = []
        Gs_ref = []
        Hs_ref = []
        es_gt = []
        iterator = iter(zip(Ps_gt, ns_gt, perm_mat_list))
        P_gt, n_gt, perm_mat = next(iterator)
        G1_gt, H1_gt, _ = build_graphs(P_gt, n_gt, stg=cfg.PAIR.GT_GRAPH_CONSTRUCT)
        Gs_gt.append(G1_gt)
        Hs_gt.append(H1_gt)
        Gs_ref.append(G1_gt)
        Hs_ref.append(H1_gt)
        for P_gt, n_gt, perm_mat in iterator:
            if cfg.PAIR.REF_GRAPH_CONSTRUCT == 'same':
                G_gt = perm_mat.dot(G1_gt)
                H_gt = perm_mat.dot(H1_gt)
                G_ref = G_gt
                H_ref = H_gt
            else:
                G_gt, H_gt, _ = build_graphs(P_gt, n_gt, stg=cfg.PAIR.GT_GRAPH_CONSTRUCT)
                G_ref, H_ref, _ = build_graphs(P_gt, n_gt, stg=cfg.PAIR.REF_GRAPH_CONSTRUCT)
            Gs_gt.append(G_gt)
            Hs_gt.append(H_gt)
            Gs_ref.append(G_ref)
            Hs_ref.append(H_ref)

        ret_dict = {'Ps': [torch.Tensor(x) for x in Ps_gt],
                    'ns': [torch.tensor(x) for x in ns_gt],
                    'es': [torch.tensor(x) for x in es_gt],
                    'gt_perm_mat': perm_mat_list,
                    'Gs': [torch.Tensor(x) for x in Gs_gt],
                    'Hs': [torch.Tensor(x) for x in Hs_gt],
                    'Gs_ref': [torch.Tensor(x) for x in Gs_ref],
                    'Hs_ref': [torch.Tensor(x) for x in Hs_ref],
                    'cls': [str(x) for x in cls]}

        imgs = [anno['image'] for anno in anno_list]
        if imgs[0] is not None:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
            ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs
        elif 'feat' in anno_list[0]['keypoints'][0]:
            feats = [np.stack([kp['feat'] for kp in anno_dict['keypoints']], axis=-1) for anno_dict in anno_list]
            ret_dict['features'] = [torch.Tensor(x) for x in feats]

        return ret_dict

class MGMCDataset(GMDataset):
    """
    Multi-Graph Matching and Clustering dataset
    """
    def __init__(self, name, length, pad=16, cls=None, problem='multi_cluster', **args):
        super(MGMCDataset, self).__init__(name, length, pad, cls, problem, **args)
        self.classes = list(combinations(self.ds.classes, cfg.PAIR.NUM_CLUSTERS))
        assert self.problem_type == 'multi_cluster'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.get_multi_cluster(idx)

    def get_multi_cluster(self, idx):
        dicts = []
        if self.cls is None or self.cls == 'none':
            cls_iterator = random.choice(self.classes)
        else:
            cls_iterator = self.cls
        for cls in cls_iterator:
            dicts.append(self.get_multi(idx, cls))
        ret_dict = {}
        for key in dicts[0]:
            ret_dict[key] = []
            for dic in dicts:
                ret_dict[key] += dic[key]
        return ret_dict


class GMRefDataset(Dataset):
    def __init__(self, name, length=None, pad=None, cls=None, **args):
        self.name = name
        self.ds = eval(self.name)(**args)

        self.obj_size = self.ds.obj_resize
        self.classes = self.ds.classes
        self.cls = None if cls == 'none' else cls

    def __len__(self):
        if self.cls is None:
            self.length = self.ds.length
        else:
            self.length = self.ds.length_of(self.cls)
        return self.length

    def __getitem__(self, idx):
        def find_cls(dataset_idx):
            for cls_idx, start_idx in enumerate(cls_start_idx[::-1]):
                if dataset_idx >= start_idx:
                    return self.classes[-cls_idx - 1]

        if self.cls is None:
            cls_start_idx = []
            cum_idx = 0
            for cls in self.classes:
                cls_start_idx.append(cum_idx)
                cum_idx += self.ds.length_of(cls)

            cls = find_cls(idx)

        else:
            cls_start_idx = [0] * len(self.classes)

            cls = self.cls

        anno_dict, perm_mat = self.ds.get_single_to_ref(idx - cls_start_idx[self.classes.index(cls)], cls)

        cls = anno_dict['cls']
        P_gt = [(kp['x'], kp['y']) for kp in anno_dict['keypoints']]

        n_gt = len(P_gt)
        n_ref = perm_mat.shape[1]

        if n_gt < 3:
            return self.__getitem__(random.randint(0, len(self) - 1))

        P_gt = np.array(P_gt)

        G_gt, H_gt, e_gt = build_graphs(P_gt, n_gt, stg=cfg.PAIR.GT_GRAPH_CONSTRUCT)
        G_ref, H_ref, e_ref = build_graphs(np.zeros((n_ref, 2)), n_ref, stg=cfg.PAIR.REF_GRAPH_CONSTRUCT)

        ret_dict = {'Ps': torch.Tensor(P_gt),
                    'ns': [torch.tensor(x) for x in (n_gt, n_ref)],
                    'es': [torch.tensor(x) for x in (e_gt, e_ref)],
                    'gt_perm_mat': perm_mat,
                    'Gs': [torch.Tensor(x) for x in (G_gt, G_ref)],
                    'Hs': [torch.Tensor(x) for x in (H_gt, H_ref)],
                    'cls': cls}

        img = anno_dict['image']
        if img is not None:
            trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
                    ])
            img = trans(img)
            ret_dict['images'] = img
        elif 'feat' in anno_dict['keypoints'][0]:
            feat = np.stack([kp['feat'] for kp in anno_dict['keypoints']], axis=-1)
            ret_dict['features'] = torch.Tensor(feat)

        return ret_dict

    def __getitem_test(self, idx):
        assert self.cls is not None

        indices = random.sample(range(0, self.ds.length_of(self.cls)), 2)

        anno_list = []
        perm_mat_list = []
        for idx in indices:
            anno_dict, perm_mat = self.ds.get_single_to_ref(idx, self.cls)
            anno_list.append(anno_dict)
            perm_mat_list.append(perm_mat)

        cls = [anno_dict['cls'] for anno_dict in anno_list]
        P_gt = [[(kp['x'], kp['y']) for kp in anno_dict['keypoints']] for anno_dict in anno_list]

        n_gt = [len(x) for x in P_gt]
        n_ref = [perm_mat.shape[1] for perm_mat in perm_mat_list]

        if n_gt < 3:
            return self.__getitem_test(random.randint(0, len(self) - 1))

        P_gt = np.array(P_gt)

        G_gt, H_gt, e_gt = build_graphs(P_gt, n_gt, stg=cfg.PAIR.GT_GRAPH_CONSTRUCT)
        G_ref, H_ref, e_ref = build_graphs(np.zeros((n_ref, 2)), n_ref, stg=cfg.PAIR.REF_GRAPH_CONSTRUCT)

        ret_dict = {'Ps': torch.Tensor(P_gt),
                    'ns': [torch.tensor(x) for x in (n_gt, n_ref)],
                    'es': [torch.tensor(x) for x in (e_gt, e_ref)],
                    'gt_perm_mat': perm_mat,
                    'Gs': [torch.Tensor(x) for x in (G_gt, G_ref)],
                    'Hs': [torch.Tensor(x) for x in (H_gt, H_ref)],
                    'cls': cls}

        img = anno_dict['image']
        if img is not None:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
            ])
            img = trans(img)
            ret_dict['images'] = img
        elif 'feat' in anno_dict['keypoints'][0]:
            feat = np.stack([kp['feat'] for kp in anno_dict['keypoints']], axis=-1)
            ret_dict['features'] = torch.Tensor(feat)

        return ret_dict


class QAPDataset(Dataset):
    def __init__(self, name, length, pad=16, cls=None, **args):
        self.name = name
        self.ds = eval(self.name)(**args, cls=cls)
        self.obj_size = self.ds.obj_resize
        self.classes = self.ds.classes
        self.cls = None if cls == 'none' else cls
        self.length = length

    def __len__(self):
        #return len(self.ds.data_list)
        return self.length

    def __getitem__(self, idx):
        Fi, Fj, perm_mat, sol, name = self.ds.get_pair(idx % len(self.ds.data_list))
        if perm_mat.size <= 2 * 2 or perm_mat.size >= cfg.PAIR.MAX_PROB_SIZE > 0:
            return self.__getitem__(random.randint(0, len(self) - 1))

        #if np.max(ori_aff_mat) > 0:
        #    norm_aff_mat = ori_aff_mat / np.mean(ori_aff_mat)
        #else:
        #    norm_aff_mat = ori_aff_mat

        ret_dict = {'Fi': Fi,
                    'Fj': Fj,
                    'gt_perm_mat': perm_mat,
                    'ns': [torch.tensor(x) for x in perm_mat.shape],
                    'solution': torch.tensor(sol),
                    'name': name}

        return ret_dict


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """
    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
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
            #pad_pattern = torch.from_numpy(np.asfortranarray(pad_pattern))
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
        elif type(inp[0]) == torch.Tensor:
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

    ret = stack(data)

    # compute CPU-intensive matrix K1, K2 here to leverage multi-processing nature of dataloader
    if 'Gs' in ret and 'Hs' in ret:
        if len(ret['Gs']) == 2:
            G1_gt, G2_gt = ret['Gs']
            H1_gt, H2_gt = ret['Hs']
            if cfg.FP16:
                sparse_dtype = np.float16
            else:
                sparse_dtype = np.float32
            K1G = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(G2_gt, G1_gt)]  # 1 as source graph, 2 as target graph
            K1H = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2_gt, H1_gt)]
            K1G = CSRMatrix3d(K1G)
            K1H = CSRMatrix3d(K1H).transpose()

            ret['Ks'] = K1G, K1H #, K1G.transpose(keep_type=True), K1H.transpose(keep_type=True)
        elif 'Gs_ref' in ret and 'Hs_ref' in ret:
            ret['KGs'] = dict()
            ret['KHs'] = dict()
            '''
            for idx_gt, idx_ref in combinations(range(len(ret['Gs'])), 2):
            #for idx_gt, idx_ref in product(range(len(ret['Gs'])), repeat=2):
                if idx_gt > idx_ref:
                    idx_gt, idx_ref = idx_ref, idx_gt
                G1_gt = ret['Gs'][idx_gt]
                H1_gt = ret['Hs'][idx_gt]
                G2_gt = ret['Gs_ref'][idx_ref]
                H2_gt = ret['Hs_ref'][idx_ref]
                if cfg.FP16:
                    sparse_dtype = np.float16
                else:
                    sparse_dtype = np.float32
                KG = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in
                       zip(G2_gt, G1_gt)]  # 1 as source graph, 2 as target graph
                KH = [kronecker_sparse(x, y).astype(sparse_dtype) for x, y in zip(H2_gt, H1_gt)]
                KG = CSRMatrix3d(KG)
                KH = CSRMatrix3d(KH).transpose()
                ret['KGs']['{},{}'.format(idx_gt, idx_ref)] = KG
                ret['KHs']['{},{}'.format(idx_gt, idx_ref)] = KH
            '''
        else:
            raise ValueError('Data type not understood.')

    if 'Fi' in ret and 'Fj' in ret:
        Fi = ret['Fi']
        Fj = ret['Fj']
        aff_mat = kronecker_torch(Fj, Fi)
        ret['ori_affmat'] = aff_mat
        dmax = (torch.max(torch.sum(aff_mat, dim=2, keepdim=True), dim=1, keepdim=True).values + 1e-5)
        ret['affmat'] = aff_mat / dmax * 1000

    return ret


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.DATALOADER_NUM, collate_fn=collate_fn,
        pin_memory=False, worker_init_fn=worker_init_fix if fix_seed else worker_init_rand
    )


if __name__ == '__main__':
    a = GMDataset('PascalVOC', length=1000 ,sets='train', obj_resize=(256, 256))
    k = a[0]
    pass