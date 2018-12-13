import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from data.pascal_voc import PascalVOC
from utils.build_graphs import build_graphs, make_grids
from utils.fgm import kronecker_sparse
from sparse_torch import CSRMatrix3d

from utils.config import cfg


class GMDataset(Dataset):
    def __init__(self, name, length, pad=16, cls=None, **args):
        self.name = name
        self.ds = eval(self.name)(**args)
        self.length = length  # NOTE images pairs are sampled randomly, so there is no exact definition of dataset size
        self.pad = pad  # Matrix size varies from different pairs. They should be padded to equal size.
        self.epad = 6 * self.pad  # pad edges. *6 as we apply delaunay triangulation to nodes.
        self.obj_size = self.ds.obj_resize
        self.classes = self.ds.classes
        self.cls = cls

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        anno_pair, perm_mat = self.ds.get_pair(self.cls if self.cls is not None else
                                               (idx % (cfg.BATCH_SIZE * len(self.classes))) // cfg.BATCH_SIZE)
        # todo this operation may affect gradient
        if perm_mat.size <= 2 * 2:
            return self.__getitem__(idx)

        imgs = [anno['image'] for anno in anno_pair]
        cls = [anno['cls'] for anno in anno_pair]
        P1_gt = [(kp['x'], kp['y']) for kp in anno_pair[0]['keypoints']]
        P2_gt = [(kp['x'], kp['y']) for kp in anno_pair[1]['keypoints']]

        n1_gt, n2_gt = len(P1_gt), len(P2_gt)

        #for P in (P1_gt, P2_gt):
        #    while len(P) < self.pad:
        #        P.append((0, 0))
        P1_gt = np.array(P1_gt)
        P2_gt = np.array(P2_gt)

        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
                ])
        imgs = [trans(img) for img in imgs]
        #perm_mat = np.pad(perm_mat, ((0, self.pad - perm_mat.shape[0]),), 'constant', constant_values=0)

        # compute CPU-intensive matrix K1, K2 here to leverage multi-processing nature of dataloader
        P1 = P2 = make_grids((0, 0), cfg.PAIR.RESCALE, cfg.PAIR.CANDIDATE_SHAPE)
        n1 = n2 = P1.shape[0]
        G1_gt, H1_gt, e1_gt = build_graphs(P1_gt, n1_gt, stg='tri')#, n_pad=self.pad, edge_pad=self.epad)
        G2_gt, H2_gt, e2_gt = build_graphs(P2_gt, n2_gt, stg='tri')#, n_pad=self.pad, edge_pad=self.epad)
        G1,    H1   , e1    = build_graphs(P1,    n1,    stg='fc')
        G2,    H2   , e2    = build_graphs(P2,    n2,    stg='fc')
        #K1G = kronecker_sparse(G2, G1_gt)  # 1 as source graph, 2 as target graph
        #K1H = kronecker_sparse(H2, H1_gt)

        def pad_edge(inarr: np.ndarray, pad_len):
            return np.pad(inarr, (0, pad_len - len(inarr)), 'constant', constant_values=0)

        return {'images': imgs,
                'Ps': [torch.Tensor(x) for x in [P1_gt, P2_gt, P1, P2]],
                'ns': [torch.tensor(x) for x in [n1_gt, n2_gt, n1, n2]],
                'es': [torch.tensor(x) for x in [e1_gt, e2_gt, e1, e2]],
                'gt_perm_mat': perm_mat,
                'Gs': [torch.Tensor(x) for x in [G1_gt, G2_gt, G1, G2]],
                'Hs': [torch.Tensor(x) for x in [H1_gt, H2_gt, H1, H2]]}
                #'Ks': [{'col': x.col, #pad_edge(x.col, x.shape[1]),
                #        'row': x.row, #pad_edge(x.row, x.shape[1]),
                #        'data': x.data, #pad_edge(x.data, x.shape[1]),
                #        'shape': torch.Tensor(x.shape)} for x in [K1G, K1H]]}


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
            pad_pattern = torch.from_numpy(np.asfortranarray(pad_pattern))
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
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    # Compute K
    G1_gt, G2_gt, G1, G2 = ret['Gs']
    H1_gt, H2_gt, H1, H2 = ret['Hs']
    K1G = [kronecker_sparse(x, y) for x, y in zip(G2, G1_gt)]  # 1 as source graph, 2 as target graph
    K1H = [kronecker_sparse(x, y) for x, y in zip(H2, H1_gt)]
    K1G = CSRMatrix3d(K1G)
    K1H = CSRMatrix3d(K1H).transpose()

    ret['Ks'] = K1G, K1H

    return ret


def worker_init_fn(worker_id):
    """
    Init random seed for dataloader workers.
    """
    random.seed(torch.initial_seed())


def get_dataloader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.BATCH_SIZE,
                                       collate_fn=collate_fn, worker_init_fn=worker_init_fn)


if __name__ == '__main__':
    a = GMDataset('PascalVOC', length=1000 ,sets='train', obj_resize=(256, 256))
    k = a[0]
    pass
