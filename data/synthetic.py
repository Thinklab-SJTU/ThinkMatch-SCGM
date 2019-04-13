import numpy as np
from utils.config import cfg
import random
from pathlib import Path
from data.base_dataset import BaseDataset
import pickle


class SyntheticDataset(BaseDataset):
    def __init__(self, sets, obj_resize):
        super(SyntheticDataset, self).__init__()
        self.classes = ['synthetic']
        self.obj_resize = obj_resize

        self.train_num = cfg.SYNTHETIC.TRAIN_NUM
        self.test_num = cfg.SYNTHETIC.TEST_NUM
        self.exp_id = cfg.SYNTHETIC.RANDOM_EXP_ID

        self.kpt_len = cfg.SYNTHETIC.KPT_NUM
        self.dimension = cfg.SYNTHETIC.DIM
        self.gt_feat_high = cfg.SYNTHETIC.FEAT_GT_UNIFORM
        self.gt_feat_low = - cfg.SYNTHETIC.FEAT_GT_UNIFORM
        self.feat_noise = cfg.SYNTHETIC.FEAT_NOISE_STD
        self.gt_pos_high = cfg.SYNTHETIC.POS_GT_UNIFORM
        self.gt_pos_low = 0
        #self.edge_density = cfg.SYNTHETIC.EDGE_DENSITY
        self.affine_dxy_high = cfg.SYNTHETIC.POS_AFFINE_DXY
        self.affine_dxy_low = - cfg.SYNTHETIC.POS_AFFINE_DXY
        self.affine_s_high = cfg.SYNTHETIC.POS_AFFINE_S_HIGH
        self.affine_s_low = cfg.SYNTHETIC.POS_AFFINE_S_LOW
        self.affine_theta_high = cfg.SYNTHETIC.POS_AFFINE_DTHETA
        self.affine_theta_low = - cfg.SYNTHETIC.POS_AFFINE_DTHETA
        self.pos_noise = cfg.SYNTHETIC.POS_NOISE_STD

        self.cache_name = 'synthetic_train{}_test{}_kpt{}_dim{}_feat{:.2f}n{:.2f}_pos{:.2f}n{:.2f}_id{}.pkl'.format(
            self.train_num, self.test_num, self.kpt_len, self.dimension,
            self.gt_feat_high, self.feat_noise, self.gt_pos_high, self.pos_noise,
            self.exp_id
        )
        self.cache_path = Path(cfg.CACHE_PATH) / 'synthetic' / self.cache_name

        if not self.cache_path.parent.exists():
            self.cache_path.parent.mkdir(parents=True)

        if self.cache_path.exists():
            print('loading dataset from {}'.format(self.cache_path))
            with self.cache_path.open(mode='rb') as f:
                data_dict = pickle.load(f)
        else:
            print('caching dataset to {}'.format(self.cache_path))
            self.data_feat = np.random.uniform(self.gt_feat_low, self.gt_feat_high, (self.dimension, self.kpt_len))
            self.data_pos = np.random.uniform(self.gt_pos_low, self.gt_pos_high, (2, self.kpt_len))
            data_dict = self.__gen_data()
            with self.cache_path.open(mode='wb') as f:
                pickle.dump(data_dict, f)
        self.data_list = data_dict[sets]

    def __gen_data(self):
        """
        Generate random data and cache them into files
        """
        data_dict = dict()
        for period, sample_num in zip(['train', 'test'], [self.train_num, self.test_num]):
            data_lst = []
            for i in range(sample_num):
                data_lst.append(self.__gen_anno_dict())
            data_dict[period] = data_lst
        return data_dict

    def __gen_anno_dict(self, is_src=False):
        """
        Generate an annotation dict according to is_ref True or False
        :param is_src: get a source point (i.e. ground truth point w/o noise) or not
        """
        data_feat = self.data_feat.copy()
        data_pos = np.concatenate((self.data_pos, np.ones((1, self.kpt_len))), axis=0)
        if not is_src:
            # feature distortion
            data_feat = data_feat + np.random.normal(0, self.feat_noise, data_feat.shape)

            # position distortion
            tx = np.random.uniform(self.affine_dxy_low, self.affine_dxy_high)
            ty = np.random.uniform(self.affine_dxy_low, self.affine_dxy_high)
            s = np.random.uniform(self.affine_s_low, self.affine_s_high)
            theta = np.random.uniform(self.affine_theta_low, self.affine_theta_high) * np.pi / 180
            aff = np.array(
                [[s * np.cos(theta), -s * np.sin(theta), tx],
                 [s * np.sin(theta), s * np.cos(theta), ty],
                 [0, 0, 1]]
            )
            data_pos = np.matmul(aff, data_pos)[:2, :]
            data_pos = data_pos + np.random.normal(0, self.pos_noise, data_pos.shape)

        keypoint_list = []
        for idx in range(self.kpt_len):
            keypoint = data_pos[:, idx]
            attr = dict()
            attr['name'] = idx
            attr['x'] = float(keypoint[0])
            attr['y'] = float(keypoint[1])
            attr['feat'] = data_feat[:, idx]
            keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict['keypoints'] = keypoint_list
        # the following keys are of no use, but kept for dataloader interface
        anno_dict['image'] = None
        anno_dict['bounds'] = self.gt_pos_low, self.gt_pos_low, self.gt_pos_high, self.gt_pos_high
        anno_dict['ori_sizes'] = (self.gt_pos_high, self.gt_pos_high)
        anno_dict['cls'] = 'synthetic'

        return anno_dict

    def get_pair(self, cls=None, shuffle=True):
        """
        Randomly get a pair of objects from synthetic data
        :param cls: no use here
        :param shuffle: random shuffle the keypoints
        :return: (pair of data, groundtruth permutation matrix)
        """
        anno_pair = []
        for anno_dict in random.sample(self.data_list, 2):
            if shuffle:
                random.shuffle(anno_dict['keypoints'])
            anno_pair.append(anno_dict)

        perm_mat = np.zeros([len(_['keypoints']) for _ in anno_pair], dtype=np.float32)
        row_list = []
        col_list = []
        for i, keypoint in enumerate(anno_pair[0]['keypoints']):
            for j, _keypoint in enumerate(anno_pair[1]['keypoints']):
                if keypoint['name'] == _keypoint['name']:
                    perm_mat[i, j] = 1
                    row_list.append(i)
                    col_list.append(j)
                    break
        row_list.sort()
        col_list.sort()
        perm_mat = perm_mat[row_list, :]
        perm_mat = perm_mat[:, col_list]
        anno_pair[0]['keypoints'] = [anno_pair[0]['keypoints'][i] for i in row_list]
        anno_pair[1]['keypoints'] = [anno_pair[1]['keypoints'][j] for j in col_list]

        return anno_pair, perm_mat
