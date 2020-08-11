"""Graph matching config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
from easydict import EasyDict as edict
import numpy as np

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

# Minibatch size
__C.BATCH_SIZE = 4

# Pairwise data loader settings.
__C.PAIR = edict()
__C.PAIR.RESCALE = (256, 256)  # rescaled image size
__C.PAIR.PADDING = 23  # padding length of keypoint pairs for batch operation
__C.PAIR.CANDIDATE_SHAPE = (16, 16)  # shape of candidates
__C.PAIR.CANDIDATE_LENGTH = np.cumprod(__C.PAIR.CANDIDATE_SHAPE)[-1]
__C.PAIR.GT_GRAPH_CONSTRUCT = 'tri'
__C.PAIR.REF_GRAPH_CONSTRUCT = 'fc'
__C.PAIR.MAX_PROB_SIZE = -1
__C.PAIR.GT_OUTLIER = False  # allow outlier in source graph
__C.PAIR.REF_OUTLIER = False  # allow outlier in reference graph (aka target graph)
__C.PAIR.SYM_ADJACENCY = True # build a symmetric adjacency matrix, else only upper right of adjacency matrix will be filled.
__C.PAIR.NUM_GRAPHS = 3
__C.PAIR.NUM_CLUSTERS = 2
__C.PAIR.TEST_ALL_GRAPHS = False
__C.PAIR.TRAIN_ALL_GRAPHS = False

# VOC2011-Keypoint Dataset
__C.VOC2011 = edict()
__C.VOC2011.KPT_ANNO_DIR = 'data/PascalVOC/annotations/'  # keypoint annotation
__C.VOC2011.ROOT_DIR = 'data/PascalVOC/VOC2011/'  # original VOC2011 dataset
__C.VOC2011.SET_SPLIT = 'data/PascalVOC/voc2011_pairs.npz'  # set split path
__C.VOC2011.CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                       'tvmonitor']

# Willow-Object Dataset
__C.WILLOW = edict()
__C.WILLOW.ROOT_DIR = 'data/WILLOW-ObjectClass'
__C.WILLOW.CLASSES = ['Car', 'Duck', 'Face', 'Motorbike', 'Winebottle']
__C.WILLOW.KPT_LEN = 10
__C.WILLOW.TRAIN_NUM = 20
__C.WILLOW.SPLIT_OFFSET = 0
__C.WILLOW.TRAIN_SAME_AS_TEST = False
__C.WILLOW.RAND_OUTLIER = 0

# Synthetic dataset
__C.SYNTHETIC = edict()
__C.SYNTHETIC.DIM = 1024
__C.SYNTHETIC.TRAIN_NUM = 100  # training graphs
__C.SYNTHETIC.TEST_NUM = 100  # testing graphs
__C.SYNTHETIC.MIXED_DATA_NUM = 10  # num of samples in mixed synthetic test
__C.SYNTHETIC.RANDOM_EXP_ID = 0  # id of random experiment
__C.SYNTHETIC.EDGE_DENSITY = 0.3  # edge_num = X * node_num^2 / 4
__C.SYNTHETIC.KPT_NUM = 10  # number of nodes (inliers)
__C.SYNTHETIC.OUT_NUM = 0 # number of outliers
__C.SYNTHETIC.FEAT_GT_UNIFORM = 1.  # reference node features in uniform(-X, X) for each dimension
__C.SYNTHETIC.FEAT_NOISE_STD = 0.1  # corresponding node features add a random noise ~ N(0, X^2)
__C.SYNTHETIC.POS_GT_UNIFORM = 256.  # reference keypoint position in image: uniform(0, X)
__C.SYNTHETIC.POS_AFFINE_DXY = 50.  # corresponding position after affine transform: t_x, t_y ~ uniform(-X, X)
__C.SYNTHETIC.POS_AFFINE_S_LOW = 0.8  # corresponding position after affine transform: s ~ uniform(S_LOW, S_HIGH)
__C.SYNTHETIC.POS_AFFINE_S_HIGH = 1.2
__C.SYNTHETIC.POS_AFFINE_DTHETA = 60.  # corresponding position after affine transform: theta ~ uniform(-X, X)
__C.SYNTHETIC.POS_NOISE_STD = 10.  # corresponding position add a random noise ~ N(0, X^2) after affine transform

# QAPLIB dataset
__C.QAPLIB = edict()
__C.QAPLIB.DIR = 'data/qapdata'
__C.QAPLIB.FEED_TYPE = 'affmat' # 'affmat' (affinity matrix) or 'adj' (adjacency matrix)
__C.QAPLIB.ONLINE_REPO = 'http://anjos.mgi.polymtl.ca/qaplib/'
__C.QAPLIB.MAX_TRAIN_SIZE = 200
__C.QAPLIB.MAX_TEST_SIZE = 100

# CUB2011 dataset
__C.CUB2011 = edict()
__C.CUB2011.ROOT_PATH = 'data/CUB_200_2011'
__C.CUB2011.CLASS_SPLIT = 'ori' # choose from 'ori' (original split), 'sup' (super class) or 'all' (all birds as one class)

#  Sub-Rome16k dataset
__C.ROME16K = edict()
__C.ROME16K.DIR = 'data/Sub-Rome16K'
__C.ROME16K.CLASSES = ['Colosseum', 'Pantheon', 'Ceiling']

# GMN model options
__C.GMN = edict()
__C.GMN.FEATURE_CHANNEL = 512
__C.GMN.PI_ITER_NUM = 50
__C.GMN.PI_STOP_THRESH = 2e-7
__C.GMN.BS_ITER_NUM = 10
__C.GMN.BS_EPSILON = 1e-10
__C.GMN.VOTING_ALPHA = 2e8
__C.GMN.GM_SOLVER = 'SM'

# PCA model options
__C.PCA = edict()
__C.PCA.FEATURE_CHANNEL = 512
__C.PCA.BS_ITER_NUM = 20
__C.PCA.BS_EPSILON = 1.0e-10
__C.PCA.SK_TAU = 0.005
__C.PCA.GNN_LAYER = 5
__C.PCA.GNN_FEAT = 1024
__C.PCA.LOSS_WEIGHTS = [0., 1.]  # [cross-module loss, final prediction loss]
__C.PCA.CROSS_ITER = False
__C.PCA.CROSS_ITER_NUM = 1

# NGM model options
__C.NGM = edict()
__C.NGM.FEATURE_CHANNEL = 512
__C.NGM.BS_ITER_NUM = 10
__C.NGM.BS_EPSILON = 1e-10
__C.NGM.SK_TAU = 0.005
__C.NGM.GNN_FEAT = [16, 16, 16]
__C.NGM.GNN_LAYER = 3
__C.NGM.GAUSSIAN_SIGMA = 1.
__C.NGM.SIGMA3 = 1.
__C.NGM.WEIGHT2 = 1.
__C.NGM.WEIGHT3 = 1.
__C.NGM.EDGE_FEATURE = 'cat' # 'cat' or 'geo'
__C.NGM.ORDER3_FEATURE = 'cat' # 'cat' or 'geo'
#__C.NGM.OUTP_SCORE = True # output the scoring matrix as prediction in testing (no Sinkhorn applied in testing)
__C.NGM.FIRST_ORDER = True
__C.NGM.EDGE_EMB = False
__C.NGM.SK_EMB = 1
__C.NGM.GUMBEL_SK = False

# GANN model options
__C.GANN = edict()
__C.GANN.FEATURE_CHANNEL = 1024
__C.GANN.SK_ITER_NUM = 20
__C.GANN.SK_TAU = 0.05
__C.GANN.SK_EPSILON = 1e-10
__C.GANN.UNIV_SIZE = 10
__C.GANN.MAX_ITER = [200, 200]
__C.GANN.INIT_TAU = [0.5, 0.5]
__C.GANN.GAMMA = 0.5
__C.GANN.BETA = [1., 0.]
__C.GANN.CONVERGE_TOL = 1e-5
__C.GANN.MIN_TAU = [1e-2, 1e-2]
__C.GANN.SCALE_FACTOR = 1.
__C.GANN.QUAD_WEIGHT = 1.
__C.GANN.CLUSTER_QUAD_WEIGHT = 1.
__C.GANN.PROJECTOR = ['sinkhorn', 'sinkhorn']

# GUROBI options
__C.GUROBI = edict()
__C.GUROBI.TIME_LIMIT = 0

#
# Training options
#

__C.TRAIN = edict()

# Iterations per epochs
__C.TRAIN.EPOCH_ITERS = 7000

# Training start epoch. If not 0, will be resumed from checkpoint.
__C.TRAIN.START_EPOCH = 0

# Total epochs
__C.TRAIN.NUM_EPOCHS = 30

# Start learning rate
__C.TRAIN.LR = 0.01

# Learning rate decay
__C.TRAIN.LR_DECAY = 0.1

# Learning rate decay step (in epochs)
__C.TRAIN.LR_STEP = [10, 20]

# SGD momentum
__C.TRAIN.MOMENTUM = 0.9

# RobustLoss normalization
__C.TRAIN.RLOSS_NORM = max(__C.PAIR.RESCALE)

# Specify a class for training
__C.TRAIN.CLASS = 'none'

# Loss function. Should be 'offset' or 'perm'
__C.TRAIN.LOSS_FUNC = 'perm'


#
# Evaluation options
#

__C.EVAL = edict()

# Evaluation epoch number
__C.EVAL.EPOCH = 30

# PCK metric
__C.EVAL.PCK_ALPHAS = []
__C.EVAL.PCK_L = float(max(__C.PAIR.RESCALE))  # PCK reference.

# Number of samples for testing. Stands for number of image pairs in each classes (VOC)
__C.EVAL.SAMPLES = 1000


#
# Visualization options
#

__C.VISUAL = edict()

# Load model from a specific epoch number for visualization
__C.VISUAL.EPOCH = 0

# Load model from file for visualization. If you want to use this, VISUAL.EPOCH must be 0.
__C.VISUAL.WEIGHT_PATH = ''

# Number of images for visualization
__C.VISUAL.NUM_IMGS = 20

# Save or plot visualized images
__C.VISUAL.SAVE = True

# Specify a exact class of image pairs to visualize
__C.VISUAL.CLASS = 'none'


#
# MISC
#

# name of backbone net
__C.BACKBONE = 'VGG16_bn'

# Parallel GPU indices ([0] for single GPU)
__C.GPUS = [0]

# num of dataloader processes
__C.DATALOADER_NUM = __C.BATCH_SIZE

# Mean and std to normalize images
__C.NORM_MEANS = [0.485, 0.456, 0.406]
__C.NORM_STD = [0.229, 0.224, 0.225]

# Data cache path
__C.CACHE_PATH = 'data/cache'

# Model name and dataset name
__C.MODEL_NAME = ''
__C.DATASET_NAME = ''
__C.DATASET_FULL_NAME = 'PascalVOC' # 'PascalVOC' or 'WillowObject'

# Module path of module
__C.MODULE = ''

# Output path (for checkpoints, running logs and visualization results)
__C.OUTPUT_PATH = ''

# The step of iteration to print running statistics.
# The real step value will be the least common multiple of this value and batch_size
__C.STATISTIC_STEP = 100

# random seed used for data loading
__C.RANDOM_SEED = 123

# enable fp16 instead of fp32 in the model (via nvidia/apex)
__C.FP16 = False

def lcm(x, y):
    """
    Compute the least common multiple of x and y. This function is used for running statistics.
    """
    greater = max(x, y)
    while True:
        if (greater % x == 0) and (greater % y == 0):
            lcm = greater
            break
        greater += 1
    return lcm


def get_output_dir(model, dataset):
    """
    Return the directory where experimental artifacts are placed.
    :param model: model name
    :param dataset: dataset name
    :return: output path (checkpoint and log), visual path (visualization images)
    """
    outp_path = os.path.join('output', '{}_{}'.format(model, dataset))
    return outp_path


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.full_load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
