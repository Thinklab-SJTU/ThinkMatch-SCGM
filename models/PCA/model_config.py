from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

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
