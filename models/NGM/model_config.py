from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# NGM model options
__C.NGM = edict()
__C.NGM.FEATURE_CHANNEL = 512
__C.NGM.SK_ITER_NUM = 10
__C.NGM.SK_EPSILON = 1e-10
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
__C.NGM.GUMBEL_SK = 0 # 0 for no gumbel, other wise for number of gumbel samples