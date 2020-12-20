from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# GUROBI options
__C.GUROBI = edict()
__C.GUROBI.TIME_LIMIT = float('inf')