import os
from datetime import datetime
from utils.print_easydict import print_easydict_str
from copy import deepcopy

repeat_times = 1

#environments = 'CUDA_VISIBLE_DEVICES=3,4 CXX=/opt/rh/devtoolset-3/root/usr/bin/gcc'
#environments = 'CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda-10.0 LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:$LD_LIBRARY_PATH PATH=/usr/local/cuda-10.0/bin/:$PATH CXX=gcc'
environments = 'CXX=g++'
python_path = '~/dl-of-gm/venv/bin/python'
cfg_file = 'experiments/nhgm_em_synthetic.yaml'
#cfg_file = 'experiments/sm_pl_synthetic.yaml'
script = 'train_eval.py'
#script = 'cache_m.py'
#script = 'eval.py --epoch 0'

ori_cfg_dict = {
    'SYNTHETIC':
        {
            'RANDOM_EXP_ID': 0,
            'TRAIN_NUM': 200,
            'TEST_NUM': 100,
            'DIM': 2, #1024,
            'KPT_NUM': 10, #20,  # keypoint num
            'OUT_NUM': 10,   # outlier num
            'FEAT_GT_UNIFORM': 1.,  # feature vector ~ uniform(-X, X)
            'FEAT_NOISE_STD': 1.5,  # feature noise ~ N(0, X^2)
            'POS_GT_UNIFORM': 1.,  # keypoint position ~ uniform(0, X)
            'POS_AFFINE_DXY': 0.,  # t_x, t_y ~ uniform(-X, X)
            'POS_AFFINE_S_LOW': 1.,  # s ~ uniform(S_LOW, S_HIGH)
            'POS_AFFINE_S_HIGH': 1.,
            'POS_AFFINE_DTHETA': 0.,  # theta ~ uniform(-X, X)
            'POS_NOISE_STD': 0.02
         }
}

def edit_cfg(cfg_name: str, edit_pairs: dict, suffix=''):
    """
    edit original config file into a tmp config file. If key name conflicts, the older one will be overwritten.
    """
    cfg_new_name = '{}_{}_{}.yaml'.format(cfg_name.strip('.yaml'), datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), suffix)
    os.system('cp {} {}'.format(cfg_name, cfg_new_name))

    append_str = print_easydict_str(edit_pairs)
    with open(cfg_new_name, mode='a') as f:
        f.write('\n# created by synthetic experiment\n')
        f.write(append_str)

    return cfg_new_name

def test_once(cfg_dict, init_t=0):
    acc_sum = 0
    for t in range(init_t, repeat_times):
        cfg_dict['SYNTHETIC']['RANDOM_EXP_ID'] = t
        new_cfg_file = edit_cfg(cfg_file, cfg_dict, str(t))
        cmd = '{} {} {} --cfg {}'.format(environments, python_path, script, new_cfg_file)

        stdout = os.popen(cmd)
        acc = 0
        for line in  stdout.readlines():
            x = line.split("average = ")
            if len(x) == 2:
                _acc = float(x[-1])
                if _acc > acc:
                    acc = _acc
        os.system('rm {}'.format(new_cfg_file))
        print('test {} accuracy = {:.4f}'.format(t, acc), flush=True)
        acc_sum += acc
    return acc_sum / repeat_times

'''
print('-' * 10)
print('FEAT_NOISE_STD', flush=True)
noise_list = [x / 100 for x in range(0, 11, 1)]
#noise_list = [x / 10 for x in range(22, 23, 1)]
for idx, noise in enumerate(noise_list):
    cfg_dict = deepcopy(ori_cfg_dict)
    cfg_dict['SYNTHETIC']['POS_NOISE_STD'] = noise
    mean_acc = test_once(cfg_dict)
    print('exp {}/{} on noise_std={}, mean acc = {:.4f}\n'.format(idx, len(noise_list), noise, mean_acc), flush=True)


print('-' * 10)
print('KPT_NUM', flush=True)
kpt_list = list(range(4, 14, 1))
#kpt_list = list(range(15, 55, 5))
for idx, kpt_num in enumerate(kpt_list):
    cfg_dict = deepcopy(ori_cfg_dict)
    cfg_dict['SYNTHETIC']['KPT_NUM'] = kpt_num
    mean_acc = test_once(cfg_dict)
    print('exp {}/{} on kpt_num={}, mean acc = {:.4f}\n'.format(idx, len(kpt_list), kpt_num, mean_acc), flush=True)
'''

print('-' * 10)
print('OUT_NUM', flush=True)
out_list = list(range(2, 20, 2))
for idx, out_num in enumerate(out_list):
    cfg_dict = deepcopy(ori_cfg_dict)
    cfg_dict['SYNTHETIC']['OUT_NUM'] = out_num
    mean_acc = test_once(cfg_dict)
    print('exp {}/{} on out_num={}, mean acc = {:.4f}\n'.format(idx, len(out_list), out_num, mean_acc), flush=True)

