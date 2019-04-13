import os
from datetime import datetime
from utils.print_easydict import print_easydict_str
from copy import deepcopy

repeat_times = 10

#environments = 'CUDA_VISIBLE_DEVICES=3 CXX=/opt/rh/devtoolset-3/root/usr/bin/gcc'
environments = 'CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda-10.0 LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:$LD_LIBRARY_PATH PATH=/usr/local/cuda-10.0/bin/:$PATH CXX=gcc'
python_path = '~/dl-of-gm/venv/bin/python'
cfg_file = 'experiments/sm_pl_synthetic.yaml'

ori_cfg_dict = {
    'SYNTHETIC':
        {
            'RANDOM_EXP_ID': 0,
            'TRAIN_NUM': 200,
            'TEST_NUM': 100,
            'DIM': 1024,
            'KPT_NUM': 20,  # keypoint num
            'FEAT_GT_UNIFORM': 1.,  # feature vector ~ uniform(-X, X)
            'FEAT_NOISE_STD': 1.5,  # feature noise ~ N(0, X^2)
            'POS_GT_UNIFORM': 256.,  # keypoint position ~ uniform(0, X)
            'POS_AFFINE_DXY': 50.,  # t_x, t_y ~ uniform(-X, X)
            'POS_AFFINE_S_LOW': 0.8,  # s ~ uniform(S_LOW, S_HIGH)
            'POS_AFFINE_S_HIGH': 1.2,
            'POS_AFFINE_DTHETA': 60.,  # theta ~ uniform(-X, X)
            'POS_NOISE_STD': 10.
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

def test_once(cfg_dict):
    acc_sum = 0
    for t in range(repeat_times):
        cfg_dict['SYNTHETIC']['RANDOM_EXP_ID'] = t
        new_cfg_file = edit_cfg(cfg_file, cfg_dict, str(t))
        cmd = '{} {} train_eval.py --cfg {}'.format(environments, python_path, new_cfg_file)
        stdout = os.popen(cmd)
        acc_line = stdout.readlines()[-2]
        acc = float(acc_line.split('=')[-1])
        os.system('rm {}'.format(new_cfg_file))
        print('test {} accuracy = {:.4f}'.format(t, acc), flush=True)
        acc_sum += acc
    return acc_sum / repeat_times


print('-' * 10)
print('FEAT_NOISE_STD', flush=True)
noise_list = [x / 10 for x in range(10, 26, 1)]
#noise_list = [x / 10 for x in range(21, 26, 1)]
for idx, noise in enumerate(noise_list):
    cfg_dict = deepcopy(ori_cfg_dict)
    cfg_dict['SYNTHETIC']['FEAT_NOISE_STD'] = noise
    mean_acc = test_once(cfg_dict)
    print('exp {}/{} on noise_std={}, mean acc = {:.4f}\n'.format(idx, len(noise_list), noise, mean_acc), flush=True)


print('-' * 10)
print('KPT_NUM', flush=True)
kpt_list = list(range(5, 55, 5))
#kpt_list = list(range(35, 55, 5))
for idx, kpt_num in enumerate(kpt_list):
    cfg_dict = deepcopy(ori_cfg_dict)
    cfg_dict['SYNTHETIC']['KPT_NUM'] = kpt_num
    mean_acc = test_once(cfg_dict)
    print('exp {}/{} on kpt_num={}, mean acc = {:.4f}\n'.format(idx, len(kpt_list), kpt_num, mean_acc), flush=True)
