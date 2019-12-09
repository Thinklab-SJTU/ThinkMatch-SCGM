import os
from datetime import datetime
from utils.print_easydict import print_easydict_str
from copy import deepcopy
import xlwt

repeat_times = 1

#environments = 'CUDA_VISIBLE_DEVICES=3,4 CXX=/opt/rh/devtoolset-3/root/usr/bin/gcc'
#environments = 'CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda-10.0 LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:$LD_LIBRARY_PATH PATH=/usr/local/cuda-10.0/bin/:$PATH CXX=gcc'
environments = 'CXX=g++'
#python_path = '~/dl-of-gm/venv/bin/python'
python_path = '/opt/conda/bin/python'
#cfg_file = 'experiments/nhgm_synthetic.yaml'
cfg_file = 'experiments/sm_pl_synthetic.yaml'
#script = 'train_eval.py'
#script = 'cache_m.py'
script = 'eval.py --epoch 0'

ori_cfg_dict_cvpr20 = {
    'SYNTHETIC':
        {
            'RANDOM_EXP_ID': 0,
            'TRAIN_NUM': 200,
            'TEST_NUM': 100,
            'DIM': 2, #1024,
            'KPT_NUM': 10, #20,  # keypoint num
            'OUT_NUM': 0,   # outlier num
            'FEAT_GT_UNIFORM': 1.,  # feature vector ~ uniform(-X, X)
            'FEAT_NOISE_STD': 1.5,  # feature noise ~ N(0, X^2)
            'POS_GT_UNIFORM': 1.,  # keypoint position ~ uniform(0, X)
            'POS_AFFINE_DXY': 0.,  # t_x, t_y ~ uniform(-X, X)
            'POS_AFFINE_S_LOW': .9,  # s ~ uniform(S_LOW, S_HIGH)
            'POS_AFFINE_S_HIGH': 1.1,
            'POS_AFFINE_DTHETA': 0.,  # theta ~ uniform(-X, X)
            'POS_NOISE_STD': 0.02
         },
    'PAIR':
        {
            'RESCALE': '\n    - 256\n    - 256',
            'CANDIDATE_SHAPE': '\n    - 16\n    - 16',
            'GT_GRAPH_CONSTRUCT': 'tri',
            'REF_GRAPH_CONSTRUCT': 'fc',
            'NUM_GRAPHS': 4,
        }
}

ori_cfg_dict_nips19 = {
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

ori_cfg_dict_iccv19 = {
    'SYNTHETIC':
        {
            'RANDOM_EXP_ID': 0,
            'TRAIN_NUM': 200,
            'TEST_NUM': 100,
            'DIM': 1024,
            'KPT_NUM': 20,  # keypoint num
            'OUT_NUM': 0,   # outlier num
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

ori_cfg_dict = ori_cfg_dict_cvpr20

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
    obj_sum = 0
    for t in range(init_t, repeat_times):
        cfg_dict['SYNTHETIC']['RANDOM_EXP_ID'] = t
        new_cfg_file = edit_cfg(cfg_file, cfg_dict, str(t))
        cmd = '{} {} {} --cfg {}'.format(environments, python_path, script, new_cfg_file)

        try:
            stdout = os.popen(cmd)
            acc = 0
            obj = 0
            for line in stdout.readlines():
                x = line.split("average accuracy = ")
                if len(x) == 2:
                    _acc = float(x[-1])
                    if _acc > acc:
                        acc = _acc
                x = line.split('average objscore = ')
                if len(x) == 2:
                    _obj = float(x[-1])
                    if _obj > obj:
                        obj = _obj
            os.system('rm {}'.format(new_cfg_file))
            print('test {} accuracy = {:.4f}'.format(t, acc), flush=True)
            print('test {} objscore = {:.4f}'.format(t, obj), flush=True)
            acc_sum += acc
            obj_sum += obj
        except KeyboardInterrupt as err:
            os.system('rm {}'.format(new_cfg_file))
            raise err
    return acc_sum / repeat_times, obj_sum /repeat_times

wb = xlwt.Workbook()

method_list = ['sm', 'rrwm', 'ngm', 'nhgm', 'nmgm_eval', 'nmgm',
               'ngm_vanilla']
cfg_list = ['experiments/sm_pl_synthetic.yaml', 'experiments/rrwm_pl_synthetic.yaml', 'experiments/ngm_synthetic.yaml', 'experiments/nhgm_synthetic.yaml', 'experiments/nmgm_synthetic.yaml', 'experiments/nmgm_synthetic.yaml',
            'experiments/ngm_vanilla_synthetic.yaml']
script_list = ['eval.py --epoch 0', 'eval.py --epoch 0', 'train_eval.py', 'train_eval.py', 'eval_multi.py --epoch 10', 'train_eval_multi.py',
               'train_eval.py']

print('-' * 10)
print('POS_NOISE_STD', flush=True)
ws_acc = wb.add_sheet('POS_NOISE_STD_acc')
ws_obj = wb.add_sheet('POS_NOISE_STD_obj')
for idx, method in enumerate(method_list):
    ws_acc.write(idx+1, 0, method)
    ws_obj.write(idx+1, 0, method)
noise_list = [x / 100 for x in range(0, 11, 1)]
#noise_list = [x / 10 for x in range(22, 23, 1)]
for idx, noise in enumerate(noise_list):
    ws_acc.write(0, idx+1, noise)
    ws_obj.write(0, idx+1, noise)
    for jdx, (method, cfg_file, script) in enumerate(zip(method_list, cfg_list, script_list)):
        #if method != 'sm' and method != 'rrwm' and method != 'nhgm':
        if method != 'sm':
            continue
        cfg_dict = deepcopy(ori_cfg_dict)
        cfg_dict['SYNTHETIC']['POS_NOISE_STD'] = noise
        mean_acc, mean_obj = test_once(cfg_dict)
        ws_acc.write(jdx+1, idx+1, mean_acc)
        ws_obj.write(jdx+1, idx+1, mean_obj)
        print('{} exp {}/{} on noise_std={}, mean acc = {:.4f}, mean obj = {:.4f}\n'.format(method, idx, len(noise_list), noise, mean_acc, mean_obj), flush=True)
        wb.save('syn_exp_rrwm11.xls')

wb.save('syn_exp_rrwm11.xls')
exit(0)

print('-' * 10)
print('KPT_NUM', flush=True)
kpt_list = list(range(4, 14, 1))
#kpt_list = list(range(15, 55, 5))
for idx, kpt_num in enumerate(kpt_list):
    cfg_dict = deepcopy(ori_cfg_dict)
    cfg_dict['SYNTHETIC']['KPT_NUM'] = kpt_num
    mean_acc = test_once(cfg_dict)
    print('exp {}/{} on kpt_num={}, mean acc = {:.4f}\n'.format(idx, len(kpt_list), kpt_num, mean_acc), flush=True)


print('-' * 10)
print('OUT_NUM', flush=True)
ws_acc = wb.add_sheet('OUT_NUM_acc')
ws_obj = wb.add_sheet('OUT_NUM_obj')
for idx, method in enumerate(method_list):
    ws_acc.write(idx+1, 0, method)
    ws_obj.write(idx+1, 0, method)
out_list = list(range(0, 20, 2))
for idx, out_num in enumerate(out_list):
    ws_acc.write(0, idx + 1, out_num)
    ws_obj.write(0, idx + 1, out_num)
    for jdx, (method, cfg_file, script) in enumerate(zip(method_list, cfg_list, script_list)):
        if method != 'nmgm':
            continue
        cfg_dict = deepcopy(ori_cfg_dict)
        cfg_dict['SYNTHETIC']['OUT_NUM'] = out_num
        mean_acc, mean_obj = test_once(cfg_dict)
        ws_acc.write(jdx + 1, idx + 1, mean_acc)
        ws_obj.write(jdx + 1, idx + 1, mean_obj)
        print('{} exp {}/{} on out_num={}, mean acc = {:.4f}, mean obj = {:.4f}\n'.format(method, idx, len(out_list), out_num, mean_acc, mean_obj), flush=True)
        wb.save('syn_exp_out_0911.xls')

wb.save('syn_exp_out_0911_nmgm.xls')
exit(0)

print('-' * 10)
print('GRAPH_NUM', flush=True)
ws_acc = wb.add_sheet('GRAPH_NUM_acc')
ws_obj = wb.add_sheet('GRAPH_NUM_obj')
for idx, method in enumerate(method_list):
    ws_acc.write(idx+1, 0, method)
    ws_obj.write(idx+1, 0, method)
graph_list = list(range(0, 20, 2))
for idx, graph_num in enumerate(graph_list):
    ws_acc.write(0, idx + 1, graph_num)
    ws_obj.write(0, idx + 1, graph_num)
    for jdx, (method, cfg_file, script) in enumerate(zip(method_list, cfg_list, script_list)):
        if graph_num == 2:
            if method == 'nhgm' or method == 'ngm_vanilla':
                continue
        else:
            if method != 'nmgm':
                continue
        cfg_dict = deepcopy(ori_cfg_dict)
        cfg_dict['PAIR']['NUM_GRAPHS'] = graph_num
        mean_acc, mean_obj = test_once(cfg_dict)
        ws_acc.write(jdx + 1, idx + 1, mean_acc)
        ws_obj.write(jdx + 1, idx + 1, mean_obj)
        print('{} exp {}/{} on num_graph={}, mean acc = {:.4f}, mean obj = {:.4f}\n'.format(method, idx, len(graph_list), graph_num, mean_acc, mean_obj), flush=True)
        wb.save('syn_exp_num_graph.xls')

wb.save('syn_exp_num_graph.xls')
exit(0)
