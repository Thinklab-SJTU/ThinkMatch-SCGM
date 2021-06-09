import torch
import numpy as np
from pathlib import Path

from src.dataset.data_loader import GMDataset, get_dataloader
from src.utils.model_sl import load_model
from src.parallel import DataParallel
from src.lap_solvers.hungarian import hungarian
from src.utils.data_to_cuda import data_to_cuda
import matplotlib
try:
    import _tkinter
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

plt.rcParams["font.family"] = "serif"

from src.utils.config import cfg

def vertical_subplt(a,b,c):
    plt.subplot(b, a, (c // b) + c % b * a)

def visualize_model(models, dataloader, device, num_images=6, set='test', cls=None, save_img=False):
    print('Visualizing model...')
    assert set in ('train', 'test')

    for model in models:
        model.eval()
    images_so_far = 0

    #names = ['source', 'GMN', 'PCA-GM', 'IPCA-GM']
    names = ['source', 'GMN', 'PCA-GM', 'NGM', 'NGM-v2', 'NHGM-v2']
    num_cols = num_images // 2 #+ 1

    old_cls = dataloader[set].dataset.cls
    if cls is not None:
        dataloader[set].dataset.cls = cls

    visualize_path = Path(cfg.OUTPUT_PATH) / 'visual'
    if save_img:
        if not visualize_path.exists():
            visualize_path.mkdir(parents=True)
    #with torch.no_grad():
        #for idx in range(len(names)):
        #    ax = plt.subplot(len(names), num_cols, idx * num_cols + 1)
        #    ax.axis('off')
        #    plt.text(0.5, 0.5, names[idx], fontsize=40, horizontalalignment='center')
    for cls in range(20):
        fig = plt.figure(figsize=(50, 35), dpi=120)
        dataloader[set].dataset.cls = cls
        images_so_far = 0

        for i, inputs in enumerate(dataloader[set]):
            if models[0].module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)
            assert 'images' in inputs
            data1, data2 = inputs['images']
            P1_gt, P2_gt = inputs['Ps']
            n1_gt, n2_gt = inputs['ns']
            perm_mat = inputs['gt_perm_mat']

            pred_perms = []
            for model in models:
                outputs = model(inputs)
                pred_perms.append(outputs['perm_mat'])

            for j in range(inputs['batch_size']):
                if n1_gt[j] <= 4:
                    print('graph too small.')
                    continue

                matched = []
                for idx, pred_perm in enumerate(pred_perms):
                    matched_num = torch.sum(pred_perm[j, :n1_gt[j], :n2_gt[j]] * perm_mat[j, :n1_gt[j], :n2_gt[j]])
                    matched.append(matched_num)

                #if random.choice([0, 1, 2]) >= 1:
                #    if not ((matched[0] < matched[1] or matched[0] < matched[2]) and (matched[0] <= matched[1] and matched[0] <= matched[2])):
                #        continue
                #else:
                #    if not (matched[0] <= matched[1] and matched[0] <= matched[2]):
                #        continue
                #cls = dataloader[set].dataset.cls
                #if cls != 10 and cls != 2 and cls != 5 and cls != 12 and cls != 18 and cls != 19 and not (matched[-1] >= matched[-2] >= matched[2] >= matched[1] > matched[0]):
                #    continue

                if random.choice([0, 1, 2]) >= 1:
                    if dataloader[set].dataset.cls != 10 and dataloader[set].dataset.cls != 19 and not (matched[4] >= matched[3] >= matched[2] >= matched[1] > matched[0]):
                        print('performance not good.')
                        continue

                images_so_far += 1
                print(chr(13) + 'Visualizing {:4}/{:4}'.format(images_so_far, num_images))  # chr(13)=CR

                colorset = np.random.rand(n1_gt[j], 3)
                #ax = plt.subplot(1 + len(s_pred_perms), num_cols, images_so_far + 1)
                #ax.axis('off')
                #plt.title('source')
                #plot_helper(data1[j], P1_gt[j], n1_gt[j], ax, colorset)

                for idx, pred_perm in enumerate(pred_perms):
                    #ax = plt.subplot(1 + len(s_pred_perms), num_cols, (idx + 1) * num_cols + images_so_far + 1)
                    if images_so_far > num_cols:
                        ax = plt.subplot(len(pred_perms) * 2, num_cols, (idx + len(pred_perms)) * num_cols + images_so_far - num_cols)
                    else:
                        ax = plt.subplot(len(pred_perms) * 2, num_cols, idx * num_cols + images_so_far)
                    ax.axis('off')
                    #plt.title('predict')
                    #plot_helper(data2[j], P2_gt[j], n1_gt[j], ax, colorset, 'tgt', s_pred_perm[j], perm_mat[j])
                    plot_2graph_helper(data1[j], data2[j], P1_gt[j], P2_gt[j], n1_gt[j], ax, colorset, pred_perm[j], perm_mat[j], names[idx+1])

                #ax = plt.subplot(2 + len(s_pred_perms), num_images + 1, (len(s_pred_perms) + 1) * num_images + images_so_far)
                #ax.axis('off')
                #plt.title('groundtruth')
                #plot_helper(data2[j], P2_gt[j], n1_gt[j], ax, colorset, 'tgt', perm_mat[j])

                if not save_img:
                    plt.show()
                    print("Press Enter to continue...", end='', flush=True)  # prevent new line
                    input()

                if images_so_far == num_images:
                    fig.savefig(str(visualize_path / '{}_{:0>4}.jpg'.format(dataloader[set].dataset.cls, images_so_far)), bbox_inches='tight')
                    break

                #dataloader[set].dataset.cls += 1
            if images_so_far == num_images:
                break

    dataloader[set].dataset.cls = old_cls


def plot_helper(img, P, n, ax, colorset, mode='src', pmat=None, gt_pmat=None):
    assert mode in ('src', 'tgt')
    if mode == 'tgt':
        assert pmat is not None
    img = tensor2np(img.cpu())
    plt.imshow(img)

    P = P.cpu().numpy()
    if mode == 'src':
        for i in range(n):
            mark = plt.Circle(P[i], 7, edgecolor='w', facecolor=colorset[i])
            ax.add_artist(mark)
    else:
        pmat = pmat.cpu().numpy()
        gt_pmat = gt_pmat.cpu().numpy()
        idx = np.argmax(pmat, axis=-1)
        idx_gt = np.argmax(gt_pmat, axis=-1)
        matched = 0
        for i in range(n):
            mark = plt.Circle(P[idx[i]], 7, edgecolor='w' if idx[i] == idx_gt[i] else 'r', facecolor=colorset[i])
            ax.add_artist(mark)
            if idx[i] == idx_gt[i]:
                matched += 1
        plt.title('{:d}/{:d}'.format(matched, n), y=-0.2, fontsize=25)

def plot_2graph_helper(imgsrc, imgtgt, Psrc, Ptgt, n, ax, colorset, pmat, gt_pmat, method=""):
    imgcat = torch.cat((imgsrc, imgtgt), dim=2)
    imgcat = tensor2np(imgcat.cpu())
    plt.imshow(imgcat)

    Psrc = Psrc.cpu().numpy()
    Ptgt = Ptgt.cpu().numpy()
    Ptgt[:, 0] += imgsrc.shape[2]
    pmat = pmat.cpu().numpy()
    gt_pmat = gt_pmat.cpu().numpy()
    idx = np.argmax(pmat, axis=-1)
    idx_gt = np.argmax(gt_pmat, axis=-1)
    matched = 0
    for i in range(n):
        # src
        i_is_matched = idx[i] == idx_gt[i]
        mark = plt.Circle(Psrc[i], 7, edgecolor='g' if i_is_matched else 'r', facecolor="None")
        ax.add_artist(mark)
        #tgt
        mark = plt.Circle(Ptgt[idx[i]], 7, edgecolor='g' if i_is_matched else 'r', facecolor="None")
        ax.add_artist(mark)
        l = matplotlib.lines.Line2D([Psrc[i][0], Ptgt[idx[i]][0]], [Psrc[i][1], Ptgt[idx[i]][1]], color='g' if i_is_matched else 'r')
        ax.add_line(l)
        if idx[i] == idx_gt[i]:
            matched += 1
    plt.title('{} {}: {:d}/{:d}'.format(method, cfg.PascalVOC.CLASSES[dataloader['test'].dataset.cls], matched, n), y=-0.3, fontsize=20)


def tensor2np(inp):
    """Tensor to numpy array for plotting"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(cfg.NORM_MEANS)
    std = np.array(cfg.NORM_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


if __name__ == '__main__':
    from src.utils.parse_args import parse_args
    args = parse_args('Deep learning of graph matching visualization code.')

    import importlib
    from src.utils.config import cfg_from_file

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    image_dataset = {
        x: GMDataset(cfg.DATASET_FULL_NAME,
                     sets=x,
                     problem=cfg.PROBLEM.TYPE,
                     length=dataset_len[x],
                     cls=cfg.TRAIN.CLASS if x == 'train' else cfg.EVAL.CLASS,
                     obj_resize=cfg.PROBLEM.RESCALE)
        for x in ('train', 'test')}
    cfg.DATALOADER_NUM = 0
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_paths = ['/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_gmn_voc.pt',
                   '/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_pca_voc.pt',
                   #'/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_ipca_voc.pt']
                   '/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_ngm_voc.pt',
                   '/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_ngmv2_voc.pt',
                   '/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_nhgmv2_voc.pt',
                   ]

    cfg_files = ['experiments/vgg16_gmn_voc.yaml',
                 'experiments/vgg16_pca_voc.yaml',
                 #'experiments/vgg16_ipca_voc.yaml',
                 'experiments/vgg16_ngm_voc.yaml',
                 'experiments/vgg16_ngmv2_voc.yaml',
                 'experiments/vgg16_nhgmv2_voc.yaml',
                 ]
    models = []

    for i, (model_path, cfg_file) in enumerate(zip(model_paths, cfg_files)):
        cfg_from_file(cfg_file)

        mod = importlib.import_module(cfg.MODULE)
        Net = mod.Net

        model = Net()
        model = model.to(device)
        model = DataParallel(model, device_ids=cfg.GPUS)

        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)
        models.append(model)

    visualize_model(models, dataloader, device,
                    num_images=cfg.VISUAL.NUM_IMGS,
                    cls=cfg.VISUAL.CLASS if cfg.VISUAL.CLASS != 'none' else None,
                    save_img=cfg.VISUAL.SAVE)
