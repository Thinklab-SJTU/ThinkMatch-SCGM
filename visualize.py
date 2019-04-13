import torch
import numpy as np
from pathlib import Path

from data.data_loader import GMDataset, get_dataloader
from utils.model_sl import load_model
from parallel import DataParallel
from GMN.bi_stochastic import BiStochastic

import matplotlib
try:
    import _tkinter
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.config import cfg


def visualize_model(model, dataloader, device, num_images=6, set='train', cls=None, save_img=False):
    print('Visualizing model...')
    assert set in ('train', 'test')

    was_training = model.training
    model.eval()
    images_so_far = 0

    old_cls = dataloader[set].dataset.cls
    if cls is not None:
        dataloader[set].dataset.cls = cls

    lap_solver = BiStochastic(max_iter=20)

    visualize_path = Path(cfg.OUTPUT_PATH) / 'visual'
    if save_img:
        if not visualize_path.exists():
            visualize_path.mkdir(parents=True)
    with torch.no_grad():
        for i, inputs in enumerate(dataloader[set]):
            if 'images' in inputs:
                data1, data2 = [_.cuda() for _ in inputs['images']]
                inp_type = 'img'
            elif 'features' in inputs:
                data1, data2 = [_.cuda() for _ in inputs['features']]
                inp_type = 'feat'
            else:
                raise ValueError('no valid data key (\'images\' or \'features\') found from dataloader!')
            P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
            n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
            e1_gt, e2_gt = [_.cuda() for _ in inputs['es']]
            G1_gt, G2_gt = [_.cuda() for _ in inputs['Gs']]
            H1_gt, H2_gt = [_.cuda() for _ in inputs['Hs']]
            KG, KH = [_.cuda() for _ in inputs['Ks']]
            perm_mat = inputs['gt_perm_mat'].cuda()

            s_pred, _ = model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH, inp_type)
            if type(s_pred) is list:
                s_pred = s_pred[-1]
            s_pred_perm = lap_solver(s_pred, n1_gt, n2_gt, exp=True)

            for j in range(s_pred.size()[0]):
                images_so_far += 1
                print(chr(13) + 'Visualizing {:4}/{:4}'.format(images_so_far, num_images))  # chr(13)=CR

                fig = plt.figure()

                colorset = np.random.rand(n1_gt[j], 3)
                ax = plt.subplot(1, 3, 1)
                ax.axis('off')
                plt.title('source')
                plot_helper(data1[j], P1_gt[j], n1_gt[j], ax, colorset)

                ax = plt.subplot(1, 3, 2)
                ax.axis('off')
                plt.title('predict')
                plot_helper(data2[j], P2_gt[j], n1_gt[j], ax, colorset, 'tgt', s_pred_perm[j])

                ax = plt.subplot(1, 3, 3)
                ax.axis('off')
                plt.title('groundtruth')
                plot_helper(data2[j], P2_gt[j], n1_gt[j], ax, colorset, 'tgt', perm_mat[j])

                if save_img:
                    fig.savefig(str(visualize_path / '{:0>4}.pdf'.format(images_so_far)), bbox_inches='tight')
                else:
                    plt.show()
                    print("Press Enter to continue...", end='', flush=True)  # prevent new line
                    input()

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    dataloader[set].dataset.cls = old_cls
                    return

    dataloader[set].dataset.cls = old_cls


def plot_helper(img, P, n, ax, colorset, mode='src', pmat=None):
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
        idx = np.argmax(pmat, axis=-1)
        for i in range(n):
            mark = plt.Circle(P[idx[i]], 7, edgecolor='w', facecolor=colorset[i])
            ax.add_artist(mark)


def tensor2np(inp):
    """Tensor to numpy array for plotting"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(cfg.NORM_MEANS)
    std = np.array(cfg.NORM_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


if __name__ == '__main__':
    from utils.parse_args import parse_args
    args = parse_args('Deep learning of graph matching visualization code.')

    import importlib

    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS, 'test': cfg.EVAL.EPOCH_ITERS}
    image_dataset = {
        x: GMDataset('PascalVOC',
                     sets=x,
                     length=dataset_len[x],
                     pad=cfg.PAIR.PADDING,
                     obj_resize=cfg.PAIR.RESCALE)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    model_path = None
    if cfg.VISUAL.EPOCH != 0:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(cfg.VISUAL.EPOCH))
    elif len(cfg.VISUAL.WEIGHT_PATH) != 0:
        model_path = cfg.VISUAL.WEIGHT_PATH
    if model_path is not None:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    visualize_model(model, dataloader, device,
                    num_images=cfg.VISUAL.NUM_IMGS,
                    cls=cfg.VISUAL.CLASS if cfg.VISUAL.CLASS != 'none' else None,
                    save_img=cfg.VISUAL.SAVE)
