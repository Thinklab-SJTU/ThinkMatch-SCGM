import torch
import numpy as np
from pathlib import Path

from data.data_loader import GMDataset
from GMN.model import Net
from utils.build_graphs import make_grids

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

    visualize_path = Path(cfg.OUTPUT_PATH) / 'visual'
    if save_img:
        if not visualize_path.exists():
            visualize_path.mkdir(parents=True)
    with torch.no_grad():
        for i, inputs in enumerate(dataloader[set]):
            img_src, img_tgt = inputs['images']
            P_src, P_tgt_gt = inputs['Ps']
            n_src, n_tgt_gt = inputs['ns']
            perm_mat = inputs['gt_perm_mat']

            P_tgt = make_grids((0, 0), cfg.PAIR.RESCALE, cfg.PAIR.CANDIDATE_SHAPE, device, batch=cfg.BATCH_SIZE)
            n_tgt = torch.as_tensor([x.shape[0] for x in P_tgt])

            img_src = img_src.to(device)
            img_tgt = img_tgt.to(device)
            P_src = P_src.to(device)
            P_tgt = P_tgt.to(device)
            n_src = n_src.to(device)
            n_tgt = n_tgt.to(device)
            perm_mat = perm_mat.to(device)
            P_tgt_gt = P_tgt_gt.to(device)

            s_pred = model(img_src, img_tgt, P_src, P_tgt, n_src, n_tgt)

            for j in range(s_pred.size()[0]):
                images_so_far += 1
                print(chr(13) + 'Visualizing {:4}/{:4}'.format(images_so_far, num_images))  # chr(13)=CR

                fig = plt.figure()

                colorset = np.random.rand(n_src[j], 3)
                ax = plt.subplot(1, 3, 1)
                ax.axis('off')
                plt.title('source')
                plot_helper(img_src[j], P_src[j], n_src[j], ax, colorset)

                ax = plt.subplot(1, 3, 2)
                ax.axis('off')
                plt.title('predict')
                plot_helper(img_tgt[j], P_tgt[j], n_src[j], ax, colorset, 'tgt', s_pred[j])

                ax = plt.subplot(1, 3, 3)
                ax.axis('off')
                plt.title('groundtruth')
                plot_helper(img_tgt[j], P_tgt_gt[j], n_src[j], ax, colorset, 'tgt', perm_mat[j])

                if save_img:
                    fig.savefig(str(visualize_path / '{:0>4}.jpg'.format(images_so_far)))
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

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS, 'test': cfg.EVAL.EPOCH_ITERS}
    image_dataset = {
        x: GMDataset('PascalVOC',
                     sets=x,
                     length=dataset_len[x],
                     pad=cfg.PAIR.PADDING,
                     obj_resize=cfg.PAIR.RESCALE)
        for x in ('train', 'test')}
    dataloader = {
        x: torch.utils.data.DataLoader(image_dataset[x],
                                       batch_size=cfg.BATCH_SIZE,
                                       shuffle=False,
                                       num_workers=cfg.BATCH_SIZE)
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)

    model_path = None
    if cfg.VISUAL.EPOCH != 0:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params_{:04}.pt'.format(cfg.VISUAL.EPOCH))
    elif len(cfg.VISUAL.WEIGHT_PATH) != 0:
        model_path = cfg.VISUAL.WEIGHT_PATH
    if model_path is not None:
        print('Loading model parameters from {}'.format(model_path))
        model.load_state_dict(torch.load(model_path))

    visualize_model(model, dataloader, device,
                    num_images=cfg.VISUAL.NUM_IMGS,
                    cls=cfg.VISUAL.CLASS if cfg.VISUAL.CLASS != 'none' else None,
                    save_img=cfg.VISUAL.SAVE)
