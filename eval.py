import torch
import time
from datetime import datetime
from pathlib import Path

from data.data_loader import GMDataset, get_dataloader
from GMN.model import Net
from utils.evaluation_metric import pck
from parallel import DataParallel
from utils.model_sl import load_model

from utils.config import cfg


def eval_model(model, alphas, dataloader, eval_epoch=None, verbose=False):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    classes = ds.classes
    cls_cache = ds.cls

    pcks = torch.zeros(len(classes), len(alphas), device=device)

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.cls = cls
        match_num = torch.zeros(len(alphas), device=device)
        total_num = torch.zeros(len(alphas), device=device)
        for inputs in dataloader:
            img1, img2 = inputs['images']
            P1_gt, P2_gt, P1, P2 = inputs['Ps']
            n1_gt, n2_gt, n1, n2 = inputs['ns']
            e1_gt, e2_gt, e1, e2 = inputs['es']
            G1_gt, G2_gt, G1, G2 = inputs['Gs']
            H1_gt, H2_gt, H1, H2 = inputs['Hs']

            KG, KH = inputs['Ks']

            perm_mat = inputs['gt_perm_mat']

            img1 = img1.to(device)
            img2 = img2.to(device)
            P1_gt = P1_gt.to(device)
            P2 = P2.to(device)
            n1_gt = n1_gt.to(device)
            n2 = n2.to(device)
            perm_mat = perm_mat.to(device)
            P2_gt = P2_gt.to(device)
            G1_gt, G2, H1_gt, H2 = G1_gt.to(device), G2.to(device), H1_gt.to(device), H2.to(device)
            KG, KH = KG.to(device), KH.to(device)

            batch_num = img1.size(0)

            iter_num = iter_num + batch_num

            thres = torch.empty(batch_num, len(alphas), device=device)
            for b in range(batch_num):
                thres[b] = alphas * cfg.EVAL.PCK_L

            with torch.set_grad_enabled(False):
                s_pred, _ = model(img1, img2, P1_gt, P2, G1_gt, G2, H1_gt, H2, n1_gt, n2, KG, KH)

            _, _match_num, _total_num = pck(P2, P2_gt, s_pred, thres, n1_gt)
            match_num += _match_num
            total_num += _total_num

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP / (time.time() - running_since)
                print('Class {:<8} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()

        pcks[i] = match_num / total_num
        if verbose:
            print('Class {} PCK@{{'.format(cls) +
                  ', '.join(list(map('{:.2f}'.format, alphas.tolist()))) + '} = {' +
                  ', '.join(list(map('{:.4f}'.format, pcks[i].tolist()))) + '}')

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    # print result
    for i in range(len(alphas)):
        print('PCK@{:.2f}'.format(alphas[i]))
        for cls, single_pck in zip(classes, pcks[:, i]):
            print('{} = {:.4f}'.format(cls, single_pck))
        print('average = {:.4f}'.format(torch.mean(pcks[:, i])))

    return pcks


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args

    args = parse_args('Deep learning of graph matching evaluation code.')

    torch.manual_seed(cfg.RANDOM_SEED)

    image_dataset = GMDataset('PascalVOC',
                              sets='test',
                              length=cfg.EVAL.EPOCH_ITERS,
                              pad=cfg.PAIR.PADDING,
                              obj_resize=cfg.PAIR.RESCALE)
    dataloader = get_dataloader(image_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        alphas = torch.tensor(cfg.EVAL.PCK_ALPHAS, dtype=torch.float32, device=device)
        classes = dataloader.dataset.classes
        pcks = eval_model(model, alphas, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True)
