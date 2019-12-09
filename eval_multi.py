import torch
import time
from datetime import datetime
from pathlib import Path
import itertools

from GMN.bi_stochastic import BiStochastic
from utils.hungarian import hungarian
from data.data_loader import GMDataset, get_dataloader
from utils.evaluation_metric import pck, matching_accuracy
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

    #lap_solver = BiStochastic(max_iter=20)
    lap_solver = hungarian

    pcks = torch.zeros(len(classes), len(alphas), device=device)
    accs = torch.zeros(len(classes), device=device)

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.cls = cls
        pck_match_num = torch.zeros(len(alphas), device=device)
        pck_total_num = torch.zeros(len(alphas), device=device)
        acc_match_num = torch.zeros(1, device=device)
        acc_total_num = torch.zeros(1, device=device)
        for inputs in dataloader:
            if 'images' in inputs:
                data = [_.cuda() for _ in inputs['images']]
                inp_type = 'img'
            elif 'features' in inputs:
                data = [_.cuda() for _ in inputs['features']]
                inp_type = 'feat'
            else:
                raise ValueError('no valid data key (\'images\' or \'features\') found from dataloader!')
            Ps_gt = [_.cuda() for _ in inputs['Ps']]
            ns_gt = [_.cuda() for _ in inputs['ns']]
            es_gt = [_.cuda() for _ in inputs['es']]
            Gs_gt = [_.cuda() for _ in inputs['Gs']]
            Hs_gt = [_.cuda() for _ in inputs['Hs']]
            Gs_ref = [_.cuda() for _ in inputs['Gs_ref']]
            Hs_ref = [_.cuda() for _ in inputs['Hs_ref']]
            KGs = {_: inputs['KGs'][_].cuda() for _ in inputs['KGs']}
            KHs = {_: inputs['KHs'][_].cuda() for _ in inputs['KHs']}
            perm_mats = [_.cuda() for _ in inputs['gt_perm_mat']]

            batch_num = data[0].size(0)

            iter_num = iter_num + 1

            thres = torch.empty(batch_num, len(alphas), device=device)
            for b in range(batch_num):
                thres[b] = alphas * cfg.EVAL.PCK_L

            with torch.set_grad_enabled(False):
                pred = model(data, Ps_gt, Gs_gt, Hs_gt, Gs_ref, Hs_ref, KGs, KHs, ns_gt, iter_times=2, type=inp_type)
                s_pred_list, indices = pred

            for s_pred, (gt_idx_src, gt_idx_tgt) in zip(s_pred_list, indices):
                pred_perm_mat = lap_solver(s_pred, ns_gt[gt_idx_src], ns_gt[gt_idx_tgt])
                gt_perm_mat = torch.bmm(perm_mats[gt_idx_src], perm_mats[gt_idx_tgt].transpose(1, 2))
                _, _acc_match_num, _acc_total_num = matching_accuracy(pred_perm_mat, gt_perm_mat, ns_gt[gt_idx_src])
                acc_match_num += _acc_match_num
                acc_total_num += _acc_total_num

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print('Class {:<8} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()

        pcks[i] = pck_match_num / pck_total_num
        accs[i] = acc_match_num / acc_total_num
        if verbose:
            print('Class {} PCK@{{'.format(cls) +
                  ', '.join(list(map('{:.2f}'.format, alphas.tolist()))) + '} = {' +
                  ', '.join(list(map('{:.4f}'.format, pcks[i].tolist()))) + '}')
            print('Class {} acc = {:.4f}'.format(cls, accs[i]))

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    # print result
    for i in range(len(alphas)):
        print('PCK@{:.2f}'.format(alphas[i]))
        for cls, single_pck in zip(classes, pcks[:, i]):
            print('{} = {:.4f}'.format(cls, single_pck))
        print('average PCK = {:.4f}'.format(torch.mean(pcks[:, i])))

    print('Matching accuracy')
    for cls, single_acc in zip(classes, accs):
        print('{} = {:.4f}'.format(cls, single_acc))
    print('average accuracy = {:.4f}'.format(torch.mean(accs)))

    return accs


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    image_dataset = GMDataset(cfg.DATASET_FULL_NAME,
                              sets='test',
                              length=cfg.EVAL.SAMPLES,
                              pad=cfg.PAIR.PADDING,
                              pair=False,
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
        print_easydict(cfg)
        alphas = torch.tensor(cfg.EVAL.PCK_ALPHAS, dtype=torch.float32, device=device)
        classes = dataloader.dataset.classes
        pcks = eval_model(model, alphas, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True)
