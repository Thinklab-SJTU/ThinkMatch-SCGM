import torch
import time
from datetime import datetime
from pathlib import Path

from lib.hungarian import hungarian
from lib.dataset.data_loader import QAPDataset, get_dataloader
from lib.evaluation_metric import objective_score
from lib.parallel import DataParallel
from lib.utils.model_sl import load_model

from lib.utils.config import cfg


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
        rel_sum = torch.zeros(1, device=device)
        rel_num = torch.zeros(1, device=device)
        for inputs in dataloader:
            if cfg.QAPLIB.FEED_TYPE == 'affmat' and 'affmat' in inputs:
                data1 = inputs['affmat'].cuda()
                data2 = None
                inp_type = 'affmat'
            elif cfg.QAPLIB.FEED_TYPE == 'adj':
                data1 = inputs['Fi'].cuda()
                data2 = inputs['Fj'].cuda()
                inp_type = 'adj'
            else:
                raise ValueError('no valid data key found from dataloader!')
            ori_affmtx = inputs['ori_affmat'].cuda()
            solution = inputs['solution'].cuda()
            name = inputs['name']
            n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
            perm_mat = inputs['gt_perm_mat'].cuda()

            batch_num = data1.size(0)

            iter_num = iter_num + 1

            fwd_since = time.time()

            if 'esc16f' in name:
                print('esc16f - 0')
                continue

            with torch.set_grad_enabled(False):
                _ = None
                pred = \
                    model(data1, data2, _, _, _, _, _, _, n1_gt, n2_gt, _, _, inp_type)
                if len(pred) == 2:
                    s_pred, d_pred = pred
                else:
                    s_pred, d_pred, affmtx = pred

            repeat_num = s_pred.shape[0] // batch_num
            repeat = lambda x: torch.repeat_interleave(x, repeat_num, dim=0)
            #repeat = lambda x : x

            if type(s_pred) is list:
                s_pred = s_pred[-1]
            s_pred_perm = lap_solver(s_pred, repeat(n1_gt), repeat(n2_gt))

            fwd_time = time.time() - fwd_since

            #_, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_perm, repeat(perm_mat), repeat(n1_gt))
            #acc_match_num += _acc_match_num
            #acc_total_num += _acc_total_num

            #obj_score = objective_score(s_pred_perm, repeat(ori_affmtx), repeat(n1_gt))
            #obj_score = obj_score.view(obj_score.shape[0] // repeat_num, repeat_num).min(dim=-1).values
            obj_score = objective_score(s_pred_perm[0::repeat_num], ori_affmtx, n1_gt)
            for ri in range(1, repeat_num):
                obj_score = torch.stack((obj_score, objective_score(s_pred_perm[ri::repeat_num], ori_affmtx, n1_gt))).min(dim=0).values
            opt_obj_score = objective_score(perm_mat, ori_affmtx, n1_gt)
            ori_obj_score = solution

            for n, x, y, z in zip(name, obj_score, opt_obj_score, ori_obj_score):
                rel = (x - z) / x
                print('{} - Solved: {:.0f}, Feas: {:.0f}, Opt/Bnd: {:.0f}, Gap: {:.0f}, Rel: {:.4f}, time: {:.3f}'.
                      format(n, x, y, z, x - z, rel, fwd_time))
                if not torch.isnan(rel):
                    rel_sum += rel
                #rel_num += 1

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
    print('mean relative: {:.4f}'.format(float(rel_sum / rel_num)))

    for i in range(len(alphas)):
        print('PCK@{:.2f}'.format(alphas[i]))
        for cls, single_pck in zip(classes, pcks[:, i]):
            print('{} = {:.4f}'.format(cls, single_pck))
        print('average = {:.4f}'.format(torch.mean(pcks[:, i])))

    print('Matching accuracy')
    for cls, single_acc in zip(classes, accs):
        print('{} = {:.4f}'.format(cls, single_acc))
    print('average = {:.4f}'.format(torch.mean(accs)))

    return accs


if __name__ == '__main__':
    from lib.utils.dup_stdout_manager import DupStdoutFileManager
    from lib.utils.parse_args import parse_args
    from lib.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    qap_dataset = QAPDataset(cfg.DATASET_FULL_NAME,
                             sets='test',
                             length=cfg.EVAL.SAMPLES,
                             pad=cfg.PAIR.PADDING,
                             obj_resize=cfg.PAIR.RESCALE)
    dataloader = get_dataloader(qap_dataset)

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
