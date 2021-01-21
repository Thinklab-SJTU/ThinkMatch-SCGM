import time
from datetime import datetime
from pathlib import Path

from src.dataset.data_loader import GMDataset, get_dataloader
from src.evaluation_metric import *
from src.parallel import DataParallel
from src.utils.model_sl import load_model
from src.utils.data_to_cuda import data_to_cuda
from src.utils.timer import Timer

from src.utils.config import cfg


def eval_model(model, alphas, dataloader, eval_epoch=None, verbose=False):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    model_path = ''
    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
    if len(cfg.PRETRAINED_PATH) > 0:
        model_path = cfg.PRETRAINED_PATH
    if len(model_path) > 0:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    classes = ds.classes

    pcks = torch.zeros(len(classes), len(alphas), device=device)
    accs = []
    precisions = []
    f1s = []
    pred_time = []
    objs = torch.zeros(len(classes), device=device)
    cluster_acc = []
    cluster_purity = []
    cluster_ri = []

    timer = Timer()

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.cls = cls
        pck_match_num = torch.zeros(len(alphas), device=device)
        pck_total_num = torch.zeros(len(alphas), device=device)
        acc_list = []
        precision_list = [] 
        f1_list = []
        pred_time_list = []
        obj_total_num = torch.zeros(1, device=device)
        cluster_acc_list = []
        cluster_purity_list = []
        cluster_ri_list = []

        for inputs in dataloader:
            if model.module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)

            batch_num = inputs['batch_size']

            iter_num = iter_num + 1

            thres = torch.empty(batch_num, len(alphas), device=device)
            for b in range(batch_num):
                thres[b] = alphas * cfg.EVAL.PCK_L

            with torch.set_grad_enabled(False):
                timer.tick()
                outputs = model(inputs)
                pred_time_list.append(torch.full((batch_num,), timer.toc() / batch_num))

            # Evaluate matching accuracy
            if cfg.PROBLEM.TYPE == '2GM':
                assert 'perm_mat' in outputs
                assert 'gt_perm_mat' in outputs

                # _, _pck_match_num, _pck_total_num = pck(P2_gt, P2_gt, torch.bmm(s_pred_perm, perm_mat.transpose(1, 2)), thres, n1_gt)
                # pck_match_num += _pck_match_num
                # pck_total_num += _pck_total_num

                acc, _, __ = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])
                acc_list.append(acc)
                precision, _, __ = matching_precision(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])
                precision_list.append(precision)
                precision_list.append(precision)
                f1 = 2 * (precision * acc) / (precision + acc)
                f1[torch.isnan(f1)] = 0
                f1_list.append(f1)

                if 'aff_mat' in outputs:
                    pred_obj_score = objective_score(outputs['perm_mat'], outputs['aff_mat'], outputs['ns'][0])
                    gt_obj_score = objective_score(outputs['gt_perm_mat'], outputs['aff_mat'], outputs['ns'][0])
                    objs[i] += torch.sum(pred_obj_score / gt_obj_score)
                    obj_total_num += batch_num
            elif cfg.PROBLEM.TYPE in ['MGM', 'MGMC']:
                assert 'graph_indices' in outputs
                assert 'perm_mat_list' in outputs
                assert 'gt_perm_mat_list' in outputs

                ns = outputs['ns']
                for x_pred, x_gt, (idx_src, idx_tgt) in \
                        zip(outputs['perm_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):
                    acc, _, __ = matching_accuracy(x_pred, x_gt, ns[idx_src])
                    acc_list.append(acc)
                    precision, _, __ = matching_precision(x_pred, x_gt, ns[idx_src])
                    precision_list.append(precision)
                    f1 = 2 * (precision * acc) / (precision + acc)
                    f1[torch.isnan(f1)] = 0
                    f1_list.append(f1)
            else:
                raise ValueError('Unknown problem type {}'.format(cfg.PROBLEM.TYPE))

            # Evaluate clustering accuracy
            if cfg.PROBLEM.TYPE == 'MGMC':
                assert 'pred_cluster' in outputs
                assert 'cls' in outputs

                pred_cluster = outputs['pred_cluster']
                cls_gt_transpose = [[] for _ in range(batch_num)]
                for batched_cls in outputs['cls']:
                    for b, _cls in enumerate(batched_cls):
                        cls_gt_transpose[b].append(_cls)
                cluster_acc_list.append(clustering_accuracy(pred_cluster, cls_gt_transpose))
                cluster_purity_list.append(clustering_purity(pred_cluster, cls_gt_transpose))
                cluster_ri_list.append(rand_index(pred_cluster, cls_gt_transpose))

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print('Class {:<8} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()

        pcks[i] = pck_match_num / pck_total_num
        accs.append(torch.cat(acc_list))
        precisions.append(torch.cat(precision_list))
        f1s.append(torch.cat(f1_list))
        objs[i] = objs[i] / obj_total_num
        pred_time.append(torch.mean(torch.cat(pred_time_list)))
        if cfg.PROBLEM.TYPE == 'MGMC':
            cluster_acc.append(torch.cat(cluster_acc_list))
            cluster_purity.append(torch.cat(cluster_purity_list))
            cluster_ri.append(torch.cat(cluster_ri_list))

        if verbose:
            print('Class {} PCK@{{'.format(cls) +
                  ', '.join(list(map('{:.2f}'.format, alphas.tolist()))) + '} = {' +
                  ', '.join(list(map('{:.4f}'.format, pcks[i].tolist()))) + '}')
            print('Class {} {}'.format(cls, format_accuracy_metric(precisions[i], accs[i], f1s[i])))
            print('Class {} norm obj score = {:.4f}'.format(cls, objs[i]))
            print('Class {} pred time = {:.4f}s'.format(cls, pred_time[i]))
            if cfg.PROBLEM.TYPE == 'MGMC':
                print('Class {} cluster acc={}'.format(cls, format_metric(cluster_acc[i])))
                print('Class {} cluster purity={}'.format(cls, format_metric(cluster_purity[i])))
                print('Class {} cluster rand index={}'.format(cls, format_metric(cluster_ri[i])))

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)

    # print result
    for i in range(len(alphas)):
        print('PCK@{:.2f}'.format(alphas[i]))
        for cls, single_pck in zip(classes, pcks[:, i]):
            print('{} = {:.4f}'.format(cls, single_pck))
        print('average PCK = {:.4f}'.format(torch.mean(pcks[:, i])))

    print('Matching accuracy')
    for cls, cls_p, cls_acc, cls_f1 in zip(classes, precisions, accs, f1s):
        print('{}: {}'.format(cls, format_accuracy_metric(cls_p, cls_acc, cls_f1)))
    print('average accuracy: {}'.format(format_accuracy_metric(torch.cat(precisions), torch.cat(accs), torch.cat(f1s))))

    if not torch.any(torch.isnan(objs)):
        print('Normalized objective score')
        for cls, cls_obj in zip(classes, objs):
            print('{} = {:.4f}'.format(cls, cls_obj))
        print('average objscore = {:.4f}'.format(torch.mean(objs)))

    if cfg.PROBLEM.TYPE == 'MGMC':
        print('Clustering accuracy')
        for cls, cls_acc in zip(classes, cluster_acc):
            print('{} = {}'.format(cls, format_metric(cls_acc)))
        print('average clustering accuracy = {}'.format(format_metric(torch.cat(cluster_acc))))

        print('Clustering purity')
        for cls, cls_acc in zip(classes, cluster_purity):
            print('{} = {}'.format(cls, format_metric(cls_acc)))
        print('average clustering purity = {}'.format(format_metric(torch.cat(cluster_purity))))

        print('Clustering rand index')
        for cls, cls_acc in zip(classes, cluster_ri):
            print('{} = {}'.format(cls, format_metric(cls_acc)))
        print('average rand index = {}'.format(format_metric(torch.cat(cluster_ri))))

    print('Predict time')
    for cls, cls_time in zip(classes, pred_time):
        print('{} = {:.4f}'.format(cls, torch.mean(cls_time)))

    return torch.Tensor(list(map(torch.mean, accs)))


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    image_dataset = GMDataset(cfg.DATASET_FULL_NAME,
                              sets='test',
                              problem=cfg.PROBLEM.TYPE,
                              length=cfg.EVAL.SAMPLES,
                              cls=cfg.EVAL.CLASS,
                              obj_resize=cfg.PROBLEM.RESCALE)
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
        pcks = eval_model(model, alphas, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True)
