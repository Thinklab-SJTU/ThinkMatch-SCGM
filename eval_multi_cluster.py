import time
from datetime import datetime
from pathlib import Path

from lib.hungarian import hungarian
from lib.dataset.data_loader import MGMCDataset, get_dataloader
from lib.evaluation_metric import *
from lib.parallel import DataParallel
from lib.utils.model_sl import load_model
from lib.utils.timer import Timer

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
    accs = [] #torch.zeros(len(classes), device=device)
    precisions = [] #torch.zeros(len(classes), device=device)
    f1s = [] #torch.zeros(len(classes), device=device)
    accs_mgm = [] #torch.zeros(len(classes), device=device)
    precisions_mgm = [] #torch.zeros(len(classes), device=device)
    f1s_mgm = [] #torch.zeros(len(classes), device=device)
    cluster_acc = [] #torch.zeros(len(classes), device=device)
    cluster_purity = [] #torch.zeros(len(classes), device=device)
    cluster_ri = [] #torch.zeros(len(classes), device=device)
    inf_time = []

    timer = Timer()

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.cls = cls
        pck_match_num = torch.zeros(len(alphas), device=device)
        pck_total_num = torch.zeros(len(alphas), device=device)
        acc_pairwise_list = []
        precision_pairwise_list = []
        f1_pairwise_list = []
        acc_mgm_list = []
        precision_mgm_list = []
        f1_mgm_list = []
        cluster_acc_list = [] #torch.zeros(1, device=device)
        cluster_purity_list = [] #torch.zeros(1, device=device)
        cluster_ri_list = [] #torch.zeros(1, device=device)
        inf_time_list = []
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
            cls_gt = [_ for _ in inputs['cls']]

            batch_num = data[0].size(0)

            iter_num = iter_num + 1

            thres = torch.empty(batch_num, len(alphas), device=device)
            for b in range(batch_num):
                thres[b] = alphas * cfg.EVAL.PCK_L

            with torch.set_grad_enabled(False):
                timer.tick()
                pred = model(
                    data, Ps_gt, Gs_gt, Hs_gt,
                    Gs_ref=Gs_ref, Hs_ref=Hs_ref, KGs=KGs, KHs=KHs,
                    ns=ns_gt,
                    gt_cls=cls_gt,
                    iter_times=2,
                    type=inp_type,
                    pretrain_backbone=False,
                    num_clusters=cfg.PAIR.NUM_CLUSTERS,
                    return_cluster=True
                )
                s_pred_list, indices, s_pred_list_mgm, pred_cluster = pred
                pred_t = timer.toc()

            #pred_num = len(s_pred_list[0]) # todo

            inf_time_list.append(torch.full((batch_num,), pred_t / batch_num))
            #inf_time_list.append(torch.full((pred_num,), pred_t / batch_num)) # todo

            #ns_gt = [torch.repeat_interleave(x, pred_num, dim=0) for x in ns_gt] #todo

            # pairwise accuracy
            _acc_pairwise_list = []
            _precision_pairwise_list = []
            _f1_pairwise_list = []
            for s_pred, (gt_idx_src, gt_idx_tgt) in zip(s_pred_list, indices):
                pred_perm_mat = lap_solver(s_pred, ns_gt[gt_idx_src], ns_gt[gt_idx_tgt])
                gt_perm_mat = torch.bmm(perm_mats[gt_idx_src], perm_mats[gt_idx_tgt].transpose(1, 2))
                #gt_perm_mat = torch.repeat_interleave(gt_perm_mat, pred_num, dim=0) # todo
                acc, _, __ = matching_accuracy(pred_perm_mat, gt_perm_mat, ns_gt[gt_idx_src])
                _acc_pairwise_list.append(acc)
                precision, _, __ = matching_precision(pred_perm_mat, gt_perm_mat, ns_gt[gt_idx_src])
                _precision_pairwise_list.append(precision)
                f1 = 2 * (precision * acc) / (precision + acc)
                f1[torch.isnan(f1)] = 0
                _f1_pairwise_list.append(f1)
            acc_pairwise_list.append(torch.mean(torch.stack(_acc_pairwise_list, dim=-1), dim=-1))
            precision_pairwise_list.append(torch.mean(torch.stack(_precision_pairwise_list, dim=-1), dim=-1))
            f1_pairwise_list.append(torch.mean(torch.stack(_f1_pairwise_list, dim=-1), dim=-1))

            # multi-matching accuracy
            _acc_mgm_list = []
            _precision_mgm_list = []
            _f1_mgm_list = []
            for s_pred, (gt_idx_src, gt_idx_tgt) in zip(s_pred_list_mgm, indices):
                pred_perm_mat = lap_solver(s_pred, ns_gt[gt_idx_src], ns_gt[gt_idx_tgt])
                gt_perm_mat = torch.bmm(perm_mats[gt_idx_src], perm_mats[gt_idx_tgt].transpose(1, 2))
                #gt_perm_mat = torch.repeat_interleave(gt_perm_mat, pred_num, dim=0) # todo
                acc, _acc_match_num, _acc_total_num = matching_accuracy(pred_perm_mat, gt_perm_mat, ns_gt[gt_idx_src])
                _acc_mgm_list.append(acc)
                precision, _, __ = matching_precision(pred_perm_mat, gt_perm_mat, ns_gt[gt_idx_src])
                _precision_mgm_list.append(precision)
                f1 = 2 * (precision * acc) / (precision + acc)
                f1[torch.isnan(f1)] = 0
                _f1_mgm_list.append(f1)
            acc_mgm_list.append(torch.mean(torch.stack(_acc_mgm_list, dim=-1), dim=-1))
            precision_mgm_list.append(torch.mean(torch.stack(_precision_mgm_list, dim=-1), dim=-1))
            f1_mgm_list.append(torch.mean(torch.stack(_f1_mgm_list, dim=-1), dim=-1))

            #todo
            #for iii in range(pred_num):
            #    print('mgm f1 {} {:.4f}'.format(iii, torch.mean(torch.cat(f1_mgm_list)[iii::pred_num])))
            print('mgm f1 {:.4f}'.format(torch.mean(torch.cat(f1_mgm_list))))

            #cls_gt = [_ * pred_num for _ in cls_gt] #todo
            # clustering metric
            cls_gt_transpose = [[] for _ in range(batch_num)]
            #cls_gt_transpose = [[] for _ in range(pred_num)] # todo
            for batched_cls in cls_gt:
                for b, _cls in enumerate(batched_cls):
                    cls_gt_transpose[b].append(_cls)
            cluster_acc_list.append(clustering_accuracy(pred_cluster, cls_gt_transpose))
            cluster_purity_list.append(clustering_purity(pred_cluster, cls_gt_transpose))
            cluster_ri_list.append(rand_index(pred_cluster, cls_gt_transpose))

            #todo
            #for b in range(pred_num):
            for b in range(batch_num):
                print('CA {} {:.4f}'.format(b, torch.mean(clustering_accuracy(pred_cluster[b:b+1], cls_gt_transpose[b:b+1]))))
                print('CP {} {:.4f}'.format(b, torch.mean(clustering_purity(pred_cluster[b:b+1], cls_gt_transpose[b:b+1]))))
                print('RI {} {:.4f}'.format(b, torch.mean(rand_index(pred_cluster[b:b+1], cls_gt_transpose[b:b+1]))))

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print('Class {:<8} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()

        pcks[i] = pck_match_num / pck_total_num
        accs.append(torch.cat(acc_pairwise_list))
        precisions.append(torch.cat(precision_pairwise_list))
        f1s.append(torch.cat(f1_pairwise_list))
        accs_mgm.append(torch.cat(acc_mgm_list))
        precisions_mgm.append(torch.cat(precision_mgm_list))
        f1s_mgm.append(torch.cat(f1_mgm_list))
        cluster_acc.append(torch.cat(cluster_acc_list))
        cluster_purity.append(torch.cat(cluster_purity_list))
        cluster_ri.append(torch.cat(cluster_ri_list))
        inf_time.append(torch.cat(inf_time_list))

        if verbose:
            print('Class {} PCK@{{'.format(cls) +
                  ', '.join(list(map('{:.2f}'.format, alphas.tolist()))) + '} = {' +
                  ', '.join(list(map('{:.4f}'.format, pcks[i].tolist()))) + '}')
            print('Class {} {}'.format(cls, format_accuracy_metric(precisions[i], accs[i], f1s[i])))
            print('Class {} mgm {}'.format(cls, format_accuracy_metric(precisions_mgm[i], accs_mgm[i], f1s_mgm[i])))
            print('Class {} cluster acc={}'.format(cls, format_metric(cluster_acc[i])))
            print('Class {} cluster purity={}'.format(cls, format_metric(cluster_purity[i])))
            print('Class {} cluster rand index={}'.format(cls, format_metric(cluster_ri[i])))
            print('Class {} inference time={}'.format(cls, format_metric(inf_time[i])))

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
    for cls, cls_p, cls_acc, cls_f1 in zip(classes, precisions, accs, f1s):
        print('{}: {}'.format(cls, format_accuracy_metric(cls_p, cls_acc, cls_f1)))
    print('average accuracy: {}'.format(format_accuracy_metric(torch.cat(precisions), torch.cat(accs), torch.cat(f1s))))

    print('MGM Matching accuracy')
    for cls, cls_p, cls_acc, cls_f1 in zip(classes, precisions_mgm, accs_mgm, f1s_mgm):
        print('{}: {}'.format(cls, format_accuracy_metric(cls_p, cls_acc, cls_f1)))
    print(u'average accuracy: {}'.format(format_accuracy_metric(torch.cat(precisions_mgm), torch.cat(accs_mgm), torch.cat(f1s_mgm))))

    if cfg.PAIR.NUM_CLUSTERS > 1:
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

    print('Inference time')
    for cls, cls_time in zip(classes, inf_time):
        print('{} = {}'.format(cls, format_metric(cls_time)))

    return torch.Tensor(list(map(torch.mean, accs)))


if __name__ == '__main__':
    from lib.utils.dup_stdout_manager import DupStdoutFileManager
    from lib.utils.parse_args import parse_args
    from lib.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    torch.manual_seed(cfg.RANDOM_SEED)
    import numpy as np
    np.random.seed(cfg.RANDOM_SEED)
    import random
    random.seed(cfg.RANDOM_SEED)

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    image_dataset = MGMCDataset(cfg.DATASET_FULL_NAME,
                                sets='train',
                                length=cfg.EVAL.SAMPLES,
                                pad=cfg.PAIR.PADDING,
                                problem='multi_cluster',
                                obj_resize=cfg.PAIR.RESCALE)
    #image_dataset.classes = [['Car', 'Duck', 'Motorbike']] #, 'Winebottle']]
    #image_dataset.classes = [['051.Horned_Grebe', '113.Baird_Sparrow', '143.Caspian_Tern']]
    #image_dataset.classes = [['Grebe', 'Sparrow', 'Tern']]
    cls_len = len(image_dataset.classes)
    if cls_len > 200:
        image_dataset.classes = image_dataset.classes[::(cls_len//200)]
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
