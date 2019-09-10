import torch
import time
from datetime import datetime
from pathlib import Path

from GMN.bi_stochastic import BiStochastic
from utils.hungarian import hungarian
from data.data_loader import GMRefDataset, get_dataloader
from utils.evaluation_metric import pck, matching_accuracy
from parallel import DataParallel
from utils.model_sl import load_model
from NGM.refmodel import RefGraph

from utils.config import cfg


def eval_model(model, refmodel, alphas, dataloader, eval_epoch=None, verbose=False):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

        refmodel_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'refs_{:04}.pt'.format(eval_epoch))
        print('Loading refmodel parameters from {}'.format(refmodel_path))
        load_model(refmodel, refmodel_path)

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    classes = ds.classes
    cls_cache = ds.cls

    lap_solver = hungarian

    accs = torch.zeros(len(classes), device=device)

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.cls = cls
        acc_match_num = torch.zeros(1, device=device)
        acc_total_num = torch.zeros(1, device=device)
        for inputs in dataloader:
            if 'images' in inputs:
                data1 = inputs['images'].cuda()
                inp_type = 'img'
            elif 'features' in inputs:
                data1 = inputs['features'].cuda()
                inp_type = 'feat'
            else:
                raise ValueError('no valid data key (\'images\' or \'features\') found from dataloader!')
            cls = inputs['cls']
            P1_gt = inputs['Ps']
            n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
            e1_gt, e2_gt = [_.cuda() for _ in inputs['es']]
            G1_gt, G2_gt = [_.cuda() for _ in inputs['Gs']]
            H1_gt, H2_gt = [_.cuda() for _ in inputs['Hs']]
            KG, KH = [_.cuda() for _ in inputs['Ks']]
            perm_mat = inputs['gt_perm_mat'].cuda()

            data2 = refmodel.get_ref(cls)

            batch_num = data1.size(0)

            iter_num = iter_num + 1

            with torch.set_grad_enabled(False):
                pred = \
                    model(data1, data2, P1_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH, inp_type)
                if len(pred) == 2:
                    s_pred_score, d_pred = pred
                else:
                    s_pred_score, _, d_pred = pred

            if type(s_pred_score) is list:
                s_pred_score = s_pred_score[-1]
            s_pred_perm = lap_solver(s_pred_score, n1_gt, n2_gt)

            def self_mul_helper(t):
                return t
                new_t = torch.zeros(t.shape[0], t.shape[1], t.shape[1])
                for i in range(t.shape[0]):
                    new_t[i] = torch.matmul(t[i], t[(i + 1) % t.shape[0]].transpose(0, 1))
                return new_t

            _, _acc_match_num, _acc_total_num = matching_accuracy(self_mul_helper(s_pred_perm), self_mul_helper(perm_mat), n1_gt)
            acc_match_num += _acc_match_num
            acc_total_num += _acc_total_num


            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print('Class {:<8} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()

        accs[i] = acc_match_num / acc_total_num
        if verbose:
            print('Class {} acc = {:.4f}'.format(cls, accs[i]))

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    # print result
    print('Matching accuracy')
    for cls, single_acc in zip(classes, accs):
        print('{} = {:.4f}'.format(cls, single_acc))
    print('average = {:.4f}'.format(torch.mean(accs)))

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

    image_dataset = GMRefDataset(cfg.DATASET_FULL_NAME,
                              sets='test',
                              length=cfg.EVAL.SAMPLES,
                              pad=cfg.PAIR.PADDING,
                              obj_resize=cfg.PAIR.RESCALE)
    dataloader = get_dataloader(image_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    refmodel = RefGraph(image_dataset.ds.classes_kpts,
                        cfg.NGM.FEATURE_CHANNEL, cfg.NGM.FEATURE_CHANNEL)
    refmodel = refmodel.cuda()

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        alphas = torch.tensor(cfg.EVAL.PCK_ALPHAS, dtype=torch.float32, device=device)
        classes = dataloader.dataset.classes
        pcks = eval_model(model, refmodel, alphas, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True)
