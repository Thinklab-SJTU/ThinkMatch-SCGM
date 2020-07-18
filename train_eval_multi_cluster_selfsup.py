import torch.optim as optim
import time
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

from src.dataset.data_loader import MGMCDataset, get_dataloader
from src.loss_func import *
from src.evaluation_metric import matching_accuracy
from src.parallel import DataParallel
from src.utils.model_sl import load_model, save_model
from eval_multi_cluster import eval_model
from src.lap_solvers.hungarian import hungarian

from src.utils.config import cfg

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')


def train_eval_model(model,
                     criterion,
                     optimizer,
                     dataloader,
                     tfboard_writer,
                     num_epochs=25,
                     resume=False,
                     start_epoch=0):
    print('Start training...')

    since = time.time()
    dataset_size = len(dataloader['train'].dataset)
    lap_solver = hungarian

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    alphas = torch.tensor(cfg.EVAL.PCK_ALPHAS, dtype=torch.float32, device=device)  # for evaluation

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        assert start_epoch != 0
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY,
                                               last_epoch=cfg.TRAIN.START_EPOCH - 1)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0
        det_anomaly = False

        # Iterate over data.
        for inputs in dataloader['train']:
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
            KGs = {_ : inputs['KGs'][_].cuda() for _ in inputs['KGs']}
            KHs = {_ : inputs['KHs'][_].cuda() for _ in inputs['KHs']}
            perm_mats = [_.cuda() for _ in inputs['gt_perm_mat']]

            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                with torch.autograd.set_detect_anomaly(det_anomaly):
                    # forward
                    pred = model(
                        data,
                        Ps_gt, Gs_gt, Hs_gt,
                        Gs_ref=Gs_ref, Hs_ref=Hs_ref,
                        KGs=KGs, KHs=KHs,
                        ns=ns_gt,
                        gt_cls=None,
                        iter_times=2,
                        type=inp_type,
                        pretrain_backbone=False,
                        num_clusters=cfg.PAIR.NUM_CLUSTERS,
                        return_cluster=True
                    )
                    if len(pred) == 2:
                        s_pred_list, indices = pred
                    else:
                        s_pred_list, indices, s_pred_list_mgm, cluster_v = pred

                    if cfg.TRAIN.LOSS_FUNC == 'perm' or cfg.TRAIN.LOSS_FUNC == 'focal' or cfg.TRAIN.LOSS_FUNC == 'hung' or cfg.TRAIN.LOSS_FUNC == 'innp':
                        loss = torch.zeros(1).to(device)
                        assert type(s_pred_list) is list
                        assert type(s_pred_list_mgm) is list
                        for s_pred, sd_pred, (gt_idx_src, gt_idx_tgt) in zip(s_pred_list, s_pred_list_mgm, indices):
                            l = criterion(s_pred, sd_pred, ns_gt[gt_idx_src], ns_gt[gt_idx_tgt])
                            loss += l
                        loss /= len(s_pred_list)

                    else:
                        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

                    # backward + optimize
                    if cfg.FP16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    det_anomaly = False

                    for param in model.parameters():
                        if param.requires_grad and param.grad is not None and torch.any(torch.isnan(param.grad)):
                            det_anomaly = True
                            break
                    if not det_anomaly:
                        optimizer.step()

                    # training accuracy statistic
                    matched_num = 0
                    total_num = 0
                    for s_pred, (gt_idx_src, gt_idx_tgt) in zip(s_pred_list, indices):
                        pred_perm_mat = lap_solver(s_pred, ns_gt[gt_idx_src], ns_gt[gt_idx_tgt])
                        gt_perm_mat = torch.bmm(perm_mats[gt_idx_src], perm_mats[gt_idx_tgt].transpose(1, 2))
                        _, mn, tn = matching_accuracy(pred_perm_mat, gt_perm_mat, ns_gt[gt_idx_src])
                        matched_num += mn
                        total_num += tn
                    acc = matched_num / total_num

                    # tfboard writer
                    loss_dict = dict()
                    loss_dict['loss'] = loss.item()
                    tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)
                    #accdict = {'PCK@{:.2f}'.format(a): p for a, p in zip(alphas, pck)}
                    accdict = dict()
                    accdict['matching accuracy'] = acc
                    tfboard_writer.add_scalars(
                        'training accuracy',
                        accdict,
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )

                    # statistics
                    running_loss += loss.item() * perm_mats[0].size(0)
                    epoch_loss += loss.item() * perm_mats[0].size(0)

                    if iter_num % cfg.STATISTIC_STEP == 0:
                        running_speed = cfg.STATISTIC_STEP * perm_mats[0].size(0) / (time.time() - running_since)
                        print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                              .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / perm_mats[0].size(0)))
                        tfboard_writer.add_scalars(
                            'speed',
                            {'speed': running_speed},
                            epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                        )
                        running_loss = 0.0
                        running_since = time.time()

        epoch_loss = epoch_loss / dataset_size

        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()

        # Eval in each epoch
        accs = eval_model(model, alphas, dataloader['test'])
        acc_dict = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['train'].dataset.classes, accs)}
        acc_dict['average'] = torch.mean(accs)
        tfboard_writer.add_scalars(
            'Eval acc',
            acc_dict,
            (epoch + 1) * cfg.TRAIN.EPOCH_ITERS
        )

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching training & evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    image_dataset = {
        x: MGMCDataset(cfg.DATASET_FULL_NAME,
                       sets=x,
                       length=dataset_len[x],
                       pad=cfg.PAIR.PADDING,
                       cls=cfg.TRAIN.CLASS if x == 'train' else None,
                       problem='multi_cluster',
                       obj_resize=cfg.PAIR.RESCALE)
        for x in ('train', 'test')}
    #for x in ('train', 'test'):
        #image_dataset[x].classes = [['Car', 'Duck', 'Motorbike']]
        #image_dataset[x].classes = [['051.Horned_Grebe', '113.Baird_Sparrow', '143.Caspian_Tern']]
        #image_dataset[x].classes = [['Grebe', 'Sparrow', 'Tern']]

    cls_len = len(image_dataset['test'].classes)
    if cls_len > 200:
        image_dataset['test'].classes = image_dataset['test'].classes[::(cls_len//200)]
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.cuda()

    if cfg.TRAIN.LOSS_FUNC == 'offset':
        criterion = RobustLoss(norm=cfg.TRAIN.RLOSS_NORM)
    elif cfg.TRAIN.LOSS_FUNC == 'perm':
        criterion = CrossEntropyLoss()
    elif cfg.TRAIN.LOSS_FUNC == 'hung':
        criterion = CrossEntropyLossHung()
    elif cfg.TRAIN.LOSS_FUNC == 'focal':
        criterion = FocalLoss(alpha=.5, gamma=0.)
    elif cfg.TRAIN.LOSS_FUNC == 'innp':
        criterion = InnerProductLoss()
    else:
        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)

    if cfg.FP16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to enable FP16.")
        model, optimizer = amp.initialize(model, optimizer)

    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(model, criterion, optimizer, dataloader, tfboardwriter,
                             num_epochs=cfg.TRAIN.NUM_EPOCHS,
                             resume=cfg.TRAIN.START_EPOCH != 0,
                             start_epoch=cfg.TRAIN.START_EPOCH)
