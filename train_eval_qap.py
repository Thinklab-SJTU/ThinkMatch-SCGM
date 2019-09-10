import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter
import scipy.sparse as ssp

from data.data_loader import QAPDataset, get_dataloader
from GMN.bi_stochastic import BiStochastic
from GMN.permutation_loss import CrossEntropyLoss
from utils.evaluation_metric import pck as eval_pck, matching_accuracy
from parallel import DataParallel
from utils.model_sl import load_model, save_model
from eval_qap import eval_model
from utils.hungarian import hungarian

from utils.config import cfg


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

        scheduler.step()
        model.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        for inputs in dataloader['train']:
            if 'affmat' in inputs:
                data = inputs['affmat'].cuda()
                inp_type = 'affmat'
            else:
                raise ValueError('no valid data key found from dataloader!')
            n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
            perm_mat = inputs['gt_perm_mat'].cuda()

            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                _ = None
                pred = \
                    model(data, _, _, _, _, _, _, _, n1_gt, n2_gt, _, _, inp_type)
                if len(pred) == 2:
                    s_pred, d_pred = pred
                    s_pred_score = s_pred
                else:
                    s_pred_score, s_pred, d_pred = pred

                multi_loss = []
                if cfg.TRAIN.LOSS_FUNC == 'perm':
                    loss = criterion(s_pred, perm_mat, n1_gt, n2_gt)
                elif cfg.TRAIN.LOSS_FUNC == 'obj':
                    loss = criterion(d_pred)
                else:
                    raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

                if type(s_pred_score) is list:
                    s_pred_score = s_pred_score[-1]

                # backward + optimize
                if cfg.FP16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()

                if cfg.MODULE == 'NGM.hypermodel':
                    tfboard_writer.add_scalars(
                        'weight',
                        {'w2': model.module.weight2, 'w3': model.module.weight3},
                        epoch * dataset_size + iter_num
                    )

                # training accuracy statistic
                acc, _, __ = matching_accuracy(lap_solver(s_pred_score, n1_gt, n2_gt), perm_mat, n1_gt)

                # tfboard writer
                loss_dict = {'loss_{}'.format(i): l.item() for i, l in enumerate(multi_loss)}
                loss_dict['loss'] = loss.item()
                tfboard_writer.add_scalars('loss', loss_dict, epoch * dataset_size + iter_num)
                #accdict = {'PCK@{:.2f}'.format(a): p for a, p in zip(alphas, pck)}
                accdict = dict()
                accdict['matching accuracy'] = acc
                tfboard_writer.add_scalars(
                    'training accuracy',
                    accdict,
                    epoch * dataset_size + iter_num
                )

                # statistics
                running_loss += loss.item() * perm_mat.size(0)
                epoch_loss += loss.item() * perm_mat.size(0)

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * perm_mat.size(0) / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                          .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / perm_mat.size(0)))
                    tfboard_writer.add_scalars(
                        'speed',
                        {'speed': running_speed},
                        epoch * dataset_size + iter_num
                    )
                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size

        #loss_dict = dict()
        #loss_dict['loss'] = loss.item()
        #tfboard_writer.add_scalars('loss', loss_dict, epoch * dataset_size + iter_num)

        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()

        # Eval in each epoch
        #print('training set')
        #accs = eval_model(model, alphas, dataloader['train'])
        #print()
        #print('testing set')
        accs = eval_model(model, alphas, dataloader['test'])
        acc_dict = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['train'].dataset.classes, accs)}
        acc_dict['average'] = torch.mean(accs)
        tfboard_writer.add_scalars(
            'Eval acc',
            acc_dict,
            (epoch + 1) * dataset_size
        )

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching training & evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    qap_dataset = {
        x: QAPDataset(cfg.DATASET_FULL_NAME,
                      sets=x,
                      length=dataset_len[x],
                      pad=cfg.PAIR.PADDING,
                      cls=cfg.TRAIN.CLASS if x == 'train' else None,
                      obj_resize=cfg.PAIR.RESCALE)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(qap_dataset[x], fix_seed=(x == 'test'), shuffle=(x == 'train'))
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.cuda()

    if cfg.TRAIN.LOSS_FUNC == 'perm':
        criterion = CrossEntropyLoss()
    elif cfg.TRAIN.LOSS_FUNC == 'obj':
        criterion = torch.sum
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
