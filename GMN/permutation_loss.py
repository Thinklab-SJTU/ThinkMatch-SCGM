import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.hungarian import hungarian


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)

        try:
            assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_perm)
            raise err

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_perm[b, :pred_ns[b], :gt_ns[b]],
                gt_perm[b, :pred_ns[b], :gt_ns[b]],
                reduction='sum')
            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum


class CrossEntropyLossHung(nn.Module):
    """
    Cross entropy loss between two permutations.
    """
    def __init__(self):
        super(CrossEntropyLossHung, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        lap_solver = hungarian

        dis_pred = lap_solver(pred_perm, pred_ns, gt_ns)
        # dis_pred = dis_pred.detach()

        # pdb.set_trace()
        ali_perm = dis_pred + gt_perm
        # ali_perm[ali_perm > 1.0] = 1.0 # Hung
        ali_perm[ali_perm > 1.0] = 0.9 # Hung+
        pred_perm = torch.mul(ali_perm, pred_perm)
        gt_perm = torch.mul(ali_perm, gt_perm)
        # pdb.set_trace()
        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        # pdb.set_trace()
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_perm[b, :pred_ns[b], :gt_ns[b]],
                gt_perm[b, :pred_ns[b], :gt_ns[b]],
                reduction='sum')
            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)
        # pdb.set_trace()
        return loss / n_sum
