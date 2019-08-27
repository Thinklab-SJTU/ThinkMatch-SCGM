import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss between two permutations.
    """
    def __init__(self, alpha=1., gamma=0., eps=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            x = pred_perm[b, :pred_ns[b], :gt_ns[b]]
            y = gt_perm[b, :pred_ns[b], :gt_ns[b]]
            loss += torch.sum(
                - self.alpha * (1 - x) ** self.gamma * y * torch.log(x + self.eps)
                - (1 - self.alpha) * x ** self.gamma * (1 - y) * torch.log(1 - x + self.eps)
            )
            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum
