import torch.nn as nn
import torch.nn.functional as F
import torch


def simclr_loss(z1, z2):
    """

    :param z1: B * N * D
    :param z2: B * N * D
    B is the batch size, N is the number of nodes, D is the number of channel
    :return:
    """
    B, N, D = z1.size()
    z1 = z1.view(B * N, D)
    z2 = z2.view(B * N, D)
    temperature = 0.5
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    batch_size = z1.size(0)
    out = torch.cat([z1, z2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

if __name__ == '__main__':
    z1 = torch.randn([5, 109, 256])
    z2 = torch.randn([5, 109, 256])
    print(simclr_loss(z1, z2))