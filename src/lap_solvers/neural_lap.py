import torch
import torch.nn as nn
import torch.nn.functional as nnF
import math


class NeuralLAP(nn.Module):
    """
    Neural network solver for LAP.
    Parameter:
    Input: input matrix s
    Output: bi-stochastic matrix s / perm-matrix p
    """
    def __init__(self, net_dim=16, sim_dim=8, num_layers=3):
        super(NeuralLAP, self).__init__()
        self.lap_dim = net_dim
        self.sim_dim = sim_dim
        self.lap_cross = nn.Linear(self.lap_dim, self.lap_dim)
        self.lap_self = nn.Linear(self.lap_dim, self.lap_dim)

        self.num_layers = num_layers
        for i in range(self.num_layers):
            lap_msg = nn.Linear(self.lap_dim if i > 0 else 1, self.lap_dim, bias=False)
            lap_self = nn.Linear(self.lap_dim if i > 0 else 1, self.lap_dim, bias=False)
            self.add_module('lap_msg_{}'.format(i), lap_msg)
            self.add_module('lap_self_{}'.format(i), lap_self)

        self.similarity_measure = nn.Parameter(torch.Tensor(self.sim_dim, self.lap_dim, self.lap_dim))
        #self.lap_classifier = nn.Sequential(nn.Linear(self.sim_dim, 1), nn.Sigmoid())
        self.lap_classifier = nn.Sequential(nn.Linear(self.lap_dim, 1), nn.Sigmoid())

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.lap_dim)
        self.similarity_measure.data.uniform_(-stdv, stdv)

    def forward(self, s, nrows=None, ncols=None, dtype=torch.float32):
        batch_size = s.shape[0]
        device = s.device

        assert s.shape[2] >= s.shape[1]

        emb = s.unsqueeze(-1)

        for i in range(self.num_layers):
            lap_msg = getattr(self, 'lap_msg_{}'.format(i))
            lap_self = getattr(self, 'lap_self_{}'.format(i))
            msg = lap_msg(emb)
            emb = (msg.sum(dim=1, keepdim=True) + msg.sum(dim=2, keepdim=True) - 2 * msg) / (s.shape[1] + s.shape[2] - 2) #+ lap_self(emb)

        s = self.lap_classifier(emb).squeeze(-1)

        '''
        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        emb1 = torch.ones((batch_size, s.shape[1], self.lap_dim), device=device)
        emb2 = torch.ones((batch_size, s.shape[2], self.lap_dim), device=device)
        new_emb1 = torch.zeros_like(emb1)
        new_emb2 = torch.zeros_like(emb2)
        for b in range(batch_size):
            sb = s[b, :nrows[b], :ncols[b]]
            new_emb1[b, :nrows[b], :] = nnF.relu(torch.mm(nnF.softmax(sb, dim=0), self.lap_cross(emb2[b, :ncols[b], :]))) + self.lap_self(emb1[b, :nrows[b], :])
            new_emb2[b, :ncols[b], :] = nnF.relu(torch.mm(nnF.softmax(sb.t(), dim=0), self.lap_cross(emb1[b, :nrows[b], :]))) + self.lap_self(emb2[b, :ncols[b], :])
        #lap_emb = new_emb1.unsqueeze(2) - new_emb2.unsqueeze(1)
        lap_emb = torch.matmul(new_emb1.unsqueeze(1), self.similarity_measure.unsqueeze(0))
        lap_emb = torch.matmul(lap_emb, new_emb2.transpose(1, 2).unsqueeze(1))
        lap_emb = lap_emb.permute(0, 2, 3, 1)
        s = self.lap_classifier(lap_emb).squeeze(-1)
        '''

        return s#, new_p


if __name__ == '__main__':
    import torch.optim as optim
    from src.lap_solvers.hungarian import hungarian
    from src.lap_solvers.greedy_lap import greedy_lap
    from src.loss_func import CrossEntropyLoss

    steps = 1000
    prob_size = 10
    batch_size = 50
    layers = 5
    device = torch.device("cuda:0")

    neural_lap = NeuralLAP(16, num_layers=layers).to(device)
    criterion =  CrossEntropyLoss()
    optimizer = optim.SGD(neural_lap.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    #optimizer = optim.SGD(neural_lap.parameters(), lr=1e-2, momentum=0., nesterov=False)


    for i in range(steps):
        optimizer.zero_grad()

        problem = torch.randint(0, 10, size=(batch_size, prob_size, prob_size), dtype=torch.float, device=device) / 10
        gt = hungarian(problem)
        pred = neural_lap(problem)

        pred_bi = greedy_lap(pred)

        loss = criterion(pred, gt, torch.Tensor([prob_size]*batch_size).to(dtype=torch.int))
        loss.backward()

        optimizer.step()

        opt_sln = torch.sum(gt * problem)
        pred_sln = torch.sum(hungarian(pred) * problem)
        acc = torch.sum(gt * hungarian(pred)) / torch.sum(gt * gt)

        pred_sln_greedy = torch.sum(pred_bi * problem)
        acc_greedy = torch.sum(gt * pred_bi) / torch.sum(gt * gt)

        print('iter={}, loss={:.4f}, HUNG acc={:.4f}, gap={:.8f}, GREE acc={:.4f}, gap={:.8f}'.format(
            i, loss,
            acc,  (opt_sln - pred_sln) / batch_size,
            acc_greedy, (opt_sln - pred_sln_greedy) / batch_size
        ))


