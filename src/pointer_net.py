import torch
import torch.nn as nn
import math

class PointerNet(nn.Module):
    """
    Pointer Network, given two sequences, compute a matching matrix between elements in two sequences via attention.
    """
    def __init__(self, inp_size, hid_size, alpha=1.):
        super(PointerNet, self).__init__()
        self.encoder = nn.LSTM(inp_size, hid_size, 1, batch_first=True)
        self.decoder = nn.LSTM(inp_size, hid_size, 1, batch_first=True)
        self.softmax = nn.Softmax(dim=-1)

        hid_scale = 1 / math.sqrt(hid_size)
        self.att_v = nn.Parameter(torch.empty(hid_size).uniform_(-hid_scale, hid_scale))
        self.att_W1 = nn.Parameter(torch.empty(hid_size, hid_size).uniform_(-hid_scale, hid_scale))
        self.att_W2 = nn.Parameter(torch.empty(hid_size, hid_size).uniform_(-hid_scale, hid_scale))

        self.alpha = alpha

        #self.init_hidden = torch.empty(hid_size).uniform_(-hid_scale, hid_scale).unsqueeze(0)
        #self.init_cell = torch.zeros(hid_size).unsqueeze(0)

    def forward(self, seq_src, seq_tgt, ns_src, ns_tgt):
        max_node_len = seq_src.shape[1]
        batch_size = seq_src.shape[0]

        seq_src_pack = nn.utils.rnn.pack_padded_sequence(seq_src, ns_src, batch_first=True, enforce_sorted=False)
        seq_tgt_pack = nn.utils.rnn.pack_padded_sequence(seq_tgt, ns_tgt, batch_first=True, enforce_sorted=False)
        src_outp, encoder_state = self.encoder(seq_src_pack) #, (self.init_hidden, self.init_cell))
        tgt_outp, _ = self.decoder(seq_tgt_pack, encoder_state)

        src_outp, _ = nn.utils.rnn.pad_packed_sequence(src_outp, batch_first=True, total_length=max_node_len)
        tgt_outp, _ = nn.utils.rnn.pad_packed_sequence(tgt_outp, batch_first=True, total_length=max_node_len)

        # greedy LAP solver
        col_slice = []
        for b in range(batch_size):
            col_slice.append(list(range(0, ns_tgt[b] if ns_tgt is not None else max_node_len)))

        new_bs = torch.zeros(batch_size, max_node_len, max_node_len, device=seq_src.device)
        for r in range(max_node_len):
            for b in range(batch_size):
                if ns_src is None or r < ns_src[b]:
                    u = torch.matmul(
                        self.att_v.unsqueeze(0).unsqueeze(0),
                        torch.matmul(self.att_W1.unsqueeze(0), src_outp[b, col_slice[b]].unsqueeze(-1)) +
                        torch.matmul(self.att_W2.unsqueeze(0), tgt_outp[b, r:r+1].unsqueeze(-1))
                    ).squeeze()

                    new_bs[b, r, col_slice[b]] = self.softmax(u * self.alpha)
                    max_idx = col_slice[b][torch.argmax(new_bs[b, r, col_slice[b]])]
                    col_slice[b].remove(max_idx)

        return new_bs


