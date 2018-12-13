import torch
import torch.nn as nn


class Voting(nn.Module):
    """
    Voting Layer computes a new row-stotatic matrix with softmax. A large number (alpha) is multiplied to the input
    stochastic matrix to scale up the difference.
    Parameter: value multiplied before softmax alpha
               threshold that will ignore such points while calculating displacement in pixels pixel_thresh
    Input: permutation or doubly stochastic matrix s
           ///point set on source image P_src
           ///point set on target image P_tgt
           ground truth number of effective points in source image ns_gt
    Output: softmax matrix s
    """
    def __init__(self, alpha=200, pixel_thresh=None):
        super(Voting, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=2)  # Voting among columns
        self.pixel_thresh = pixel_thresh

    def forward(self, s, ns_gt):
        # TODO discard dummy nodes & far away nodes
        discard_mask = torch.zeros_like(s)
        # filter dummy nodes
        for b, n in enumerate(ns_gt):
            discard_mask[b, 0:n, :] = 1

        s = s * discard_mask
        s = self.softmax(self.alpha * s)

        return s
