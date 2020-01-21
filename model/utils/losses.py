import torch.nn as nn


class BayesLoss(nn.Module):
    def __init__(self, loss=nn.BCEWithLogitsLoss()):
        super(BayesLoss, self).__init__()
        self.loss = loss

    def forward(self, logits, label, kl, beta):
        nll = self.loss(logits, label)
        loss = nll + beta * kl
        return loss