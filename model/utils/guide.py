import torch.nn as nn
import torch
import pyro.distributions as distribution
import torch.nn.functional as F
import copy


class MeanFieldNormal(nn.Module):
    def __init__(self, shape, loc=None, scale=None, event=1):
        super(MeanFieldNormal, self).__init__()
        self.event = event
        if loc is None:
            self.loc = nn.Parameter(torch.normal(mean=torch.zeros(shape), std=torch.ones(shape) * 0.1), requires_grad=True)
        else:
            self.loc = nn.Parameter(loc, requires_grad=True)

        if scale is None:
            self.scale = nn.Parameter(torch.normal(mean=torch.ones(shape) * -3, std=0.1*torch.ones(shape)), requires_grad=True)
        else:
            self.scale = nn.Parameter(scale, requires_grad=True)

    def rsample(self):
        sigma = F.softplus(self.scale)
        dist = distribution.Normal(loc=self.loc, scale=sigma).to_event(self.event)
        return dist.rsample()

    def log_prob(self, value):
        sigma = F.softplus(self.scale)
        dist = distribution.Normal(loc=self.loc, scale=sigma).to_event(self.event)
        return dist.log_prob(value)

    def forward(self, x):
        raise NotImplementedError

    def distribution(self):
        sigma = F.softplus(self.scale)
        dist = torch.distributions.Normal(loc=self.loc, scale=sigma)
        return dist

