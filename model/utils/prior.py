import torch.nn as nn
import torch
from pyro import distributions


class Normal(nn.Module):
    def __init__(self, shape, loc=None, scale=None, event=1):
        super(Normal, self).__init__()
        self.event = event
        if loc is None:
            self.loc = nn.Parameter(torch.zeros(shape), requires_grad=False)
        else:
            if loc.shape == shape:
                self.loc = nn.Parameter(loc, requires_grad=False)
            else:
                raise ValueError('loc shape %s do not match shape %s' % (str(loc.shape), str(shape)))

        if scale is None:
            self.scale = nn.Parameter(torch.ones(shape), requires_grad=False)
        else:
            if scale.shape == shape:
                self.scale = nn.Parameter(scale, requires_grad=False)
            else:
                raise ValueError('scale shape %s do not match shape %s' % (str(scale.shape), str(shape)))

    def log_prob(self, value):
        dist = distributions.Normal(loc=self.loc, scale=self.scale).to_event(self.event)
        return dist.log_prob(value)

    def rsample(self):
        dist = distributions.Normal(loc=self.loc, scale=self.scale).to_event(self.event)
        return dist.rsample()

    def sample(self):
        dist = distributions.Normal(loc=self.loc, scale=self.scale).to_event(self.event)
        return dist.sample()

    def forward(self, x):
        raise NotImplementedError

    def distribution(self):
        dist = torch.distributions.Normal(loc=self.loc, scale=self.scale)
        return dist

