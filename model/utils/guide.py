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
        dist = torch.distributions.Normal(loc=self.loc, scale=self.scale)
        return dist


class PlainNormalFlow(nn.Module):
    def __init__(self, shape, flow=1, event=1):
        super(PlainNormalFlow, self).__init__()
        self.event = event
        self.loc = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(shape), requires_grad=False)
        plf_dist = distribution.PlanarFlow(shape[-1])
        self.layers = nn.ModuleList([copy.deepcopy(plf_dist) for _ in range(flow)])

    def rsample(self):
        base_dist = distribution.Normal(loc=self.loc, scale=self.scale)
        plf_dist = distribution.TransformedDistribution(base_dist, [each for each in self.layers]).to_event(self.event)
        return plf_dist.rsample()

    def log_prob(self, value):
        base_dist = distribution.Normal(loc=self.loc, scale=self.scale)
        plf_dist = distribution.TransformedDistribution(base_dist, [each for each in self.layers])
        return plf_dist.log_prob(value)

    def forward(self, x):
        raise NotImplementedError


class PlainNormalFlowTrainable(nn.Module):
    def __init__(self, shape, loc=None, scale=None, flow=1, event=1):
        super(PlainNormalFlowTrainable, self).__init__()
        self.event = event

        if loc is None:
            self.loc = nn.Parameter(torch.normal(mean=torch.zeros(shape), std=torch.ones(shape) * 0.1),
                                    requires_grad=True)
        else:
            self.loc = nn.Parameter(loc, requires_grad=True)

        if scale is None:
            self.scale = nn.Parameter(torch.normal(mean=torch.ones(shape) * -3, std=0.1 * torch.ones(shape)),
                                      requires_grad=True)
        else:
            self.scale = nn.Parameter(scale, requires_grad=True)

        plf_dist = distribution.PlanarFlow(shape[-1])
        self.layers = nn.ModuleList([copy.deepcopy(plf_dist) for _ in range(flow)])

    def rsample(self):
        sigma = F.softplus(self.scale)
        base_dist = distribution.Normal(loc=self.loc, scale=sigma)
        plf_dist = distribution.TransformedDistribution(base_dist, [each for each in self.layers]).to_event(self.event)
        return plf_dist.rsample()

    def log_prob(self, value):
        sigma = F.softplus(self.scale)
        base_dist = distribution.Normal(loc=self.loc, scale=sigma)
        plf_dist = distribution.TransformedDistribution(base_dist, [each for each in self.layers])
        return plf_dist.log_prob(value)

    def forward(self, x):
        raise NotImplementedError

