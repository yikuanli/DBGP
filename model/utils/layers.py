import torch.nn as nn
from model.utils import guide, prior as model
import torch.nn.functional as F
from model.utils.divergence import KL
from torch._jit_internal import weak_script_method
import torch


class Linear(nn.Module):
    def __init__(self, in_feature, out_feature, prior=None, posterior=None, divergence=None):
        super(Linear, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        if prior is None:
            self.weight_prior = model.Normal(shape=(out_feature, in_feature))
        else:
            self.weight_prior = prior

        if posterior is None:
            self.weight_posterior = guide.MeanFieldNormal(shape=(out_feature, in_feature))
        else:
            self.weight_posterior = posterior

        if divergence is None:
            self.divergence = KL
        else:
            self.divergence = divergence

    def forward(self, inputs):
        weight_posterior = self.weight_posterior.rsample()
        output = F.linear(input=inputs, weight=weight_posterior)
        elbo = self.divergence(self.weight_prior.distribution(), self.weight_posterior.distribution())
        return output, elbo



class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None, prior=None, posterior=None, divergence=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx


        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        if prior is None:
            self.weight_prior = model.Normal(shape=(num_embeddings, embedding_dim))
        else:
            self.weight_prior = prior

        if posterior is None:
            self.weight_posterior = guide.MeanFieldNormal(shape=(num_embeddings, embedding_dim))
        else:
            self.weight_posterior = posterior

        if divergence is None:
            self.divergence = KL
        else:
            self.divergence = divergence

        self.sparse = sparse

    # def reset_parameters(self):
    #     nn.init.normal_(self.weight)
    #     if self.padding_idx is not None:
    #         with torch.no_grad():
    #             self.weight[self.padding_idx].fill_(0)

    @weak_script_method
    def forward(self, input):
        weight_posterior = self.weight_posterior.rsample()

        embedding = F.embedding(input, weight_posterior, self.padding_idx, self.max_norm,self.norm_type,
                    self.scale_grad_by_freq, self.sparse)

        elbo = self.divergence(self.weight_prior.distribution(), self.weight_posterior.distribution())
        return embedding, elbo


class TokenLinear(nn.Module):
    def __init__(self, divergence=None, std_prior=1):
        super(TokenLinear, self).__init__()
        if divergence is None:
            self.divergence = KL
        else:
            self.divergence = divergence

        self.std_prior = std_prior

    def forward(self, inputs_mu, inputs_sig):
        prior = torch.distributions.Normal(loc=torch.zeros_like(inputs_mu), scale=torch.ones_like(inputs_mu)*self.std_prior)

        sigma = F.softplus(inputs_sig)

        posterior = torch.distributions.Normal(loc=inputs_mu, scale=sigma)

        # print('inputs_mu shape : {}'.format(inputs_mu.shape))
        # print('inputs_sig shape : {}'.format(inputs_sig.shape))

        output = posterior.rsample()

        # print('prior:', prior)
        # print('posterior:', posterior)

        elbo = torch.distributions.kl_divergence(posterior, prior).sum()

        # elbo = self.divergence(prior, posterior)

        return output, elbo