import torch.distributions as dist

def KL(prior, posterior):
    kl = dist.kl_divergence(posterior, prior).sum()
    return kl