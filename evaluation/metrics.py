import torch
import numpy as np
from scipy import stats as st
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def uncertain_cal(prob, dim=1):
    aleatoric = torch.mean(prob*(1-prob), dim=dim)
    epistemic = torch.mean(prob**2, dim=dim) - torch.mean(prob, dim=dim)**2
    predictive = aleatoric + epistemic
    variance = torch.var(prob, dim=dim)
    return aleatoric, epistemic, predictive, variance


def calibration(y, p_mean, num_bins=10):
  """Compute the calibration.
  References:
  https://arxiv.org/abs/1706.04599
  https://arxiv.org/abs/1807.00263
  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_mean: numpy array, size (?, num_classes)
           containing the mean output predicted probabilities
    num_bins: number of bins
  Returns:
    cal: a dictionary
      {reliability_diag: realibility diagram
       ece: Expected Calibration Error
       mce: Maximum Calibration Error
      }
  """
  # Compute for every test sample x, the predicted class.
  class_pred = np.argmax(p_mean, axis=1)
  # and the confidence (probability) associated with it.
  conf = np.max(p_mean, axis=1)
  # Convert y from one-hot encoding to the number of the class
  y = np.argmax(y, axis=1)
  # Storage
  acc_tab = np.zeros(num_bins)  # empirical (true) confidence
  mean_conf = np.zeros(num_bins)  # predicted confidence
  nb_items_bin = np.zeros(num_bins)  # number of items in the bins
  tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
  for i in np.arange(num_bins):  # iterate over the bins
    # select the items where the predicted max probability falls in the bin
    # [tau_tab[i], tau_tab[i + 1)]
    sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
    nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
    # select the predicted classes, and the true classes
    class_pred_sec, y_sec = class_pred[sec], y[sec]
    # average of the predicted max probabilities
    mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
    # compute the empirical confidence
    acc_tab[i] = np.mean(
        class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

  # Cleaning
  mean_conf = mean_conf[nb_items_bin > 0]
  acc_tab = acc_tab[nb_items_bin > 0]
  nb_items_bin = nb_items_bin[nb_items_bin > 0]

  # Reliability diagram
  reliability_diag = (mean_conf, acc_tab)
  # Expected Calibration Error
  ece = np.average(
      np.absolute(mean_conf - acc_tab),
      weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))
  # Maximum Calibration Error
  mce = np.max(np.absolute(mean_conf - acc_tab))
  # Saving
  cal = {'reliability_diag': reliability_diag,
         'ece': ece,
         'mce': mce}
  return cal


def distributions_js(mu_p, std_p, mu_q, std_q, n_samples=10 ** 5):
    # jensen shannon divergence. (Jensen shannon distance is the square root of the divergence)
    # all the logarithms are defined as log2 (because of information entrophy)
    distribution_p = st.norm(loc=mu_p, scale=std_p)
    distribution_q = st.norm(loc=mu_q, scale=std_q)

    X = distribution_p.rvs(n_samples)
    p_X = distribution_p.pdf(X)
    q_X = distribution_q.pdf(X)
    log_mix_X = np.log2(p_X + q_X)

    Y = distribution_q.rvs(n_samples)
    p_Y = distribution_p.pdf(Y)
    q_Y = distribution_q.pdf(Y)
    log_mix_Y = np.log2(p_Y + q_Y)

    return (np.log2(p_X).mean() - (log_mix_X.mean() - np.log2(2))
            + np.log2(q_Y).mean() - (log_mix_Y.mean() - np.log2(2))) / 2


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, labels):
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            print('lower {} higher {}'.format(bin_lower, bin_upper))
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            print('prob_in_bin {}'.format(prop_in_bin))
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                print('accuracy in bin {}'.format(accuracy_in_bin))
                avg_confidence_in_bin = confidences[in_bin].mean()
                print('average confidence in bin {}'.format(avg_confidence_in_bin))
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


def reliability_curve(y_true, y_score, bins=10, normalize=False):
    if normalize:  # Normalize scores into bin [0, 1]
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    center = np.empty(bins)
    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    nb_items_bin = np.zeros(bins)

    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                 y_score <= threshold + bin_width / 2)
        nb_items_bin[i] = np.sum(bin_idx)
        # Store mean y_score and mean empirical probability of positive class
        y_score_bin_mean[i] = y_score[bin_idx].mean() if nb_items_bin[i] > 0 else np.nan
        empirical_prob_pos[i] = y_true[bin_idx].mean() if nb_items_bin[i] > 0 else np.nan
        center[i] = threshold if nb_items_bin[i] > 0 else np.nan

    return y_score_bin_mean, empirical_prob_pos, bin_centers


def expected_calibration_error(y_true, y_score, bins=10, normalize=False):
    if normalize:  # Normalize scores into bin [0, 1]
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    center = np.empty(bins)
    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    nb_items_bin = np.zeros(bins)

    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                 y_score <= threshold + bin_width / 2)
        nb_items_bin[i] = np.sum(bin_idx)
        # Store mean y_score and mean empirical probability of positive class
        y_score_bin_mean[i] = y_score[bin_idx].mean() if nb_items_bin[i] > 0 else np.nan
        empirical_prob_pos[i] = y_true[bin_idx].mean() if nb_items_bin[i] > 0 else np.nan
        center[i] = threshold if nb_items_bin[i] > 0 else np.nan

    mean_conf = y_score_bin_mean[nb_items_bin > 0]
    acc_tab = empirical_prob_pos[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]

    ece = np.average(
        np.absolute(mean_conf - acc_tab),
        weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))

    return ece


def accuracy_confidence(prob, label, num_bins=10):
    pos = prob
    neg = 1-prob
    prob = np.stack([neg, pos], axis=1)

    class_pred = np.argmax(prob, axis=1)
    conf = np.max(prob, axis=1)

    acc_tab = np.zeros(num_bins)
    tau_tab = np.linspace(0, 1, num_bins+1)
    nb_items_bin = np.zeros(num_bins)

    for i in np.arange(num_bins):
        sec = (tau_tab[-1] > conf) & (conf >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)
        class_pred_sec, y_sec = class_pred[sec], label[sec]
        acc_tab[i] = np.mean(class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

    return tau_tab[:-1], acc_tab


def roc_confidence(prob, label, num_bins=10):
    roc_tab = np.zeros(num_bins)
    tau_tab = np.linspace(0, 1, num_bins + 1)
    nb_items_bin = np.zeros(num_bins)
    bin_center = np.zeros(num_bins)

    for i in np.arange(num_bins):
        sec = (tau_tab[i+1] > prob) & (prob >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)
        score_sec, y_sec = prob[sec], label[sec]

        if len(set(label[sec]))>1:
            roc_tab[i] = roc_auc_score(y_true=y_sec, y_score=score_sec) if nb_items_bin[i] > 0 else np.nan
            bin_center[i] = (tau_tab[i + 1] + tau_tab[i]) / 2 if nb_items_bin[i] > 0 else np.nan

    return bin_center, tau_tab, roc_tab
