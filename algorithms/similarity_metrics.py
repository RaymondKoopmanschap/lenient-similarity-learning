import random
import numpy as np
from scipy.stats import norm
from scipy.spatial import distance


def prop_metric(dist, reward):
    return dist.count(reward) / len(dist)


def dif_hist(cur_samples, next_samples, bins):
    cur_hist, _ = np.histogram(cur_samples, bins)
    next_hist, _ = np.histogram(next_samples, bins)
    cur_samples = cur_hist / len(cur_samples)
    next_samples = next_hist / len(next_samples)
    total_diff = sum(abs(cur_samples - next_samples))
    return 1 - total_diff / 2  # 1 - diff because it is a similarity metric


def emd(cur_samples, next_samples, bins=None):
    # emd = sum([abs(cur_samples[x] - next_samples[x]) for x in range(length)]) / length
    if bins is not None:
        cur_hist, _ = np.histogram(cur_samples, bins)
        next_hist, _ = np.histogram(next_samples, bins)
        cur_samples = cur_hist / len(cur_samples)
        next_samples = next_hist / len(next_samples)
    else:
        cur_samples = np.sort(cur_samples)
        next_samples = np.sort(next_samples)
    emd_metric = np.sum(abs(cur_samples - next_samples))
    if emd_metric > 1:
        emd_metric = 1
    emd_metric = 1 - emd_metric
    return emd_metric


def ks(cur_samples, next_samples, bins=None):
    if bins is not None:
        cur_samples = np.digitize(cur_samples, bins)
        next_samples = np.digitize(next_samples, bins)
    cur_samples = np.sort(cur_samples)
    next_samples = np.sort(next_samples)
    n1 = cur_samples.shape[0]
    n2 = next_samples.shape[0]
    tot_samples = np.concatenate([cur_samples, next_samples])
    cdf1 = np.searchsorted(cur_samples, tot_samples, side='right') / n1
    cdf2 = np.searchsorted(next_samples, tot_samples, side='right') / n2
    ks_metric = max(abs(cdf1 - cdf2))
    return 1 - ks_metric


def find_quantile(base_samples, target_samples, tau):
    """input: target samples, distribution samples and tau are the uniform samples used for the distribution samples
       output: the quantiles (probability) of the respective target samples: F(a) and F(b)"""
    # All the target sample indices that are out of bounds are either 0 or len(dist) with np.searchsorted.
    index = np.searchsorted(base_samples, target_samples)  #
    index[index == len(base_samples)] = 0  # To make sure it doesn't go out of bounds in calculation of frac_line and
    # those values will be set to 1 anyway later on. The same is true if we get index 0.
    frac_line = (target_samples - base_samples[index - 1]) / (base_samples[index] - base_samples[index - 1] + 0.00001)
    q_dist_sample = tau[index-1]  # the begin quantile of each target sample
    quantiles = q_dist_sample + frac_line * (tau[index] - tau[index - 1])

    # Correct all the calculated quantiles that are out of bounds compared to the distribution samples.
    prob_0_indices = np.where(target_samples < base_samples[0])
    quantiles[prob_0_indices] = 0
    prob_1_indices = np.where(target_samples >= base_samples[-1])
    quantiles[prob_1_indices] = 1
    return quantiles


def tdl_for_plot(base_samples, target_samples, tau, type='sum'):
    """input: distribution and target samples coming from the return distribution. And tau are the uniform samples
              used for the distribution samples.
       output: tdl
       there are three types possible: sum, product (calculates likelihood) or log (calculates log-likelihood)"""
    base_samples = np.sort(np.array(base_samples))
    target_samples = np.sort(target_samples)
    if type == 'sum' or type == 'log':
        tdl = 0
    elif type == 'product':
        tdl = 1
    else:
        print("invalid type")
        tdl = None
    means = []
    for i in range(1, len(target_samples) - 1):
        mean_1 = (target_samples[i - 1] + target_samples[i]) / 2
        mean_2 = (target_samples[i] + target_samples[i + 1]) / 2
        means.append(mean_1)
        F_a, F_b = find_quantile(base_samples, [mean_1, mean_2], tau)
        if type == 'sum':
            tdl = tdl + (F_b - F_a)  # F(b) - F(a) is the solution of equation 11. Equation 12 is then summing those.
        elif type == 'product':
            tdl = tdl * (F_b - F_a)
        elif type == 'log':
            tdl = tdl + np.log(F_b - F_a)
        if i == len(target_samples) - 2:
            means.append(mean_2)
    if np.array_equal(target_samples, base_samples):
        return 1, means
    return tdl, means


def tdl(base_samples, target_samples, tau):
    if len(target_samples) < 2:
        return 0
    base_samples = np.sort(np.array(base_samples))
    target_samples = np.sort(target_samples)
    first_mean = (target_samples[0] + target_samples[1]) / 2
    last_mean = (target_samples[-1] + target_samples[-2]) / 2
    first_quantile, last_quantile = find_quantile(base_samples, [first_mean, last_mean], tau)
    tdl = last_quantile - first_quantile
    return tdl


def tdl_rq1(cur_samples, next_samples, dist_type, dist_params):
    """TDL specially modified for experiments in RQ1
       Always specify dist, if dist type is not None
       Possible dist types: normal"""
    cur_samples = np.sort(cur_samples)
    next_samples = np.sort(next_samples)
    first_mean = (next_samples[0] + next_samples[1]) / 2
    last_mean = (next_samples[-1] + next_samples[-2]) / 2
    if dist_type == 'normal' or dist_type == 'normal2':  # dist_params [mean_1, std_1]
        tau = norm.cdf(cur_samples, loc=dist_params[0], scale=dist_params[1])
    elif dist_type == 'uniform' or dist_type == 'uniform_wide':  # dist_params [start_1, end_1]
        tau = (cur_samples - dist_params[0]) / (dist_params[1] - dist_params[0])
    elif dist_type == 'multi-modal':  # only works for uniform distributions with shift parameter,
        # dist_params [start_dist1_first, start_dist1_second]
        tau = []
        for sample in cur_samples:
            if sample > dist_params[1]:  # dist_param is the begin position of the second uniform distribution
                tau.append((sample - dist_params[1]) * 0.5 + 0.5)  # * 0.5 because both distributions are length 1
            else:
                tau.append((sample - dist_params[0]) * 0.5)
        tau = np.asarray(tau)
    elif dist_type == 'categorical':  # dist_params [list of categories 1, probs of categories 1]
        tau = []
        for sample in cur_samples:
            id = dist_params[0].index(sample)
            begin_prob = sum(i for i in dist_params[1][:id])
            end_prob = begin_prob + dist_params[1][id]
            tau.append(random.uniform(begin_prob, end_prob))
        tau = np.sort(np.asarray(tau))
    else:
        tau = None
        print('That is not a correct distribution, please choose from: normal')
    first_quantile, last_quantile = find_quantile(cur_samples, [first_mean, last_mean], tau)
    tdl = last_quantile - first_quantile
    return tdl


def hellinger(cur_samples, next_samples, bins):
    cur_hist, _ = np.histogram(cur_samples, bins)
    next_hist, _ = np.histogram(next_samples, bins)
    cur_prob_dist = cur_hist / len(cur_samples)
    next_prob_dist = next_hist / len(next_samples)
    h = np.sqrt(np.sum((np.sqrt(cur_prob_dist) - np.sqrt(next_prob_dist)) ** 2)) / np.sqrt(2)
    return 1 - h  # to turn it into a similarity metric


def jsd(cur_samples, next_samples, bins):
    cur_hist, _ = np.histogram(cur_samples, bins)
    next_hist, _ = np.histogram(next_samples, bins)
    cur_prob_dist = cur_hist / len(cur_samples)
    next_prob_dist = next_hist / len(next_samples)
    j = distance.jensenshannon(cur_prob_dist, next_prob_dist, base=2)
    return 1 - j  # to turn into similarity metric


def ovl(cur_samples, next_samples, bins):
    cur_hist, _ = np.histogram(cur_samples, bins)
    next_hist, _ = np.histogram(next_samples, bins)
    cur_prob_dist = cur_hist / len(cur_samples)
    next_prob_dist = next_hist / len(next_samples)
    return np.sum(np.minimum(cur_prob_dist, next_prob_dist))
