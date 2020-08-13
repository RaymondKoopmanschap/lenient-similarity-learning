import matplotlib.pyplot as plt
from scipy.stats import uniform
from tqdm import tqdm
import time
from algorithms.similarity_metrics import *


def plotting(axs, data, dist1, dist2, dist_args, sample_sizes, num_samples, dist_type, metric, alpha, count1, count2):
    if len(dist_args) == 1:
        if len(sample_sizes) == 1:
            axs.hist(data, bins=10, range=(0, 1), alpha=alpha, label=metric)
            axs.set_title(f"{metric}, N = {num_samples}\n"
                          f"{dist_type} dist: [{dist1[0]}, {dist1[1]}] and [{dist2[0]}, {dist2[1]}]")
            axs.legend()
        else:
            axs[count2].hist(data, bins=10, range=(0, 1), alpha=alpha, label=metric)
            axs[count2].set_title(f"{metric}, N = {num_samples}\n"
                                  f"{dist_type} dist: [{dist1[0]}, {dist1[1]}] and [{dist2[0]}, {dist2[1]}]")
            axs[count2].legend()
    else:
        if len(sample_sizes) == 1:
            axs[count1].hist(data, bins=10, range=(0, 1), alpha=alpha, label=metric)
            axs[count1].set_title(f"{metric}, N = {num_samples}\n"
                                  f"{dist_type} dist: [{dist1[0]}, {dist1[1]}] and [{dist2[0]}, {dist2[1]}]")
            axs[count1].legend()
        else:
            axs[count2, count1].hist(data, bins=10, range=(0, 1), alpha=alpha, label=metric)
            axs[count2, count1].set_title(f"{metric}, N = {num_samples}\n"
                                          f"{dist_type} dist: [{dist1[0]}, {dist1[1]}] and [{dist2[0]}, {dist2[1]}]")
            axs[count2, count1].legend()


def calculate_boundaries(dist_args1, dist_args2, dist_type, shift):

    if dist_type == 'uniform' or dist_type == 'uniform_wide':
        low_1, high_1 = dist_args1[0], dist_args1[1]
        low_2, high_2 = dist_args2[0], dist_args2[1]
        min_r = min(low_1, low_2)
        max_r = max(high_1, high_2)
    elif dist_type == 'normal' or dist_type == 'normal2':
        mean_1, std_1 = dist_args1[0], dist_args1[1]
        mean_2, std_2 = dist_args2[0], dist_args2[1]
        min_r = min(mean_1 - 3 * std_1, mean_2 - 3 * std_2)
        max_r = max(mean_1 + 3 * std_1, mean_2 + 3 * std_2)
        # print(min_r, max_r)
    elif dist_type == 'multi-modal':
        min_r, max_r = dist_args2[0] - shift, dist_args2[0] + 0.5 + 0.5 + shift
        # print(min_r, max_r)
    elif dist_type == 'categorical':
        min_r = min(dist_args1[0] + dist_args2[0])
        max_r = max(dist_args1[0] + dist_args2[0])
        # print(min_r, max_r)
    else:
        print('invalid dist type choose: uniform, normal, multi-modal, categorical')
        min_r = None
        max_r = None
    return min_r, max_r


def change_dist_args(dist_type):
    # All are in 100, 80, 60, 40, 20, 0 format
    if dist_type == 'uniform':
        dists = [[0, 1, 0, 1], [0, 1, 0.2, 1.2], [0, 1, 0.4, 1.4], [0, 1, 0.6, 1.6], [0, 1, 0.8, 1.8], [0, 1, 1, 2]]
    elif dist_type == 'uniform_wide':
        # dists = [[-0.5, 0.5, -0.5, 0.5], [-0.5, 0.5, -0.625, 0.625], [-0.5, 0.5, -0.833, 0.833],
        #          [-0.5, 0.5, -1.25, 1.25], [-0.5, 0.5, -2.5, 2.5], [-0.5, 0.5, -10, 10]]  # last one is 5% overlap
        dists = [[0, 1, 0, 1], [0, 1, 0.0625, 1.1875], [0, 1, -0.1667, 1.5],
                 [0, 1, -0.375, 2.125], [0, 1, -1, 4],  [0, 1, -4.25, 15.75]]
    elif dist_type == 'normal':
        dists = [[0, 1, 0, 1], [0, 1, 0.5, 1], [0, 1, 1.05, 1], [0, 1, 1.68, 1], [0, 1, 2.56, 1], [0, 1, 6, 1]]
    elif dist_type == 'normal2':
        dists = [[0, 1, 0, 1], [0, 1, 0, 1.52], [0, 1, 0, 2.41], [0, 1, 0, 4.25], [0, 1, 0, 10], [0, 1, 0, 100]]
    elif dist_type == 'multi-modal':  # currently also uniform
        dists = [[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]]
    elif dist_type == 'categorical':
        category_1, category_2 = [0, 1, 2, 3], [0, 1, 2, 3]
        probs_1, probs_2, probs_3, probs_4, probs_5, probs_6 = [0.5, 0.5, 0.0, 0.0], [0.4, 0.4, 0.1, 0.1], \
                                                               [0.3, 0.3, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3], \
                                                               [0.1, 0.1, 0.4, 0.4], [0.0, 0.0, 0.5, 0.5]
        dists = [[category_1, probs_1, category_2, probs_1], [category_1, probs_1, category_2, probs_2],
                 [category_1, probs_1, category_2, probs_3], [category_1, probs_1, category_2, probs_4],
                 [category_1, probs_1, category_2, probs_5], [category_1, probs_1, category_2, probs_6]]
    else:
        print('invalid dist type choose: uniform, normal, multi-modal, categorical')
        dists = None
    return dists


def similarity_metric_simple_test(cur_samples, next_samples, bins, min_r, max_r):
    diff_metric = dif_hist(cur_samples, next_samples, bins)
    emd_metric = emd(cur_samples, next_samples, bins)
    ks_metric = ks(cur_samples, next_samples)
    hell_metric = hellinger(cur_samples, next_samples, bins)
    js_metric = jsd(cur_samples, next_samples, bins)
    tdl_metric = tdl_rq1(cur_samples, next_samples, dist_type='uniform', dist_params=[min_r, max_r])

    print(f'diff: {diff_metric}\nemd: {emd_metric}\nks: {ks_metric}\nhellinger: {hell_metric}\njs: {js_metric}'
          f'\ntdl: {tdl_metric}')


def overlap_plot(metric, means, stds, sample_sizes, d, bin_exp):
    for count, i in enumerate(sample_sizes):
        plt.xticks(np.arange(d), ['100%', '80%', '60%', '40%', '20%', '0%'])  # Adapt according to dists used
        if bin_exp:
            plt.errorbar(np.arange(d), means[count, :], label=f'{metric}, $bins$ = {i-1}')  # , yerr=stds[count, :]
        else:
            plt.errorbar(np.arange(d), means[count, :], label=f'{metric}')  # , $n$ = {i}'


def update_means_stds(metric_list, means, stds, count1, count2, count3, bin_experiments):
    mean = sum(metric_list) / len(metric_list)
    std = np.std(metric_list)
    if not bin_experiments:
        means[count2, count1] = mean
        stds[count2, count1] = std
    else:
        means[count3, count1] = mean
        stds[count3, count1] = std
    return means, stds


def calculate_deviations(metric_means):
    overlap = [1, 0.8, 0.6, 0.4, 0.2, 0]
    deviation = metric_means[0] - overlap
    pos = [round(item, 4) for item in deviation if item >= 0]
    neg = [round(abs(item), 4) for item in deviation if item < 0]
    return sum(pos), sum(neg)


def similarity_metric_experiments(metrics, dist_type, sample_sizes, num_iterations, plot_trend, plot, show_time,
                                  num_bins_list, bin_experiments):
    # Initializations
    dif_hist_list, emd_list, ks_list, hellinger_list, js_list, tdl_list, ovl_list = [], [], [], [], [], [], []
    tot_hist_time_list, tot_emd_time_list, tot_ks_time_list, tot_h_time_list, tot_js_time_list, tot_tdl_time_list, \
    tot_ovl_time_list = [], [], [], [], [], [], []
    run_time_begin = time.time()
    shifts = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # used for multi-modal

    dist_args = change_dist_args(dist_type)
    d = len(dist_args)
    if bin_experiments:
        s = len(num_bins_list)
    else:
        s = len(sample_sizes)
    hist_means, emd_means, ks_means, h_means, js_means, tdl_means, ovl_means, hist_stds, emd_stds, ks_stds, h_stds, \
        js_stds, tdl_stds, ovl_stds = np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d)), \
                                      np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d)), \
                                      np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d)), \
                                      np.zeros((s, d)), np.zeros((s, d))

    # Initialize subplots
    if plot:
        if 'dif_hist' in metrics:
            _, axs_hist = plt.subplots(len(sample_sizes), len(dist_args))
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.22, hspace=0.35)
        if 'emd' in metrics:
            _, axs_emd = plt.subplots(len(sample_sizes), len(dist_args))
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.22, hspace=0.35)
        if 'ks' in metrics:
            _, axs_ks = plt.subplots(len(sample_sizes), len(dist_args))
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.22, hspace=0.35)
        if 'hellinger' in metrics:
            _, axs_h = plt.subplots(len(sample_sizes), len(dist_args))
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.22, hspace=0.35)
        if 'js' in metrics:
            _, axs_js = plt.subplots(len(sample_sizes), len(dist_args))
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.22, hspace=0.35)
        if 'tdl' in metrics:
            _, axs_tdl = plt.subplots(len(sample_sizes), len(dist_args))
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.22, hspace=0.35)
        if 'ovl' in metrics:
            _, axs_ovl = plt.subplots(len(sample_sizes), len(dist_args))
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.22, hspace=0.35)

    for i in range(1):
        tot_hist_time = 0
        tot_emd_time = 0
        tot_ks_time = 0
        tot_h_time = 0
        tot_js_time = 0
        tot_tdl_time = 0
        tot_ovl_time = 0

        for count1, parameters in tqdm(enumerate(dist_args)):  # the different distribution parameters
            dist1, dist2 = parameters[:2], parameters[2:]
            min_r, max_r = calculate_boundaries(dist1, dist2, dist_type, shifts[count1])
            # print(min_r, max_r)
            if dist_type == 'multi-modal':
                next_dists = [uniform(loc=0 - shifts[count1], scale=0.5),
                              uniform(loc=0.5 + shifts[count1], scale=0.5)]
            for count2, num_samples in enumerate(sample_sizes):  # the different sample sizes
                for count3, num_bins in enumerate(num_bins_list):
                    # Make histogram
                    for k in range(num_iterations):
                        # Select distribution
                        if dist_type == 'uniform' or dist_type == 'uniform_wide':
                            cur_samples = sorted(np.random.uniform(low=dist1[0], high=dist1[1], size=num_samples))
                            next_samples = sorted(np.random.uniform(low=dist2[0], high=dist2[1], size=num_samples))
                        elif dist_type == 'normal' or dist_type == 'normal2':
                            cur_samples = sorted(np.random.normal(loc=dist1[0], scale=dist1[1], size=num_samples))
                            next_samples = sorted(np.random.normal(loc=dist2[0], scale=dist2[1], size=num_samples))
                        elif dist_type == 'multi-modal':
                            next_draw = np.random.choice([0, 1], num_samples, p=[0.5, 0.5])
                            cur_samples = sorted(np.random.uniform(low=0, high=1, size=num_samples))
                            next_samples = sorted([next_dists[i].rvs() for i in next_draw])
                        elif dist_type == 'categorical':
                            cur_samples = sorted(np.random.choice(dist1[0], size=num_samples, p=dist1[1]))
                            next_samples = sorted(np.random.choice(dist2[0], size=num_samples, p=dist2[1]))
                        else:
                            print(f'This distribution is not supported: {dist_type}')
                            cur_samples = None
                            next_samples = None
                        # Select and compute metrics
                        if 'dif_hist' in metrics:
                            bins = np.linspace(min_r, max_r + 0.000001, num_bins)
                            start_hist = time.time()
                            dif = dif_hist(cur_samples, next_samples, bins)
                            tot_hist_time = tot_hist_time + (time.time() - start_hist)
                            dif_hist_list.append(dif)
                        if 'emd' in metrics:
                            bins = np.linspace(min_r, max_r + 0.000001, num_bins)
                            start_emd = time.time()
                            emd_metric = emd(cur_samples, next_samples, bins)
                            tot_emd_time = tot_emd_time + (time.time() - start_emd)
                            emd_list.append(emd_metric)
                        if 'ks' in metrics:
                            start_ks = time.time()
                            ks_metric = ks(cur_samples, next_samples)
                            tot_ks_time = tot_ks_time + (time.time() - start_ks)
                            ks_list.append(ks_metric)
                        if 'hellinger' in metrics:
                            bins = np.linspace(min_r, max_r + 0.000001, num_bins)
                            start_h = time.time()
                            hellinger_metric = hellinger(cur_samples, next_samples, bins)
                            tot_h_time = tot_h_time + (time.time() - start_h)
                            hellinger_list.append(hellinger_metric)
                        if 'js' in metrics:
                            bins = np.linspace(min_r, max_r + 0.000001, num_bins)
                            start_js = time.time()
                            js_metric = jsd(cur_samples, next_samples, bins)
                            tot_js_time = tot_js_time + (time.time() - start_js)
                            js_list.append(js_metric)
                        if 'tdl' in metrics:
                            start_tdl = time.time()
                            tdl_metric = tdl_rq1(cur_samples, next_samples, dist_type, dist1)  # dist1 correct for uniform
                            tot_tdl_time = tot_tdl_time + (time.time() - start_tdl)
                            tdl_list.append(tdl_metric)
                        if 'ovl' in metrics:
                            bins = np.linspace(min_r, max_r + 0.000001, num_bins)
                            start_ovl = time.time()
                            ovl_metric = ovl(cur_samples, next_samples, bins)  # dist1 correct for uniform
                            tot_ovl_time = tot_ovl_time + (time.time() - start_ovl)
                            ovl_list.append(ovl_metric)
                    # histogram plotting
                    if plot:
                        if 'dif_hist' in metrics:
                            plotting(axs_hist, dif_hist_list, dist1, dist2, dist_args, sample_sizes, num_samples, dist_type,
                                     'dif hist', 1, count1, count2)
                        if 'emd' in metrics:
                            plotting(axs_emd, emd_list, dist1, dist2, dist_args, sample_sizes, num_samples, dist_type,
                                     'emd', 1, count1, count2)
                        if 'ks' in metrics:
                            plotting(axs_ks, ks_list, dist1, dist2, dist_args, sample_sizes, num_samples, dist_type,
                                     'ks', 1, count1, count2)
                        if 'hellinger' in metrics:
                            plotting(axs_h, hellinger_list, dist1, dist2, dist_args, sample_sizes, num_samples, dist_type,
                                     'hellinger', 1, count1, count2)
                        if 'js' in metrics:
                            plotting(axs_js, js_list, dist1, dist2, dist_args, sample_sizes, num_samples, dist_type,
                                     'js', 1, count1, count2)
                        if 'tdl' in metrics:
                            plotting(axs_tdl, tdl_list, dist1, dist2, dist_args, sample_sizes, num_samples, dist_type,
                                     'tdl', 1, count1, count2)
                        if 'ovl' in metrics:
                            plotting(axs_ovl, ovl_list, dist1, dist2, dist_args, sample_sizes, num_samples, dist_type,
                                     'ovl', 1, count1, count2)
                    # Calculating statistics
                    if plot_trend:
                        if 'dif_hist' in metrics:
                            hist_means, hist_stds = update_means_stds(dif_hist_list, hist_means, hist_stds, count1,
                                                                      count2, count3, bin_experiments)
                        if 'emd' in metrics:
                            emd_means, emd_stds = update_means_stds(emd_list, emd_means, emd_stds, count1,
                                                                    count2, count3, bin_experiments)
                        if 'ks' in metrics:
                            ks_means, ks_stds = update_means_stds(ks_list, ks_means, ks_stds, count1,
                                                                  count2, count3, bin_experiments)
                        if 'hellinger' in metrics:
                            h_means, h_stds = update_means_stds(hellinger_list, h_means, h_stds, count1,
                                                                count2, count3, bin_experiments)
                        if 'js' in metrics:
                            js_means, js_stds = update_means_stds(js_list, js_means, js_stds, count1,
                                                                  count2, count3, bin_experiments)
                        if 'tdl' in metrics:
                            tdl_means, tdl_stds = update_means_stds(tdl_list, tdl_means, tdl_stds, count1,
                                                                    count2, count3, bin_experiments)
                        if 'ovl' in metrics:
                            ovl_means, ovl_stds = update_means_stds(ovl_list, ovl_means, ovl_stds, count1,
                                                                    count2, count3, bin_experiments)
                        # print(f' list {ovl_list}')
                    dif_hist_list, emd_list, ks_list, hellinger_list, js_list, tdl_list, ovl_list = \
                        [], [], [], [], [], [], []

        run_time = (time.time() - run_time_begin)
        tot_hist_time_list.append(tot_hist_time)
        tot_emd_time_list.append(tot_emd_time)
        tot_ks_time_list.append(tot_ks_time)
        tot_h_time_list.append(tot_h_time)
        tot_js_time_list.append(tot_js_time)
        tot_tdl_time_list.append(tot_tdl_time)
        tot_ovl_time_list.append(tot_ovl_time)

    if plot_overlap_trend:
        plt.rcParams.update({'font.size': 14})
        plt.figure()
        if bin_experiments:
            iter_over = num_bins_list
        else:
            iter_over = sample_sizes

        # calculate deviation from overlap
        emd_pos, emd_neg = calculate_deviations(emd_means)
        ks_pos, ks_neg = calculate_deviations(ks_means)
        h_pos, h_neg = calculate_deviations(h_means)
        js_pos, js_neg = calculate_deviations(js_means)
        tdl_pos, tdl_neg = calculate_deviations(tdl_means)
        ovl_pos, ovl_neg = calculate_deviations(ovl_means)
        print(f'emd pos: {emd_pos}, neg: {emd_neg}\n'
              f'ks pos: {ks_pos}, neg: {ks_neg}\n'
              f'h pos: {h_pos}, neg: {h_neg}\n'
              f'js pos: {js_pos}, neg: {js_neg}\n'
              f'tdl pos: {tdl_pos}, neg: {tdl_neg}\n'
              f'ovl pos: {ovl_pos}, neg: {ovl_neg}\n')
        print([emd_pos, ks_pos, h_pos, js_pos, tdl_pos, ovl_pos])
        print([emd_neg, ks_neg, h_neg, js_neg, tdl_neg, ovl_neg])

        if 'dif_hist' in metrics:
            overlap_plot('dif hist', hist_means, hist_stds, iter_over, d, bin_experiments)
        if 'emd' in metrics:
            overlap_plot('emd', emd_means, emd_stds, iter_over, d, bin_experiments)
        if 'ks' in metrics:
            overlap_plot('ks', ks_means, ks_stds, iter_over, d, bin_experiments)
        if 'hellinger' in metrics:
            overlap_plot('hellinger', h_means, h_stds, iter_over, d, bin_experiments)
        if 'js' in metrics:
            overlap_plot('js', js_means, js_stds, iter_over, d, bin_experiments)
        if 'tdl' in metrics:
            overlap_plot('tdl', tdl_means, tdl_stds, iter_over, d, bin_experiments)
        if 'ovl' in metrics:
            overlap_plot('ovl', ovl_means, ovl_stds, iter_over, d, bin_experiments)

        plt.plot([1, 0.8, 0.6, 0.4, 0.2, 0], label='overlap line')
        plt.xlabel('percentage overlap')
        plt.ylabel('similarity metric value')
        if bin_experiments:
            plt.title(f'Similarity for different number of bins')
        else:
            plt.title(f'Similarity for distributions with different overlap')
        plt.legend()
        plt.show()

    if show_time:
        tot_hist_time_mean = np.asarray(tot_hist_time_list).mean()
        tot_emd_time_mean = np.asarray(tot_emd_time_list).mean()
        tot_ks_time_mean = np.asarray(tot_ks_time_list).mean()
        tot_h_time_mean = np.asarray(tot_h_time_list).mean()
        tot_js_time_mean = np.asarray(tot_js_time_list).mean()
        tot_tdl_time_mean = np.asarray(tot_tdl_time_list).mean()
        tot_ovl_time_mean = np.asarray(tot_ovl_time_list).mean()

        tot_hist_time_std = np.asarray(tot_hist_time_list).std()
        tot_emd_time_std = np.asarray(tot_emd_time_list).std()
        tot_ks_time_std = np.asarray(tot_ks_time_list).std()
        tot_h_time_std = np.asarray(tot_h_time_list).std()
        tot_js_time_std = np.asarray(tot_js_time_list).std()
        tot_tdl_time_std = np.asarray(tot_tdl_time_list).std()
        tot_ovl_time_std = np.asarray(tot_ovl_time_list).std()

        print(f'tot hist time mean: {tot_hist_time_mean}, std: {tot_hist_time_std}\n'
              f'tot emd time mean: {tot_emd_time_mean}, std: {tot_emd_time_std}\n'
              f'tot ks time mean: {tot_ks_time_mean}, std: {tot_ks_time_std}\n'
              f'tot h time mean: {tot_h_time_mean}, std: {tot_h_time_std}\n'
              f'tot js time mean: {tot_js_time_mean}, std: {tot_js_time_std}\n'
              f'tot tdl time mean: {tot_tdl_time_mean}, std: {tot_tdl_time_std}\n'
              f'tot ovl time mean: {tot_ovl_time_mean}, std: {tot_ovl_time_std}\n')
    print(f'run time: {run_time}')

    if plot:
        plt.show()


########################################## Adapt parameters section ###################################################
# 'dif_hist', 'emd', 'ks', 'hellinger', 'js', 'tdl', 'ovl' multiple possible
metrics1 = ['ovl', 'emd', 'ks', 'hellinger', 'js', 'tdl']
dist_type1 = 'multi-modal'  # uniform, uniform_wide, normal, normal2, multi-modal, categorical
sample_sizes1 = [50]
num_iterations1 = 1000
plot_overlap_trend = True
num_bins = [11]  # for 10 bins use 11. Always + 1
# similarity_metric_experiments(metrics1, dist_type1, sample_sizes1, num_iterations1, plot_overlap_trend, plot=False,
#                               show_time=False, num_bins_list=num_bins, bin_experiments=False)
#######################################################################################################################
pos_1 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
neg_1 = [[2.1430000000000002, 0.5163000000000001, 0.8579000000000001, 0.9486, 0.16419999999999998, 0.5695]]
pos_2 = [[0.0, 0.0004, 0.020200000000000003, 0.0013, 0.7006, 0.002]]
neg_2 = [[1.7389999999999999, 0.39909999999999995, 0.335, 0.4825, 0.0449, 0.29269999999999996]]
pos_3 = [[0.0, 0.6687, 0.1933, 0.12229999999999999, 2.9044, 0.1044]]
neg_3 = [[2.3994, 0.2028, 0.8255, 0.928, 0.0479, 0.6358]]
pos_4 = [[0.0, 1.0158, 0.0992, 0.0972, 2.9974, 0.1779]]
neg_4 = [[2.0952, 0.1955, 0.7477, 0.8556999999999999, 0.3633, 0.5216]]

results_pos = np.concatenate((pos_1, pos_2, pos_3, pos_4))
results_neg = np.concatenate((neg_1, neg_2, neg_3, neg_4))
print(results_pos)

plt.rcParams.update({'font.size': 14})
plt.subplots_adjust(left=0.15, bottom=0.11, right=0.9, top=0.87, wspace=0.22, hspace=0.35)

plt.xticks(np.arange(6), ['shifting\nuniform', 'shifting\nnormal', 'wide\nuniform', 'multi-modal\nuniform'])
plt.plot(results_pos)
plt.legend(['emd', 'ks', 'hellinger', 'js', 'tdl', 'ovl'])
plt.ylabel("Total deviation above the overlap line")
plt.title("Total positive deviation for each scenario")
plt.show()

plt.xticks(np.arange(6), ['shifting\nuniform', 'shifting\nnormal', 'wide\nuniform', 'multi-modal\nuniform'])
plt.plot(results_neg)
plt.legend(['emd', 'ks', 'hellinger', 'js', 'tdl', 'ovl'])
plt.ylabel("Total deviation below the overlap line")
plt.title("Total negative deviation for each scenario")
plt.show()