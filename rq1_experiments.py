import matplotlib.pyplot as plt
from scipy.stats import uniform
from tqdm import tqdm
import time
from algorithms.similarity_metrics import *
from argparse import ArgumentParser


def calculate_boundaries(dist_args1, dist_args2, dist_type, shift):
    """
    Calculate minimum and maximum reward possible for certain distribution types.
    For normal distribution take 3 times standard deviation. These minimum and maximum
    are used to determine the bin sizes.
    :param dist_args1: parameter of the distribution
    :param dist_args2: parameter of the distribution
    :param dist_type: string indicating which distribution type
    :param shift: shift of the multi-modal distribution
    :return: Minimum and maximum reward
    """
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
    elif dist_type == 'multi-modal':
        min_r, max_r = dist_args2[0] - shift, dist_args2[0] + 0.5 + 0.5 + shift
    elif dist_type == 'categorical':
        min_r = min(dist_args1[0] + dist_args2[0])
        max_r = max(dist_args1[0] + dist_args2[0])
    else:
        print('invalid dist type choose: uniform, normal, multi-modal, categorical')
        min_r = None
        max_r = None
    return min_r, max_r


def get_dist_args(dist_type):
    """
    Get the distribution parameters that define the distribution, the first two numbers of each list define the current
    distribution and the last two define the target distribution.
    :param dist_type: string that indicates which distribution type
    :return: list of the different distribution parameters
    """
    if dist_type == 'uniform':
        dists = [[0, 1, 0, 1], [0, 1, 0.2, 1.2], [0, 1, 0.4, 1.4], [0, 1, 0.6, 1.6], [0, 1, 0.8, 1.8], [0, 1, 1, 2]]
    elif dist_type == 'uniform_wide':
        dists = [[0, 1, 0, 1], [0, 1, 0.0625, 1.1875], [0, 1, -0.1667, 1.5],
                 [0, 1, -0.375, 2.125], [0, 1, -1, 4], [0, 1, -4.25, 15.75]]
    elif dist_type == 'normal':
        dists = [[0, 1, 0, 1], [0, 1, 0.5, 1], [0, 1, 1.05, 1], [0, 1, 1.68, 1], [0, 1, 2.56, 1], [0, 1, 6, 1]]
    elif dist_type == 'normal2':
        dists = [[0, 1, 0, 1], [0, 1, 0, 1.52], [0, 1, 0, 2.41], [0, 1, 0, 4.25], [0, 1, 0, 10], [0, 1, 0, 100]]
    elif dist_type == 'multi-modal':  # currently also uniform, the shift is applied later
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
    """
    Test the output of the 6 similarity metrics for a single set of current and next samples
    :param cur_samples: samples of the current distribution
    :param next_samples: samples of the target distribution
    :param bins: the bin sizes
    :param min_r: minimum reward
    :param max_r: maximum reward
    :return: nothing, only prints
    """
    ovl_metric = ovl(cur_samples, next_samples, bins)
    emd_metric = emd(cur_samples, next_samples, bins)
    ks_metric = ks(cur_samples, next_samples)
    hell_metric = hellinger(cur_samples, next_samples, bins)
    js_metric = jsd(cur_samples, next_samples, bins)
    tdl_metric = tdl_rq1(cur_samples, next_samples, dist_type='uniform', dist_params=[min_r, max_r])

    print(f'ovl: {ovl_metric}\nemd: {emd_metric}\nks: {ks_metric}\nhellinger: {hell_metric}\njs: {js_metric}'
          f'\ntdl: {tdl_metric}')


def overlap_plot(metric, means, sample_sizes, d, bin_exp, sample_exp):
    """
    Plot the overlap trend line
    :param metric: the similarity metric
    :param means: the means for each overlap percentage
    :param sample_sizes: the used sample size
    :param d: number of distributions
    :param bin_exp: bool, bin experiment yes or no
    :param sample_exp: bool, sample experiment yes or no
    :return: nothing, it plots
    """
    for count, i in enumerate(sample_sizes):
        plt.xticks(np.arange(d), ['100%', '80%', '60%', '40%', '20%', '0%'])  # Adapt according to dists used
        if bin_exp:
            plt.plot(np.arange(d), means[count, :], label=f'{metric}, $bins$ = {i - 1}')
        elif sample_exp:
            plt.plot(np.arange(d), means[count, :], label=f'{metric}, $n$ = {i}')
        else:
            plt.plot(np.arange(d), means[count, :], label=f'{metric}')


def update_means_stds(metric_list, means, stds, count1, count2, count3, bin_experiments):
    """
    Collect th mean and standard deviation in a matrix to later use for plotting
    :param metric_list: list with calculated values for a particular metric
    :param means: the matrix with all the means for a particular metric
    :param stds: the matrix with all the standard deviations for a particular metric
    :param count1: distribution index
    :param count2: sample size index
    :param count3: bin number index
    :param bin_experiments: bool, bin experiment yes or no
    :return: updated means and stds matrices
    """
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
    """
    Calculate the total deviation for a particular distribution type
    :param metric_means: the calculated means of a particular metric
    :return: total positive and total negative deviation
    """
    overlap = [1, 0.8, 0.6, 0.4, 0.2, 0]
    deviation = metric_means[0] - overlap
    pos = [round(item, 4) for item in deviation if item >= 0]
    neg = [round(abs(item), 4) for item in deviation if item < 0]
    return sum(pos), sum(neg)


def similarity_metric_experiments(metrics, dist_type, sample_sizes, num_iterations, plot_trend, show_time,
                                  num_bins_list, bin_experiments, sample_experiments, num_runs):
    """
    The main function used for calculating all the stuff used for research question 1
    :param metrics: the used metrics
    :param dist_type: the used dist type
    :param sample_sizes: the used sample sizes
    :param num_iterations: number of iterations
    :param plot_trend: bool, plot trend yes or no
    :param show_time: bool, show execution time yes or no
    :param num_bins_list: list with the number of bins
    :param bin_experiments: bool, bin experiment yes or no
    :param sample_experiments: bool, sample experiment yes or no
    :param num_runs: number of runs
    :return: prints or plots the desired results
    """
    # Initializations
    dif_hist_list, emd_list, ks_list, hellinger_list, js_list, tdl_list, ovl_list = [], [], [], [], [], [], []
    tot_hist_time_list, tot_emd_time_list, tot_ks_time_list, tot_h_time_list, tot_js_time_list, tot_tdl_time_list, \
        tot_ovl_time_list = [], [], [], [], [], [], []
    run_time_begin = time.time()
    shifts = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # used for multi-modal

    dist_args = get_dist_args(dist_type)
    d = len(dist_args)
    if bin_experiments:
        s = len(num_bins_list)
    else:
        s = len(sample_sizes)
    emd_means, ks_means, h_means, js_means, tdl_means, ovl_means, emd_stds, ks_stds, h_stds, \
    js_stds, tdl_stds, ovl_stds = np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d)), \
                                  np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d)), \
                                  np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d)), np.zeros((s, d))

    for i in range(num_runs):
        tot_emd_time = 0
        tot_ks_time = 0
        tot_h_time = 0
        tot_js_time = 0
        tot_tdl_time = 0
        tot_ovl_time = 0

        for count1, parameters in tqdm(enumerate(dist_args)):  # the different distribution parameters
            dist1, dist2 = parameters[:2], parameters[2:]
            min_r, max_r = calculate_boundaries(dist1, dist2, dist_type, shifts[count1])
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
                            tdl_metric = tdl_rq1(cur_samples, next_samples, dist_type,
                                                 dist1)  # dist1 correct for uniform
                            tot_tdl_time = tot_tdl_time + (time.time() - start_tdl)
                            tdl_list.append(tdl_metric)
                        if 'ovl' in metrics:
                            bins = np.linspace(min_r, max_r + 0.000001, num_bins)
                            start_ovl = time.time()
                            ovl_metric = ovl(cur_samples, next_samples, bins)  # dist1 correct for uniform
                            tot_ovl_time = tot_ovl_time + (time.time() - start_ovl)
                            ovl_list.append(ovl_metric)
                    # Calculating statistics
                    if plot_trend:
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
                    emd_list, ks_list, hellinger_list, js_list, tdl_list, ovl_list = [], [], [], [], [], []

        run_time = (time.time() - run_time_begin)
        tot_emd_time_list.append(tot_emd_time)
        tot_ks_time_list.append(tot_ks_time)
        tot_h_time_list.append(tot_h_time)
        tot_js_time_list.append(tot_js_time)
        tot_tdl_time_list.append(tot_tdl_time)
        tot_ovl_time_list.append(tot_ovl_time)

    if plot_trend:
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
        print(f'Absolute deviations for each similarity metric:')
        if 'emd' in metrics:
            overlap_plot('emd', emd_means, iter_over, d, bin_experiments, sample_experiments)
            print(f'emd pos: {emd_pos}, neg: {emd_neg}')
        if 'ks' in metrics:
            overlap_plot('ks', ks_means, iter_over, d, bin_experiments, sample_experiments)
            print(f'ks pos: {ks_pos}, neg: {ks_neg}')
        if 'hellinger' in metrics:
            overlap_plot('hellinger', h_means, iter_over, d, bin_experiments, sample_experiments)
            print(f'h pos: {h_pos}, neg: {h_neg}')
        if 'js' in metrics:
            overlap_plot('js', js_means, iter_over, d, bin_experiments, sample_experiments)
            print(f'js pos: {js_pos}, neg: {js_neg}')
        if 'tdl' in metrics:
            overlap_plot('tdl', tdl_means, iter_over, d, bin_experiments, sample_experiments)
            print(f'tdl pos: {tdl_pos}, neg: {tdl_neg}')
        if 'ovl' in metrics:
            overlap_plot('ovl', ovl_means, iter_over, d, bin_experiments, sample_experiments)
            print(f'ovl pos: {ovl_pos}, neg: {ovl_neg}')

        plt.plot([1, 0.8, 0.6, 0.4, 0.2, 0], label='true overlap')
        plt.xlabel('percentage overlap')
        plt.ylabel('similarity value')
        if bin_experiments:
            plt.title(f'Similarity for different number of bins')
        else:
            plt.title(f'Similarity for distributions with different overlap')
        plt.legend()
        plt.show()

    if show_time:
        tot_emd_time_mean = np.asarray(tot_emd_time_list).mean()
        tot_ks_time_mean = np.asarray(tot_ks_time_list).mean()
        tot_h_time_mean = np.asarray(tot_h_time_list).mean()
        tot_js_time_mean = np.asarray(tot_js_time_list).mean()
        tot_tdl_time_mean = np.asarray(tot_tdl_time_list).mean()
        tot_ovl_time_mean = np.asarray(tot_ovl_time_list).mean()

        tot_emd_time_std = np.asarray(tot_emd_time_list).std()
        tot_ks_time_std = np.asarray(tot_ks_time_list).std()
        tot_h_time_std = np.asarray(tot_h_time_list).std()
        tot_js_time_std = np.asarray(tot_js_time_list).std()
        tot_tdl_time_std = np.asarray(tot_tdl_time_list).std()
        tot_ovl_time_std = np.asarray(tot_ovl_time_list).std()

        print(f'tot emd time mean: {tot_emd_time_mean}, std: {tot_emd_time_std}\n'
              f'tot ks time mean: {tot_ks_time_mean}, std: {tot_ks_time_std}\n'
              f'tot h time mean: {tot_h_time_mean}, std: {tot_h_time_std}\n'
              f'tot js time mean: {tot_js_time_mean}, std: {tot_js_time_std}\n'
              f'tot tdl time mean: {tot_tdl_time_mean}, std: {tot_tdl_time_std}\n'
              f'tot ovl time mean: {tot_ovl_time_mean}, std: {tot_ovl_time_std}\n')
    print(f'run time: {run_time}')


def main(args):
    # 'dif_hist', 'emd', 'ks', 'hellinger', 'js', 'tdl', 'ovl' multiple possible
    metric_dict = {'o': 'ovl', 'e': 'emd', 'k': 'ks', 'h': 'hellinger', 'j': 'js', 't': 'tdl'}
    metrics1 = []
    for metric in args.metrics:
        if metric not in metric_dict:
            print("Please choose a valid metric: o, e, k, h, j or t and write them next to each other.\n"
                  "Example: --metrics: k h j\nNon-valid metrics will be ignored")
        else:
            metrics1.append(metric_dict[metric])
    dist_type1 = args.dist_type  # uniform, uniform_wide, normal, normal2, multi-modal, categorical
    sample_sizes = args.sample_sizes
    if isinstance(sample_sizes, int):
        sample_sizes = [sample_sizes]
    num_bins = args.num_bins
    if isinstance(num_bins, int):
        num_bins = [num_bins]
    print(f'Overview of the chosen settings\n'
          f'Metrics: {metrics1}\n'
          f'Distribution type: {args.dist_type}\n'
          f'Sample sizes: {args.sample_sizes}\n'
          f'Number of bins: {args.num_bins}\n')
    for count, i in enumerate(num_bins):
        num_bins[count] = i + 1

    similarity_metric_experiments(metrics1, dist_type1, sample_sizes, args.num_iterations, args.plot_overlap_trend,
                                  show_time=args.show_time, num_bins_list=num_bins,
                                  bin_experiments=args.bin_experiments, sample_experiments=args.sample_experiments,
                                  num_runs=args.num_runs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--metrics", type=str, nargs='+', default='o', help='Provide the first letter of the '
                                                                                  'metric and put them behind each '
                                                                                  'other with spaces. Possible letters '
                                                                                  'are: o, e, k, h, j or t')
    parser.add_argument("-d", "--dist_type", type=str, default='uniform', help="Select the distribution type, possible "
                                                                               "options are: uniform, uniform_wide, "
                                                                               "normal, normal2, multi-modal, "
                                                                               "categorical")
    parser.add_argument("-s", "--sample_sizes", type=int, nargs='+', default=50, help="Specify sample sizes, can be "
                                                                                      "one or multiple, e.g.: 50 100")
    parser.add_argument("-i", "--num_iterations", type=int, default=1000, help="Specify number of iterations")
    parser.add_argument("-b", "--num_bins", type=int, nargs='+', default=10, help="Speciy number of bins, can be one or"
                                                                                  " multiple")
    parser.add_argument("--show_time", type=bool, default=False, help='Show execution time + standard deviation')
    parser.add_argument("--plot_overlap_trend", type=bool, default=False, help="Show overlap trend")
    parser.add_argument("--bin_experiments", type=bool, default=False, help="If you specified multiple bins, then you "
                                                                            "can here indicate to show the results "
                                                                            "of these")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of times the experiments get executed")
    parser.add_argument("--sample_experiments", type=bool, default=False, help="Specify if you run the sample "
                                                                               "experiment")
    args = parser.parse_args()
    main(args)
