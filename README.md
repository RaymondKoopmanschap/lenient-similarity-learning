# Aim of the project
The goal of this repository is to allow interested readers to reproduce the results of my thesis and to experiment with different hyperparameters. Note that the
code is not particularly written with the goal of reuseability for other projects in mind. That being said, the code is documented and should be understandable. 
It also has a requirements.txt so to get started just clone the repository and type
`pip install -r requirements.txt` in your virtual environment.

My thesis can be found here: [Lenient Similarity Learning for Cooperative Multi-Agent Reinforcement Learning](https://scripties.uba.uva.nl/search?id=716643) 
and extends [Lenient learning](https://dl.acm.org/doi/abs/10.5555/2946645.3007037) with a similarity metric which was introduced by a paper called
[Likelihood Quantile Networks for Coordinating Multi-Agent Reinforcement Learning](http://ifaamas.org/Proceedings/aamas2020/pdfs/p798.pdf). 
A concise summary of the thesis is given by my thesis defense which can be found here:<br>
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/6kyZ9ac8bk4/0.jpg)](https://www.youtube.com/watch?v=6kyZ9ac8bk4)

The rough outline of my thesis is that I explain why a similarity metric is useful and compare different similarity in their effectiveness. Next, I incorporate the 
similarity metric into lenient learning and compare it with the version without a similarity metric. Additionally, I also show in which setting the similarity 
metric can fail and finally compare lenient similarity learning with different similarity metrics. 

# Structure of the project
The structure of the project is pretty simple. There are two main files:
<ol>
<li> <strong>rq1_experiments.py</strong>: These allow for the execution of all the experiments of research question 1
where I compare the effectiviness of separating environment stochasticity from miscoordination</li>
<li><strong>rq2_and_3_experiments.py</strong>: These are for research question 2 and 3 
where I incorporate the similarity metric into lenient learning and tests this in a reinforcement learning setting</li>
</ol>

Then we have 3 packages:
<ol>
<li> <strong>algorithms</strong>: This directory include all the similarity metrics, and 
the MARL_algorithms.py contains seven different algorithms depending on which parameters
   are chosen, this will be explained later.</li>
<li> <strong>enviroments</strong>: This directory contains all the reinforcement learning 
environments that are used: the Climb Game, extended Climb Game, the extended stochastic
climb game and relative overgeneralization 3, the last environment is not used in the thesis but was the
most difficult environment for lenient learning</li>
<li><strong>helper_functions</strong>: This one contains all the helper functions divided into 
general helper functions and plotting</li>
</ol>

### Options for the MARL algorithms
Below the 7 different options are given, these are explained in more detail in my thesis section 5.3.2.

            """ the code below can indicate seven different algorithms
                lenient learning:            (ll)   if alpha_sim = alpha      and prob = temp prob
                lenient similarity learning: (lsl)  if alpha_sim = alpha      and prob = l
                lenient hysteretic learning: (lhl)  if alpha_sim = 0          and prob = temp prob
                lenient hyst. sim. learning: (lhsl) if alpha_sim = alpha * l  and prob = temp prob
                hysteretic learning:         (hl)   if alpha_sim = 0          and prob = 1 
                hysteretic sim. learning:    (hsl)  if alpha_sim = alpha * l  and prob = 1 (change ret dist)
                lenient sim. Daan learning:  (lsdl) if alpha_sim = alpha      and prob = min(l, temp prob)"""

# Explanation of the RQ1 and RQ2&3 scripts and how to reproduce each experiment
In this section, we will explain how the RQ1 and RQ2&3 scripts can be used by reproducing
the experiments in my thesis. The format we use is to give the corresponding section and figure in
the thesis that shows the experiment and then how to reproduce it. 

### RQ1 experiments (section 5.2 in thesis)
Section 5.2.1 Analysis of the Time Difference Likelihood, 
figure 7b: use the  `tdl_for_plot(...)` function with the same input as the thesis.

Section 5.2.2 Computational efficiency, figure 9<br>
`python rq1_experiments.py --metrics o e k h j t --dist_type uniform --sample_sizes 10 
--num_iterations 10000 --num_bins 10 --show_time True --num_runs 10`<br>
This calculates the computation time for 10 samples, this has to be executed for 10, 20, 50, 100
,200, 500 and 1000 and these numbers have to be plotted to obtain figure 9. 

--metrics indicate which metrics you want to use: Overlapping coefficienct, Earth Mover's distance, 
Kolmogorov-Smirnov, Hellinger distance, Jensen-Shannon distance or the Time Difference Likelihood. 
Multiple letters indicate multiple metrics, dist_type is which distributions you want to compare, 
see the description in my thesis for an exact explanation and show_time True shows the computation
time. 

Section 5.2.4 figure 10, the sample experiment
`python rq1_experiments.py -m o --dist_type uniform --sample_sizes 10 20 50 100 500 
--num_iterations 1000 --plot_overlap_trend True --sample_experiments True`
In contrast to the computational efficiency experiment, here we can specify multiple sample sizes.
Additionally, the plot_overlap_trend is set to True to show the overlap plot and the
sample experiments is set to True

Section 5.2.4 figure 11, the number of bins experiment<br>
`python rq1_experiments.py -m o --dist_type uniform --sample_sizes 50 --num_iterations 1000 
--plot_overlap_trend True --num_bins 3 5 10 20 50 100 --bin_experiments True
`
Section 5.2.4 figure 12, 13, 14a/b, 15
For figure 12 it is <br>
`python rq1_experiments.py -m o e k h j t --dist_type uniform --plot_overlap_trend True 
--sample_sizes 50 --num_bins 10 --num_iterations 1000
`<br> which can be reduced to <br>
`python rq1_experiments.py -m o e k h j t --dist_type uniform --plot_overlap_trend True` 
<br>because we use the default settings. 
Now for figure 13 dist_type normal, for figure 14a/b dist_type uniform_wide and additionally 
num_bins of 5 and 20, and for figure 15 dist_type multi-modal

Additionally, the positive and negative total deviation for each scenario is printed. If all these
values are plotted we obtain figure 16a and 16b.

### RQ2&3 preliminary experiments (section 5.1)
Section 5.1.2 figure 2, Q-learning<br>
`python rq2_and_3_experiments.py --e_decay 0.9996 --n_runs 100 --num_episodes 10000 --iter_avg 50 
--game_type det --game CG --algo_name hl --beta 0.1 --ylim -10`<br>
Which after using the default settings reduces to<br>
`python rq2_and_3_experiments.py --e_decay 0.9996 --num_episodes 10000 --game_type det --game CG 
--algo_name hl --beta 0.1 --ylim -10`<br>
e_decay indicates the epsilon decay, game type is deterministic, partially stochastic or fully
stochasticy, game is Climb Game (CB), algo_name is hysteretic learning (hl) **but** with beta=0.1 
which is the same as alpha, this let hysteretic learning become Q-learning again. 

The plots for hysteretic learning and lenient learning in this section are similar, just specify the
correct algo_name and parameters. 

### RQ2&3 experiments (section 5.3)
**Grid searches and heatmaps**<br>
This section uses grid searches, for example section 5.3.2 table 19 can be obtained by:<br>
`python rq2_and_3_experiments.py --grid_search normal --num_episodes 15000 --game_type det 
--game ECG --algo_name ll`
There are two grid searches used, normal and lsl. Lsl is normal without the temperature decays. 
Additionally, for the lsl the similarity metric can be specified by 
--sim_metric ovl (or emd, ks, tdl, hellinger, jsd). The results will be stored into a directory with
a csv fie for the correct policies and sample efficiency which are used for the heatmaps. 
Additionally the mean and standard deviation of the sample efficiencies of all the runs are also
stored in a separate csv. This option can be disabled by using --write_to_csv False. 
All the other heatmaps can be obtained in a similar way. 

**Similarity value plots**<br>
The similarity value plots, for example section 5.3.3 figure 20 can be obtained by<br>
`python rq2_and_3_experiments.py --e_decay 0.9997 --num_episodes 15000 --game_type ps --game ECG --algo_name lsl 
--n_runs 1 --plot_sim_value True --debug_run 0`<br>
We have specified n_runs=1, indicating a single run and plot_sim_value = True. Subsequently, debug_run
shows for which run we show the Q-value and similarity value plots. This can also be used to debug
since you can single out a failed run when you run 100 runs.

**Histogram plots**<br>
Finally, the histogram plots, below the reproduction of figure 21<br>
`python rq2_and_3_experiments.py --e_decay 0.9997 --num_episodes 15000 --game_type ps --game ECG 
--algo_name lsl --n_runs 1 --debug_run 0 --agent 0 --vis_iters 75 93 106 811`<br>
Here you also need to specify the --debug_run parameter and additionally for which agent you want
to visualize and which iterations (vis_iters). Only iterations are visualized where the agent needed
to calculate the similarity metric, i.e. when delta < 0. 

You should now able to reproduce all the plots in my thesis and play with the hyperparameters yourself.
More documentation of the possibilities on all parameters can be found in the code. 