import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import uniform, norm

np.random.seed(3)
size = 30
data1 = np.random.normal(loc=0, scale=10.0, size=size+1)
data2 = np.random.normal(loc=0, scale=10.0, size=size+1)
data1.sort(), data2.sort()

tot_dist = np.concatenate([data1, data2])
cdf1 = np.searchsorted(data1, tot_dist, side='right') / size
cdf2 = np.searchsorted(data2, tot_dist, side='right') / size
loc = np.argmax(abs(cdf1 - cdf2))
print(loc)
val = tot_dist[loc] + 0.2
print(val)
loc_cdf1 = np.searchsorted(data1, val, side='right') / size
loc_cdf2 = np.searchsorted(data2, val, side='right') / size
min_loc = min(loc_cdf1, loc_cdf2)
max_loc = max(loc_cdf1, loc_cdf2)
print(min_loc)
print(max_loc)
##################################################
# Cumulative distributions, stepwise:
axes1 = plt.step(data1, np.arange(data1.size)/size, label='empirical cdf $F_1(x)$')
axes2 = plt.step(data2, np.arange(data2.size)/size, label='empirical cdf $F_2(x)$')
matplotlib.rcParams.update({'font.size': 22})
# plt.vlines(x=val+0.25, ymin=min_loc, ymax=max_loc, color='black')
plt.annotate(s='', xy=(val, min_loc), xytext=(val, max_loc), arrowprops=dict(arrowstyle='<->'))
plt.text(x=val, y=(min_loc+max_loc)/2, s='$D$', fontsize=20)
plt.text(x=-25.9, y=0.83, s='$D = sup_x|F_1(x) - F_2(x)|$')

plt.title('Two-sampled Kolmogorov-Smirnov test statistic $D$')
plt.ylabel('Probability', fontsize=22)
plt.xlabel('Value', fontsize=22)
plt.legend(loc='upper left')
plt.grid()

plt.show()

### Multi-modal stuff

mu_1 = 0
mu_2 = 10
mu_3 = 5
variance = 0.5
sigma = np.sqrt(variance)
x_1 = np.linspace(mu_1 - 5*sigma, mu_1 + 7*sigma, 100)
x_2 = np.linspace(mu_2 - 7*sigma, mu_2 + 5*sigma, 100)
x_3 = np.linspace(mu_3 - 7*sigma, mu_3 + 7*sigma, 100)
plt.rcParams.update({'font.size': 14})
plt.plot(x_1, norm.pdf(x_1, mu_1, sigma)/2, color='C1', label='target distribution')
plt.plot(x_2, norm.pdf(x_2, mu_2, sigma)/2, color='C1')
plt.plot(x_3, norm.pdf(x_3, mu_3, sigma), color='C0', label='base distribution')
plt.title('Multi-modal and normal distribution')
plt.xlabel('value')
plt.ylabel('probability')
plt.legend()
plt.show()