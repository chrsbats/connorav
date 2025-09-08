import connorav
import numpy as np
from scipy import stats

N = 10000

# moments desired
mu = 1
std = 0.2
skew = 0.0
kurt = 6

def describe(x):
    out = dict(
        N=len(x),
        MU=np.mean(x),
        STD=np.std(x),
        SKEW=stats.skew(x),
        KURT=stats.kurtosis(x),
    )
    return out

# creating distribution
for i in range(15):
    d = connorav.MSSKDistribution(mu, std, skew, kurt)
    x_rand = d.rvs(N)
    print(stats.kurtosis(x_rand))

    # instancing distrution
    # N = 10000

from matplotlib import pyplot as plt

d = connorav.MSSKDistribution(mu, std, skew, kurt)
xs = np.linspace(d.ppf(1e-4), d.ppf(1 - 1e-4), 400)
ys = d.pdf(xs)
plt.plot(xs, ys)
plt.title("MSSKDistribution PDF")
plt.show()


