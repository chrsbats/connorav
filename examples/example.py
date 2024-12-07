import connorav
import numpy as np
from scipy impor stats

# moments desired
mu = 1
std = 0.2
skew = 0.0
kurt = 6

def describe(x):
    out = dict(
        N = len(x)
        MU = np.mean(x)
        STD = np.std(x)
        SKEW = stats.skew(x)
        KURT = stats.kurtosis(x)
    )
    return out

# creating distribution
for i in range(15):
    d = connorav.MSSKDistribution(mu, std, skew, kurt)
    x_rand = d.rvs(N)
    print(stats.kurtosis(x_rand))

    # instancing distrution
    # N = 10000

# ============================================================
# Statsmodels

from statsmodels.sandbox.distributions.extras import pdf_mvsk
import numpy as np
from matplotlib import pyplot as plt

pdf = pdf_mvsk([mu, std, skew, kurt])
x = np.linspace(pdf(1e-4), pdf(1-1e-4))
y = np.array([pdf(xi) for xi in x])

plt.plot(x,y)
plt.show()


