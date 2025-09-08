# Fast generation of COrrelated Non-NOrmal RAndom Variates (CONNORAV)
[![PyPI](https://img.shields.io/pypi/v/connorav.svg)](https://pypi.org/project/connorav/)
[![Python](https://img.shields.io/pypi/pyversions/connorav.svg)](https://pypi.org/project/connorav/)
[![Build](https://github.com/chrsbats/connorav/actions/workflows/ci.yml/badge.svg)](https://github.com/chrsbats/connorav/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.TXT)

Specify your distributions in terms of mean, standard deviation, skew, kurtosis and a correlation matrix.   CONNORAV can generate random variates fitting these distribution descriptions in a fast and accurate manner.

## Requirements

- Python >= 3.8
- NumPy:
  - For Python < 3.10: >=1.22,<2.0
  - For Python >= 3.10: >=1.26 (2.x supported)
- SciPy:
  - For Python < 3.10: >=1.10,<1.12
  - For Python >= 3.10: >=1.13


## Motivation

Fitting a distribution to mean, standard deviation, skew and kurtosis is a surprisingly tricky proposition, which is a little surprising since these are the most common descriptors used when describing non-normal distributions.  CONNORAV achieves this using the optimization techinique described by Tuenter (2001) to fit these statistics to an analytical Johnson SU distribution.

- We match the first four moments to a Johnson SU distribution using:
- H. J. H. Tuenter (2001), “An algorithm to determine the parameters of SU-curves in the Johnson system of probability distributions by moment matching.”

While it is true that not all distributions can be described using mean, standard deviation, skew and kurtosis alone, a lot of data resembles these shapes and the statistics are very easy to measure.

Once the distributions have been specified, non-normal correlated random variates can be generated via the copula trick.   This is extremely useful for monte-carlo analysis and risk assessment. 


## Examples

Generate a distribution and test the random number generator.

    In [1]: from connorav import MSSKDistribution

    In [2]: dist = MSSKDistribution(2.5,1.3,-0.3,2.0)

Alternatively you can construct using an array or tuple 

    In [2]: dist = MSSKDistribution([2.5,1.3,-0.3,2.0])

    In [3]: dist.mean()
    Out[3]: 2.4999999999942295

    In [4]: dist.std()
    Out[4]: 1.3000000000127732

    In [5]: samples = dist.rvs(100000)

    In [6]: samples.mean()
    Out[6]: 2.4958419823009592

    In [7]: samples.std()
    Out[7]: 1.2991039572289937

    In [8]: from scipy.stats import skew, kurtosis

    In [9]: skew(samples)
    Out[9]: -0.2879332050448415

    In [10]: kurtosis(samples)
    Out[10]: 1.926596584590544


Lets generate some numbers.

    In [1]: import numpy

    In [2]: from connorav import CorrelatedNonNormalRandomVariates

    In [3]: num_samples = 200000

    In [4]: # The desired covariance matrix.

    In [5]: correlations = numpy.array([
       ...:         [  1.00,  0.60,  0.30],
       ...:         [  0.60,  1.00,  0.50],
       ...:         [  0.30,  0.50,  1.00]
       ...:     ])


    In [7]: moments = numpy.array([[0.0,1.0,0.0,0.0],[14.0,0.9,0.8,1.8],[7.9,0.3,0.2,1.6]])


    In [8]: rv_generator = CorrelatedNonNormalRandomVariates(moments,correlations,num_samples)
    Normal

    In [9]: rv = rv_generator.rv

    In [11]: x = numpy.corrcoef(rv)

    In [12]: print x
    [[ 1.          0.58994313  0.29320107]
     [ 0.58994313  1.          0.48837508]
     [ 0.29320107  0.48837508  1.        ]]


    In [13]: from scipy.stats import skew, kurtosis

    In [14]: for i in range(3):
       ....:         print "Moments",i
       ....:         print rv[i,:].mean()
       ....:         print rv[i,:].std()
       ....:         print skew(rv[i,:])
       ....:         print kurtosis(rv[i,:])
       ....:     
    Moments 0
    0.00153746238126
    0.999078801649
    -0.0016863652429
    -0.0135121492172
    Moments 1
    13.9980416496
    0.894414387863
    0.805404037664
    1.8027042337
    Moments 2
    7.90077062606
    0.29955446882
    0.215294195427
    1.61036003402



=======

Created by [Christopher Bates](https://github.com/chrsbats)

