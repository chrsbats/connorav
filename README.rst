=====
Fast generation of COrrelated Non-NOrmal RAndom Variates (CONNORAV)
=====

Specify your distributions in terms of mean, standard deviation, skew, kurtosis and a correlation matrix.   CONNORAV can generate random variates fitting these distribution descriptions in a fast and accurate manner.


Motivation
==========

Fitting a distribution to mean, standard deviation, skew and kurtosis is a surprisingly tricky proposition, which is a little surprising since these are the most common descriptors used when describing non-normal distributions.  CONNORAV achieves this using the optimization techinique described by Tuenter (2001) to fit these statistics to an analytical Johnson SU distribution.   While it is true that not all distributions can be described using mean, standard deviation, skew and kurtosis alone, a lot of data resembles these shapes and the statistics are very easy to measure.

Once the distributions have been specified, non-normal correlated random variates can be generated via the copula trick.   This is extremely useful for monte-carlo analysis and risk assessment. 


Examples
====



Authors
=======

Created by Christopher Bates (chrsbats@github.com).

