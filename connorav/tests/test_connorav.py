import unittest
import connorav
import numpy
from scipy.stats import norm, johnsonsu

MAX_ERROR = 0.02

class TestDist(unittest.TestCase):
    
    def setUp(self):
        self.c = connorav.MSSKDistribution(0,1.0,0.0,0.0)
        
    def test_dist(self):
        mean = 2.0
        std_dev = 1.5
        skew = 0.3
        kurt = 0.5
        a,b,loc,scale = self.c._johnsonsu_param(mean,std_dev,skew,kurt)
        x = johnsonsu(a,b,loc=loc,scale=scale)
        m1,v1,s1,k1 = x.stats(moments='mvsk')
        assert abs(mean - m1) < 0.00001
        assert abs(std_dev**2.0 - v1) < 0.00001
        assert abs(skew -s1) < 0.00001
        assert abs(kurt -k1) < 0.00001


    def test_optimize_w(self):
        #test example from HJH Tuenter's paper.
        b1 = 0.829
        b2 = 4.863
        w1 = 1.105545
        w2 = 1.334005

        w, m = self.c._optimize_w(w1,w2,b1,b2)
        assert abs(w - 1.1543) < 0.001
        assert abs(m - 0.0729) < 0.001


class TestRV(unittest.TestCase):

    def test_rv(self):

        num_samples = 200000

        # The desired covariance matrix.
        correlations = numpy.array([
                [  1.00,  0.60,  0.30],
                [  0.60,  1.00,  0.50],
                [  0.30,  0.50,  1.00]
            ])

        #kurtosis > skewness * skewness + 1 
        moments = numpy.array([[0.0,1.0,0.0,0.0],[14.0,0.9,0.8,1.8],[7.9,0.3,0.2,1.6]])

        rv_generator = connorav.CorrelatedNonNormalRandomVariates(moments,correlations,num_samples)

        rv = rv_generator.rv

        x = numpy.corrcoef(rv)
        
        error = numpy.sum((correlations - x)) ** 2.0
        assert error < MAX_ERROR

        from scipy.stats import skew, kurtosis

        stats = []
        for i in range(3):
            stats.append([rv[i,:].mean(),rv[i,:].std(),skew(rv[i,:]),kurtosis(rv[i,:])])

        stats = numpy.array(stats)

        error = numpy.sum((moments - stats)) ** 2.0
        
        assert error < MAX_ERROR




def plot_dist(samples):
    from pylab import plot, show, axis, subplot, xlabel, ylabel, grid
    #
    # Plot various projections of the samples for visual testing
    #
    y = samples
    subplot(2,2,1)
    plot(y[0], y[1], 'b.')
    ylabel('y[1]')
    axis('equal')
    grid(True)

    subplot(2,2,3)
    plot(y[0], y[2], 'b.')
    xlabel('y[0]')
    ylabel('y[2]')
    axis('equal')
    grid(True)

    subplot(2,2,4)
    plot(y[1], y[2], 'b.')
    xlabel('y[1]')
    axis('equal')
    grid(True)

    show()


if __name__ == '__main__':
    unittest.main()



