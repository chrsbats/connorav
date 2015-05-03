import unittest
import ros
import numpy
from scipy.stats import norm


class TestDist(unittest.TestCase):
    
    def setUp(self):
        mean = 2.0
        std_dev = 1.5
        skew = 0.3
        kurt = 0.5
        self.c = ContinuousDist(mean,std_dev,skew,kurt)
        
    def test_dist(self):
        a,b,loc,scale = self.c.johnsonsu_param(mean,std_dev,skew,kurt)
        x = johnsonsu(a,b,loc=loc,scale=scale)
        m1,v1,s1,k1 = x.stats(moments='mvsk')
        assert abs(mean - m1) < 0.00001
        assert abs(std_dev**2.0 - v1) < 0.00001
        assert abs(skew -s1) < 0.00001
        assert abs(kurt -k1) < 0.00001


    def test_optimize_w(self):
        #example from HJH Tuenter's paper.
        b1 = 0.829
        b2 = 4.863
        w1 = 1.105545
        w2 = 1.334005

        print "W",c.optimize_w(w1,w2,b1,b2)


class TestCorrel(unittest.TestCase):

    def test_correl(self):
        dimensions = 3
        num_samples = 20000

        # The desired covariance matrix.
        correlation = numpy.array([
                [  1.00,  0.60,  0.30],
                [  0.60,  1.00,  0.50],
                [  0.30,  0.50,  1.00]
            ])

        #kurtosis > skewness * skewness + 1 
        moments = numpy.array([[0.0,1.0,0.0,0.0],[14.0,0.9,0.8,1.8],[7.9,0.3,0.2,1.6]])

        rv = CorrelatedRandomVariates(moments,correlation,num_samples)

        rv = rv.rv

        #Testing by hand says it works :D
        
        x = numpy.corrcoef(rv.rv)
        print x

        from scipy.stats import skew, kurtosis
        print rv[0,:].mean()
        print rv[0,:].std()
        print skew(rv[0,:])
        print kurtosis(rv[0,:])
        print rv[1,:].mean()
        # rv[1,:].std()
        # skew(rv[1,:])
        # kurtosis(rv[1,:])
        # rv[2,:].mean()
        # rv[2,:].std()
        # skew(rv[2,:])
        # kurtosis(rv[2,:])





def plot_dist(samples):
    #
    # Plot various projections of the samples.
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

