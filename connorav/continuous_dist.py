from scipy.stats import johnsonsu, norm
from scipy.optimize import minimize_scalar
from scipy import integrate
import numpy

NORMAL_CUTOFF = 0.1

class ContinuousDist(object):
  
    def __init__(self,moments=[]):
        if moments:
            self.fit(moments)

    def fit(self,moments):
        self.mean = moments[0]
        self.std = moments[1]
        self.skew = moments[2]
        self.kurt = moments[3]

        if abs(self.skew) < NORMAL_CUTOFF and abs(self.kurt) < NORMAL_CUTOFF:  
            #It is hard to solve the johnson su curve when it is near normality, so just use a normal curve.
            self.dist = norm(loc=self.mean,scale=self.std)
        else:
            a,b,loc,scale = self.johnsonsu_param(self.mean,self.std,self.skew,self.kurt)
            self.dist = johnsonsu(a,b,loc=loc,scale=scale)

    def ppf(self,x):
        return self.dist.ppf(x)

    def cdf(self,x):
        return self.dist.cdf(x)

    def rvs(self,x):
        return self.dist.rvs(x)

    def optimize_w(self,w1,w2,b1,b2):
        def m_w(w):
            m = -2.0 + numpy.sqrt( 4.0 + 2.0 * ( w ** 2.0 - (b2 + 3.0) / (w ** 2.0 + 2.0 * w + 3.0)))
            return m

        def f_w(w):
            m = m_w(w)
            fw = (w - 1.0 - m) * ( w + 2.0 + 0.5 * m) ** 2.0
            return (fw - b1) ** 2.0

        
        if abs(w1 - w2) > 0.1e-6:
            solution = minimize_scalar(f_w, method='bounded',bounds=(w1,w2))
            w = solution['x']
        else:
            if w1 < 1.0001:
                w = 1.0001
            else:
                w = w1

        m = m_w(w)    
        #if m < 0.001:
        #    m = 0.001

        return w, m


    def johnsonsu_param(self,mean,std_dev,skew,kurt):
        #"An algorithm to determine the parameters of SU-curves in the johnson system of probabillity distributions by moment matching", HJH Tuenter, 2001
        
        #First convert the parameters into the moments used by Tuenter's alg. 
        u2 = std_dev ** 2.0
        u3 = skew * std_dev ** 3.0
        u4 = (kurt + 3.0) * std_dev ** 4.0
        b1 = u3 ** 2.0 / u2 ** 3.0
        b2 = kurt + 3.0

        w2 = numpy.sqrt((-1.0 + numpy.sqrt(2.0 * (b2 -1.0))))
        big_d = (3.0 + b2) * (16.0 * b2 * b2 + 87.0 * b2 + 171.0) / 27
        d = -1.0 + (7.0 + 2.0 * b2 + 2.0 * numpy.sqrt(big_d)) ** (1.0 / 3.0) - (2.0 * numpy.sqrt(big_d) - 7.0 - 2.0 * b2) ** (1.0 / 3.0)
        w1 = (-1.0 + numpy.sqrt(d) + numpy.sqrt( 4 / numpy.sqrt(d) - d - 3.0)) / 2.0
        if (w1 - 1.0) * ((w1 + 2.0) ** 2.0) < b1:
            #no curve will fit
            print mean,std_dev,skew,kurt
            raise Exception

        w, mw = self.optimize_w(w1,w2,b1,b2)

        z = ((w + 1.0) / (2.0 * w )) * ( ((w - 1.0) / mw) - 1.0) 
        if z < 0.0:
            z = 0.0
        omega = -1.0 * numpy.sign(u3) * numpy.arcsinh(numpy.sqrt(z))
        
        a = omega / numpy.sqrt(numpy.log(w))
        b = 1.0 / numpy.sqrt(numpy.log(w))

        z =  w - 1.0 - mw
        if z < 0.0:
            z = 0.0
        loc = mean - numpy.sign(u3) * (std_dev / (w -1.0)) * numpy.sqrt(z)
        
        scale = std_dev / (w - 1.0) * numpy.sqrt( (2.0 * mw) / ( w + 1.0))

        return a,b,loc,scale

    def cvar(self,upper=0.05,samples=64,lower=0.00001):
        interval = (upper - lower) / float(samples)
        ppfs = self.dist.ppf(numpy.arange(lower, upper+interval, interval))
        result = integrate.romb(ppfs, dx=interval)
        return result

            






