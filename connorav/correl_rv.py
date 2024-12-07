import numpy
from scipy.linalg import eigh, cholesky
from scipy.stats import norm, johnsonsu
from .distribution import MSSKDistribution



class CorrelatedNonNormalRandomVariates(object):

    def __init__(self,moments,correlations,num_samples, method='cholesky'):

        #List of 4 moments array.
        #NXN correlation array
        self.dimensions = moments.shape[0]
        if self.dimensions != correlations.shape[0]:
            raise Exception("Error in dimensions")
        if self.dimensions != correlations.shape[1]:
            raise Exception("Error in dimensions")

        
        self.moments = moments
        self.correlations = correlations
        self.distributions = map(MSSKDistribution,moments.tolist())

        self.generate(num_samples,method)
        

    def generate(self,num_samples,method='cholesky'):
        rv = self._uniform_correlated(self.dimensions,self.correlations,num_samples,method)
        rv = rv.tolist()
        for d in range(self.dimensions):
            rv[d] = self.distributions[d].ppf(rv[d])
        self.rv = numpy.array(rv)
        return self.rv
        

    def _normal_correlated(self,dimensions,correlation,num_samples,method='cholesky'):

        # Generate samples from three independent normally distributed random
        # variables (with mean 0 and std. dev. 1).
        x = norm.rvs(size=(dimensions, num_samples))

        # We need a matrix `c` for  which `c*c^T = r`.  We can use, for example,
        # the Cholesky decomposition, or the we can construct `c` from the
        # eigenvectors and eigenvalues.
        if method == 'cholesky':
            # Compute the Cholesky decomposition.
            c = cholesky(correlation, lower=True)
        else:
            # Compute the eigenvalues and eigenvectors.
            evals, evecs = eigh(correlation)
            # Construct c, so c*c^T = r.
            c = numpy.dot(evecs, np.diag(np.sqrt(evals)))

        # Convert the data to correlated random variables. 
        y = numpy.dot(c, x)
        return y

    def _uniform_correlated(self,dimensions,correlation,num_samples,method='cholesky'):
        normal_samples = self._normal_correlated(dimensions,correlation,num_samples,method) 
        x = norm.cdf(normal_samples)
        return x

