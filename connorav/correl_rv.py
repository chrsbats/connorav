import numpy
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
from .distribution import MSSKDistribution



class CorrelatedNonNormalRandomVariates(object):

    def __init__(self, moments, correlations, num_samples, method='cholesky', corr_kind='pearson'):

        #List of 4 moments array.
        #NXN correlation array
        self.dimensions = moments.shape[0]
        if self.dimensions != correlations.shape[0]:
            raise Exception("Error in dimensions")
        if self.dimensions != correlations.shape[1]:
            raise Exception("Error in dimensions")

        
        self.moments = moments
        self.distributions = list(map(MSSKDistribution, moments.tolist()))
        self.correlations = self._to_latent_corr(correlations, corr_kind)
        self.generate(num_samples, method)
        

    def generate(self, num_samples, method='cholesky'):
        u = self._uniform_correlated(self.dimensions, self.correlations, num_samples, method)
        # numerical safety: keep uniforms strictly inside (0,1)
        eps = 1e-12
        u = numpy.clip(u, eps, 1.0 - eps)
        rows = [self.distributions[i].ppf(u[i, :]) for i in range(self.dimensions)]
        self.rv = numpy.vstack(rows)
        return self.rv
        

    def _to_latent_corr(self, corr, kind):
        m = numpy.array(corr, dtype=float, copy=True)
        if m.shape[0] != m.shape[1]:
            raise Exception("Error in dimensions")
        # Ensure symmetry and unit diagonal
        m = (m + m.T) / 2.0
        numpy.fill_diagonal(m, 1.0)
        k = (kind or 'pearson').lower()
        if k in ('pearson', 'latent', 'z', 'gaussian'):
            mapped = m
        elif k == 'spearman':
            mapped = 2.0 * numpy.sin(numpy.pi * m / 6.0)
            numpy.fill_diagonal(mapped, 1.0)
        elif k == 'kendall':
            mapped = numpy.sin(numpy.pi * m / 2.0)
            numpy.fill_diagonal(mapped, 1.0)
        else:
            raise ValueError(f"Unknown corr_kind: {kind}")
        # Numerical guards
        mapped = numpy.clip(mapped, -0.999999, 0.999999)
        mapped = (mapped + mapped.T) / 2.0
        numpy.fill_diagonal(mapped, 1.0)
        return mapped

    def _normal_correlated(self,dimensions,correlation,num_samples,method='cholesky'):

        # Generate samples from three independent normally distributed random
        # variables (with mean 0 and std. dev. 1).
        half = (num_samples + 1) // 2
        x_half = numpy.random.standard_normal(size=(dimensions, half))
        x = numpy.concatenate([x_half, -x_half], axis=1)[:, :num_samples]

        # We need a matrix `c` for  which `c*c^T = r`.  We can use, for example,
        # the Cholesky decomposition, or the we can construct `c` from the
        # eigenvectors and eigenvalues.
        if method == 'cholesky':
            # Compute the Cholesky decomposition.
            c = cholesky(correlation, lower=True)
        elif method in ('eigh', 'eigen'):
            # Compute the eigenvalues and eigenvectors.
            evals, evecs = eigh(correlation)
            # Clip tiny negative eigenvalues for numerical stability
            evals = numpy.maximum(evals, 0.0)
            # Construct c, so c*c^T = r.
            c = numpy.dot(evecs, numpy.diag(numpy.sqrt(evals)))
        else:
            raise ValueError("Unknown method: {} (use 'cholesky' or 'eigh')".format(method))

        # Convert the data to correlated random variables. 
        y = numpy.dot(c, x)
        return y

    def _uniform_correlated(self,dimensions,correlation,num_samples,method='cholesky'):
        normal_samples = self._normal_correlated(dimensions,correlation,num_samples,method) 
        x = norm.cdf(normal_samples)
        return x


