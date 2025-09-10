import unittest
import pytest
import connorav
import numpy
from scipy.stats import norm, johnsonsu

MAX_ERROR = 0.08

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
        
        error = numpy.sum((correlations - x) ** 2.0)
        assert error < MAX_ERROR

        from scipy.stats import skew, kurtosis

        stats = []
        for i in range(3):
            stats.append([rv[i,:].mean(),rv[i,:].std(),skew(rv[i,:]),kurtosis(rv[i,:])])

        stats = numpy.array(stats)

        error = numpy.sum((moments - stats) ** 2.0)
        
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


# Additional tests to increase coverage and exercise branches

def test_correlated_eigh_method_shapes():
    num_samples = 1000
    correlations = numpy.array([
        [1.00, 0.60, 0.30],
        [0.60, 1.00, 0.50],
        [0.30, 0.50, 1.00],
    ])
    moments = numpy.array([
        [0.0, 1.0, 0.0, 0.0],
        [14.0, 0.9, 0.8, 1.8],
        [7.9, 0.3, 0.2, 1.6],
    ])
    rv_generator = connorav.CorrelatedNonNormalRandomVariates(moments, correlations, num_samples, method="eigh")
    assert rv_generator.rv.shape == (3, num_samples)


def test_correlated_invalid_method_raises():
    num_samples = 10
    correlations = numpy.eye(2)
    moments = numpy.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    with pytest.raises(ValueError):
        connorav.CorrelatedNonNormalRandomVariates(moments, correlations, num_samples, method="invalid")


def test_correlated_dimension_mismatch_rows():
    num_samples = 10
    moments = numpy.array([
        [0.0, 1.0, 0.0, 0.0],
        [14.0, 0.9, 0.8, 1.8],
        [7.9, 0.3, 0.2, 1.6],
    ])
    correlations = numpy.eye(2)  # wrong first dimension
    with pytest.raises(Exception):
        connorav.CorrelatedNonNormalRandomVariates(moments, correlations, num_samples)


def test_correlated_dimension_mismatch_cols():
    num_samples = 10
    moments = numpy.array([
        [0.0, 1.0, 0.0, 0.0],
        [14.0, 0.9, 0.8, 1.8],
        [7.9, 0.3, 0.2, 1.6],
    ])
    correlations = numpy.zeros((3, 2))  # wrong second dimension
    with pytest.raises(Exception):
        connorav.CorrelatedNonNormalRandomVariates(moments, correlations, num_samples)


def test_distribution_wrappers_and_cvar_cover():
    d = connorav.MSSKDistribution(0.0, 1.0, 0.0, 0.0)  # normal path
    # sampling and basic methods
    s = d.rvs(1000)
    assert isinstance(s, numpy.ndarray)
    # density and distribution functions
    for x in [0.0, 1.0, -1.0]:
        assert numpy.isfinite(d.pdf(x))
        assert numpy.isfinite(d.logpdf(x))
        assert 0.0 <= d.cdf(x) <= 1.0
        assert numpy.isfinite(d.logcdf(x))
        assert 0.0 <= d.sf(x) <= 1.0
        assert numpy.isfinite(d.logsf(x))
    # quantile and survival quantile
    for p in [0.1, 0.5, 0.9]:
        q = d.ppf(p)
        assert numpy.isfinite(q)
        assert numpy.isfinite(d.isf(1 - p))
    # moments
    assert numpy.isfinite(d.mean())
    assert numpy.isfinite(d.median())
    assert numpy.isfinite(d.std())
    assert numpy.isfinite(d.var())
    m, s, sk, k = d.stats()
    assert (m, s, sk, k) == (0.0, 1.0, 0.0, 0.0)
    # cvar integration helper
    cvar = d._cvar(upper=0.1, samples=64)
    assert numpy.isfinite(cvar)


def test_optimize_w_else_branches():
    d = connorav.MSSKDistribution(0.0, 1.0, 0.0, 0.0)
    # Force the "else" branch by making w1 ~ w2 (difference <= 1e-7)
    w, m = d._optimize_w(1.0, 1.0 + 1e-8, b1=0.0, b2=0.0)
    assert w == pytest.approx(1.0001)
    assert numpy.isfinite(m)
    w, m = d._optimize_w(1.5, 1.5 + 1e-8, b1=0.0, b2=0.0)
    assert w == pytest.approx(1.5)
    assert numpy.isfinite(m)


def test_johnsonsu_param_clamps(monkeypatch):
    d = connorav.MSSKDistribution(0.0, 1.0, 0.0, 0.0)
    # Force negative z values so the function clamps to 0.0 in two places
    monkeypatch.setattr(d, "_optimize_w", lambda w1, w2, b1, b2: (1.1, 100.0))
    a, b, loc, scale = d._johnsonsu_param(0.0, 1.0, 0.1, 1.0)
    for v in (a, b, loc, scale):
        assert numpy.isfinite(v)


def test_mssk_init_bad_iterable_raises():
    with pytest.raises(TypeError):
        connorav.MSSKDistribution([1.0, 2.0, 3.0])


def test_johnsonsu_param_invalid_parameters_raise():
    d = connorav.MSSKDistribution(0.0, 1.0, 0.0, 0.0)
    # Extremely large skew makes b1 very large, violating the feasibility condition.
    with pytest.raises(Exception):
        d._johnsonsu_param(0.0, 1.0, skew=1e6, kurt=0.0)


def test_corr_kind_spearman_preserves_rank():
    num_samples = 100000
    # target Spearman matrix for 2 vars
    rho_s = 0.6
    correlations_s = numpy.array([[1.0, rho_s],
                                  [rho_s, 1.0]])
    moments = numpy.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    rv_generator = connorav.CorrelatedNonNormalRandomVariates(
        moments, correlations_s, num_samples, method="eigh", corr_kind="spearman"
    )
    from scipy.stats import spearmanr
    r_s, _ = spearmanr(rv_generator.rv[0, :], rv_generator.rv[1, :])
    assert abs(r_s - rho_s) < 0.03


def test_corr_kind_kendall_preserves_rank():
    num_samples = 100000
    tau = 0.4
    correlations_k = numpy.array([[1.0, tau],
                                  [tau, 1.0]])
    moments = numpy.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    rv_generator = connorav.CorrelatedNonNormalRandomVariates(
        moments, correlations_k, num_samples, method="eigh", corr_kind="kendall"
    )
    from scipy.stats import kendalltau
    tau_hat, _ = kendalltau(rv_generator.rv[0, :], rv_generator.rv[1, :])
    assert abs(tau_hat - tau) < 0.03


def test_corr_kind_invalid_raises():
    num_samples = 10
    moments = numpy.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    correlations = numpy.eye(2)
    with pytest.raises(ValueError):
        connorav.CorrelatedNonNormalRandomVariates(
            moments, correlations, num_samples, corr_kind="not-a-kind"
        )


if __name__ == '__main__':
    unittest.main()



def test__to_latent_corr_non_square_raises():
    # Create a valid instance first (so __init__ dimension checks pass)
    moments = numpy.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    correlations = numpy.eye(2)
    rv_generator = connorav.CorrelatedNonNormalRandomVariates(moments, correlations, num_samples=10)

    # Now exercise the non-square guard inside _to_latent_corr (line ~40)
    bad_corr = numpy.zeros((2, 3))
    with pytest.raises(Exception):
        rv_generator._to_latent_corr(bad_corr, kind="pearson")


def test___version_fallback_dev(monkeypatch):
    import importlib
    import importlib.metadata as ilmd
    import connorav as mod

    # Patch importlib.metadata.version to raise, then reload connorav
    orig_version = ilmd.version

    def _raise(_name):
        raise ilmd.PackageNotFoundError

    monkeypatch.setattr(ilmd, "version", _raise)
    importlib.reload(mod)
    assert mod.__version__ == "0.0.0+dev"

    # Restore and reload to avoid side effects on other tests
    monkeypatch.setattr(ilmd, "version", orig_version)
    importlib.reload(mod)

