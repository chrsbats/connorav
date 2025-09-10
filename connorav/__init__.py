from .distribution import MSSKDistribution
from .correl_rv import CorrelatedNonNormalRandomVariates

__all__ = ["MSSKDistribution", "CorrelatedNonNormalRandomVariates"]
from importlib.metadata import version, PackageNotFoundError 
try: __version__ = version("connorav") 
except PackageNotFoundError: __version__ = "0.0.0+dev"
