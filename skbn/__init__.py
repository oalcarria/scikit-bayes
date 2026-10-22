# Authors: scikit-bayes developers
# License: BSD 3 clause

try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("scikit-bayes")
    except (ImportError, PackageNotFoundError):
        __version__ = "0.0.0"
from .ande import AnDE, WeightedAnDE
from .anje import ALR, AnJE
from .discretization import DecisionTreeDiscretizer, MDLPDiscretizer
from .mixed_nb import MixedNB
from .tan import TAN

__all__ = [
    "MixedNB",
    "AnDE",
    "AnJE",
    "ALR",
    "WeightedAnDE",
    "TAN",
    "DecisionTreeDiscretizer",
    "MDLPDiscretizer",
    "__version__",
]
