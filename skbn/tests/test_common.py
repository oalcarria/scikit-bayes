"""
General tests for all estimators in skbn.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from sklearn.utils.estimator_checks import check_estimator

from skbn import AnDE, WeightedAnDE
from skbn.utils.discovery import all_estimators


@pytest.mark.parametrize("name, Estimator", all_estimators())
def test_all_estimators(name, Estimator):
    check_estimator(Estimator())


@pytest.mark.parametrize(
    "estimator",
    [
        AnDE(mean="geometric"),
        WeightedAnDE(mean="geometric"),
        WeightedAnDE(weight_scope="factor", weight_level=4),
        WeightedAnDE(mean="geometric", weight_scope="factor", weight_level=4),
        # The modular hybrid with arithmetic component weights is only
        # identifiable at the class-specific granularities
        WeightedAnDE(modular=True, weight_level=3),
        WeightedAnDE(modular=True, weight_scope="factor", weight_level=1),
    ],
    ids=repr,
)
def test_variants(estimator):
    check_estimator(estimator)
