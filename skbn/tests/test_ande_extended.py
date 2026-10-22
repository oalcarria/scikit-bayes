"""Extended tests for the AnDE and AnJE families: weighted variants, edge
cases, parallel execution and API consistency."""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_classification

from skbn import ALR, AnDE, AnJE, WeightedAnDE

# =============================================================================
# Synthetic Datasets
# =============================================================================

# 1. Mixed Data (Gaussian, Bernoulli, Categorical)
X_MIXED = np.array(
    [[0.5, 0, 0], [-1.2, 1, 1], [0.6, 1, 2], [-0.1, 0, 0], [2.5, 1, 1], [-3.0, 0, 2]]
)
y_MIXED = np.array([0, 1, 1, 0, 1, 0])

# 2. Pure Categorical (XOR-like structure)
X_CAT = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]])
y_CAT = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# 3. Pure Gaussian (XOR-like)
X_GAUSS = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
y_GAUSS = np.array([0, 1, 1, 0])

# One configuration of every estimator, with a short optimization
MODELS = {
    "AnDE": lambda: AnDE(n_dependence=1),
    "AnDE-geometric": lambda: AnDE(n_dependence=1, mean="geometric"),
    "WeightedAnDE": lambda: WeightedAnDE(n_dependence=1, max_iter=10),
    "WeightedAnDE-factor": lambda: WeightedAnDE(
        n_dependence=1, weight_scope="factor", weight_level=4, max_iter=10
    ),
    "AnJE": lambda: AnJE(n_join=2),
    "ALR": lambda: ALR(n_join=2, max_iter=10),
}


# =============================================================================
# Geometric pooling of SPnDEs
# =============================================================================


class TestGeometricAnDE:
    """Tests for AnDE with geometric pooling of its components."""

    def test_probabilities_sum_to_one(self):
        model = AnDE(n_dependence=1, mean="geometric").fit(X_MIXED, y_MIXED)
        proba = model.predict_proba(X_MIXED)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
        assert np.all(proba >= 0)

    def test_n0_is_naive_bayes_like_arithmetic(self):
        # A single component: both means pool the same joint
        geo = AnDE(n_dependence=0, mean="geometric").fit(X_MIXED, y_MIXED)
        ari = AnDE(n_dependence=0).fit(X_MIXED, y_MIXED)
        assert_allclose(geo.predict_proba(X_MIXED), ari.predict_proba(X_MIXED))

    def test_categorical_xor(self):
        model = AnDE(n_dependence=1, mean="geometric").fit(X_CAT, y_CAT)
        X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        assert_array_equal(model.predict(X_test), [0, 1, 1, 0])

    def test_gaussian_xor(self):
        model = AnDE(n_dependence=1, n_bins=2, mean="geometric")
        assert_array_equal(model.fit(X_GAUSS, y_GAUSS).predict(X_GAUSS), y_GAUSS)

    def test_differs_from_arithmetic_mean(self):
        geo = AnDE(n_dependence=1, mean="geometric").fit(X_MIXED, y_MIXED)
        ari = AnDE(n_dependence=1).fit(X_MIXED, y_MIXED)
        assert not np.allclose(geo.predict_proba(X_MIXED), ari.predict_proba(X_MIXED))


# =============================================================================
# WeightedAnDE
# =============================================================================


@pytest.mark.parametrize("mean", ["arithmetic", "geometric"])
@pytest.mark.parametrize("scope", ["component", "factor"])
class TestWeightedAnDE:
    """Tests for WeightedAnDE across poolings and weight scopes."""

    @pytest.mark.parametrize("level", [1, 2, 3, 4])
    def test_levels_fit_and_predict(self, mean, scope, level):
        model = WeightedAnDE(
            n_dependence=1,
            mean=mean,
            weight_scope=scope,
            weight_level=level,
            n_bins=2,
            max_iter=10,
        ).fit(X_MIXED, y_MIXED)
        assert model.learned_weights_.shape == (model.n_weights_,)
        assert_allclose(model.predict_proba(X_MIXED).sum(axis=1), 1.0, rtol=1e-5)

    @pytest.mark.parametrize("n_dependence", [0, 2])
    def test_other_orders(self, mean, scope, n_dependence):
        model = WeightedAnDE(
            n_dependence=n_dependence, mean=mean, weight_scope=scope, max_iter=10
        ).fit(X_MIXED, y_MIXED)
        assert len(model.ensemble_) == (1 if n_dependence == 0 else 3)
        assert set(model.predict(X_MIXED)) <= set(y_MIXED)

    def test_regularization_pulls_weights_towards_one(self, mean, scope):
        def deviation(l2_reg):
            model = WeightedAnDE(
                n_dependence=1,
                mean=mean,
                weight_scope=scope,
                weight_level=3,
                l2_reg=l2_reg,
                max_iter=50,
            ).fit(X_MIXED, y_MIXED)
            return np.abs(model.learned_weights_ - 1.0).mean()

        assert deviation(10.0) <= deviation(1e-6) + 1e-2

    def test_modular_optimization(self, mean, scope):
        model = WeightedAnDE(
            n_dependence=1,
            mean=mean,
            weight_scope=scope,
            weight_level=4,
            modular=True,
            n_bins=2,
            max_iter=10,
        ).fit(X_MIXED, y_MIXED)
        assert_allclose(model.predict_proba(X_MIXED).sum(axis=1), 1.0, rtol=1e-5)


def test_geometric_component_weights_are_non_negative():
    model = WeightedAnDE(
        n_dependence=1, mean="geometric", weight_level=3, l2_reg=0.0, max_iter=200
    ).fit(X_MIXED, y_MIXED)
    assert np.all(model.learned_weights_ >= 0)


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    @pytest.mark.parametrize("name", MODELS)
    def test_single_feature(self, name):
        X = np.array([[1.0], [2.0], [1.5], [2.5], [1.2], [2.2]])
        y = np.array([0, 1, 0, 1, 0, 1])
        model = MODELS[name]()
        if "n_dependence" in model.get_params():
            model.set_params(n_dependence=0)
        assert len(model.fit(X, y).predict(X)) == len(y)

    @pytest.mark.parametrize("name", MODELS)
    def test_many_classes(self, name):
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42,
        )
        model = MODELS[name]().set_params(n_bins=3).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 4)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    @pytest.mark.parametrize("name", MODELS)
    def test_binary_features(self, name):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5)
        y = np.array([0, 1, 1, 0] * 5)
        assert len(MODELS[name]().fit(X, y).predict(X)) == len(y)

    def test_discretization_strategies(self):
        """Test different discretization strategies."""
        X = np.random.randn(50, 3)
        y = (X.sum(axis=1) > 0).astype(int)

        for strategy in ["uniform", "quantile", "kmeans"]:
            model = AnDE(n_dependence=1, n_bins=3, strategy=strategy)
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(y)

    def test_different_n_bins(self):
        """Test different numbers of bins."""
        X = np.random.randn(50, 3)
        y = (X.sum(axis=1) > 0).astype(int)

        for n_bins in [2, 5, 10]:
            model = AnDE(n_dependence=1, n_bins=n_bins)
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == len(y)

    def test_different_alpha_smoothing(self):
        """Test different alpha smoothing values."""
        for alpha in [0.1, 1.0, 10.0]:
            model = AnDE(n_dependence=1, alpha=alpha)
            model.fit(X_MIXED, y_MIXED)
            proba = model.predict_proba(X_MIXED)
            assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_predict_on_interpolated_values(self):
        """Test prediction on new samples within training range (interpolation)."""
        X_train = np.array([[0.0, 0], [1.0, 1], [2.0, 0], [3.0, 1]] * 3)
        y_train = np.array([0, 1, 0, 1] * 3)

        # Test on values within the training range
        X_test = np.array([[0.5, 0], [1.5, 1], [2.5, 0]])

        model = AnDE(n_dependence=1, n_bins=3)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        proba = model.predict_proba(X_test)
        assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_all_same_class(self):
        """Test behavior when all training samples have the same class."""
        X = np.array([[1.0, 0], [2.0, 1], [1.5, 0], [2.5, 1]])
        y = np.array([0, 0, 0, 0])  # All same class

        model = AnDE(n_dependence=1)
        model.fit(X, y)

        # Should predict only class 0
        preds = model.predict(X)
        assert_array_equal(preds, np.zeros(4))


# =============================================================================
# Parallel Execution Tests
# =============================================================================


class TestParallelExecution:
    """Tests for parallel execution with n_jobs."""

    @pytest.mark.parametrize("cls", [AnDE, WeightedAnDE])
    def test_parallel_matches_serial(self, cls):
        serial = cls(n_dependence=1, n_jobs=1).fit(X_MIXED, y_MIXED)
        parallel = cls(n_dependence=1, n_jobs=2).fit(X_MIXED, y_MIXED)
        assert_allclose(serial.predict_proba(X_MIXED), parallel.predict_proba(X_MIXED))

    def test_ande_parallel_all_cores(self):
        model = AnDE(n_dependence=1, n_jobs=-1).fit(X_MIXED, y_MIXED)
        assert len(model.predict(X_MIXED)) == len(y_MIXED)


# =============================================================================
# API Consistency Tests
# =============================================================================


@pytest.mark.parametrize("name", MODELS)
class TestAPIConsistency:
    """Tests for scikit-learn API consistency."""

    def test_fitted_attributes(self, name):
        model = MODELS[name]().fit(X_MIXED, y_MIXED)
        assert_array_equal(model.classes_, [0, 1])
        assert model.n_features_in_ == 3

    def test_prediction_shapes(self, name):
        model = MODELS[name]().fit(X_MIXED, y_MIXED)
        assert model.predict_proba(X_MIXED).shape == (len(X_MIXED), 2)
        assert model.predict(X_MIXED).shape == (len(X_MIXED),)

    def test_log_proba_is_consistent_with_proba(self, name):
        model = MODELS[name]().fit(X_MIXED, y_MIXED)
        assert_allclose(
            np.exp(model.predict_log_proba(X_MIXED)), model.predict_proba(X_MIXED)
        )
