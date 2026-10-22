"""
Averaged n-Join Estimators (AnJE) and Accelerated Logistic Regression (ALR).

Both models follow Zaidi et al. [1]_. An n-join is the joint distribution
:math:`P(x_\\alpha | y)` of a subset :math:`\\alpha` of ``n`` attributes;
AnJE pools every n-join with a fixed weight and ALR learns one weight per
n-join, value and class by maximising the conditional log-likelihood. They are
defined on discrete attributes, so continuous attributes are discretized.

References
----------
.. [1] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., & Cerquides, J. (2016).
       ALR^n: Accelerated higher-order logistic regression. Machine Learning, 104(2-3), 151-194.
.. [2] Zaidi, N. A., Webb, G. I., Carman, M. J., Petitjean, F., Buntine, W.,
       Hynes, M., & De Sterck, H. (2017). Efficient parameter learning of
       Bayesian network classifiers. Machine Learning, 106(9-10), 1289-1329.
"""

# Authors: scikit-bayes developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
from itertools import combinations
from math import comb

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from ._core import _AttributeEncoder, _fit_weights, _row_chunks


class AnJE(ClassifierMixin, BaseEstimator):
    """
    Averaged n-Join Estimators (AnJE) [Generative].

    The geometric mean of the n-join estimators of every partition of the
    attributes into subsets of size ``n`` (Zaidi et al. [1]_, Eq. 11):

    .. math::
        P(y | x) \\propto P(y) \\prod_{\\alpha \\in \\binom{A}{n}}
        P(x_\\alpha | y)^{\\frac{(n-1)!(a-n)!}{(a-1)!}}.

    A1JE is Naive Bayes, and AnJE uses the same statistics as A(n-1)DE.
    Probabilities are m-estimates with ``m = alpha``:
    :math:`P(y) = (\\#y + m/|Y|) / (t + m)` and
    :math:`P(x_\\alpha | y) = (\\#(x_\\alpha, y) + m/|X_\\alpha|) / (\\#y + m)`.

    Continuous attributes are discretized into ``n_bins`` bins (the original
    work uses MDL discretization, available as :class:`MDLPDiscretizer` in a
    pipeline). A discrete value never seen in training falls in an empty cell
    of every n-join that contains it.

    Parameters
    ----------
    n_join : int, default=2
        Size ``n`` of the attribute subsets. With fewer attributes than
        ``n_join``, the single join of all attributes is used.
    n_bins : int, default=5
        Number of bins for continuous attributes.
    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Binning strategy for continuous attributes.
    alpha : float, default=1.0
        Weight ``m`` of the m-estimates.
    categorical_features, bernoulli_features : array-like of int, default=None
        Attributes to treat as discrete regardless of their dtype.
    gaussian_features : array-like of int, default=None
        Attributes to treat as continuous (binned) regardless of their dtype.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    joins_ : ndarray of shape (n_joins, n)
        Attribute subsets of the n-joins.
    """

    def __init__(
        self,
        n_join=2,
        n_bins=5,
        strategy="quantile",
        alpha=1.0,
        categorical_features=None,
        bernoulli_features=None,
        gaussian_features=None,
    ):
        self.n_join = n_join
        self.n_bins = n_bins
        self.strategy = strategy
        self.alpha = alpha
        self.categorical_features = categorical_features
        self.bernoulli_features = bernoulli_features
        self.gaussian_features = gaussian_features

    def fit(self, X, y):
        """Count the n-joins of the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._fit_generative(X, y)
        return self

    def _fit_generative(self, X, y):
        """Fit the tables and return the training cells and class indices."""
        X, y = validate_data(self, X, y)
        if not isinstance(self.n_join, numbers.Integral) or self.n_join < 1:
            raise ValueError(f"n_join must be a positive integer, got {self.n_join!r}")
        self.classes_ = unique_labels(y)
        y_idx = np.searchsorted(self.classes_, y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self._encoder = _AttributeEncoder(
            self.n_bins,
            self.strategy,
            gaussian_features=self.gaussian_features,
            discrete_features=list(self.categorical_features or [])
            + list(self.bernoulli_features or []),
        )
        codes = self._encoder.fit_transform(X)

        # --- Cell keys: 0 is the class prior, then one block per join ---
        n = min(self.n_join, n_features)
        self.joins_ = np.array(list(combinations(range(n_features), n)), dtype=np.int64)
        # Each attribute gets an extra code for values unseen in training
        slots = (self._encoder.cards_ + 1)[self.joins_]
        sizes = slots.prod(axis=1)
        # Row-major strides of the cells of each join
        self._join_strides = np.ones_like(slots)
        self._join_strides[:, :-1] = np.cumprod(slots[:, :0:-1], axis=1)[:, ::-1]
        self._join_offsets = 1 + np.cumsum(sizes) - sizes
        self._join_weight = 1.0 / comb(n_features - 1, n - 1)  # Eq. 10

        # --- Table rows: the cells seen in training, then one empty cell per
        # join that stands for all its unseen cells (they share its estimate) ---
        keys = self._cell_keys_of(codes)
        self._cell_keys = np.unique(keys)
        n_joins = len(self.joins_)
        self._n_rows = len(self._cell_keys) + n_joins
        self._row_join = np.concatenate(  # -1 for the prior
            [
                np.searchsorted(self._join_offsets, self._cell_keys, side="right") - 1,
                np.arange(n_joins),
            ]
        )

        # --- m-estimates (row 0 counts the classes: every sample falls in it) ---
        incidence = self._incidence(keys)
        counts = incidence.T @ np.eye(n_classes)[y_idx]
        class_counts = counts[0]
        m = self.alpha
        n_cells = self._encoder.cards_[self.joins_].prod(axis=1)  # |X_alpha|
        cell_prior = (m / n_cells)[self._row_join[1:]][:, None]
        with np.errstate(divide="ignore"):
            self._log_table = np.vstack(
                [
                    np.log((class_counts + m / n_classes) / (n_samples + m)),
                    np.log(counts[1:] + cell_prior) - np.log(class_counts + m),
                ]
            )
        return incidence, y_idx

    def _cell_keys_of(self, codes):
        """Key of the cell of every term, shape (n_samples, 1 + n_joins):
        column 0 is the class prior and column ``1 + j`` the cell of join ``j``."""
        keys = np.zeros((codes.shape[0], 1 + len(self.joins_)), dtype=np.int64)
        for rows in _row_chunks(codes.shape[0], self.joins_.size):
            keys[rows, 1:] = self._join_offsets + np.einsum(
                "ijk,jk->ij", codes[rows][:, self.joins_], self._join_strides
            )
        return keys

    def _incidence(self, keys):
        """Sparse 0/1 matrix of shape (n_samples, n_rows) marking the table
        row of every term; a cell unseen in training maps to the empty row of
        its join."""
        n_seen = len(self._cell_keys)
        pos = np.minimum(np.searchsorted(self._cell_keys, keys), n_seen - 1)
        empty = np.concatenate([[0], n_seen + np.arange(len(self.joins_))])
        rows = np.where(self._cell_keys[pos] == keys, pos, empty)
        return csr_matrix(
            (
                np.ones(rows.size),
                rows.ravel(),
                np.arange(0, rows.size + 1, keys.shape[1]),
            ),
            shape=(keys.shape[0], self._n_rows),
        )

    def _transform_incidence(self, X):
        """Incidence matrix of a validated ``X``; an unseen value takes the
        extra code of its attribute."""
        codes, unseen = self._encoder.transform(X)
        codes = np.where(unseen, self._encoder.cards_, codes)
        return self._incidence(self._cell_keys_of(codes))

    def _table(self):
        """Log-table weighted as in the AnJE: the joins by Eq. 10, the prior by 1."""
        table = self._log_table * self._join_weight
        table[0] = self._log_table[0]
        return table

    def predict_log_proba(self, X):
        """Log-probability estimates for ``X``, shape (n_samples, n_classes)."""
        check_is_fitted(self, attributes=["classes_", "joins_"])
        X = validate_data(self, X, reset=False)
        scores = self._transform_incidence(X) @ self._table()
        log_prob = scores - logsumexp(scores, axis=1, keepdims=True)
        return np.nan_to_num(log_prob, nan=-np.log(scores.shape[1]))

    def predict_proba(self, X):
        """Probability estimates for ``X``, shape (n_samples, n_classes)."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """Most probable class of every sample in ``X``."""
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


class ALR(AnJE):
    """
    Accelerated Higher-Order Logistic Regression (ALR^n) [Hybrid].

    The model of Zaidi et al. [1]_ (Eq. 13): every n-join probability of the
    AnJE gets a weight, learned by maximising the conditional log-likelihood,

    .. math::
        \\log P(y | x) = w_y \\log P(y) + \\sum_{\\alpha \\in \\binom{A}{n}}
        w_{y, \\alpha, x_\\alpha} \\log P(x_\\alpha | y) - \\log Z(x).

    The objective is convex. Weights start at 1 and are optimized with
    L-BFGS-B. The defaults reproduce the protocol of the original experiments
    [1]_ [2]_: no penalty, and the optimizer stops when the relative reduction
    of the objective falls below 1e-32 or after 10^4 iterations. ``l2_reg``
    adds the penalty towards 1 (the generative estimates) that [1]_ suggests
    for small datasets but does not evaluate.

    ``weight_level=3`` is the ALR^n of the paper. Levels 1 and 2 tie the
    weights across classes and values, following the nested
    parameterizations of Zaidi et al. [2]_ (Sec. 5.4), which the ALR paper
    does not use:

    ===== ==============================================
    Level One weight per
    ===== ==============================================
    1     n-join :math:`\\alpha`
    2     n-join and value :math:`(\\alpha, x_\\alpha)`
    3     n-join, value and class (ALR^n)
    ===== ==============================================

    The class prior is weighted like an n-join with a single value. Only the
    cells observed in training get their own weight at levels 2 and 3; the
    unseen cells of an n-join share one weight that is never trained and keeps
    its value of 1, so the number of weights is bounded by the data.

    Parameters
    ----------
    n_join : int, default=2
        Size ``n`` of the attribute subsets (ALR2 uses pairs).
    n_bins : int, default=5
        Number of bins for continuous attributes.
    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Binning strategy for continuous attributes.
    alpha : float, default=1.0
        Weight ``m`` of the m-estimates; must be positive.
    categorical_features, bernoulli_features : array-like of int, default=None
        Attributes to treat as discrete regardless of their dtype.
    gaussian_features : array-like of int, default=None
        Attributes to treat as continuous (binned) regardless of their dtype.
    weight_level : {1, 2, 3}, default=3
        Granularity of the weights.
    l2_reg : float, default=0.0
        L2 regularization towards the generative weights (w = 1), added to the
        summed conditional log-likelihood.
    max_iter : int, default=10000
        Maximum number of L-BFGS-B iterations.
    tol : float or None, default=1e-32
        Stopping tolerance on the relative reduction of the objective and on
        the projected gradient; None uses SciPy's defaults.

    Attributes
    ----------
    learned_weights_ : ndarray of shape (n_weights_,)
        Learned weights, one row per weight group and, at level 3, one column
        per class, flattened in row-major order.
    n_weights_ : int
        Number of weights.
    n_iter_ : int
        Number of L-BFGS-B iterations.
    """

    def __init__(
        self,
        n_join=2,
        n_bins=5,
        strategy="quantile",
        alpha=1.0,
        categorical_features=None,
        bernoulli_features=None,
        gaussian_features=None,
        weight_level=3,
        l2_reg=0.0,
        max_iter=10000,
        tol=1e-32,
    ):
        super().__init__(
            n_join,
            n_bins,
            strategy,
            alpha,
            categorical_features,
            bernoulli_features,
            gaussian_features,
        )
        self.weight_level = weight_level
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """Fit the AnJE tables, then the discriminative weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.weight_level not in (1, 2, 3):
            raise ValueError(f"weight_level must be 1, 2 or 3, got {self.weight_level}")
        if not self.alpha > 0:
            raise ValueError(
                "ALR needs alpha > 0: an empty cell would have log-probability -inf"
            )
        incidence, y_idx = self._fit_generative(X, y)
        n_classes = len(self.classes_)

        # Weight group of every table row
        if self.weight_level == 1:
            self._groups = 1 + self._row_join  # the prior row has join -1
        else:
            self._groups = np.arange(self._n_rows)
        self._n_columns = n_classes if self.weight_level == 3 else 1
        self.n_weights_ = (int(self._groups.max()) + 1) * self._n_columns

        Y = np.eye(n_classes)[y_idx]
        self.learned_weights_, self.n_iter_ = _fit_weights(
            lambda theta: self._nll_and_grad(theta, incidence, Y),
            np.ones(self.n_weights_),
            self.l2_reg,
            self.max_iter,
            tol=self.tol,
        )
        return self

    def _nll_and_grad(self, theta, incidence, Y):
        """Conditional negative log-likelihood and its gradient."""
        weights = theta.reshape(-1, self._n_columns)
        scores = incidence @ (weights[self._groups] * self._log_table)
        log_norm = logsumexp(scores, axis=1, keepdims=True)
        nll = -np.sum((scores - log_norm) * Y)
        # dNLL/dweight of a row: its log-probability times the residuals of
        # the samples that fall in it
        drow = (incidence.T @ (np.exp(scores - log_norm) - Y)) * self._log_table
        if self._n_columns == 1:
            drow = drow.sum(axis=1, keepdims=True)
        grad = np.column_stack(
            [
                np.bincount(self._groups, weights=col, minlength=len(weights))
                for col in drow.T
            ]
        )
        return nll, grad.ravel()

    def _table(self):
        weights = self.learned_weights_.reshape(-1, self._n_columns)[self._groups]
        return weights * self._log_table

    def predict_log_proba(self, X):
        check_is_fitted(self, attributes=["classes_", "joins_", "learned_weights_"])
        return super().predict_log_proba(X)
