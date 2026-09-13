from __future__ import annotations

from abc import ABC, abstractmethod
import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


class BaseReducer(ABC):
    """Common plug-in interface for dimensionality reduction experiments."""

    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BaseReducer":
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(X, y).transform(X)


class IdentityReducer(BaseReducer):
    name = "none"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IdentityReducer":
        self.n_components_ = X.shape[1]
        self.explained_variance_ = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)


class PCAReducer(BaseReducer):
    def __init__(self, n_components: int | float, name: str):
        self.name = name
        self.n_components = n_components
        self.model: PCA | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PCAReducer":
        components = self.n_components
        if isinstance(components, int):
            components = min(components, X.shape[0] - 1, X.shape[1])
        self.model = PCA(n_components=components, svd_solver="full")
        self.model.fit(X)
        self.n_components_ = int(self.model.n_components_)
        self.explained_variance_ = float(self.model.explained_variance_ratio_.sum())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA reducer is not fitted")
        return self.model.transform(X)


class SVDReducer(BaseReducer):
    def __init__(self, n_components: int, seed: int):
        self.name = f"svd{n_components}"
        self.n_components = n_components
        self.seed = seed
        self.model: TruncatedSVD | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SVDReducer":
        components = min(self.n_components, X.shape[0] - 1, X.shape[1] - 1)
        self.model = TruncatedSVD(n_components=max(1, components), random_state=self.seed)
        self.model.fit(X)
        self.n_components_ = int(self.model.n_components)
        self.explained_variance_ = float(self.model.explained_variance_ratio_.sum())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SVD reducer is not fitted")
        return self.model.transform(X)


class PLSReducer(BaseReducer):
    def __init__(self, n_components: int):
        self.name = f"pls{n_components}"
        self.n_components = n_components
        self.model: PLSRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PLSReducer":
        if y is None:
            raise ValueError("PLS requires supervised targets")
        components = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.model = PLSRegression(n_components=max(1, components), scale=False)
        self.model.fit(X, y)
        self.n_components_ = int(components)
        self.explained_variance_ = np.nan
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PLS reducer is not fitted")
        return self.model.transform(X)


class AutoencoderReducer(BaseReducer):
    """Small sklearn autoencoder with an exposed bottleneck representation."""

    def __init__(self, bottleneck: int, seed: int):
        self.name = f"autoencoder{bottleneck}"
        self.bottleneck = bottleneck
        self.seed = seed
        self.model: MLPRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AutoencoderReducer":
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, self.bottleneck, 64),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=min(64, len(X)),
            learning_rate_init=0.001,
            max_iter=180,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=18,
            random_state=self.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.model.fit(X, X)
        self.n_components_ = self.bottleneck
        reconstruction = self.model.predict(X)
        denominator = float(np.var(X)) or 1.0
        self.explained_variance_ = 1.0 - float(np.mean((X - reconstruction) ** 2)) / denominator
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Autoencoder reducer is not fitted")
        hidden = np.asarray(X, dtype=float)
        # The bottleneck is the output of the second hidden layer.
        for layer in range(2):
            hidden = hidden @ self.model.coefs_[layer] + self.model.intercepts_[layer]
            hidden = np.maximum(hidden, 0.0)
        return hidden


def make_reducer(name: str, seed: int = 42) -> BaseReducer:
    registry = {
        "none": lambda: IdentityReducer(),
        "pca50": lambda: PCAReducer(50, "pca50"),
        "pca95": lambda: PCAReducer(0.95, "pca95"),
        "svd50": lambda: SVDReducer(50, seed),
        "pls20": lambda: PLSReducer(20),
        "autoencoder16": lambda: AutoencoderReducer(16, seed),
    }
    if name not in registry:
        raise ValueError(f"Unknown reducer: {name}. Available: {sorted(registry)}")
    return registry[name]()


def available_reducers() -> list[str]:
    return ["none", "pca50", "pca95", "svd50", "pls20", "autoencoder16"]
