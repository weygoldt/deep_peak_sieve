"""Base class for all embedder classes."""

from abc import ABC, abstractmethod

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP


class BaseEmbedder(ABC):
    """Base class for all embedder classes."""

    def __init__(self, model_name: str, model_path: str) -> None:
        self.model_name = model_name
        self.model_path = model_path

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit the embedder to the data.

        Parameters
        ----------
        data: np.ndarray
            1D signals to be embedded.
        """
        pass

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load the embedder from a file.

        Parameters
        ----------
        model_path: str
            Path to load the embedder from.
        """
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> list:
        """Embed 1D signals into a latent space.

        Parameters
        ----------
        data: np.ndarray
            1D signals to be embedded.

        Returns
        -------
        np.ndarray
            Embedded signals.
        """
        pass


class UmapEmbedder(BaseEmbedder):
    """UMAP embedder for 1D signals."""

    def __init__(self, model_name: str, model_path: str) -> None:
        super().__init__(model_name, model_path)
        self.model = UMAP()
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.scaler_path = model_path.replace(".joblib", "_scaler.joblib")

    def fit(self, data: np.ndarray) -> None:
        """Fit the embedder to the data.

        Parameters
        ----------
        data: np.ndarray
            1D signals to be embedded.
        """
        reshaped_data = data.reshape(data.shape[0], -1)
        scaled_data = self.scaler.fit_transform(reshaped_data)
        data = scaled_data.reshape(data.shape[0], data.shape[1], data.shape[2])
        self.model.fit(data)

        # save model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load(self, model_path: str) -> None:
        """Load the embedder from a file.

        Parameters
        ----------
        model_path: str
            Path to load the embedder from.
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict(self, data: np.ndarray) -> list:
        """Embed 1D signals into a latent space.

        Parameters
        ----------
        data: np.ndarray
            1D signals to be embedded.

        Returns
        -------
        np.ndarray
            Embedded signals.
        """
        reshaped_data = data.reshape(data.shape[0], -1)
        scaled_data = self.scaler.transform(reshaped_data)
        data = scaled_data.reshape(data.shape[0], data.shape[1], data.shape[2])
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains NaN or infinite values.")
        return self.model.transform(data)


class UmapEmbedder(BaseEmbedder):
    """UMAP embedder for 1D signals."""

    def __init__(self, model_name: str, model_path: str) -> None:
        super().__init__(model_name, model_path)
        self.model = UMAP()

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Z-score the data before embedding.

        Parameters
        ----------
        data: np.ndarray
            1D signals to be embedded.

        Returns
        -------
        np.ndarray
            Preprocessed data.
        """
        return super().preprocess(data)
