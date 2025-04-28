import numpy as np
from umap import UMAP

from thunderpulse.embedding.base import BaseEmbedder


class UmapEmbedder(BaseEmbedder):
    """UMAP embedder for 1D signals."""

    def __init__(self, model_name: str, model_path: str) -> None:
        super().__init__(model_name, model_path)
        self.model = UMAP()

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Z-score the data before embedding.

        Parameters:
        ----------
        data: np.ndarray
            1D signals to be embedded.

        Returns:
        -------
        np.ndarray
            Preprocessed data.
        """

        return super().preprocess(data)
