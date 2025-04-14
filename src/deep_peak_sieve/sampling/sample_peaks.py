
from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import typer 
from deep_peak_sieve.utils.loggers import get_logger, get_progress, configure_logging

app = typer.Typer(pretty_exceptions_show_locals=False)
log = get_logger(__name__)


class BaseSampler:
    """Base class for different peak samplers for labeling peaks."""

    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def sample(self) -> np.ndarray:
        pass

    @abstractmethod
    def plot(self) -> None:
        pass


def main(path):
    pass
    
if __name__ == "__main__":
