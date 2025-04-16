import logging
import pkgutil
import importlib
from pathlib import Path
from rich.logging import RichHandler
from rich.progress import (
    SpinnerColumn,
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def discover_package_modules():
    """Dynamically discovers all modules within the current package."""
    package_name = Path(
        __file__
    ).parent.parent.parent.parent.name  # get the package name
    package = importlib.import_module(package_name)
    # print(f"Package name: {package_name}")
    package_path = Path(package.__file__).parent
    modules = []

    for _, module_name, _ in pkgutil.walk_packages(
        [str(package_path)], prefix=f"{package_name}."
    ):
        modules.append(module_name)

    return modules


def configure_logging(
    verbosity: int, log_to_file: bool = False, log_file="wavetracker.log"
):
    """Configures logging globally, ensuring third-party libraries remain quiet."""
    level = logging.WARNING  # Default level

    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
        print("Debugging enabled.")
    else:
        print("Invalid verbosity level. Defaulting to WARNING.")

    handlers = [RichHandler()]

    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    # Configure the root logger (keeps third-party libraries quiet)
    logging.basicConfig(
        level=logging.WARNING,  # Keep root logger at WARNING to prevent spam
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,  # Ensures all loggers adopt the new settings
    )

    # Dynamically discover and set logging level only for the user's package
    for package in discover_package_modules():
        logging.getLogger(package).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Returns a logger with RichHandler for consistent formatting."""
    logger = logging.getLogger(name)
    return logger


def get_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description:<30}"),  # Task description
        BarColumn(),  # Visual progress bar
        TextColumn(" | Completed: "),
        MofNCompleteColumn(),  # M/N complete count
        TextColumn(" | Percent: "),
        TextColumn(
            "[progress.percentage]{task.percentage:>3.0f}%"
        ),  # Percentage complete
        TextColumn(" | Time Elapsed: "),
        TimeElapsedColumn(),  # Time elapsed
        TextColumn(" | ETA: "),
        TimeRemainingColumn(),  # Estimated time remaining
        expand=True,
    )
