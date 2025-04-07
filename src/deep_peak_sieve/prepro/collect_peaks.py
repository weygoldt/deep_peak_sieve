from pathlib import Path
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from audioio.audioloader import AudioLoader
from deep_peak_sieve.utils.loggers import get_logger, get_progress, configure_logging
from IPython import embed

log = get_logger(__name__)


def collect_peaks(path: Path, buffersize: int = 60 * 1):
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")
    elif path.is_dir():
        log.info("Dataset is a directory, processing each file.")
        files = sorted(list(path.glob("*.wav")))
        files = [str(file) for file in files]
        data = AudioLoader(files)
    else:
        log.info("Dataset is a single file.")
        data = AudioLoader(str(path))

    blocksize = int(np.ceil(data.rate * buffersize))
    overlap = int(np.ceil(blocksize // 2))
    log.info(
        f"Buffer size: {buffersize} seconds\nBlock size: {blocksize} samples\nsampling rate: {data.rate} Hz\n{data.channels} channels\n"
    )

    for i, block in enumerate(data.blocks(block_size=blocksize, noverlap=overlap)):
        if data.channels == 1:
            block = np.expand_dims(block, axis=1)

        std = np.std(block)

        peaks = [
            find_peaks(np.abs(block[:, ch]), prominence=std * 3)[0]
            for ch in range(data.channels)
        ]

        fig, axs = plt.subplots(
            data.channels,
            1,
            figsize=(8, 4),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        for ch in range(data.channels):
            axs[ch].plot(block[:, ch], label=f"Channel {ch}")
            axs[ch].plot(peaks[ch], block[peaks[ch], ch], "x", label="Peaks")
        plt.show()


def main():
    configure_logging(verbosity=10)
    path = Path(__file__).parent.parent.parent.parent / "testdata"
    collect_peaks(path=path)
