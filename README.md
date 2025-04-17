# Deep Peak Sieve

Deep Peak Sieve is a peak detector for 1D signals (on steroids 💊).

## Overview 🔎

The goal of this project is to provide a peak detection algorithm that is
robust to noise and can be used in a variety of applications. It is to achieve
the accuracy of supervised deep learning while purposely working against
missing interesting events due to biased training data. And to prevent hand
tuned features as heuristics and instead rely on learned feature extraction
from the data to make decisions.

The pipeline consists of several steps:

1. **Preprocessing**: The raw data is preprocessed to remove noise and artifacts by filtering and smoothing.
1. **Feature Extraction**: Peaks are detected on a per-channel basis using a low amplitude threshold and temporal constraints.
1. **Embedding**: The detected peaks are embedded in a latent space using a Variational Autoencoder (VAE).
1. **Labeling**: The latent space is regularly sampled and a mini GUI is provided to label the detected peaks as either noise or valid peaks. Options include regular sampling, random sampling or sampling based on the structure of the latent space (such as biasing towards high density or low density regions).
1. **Classification**: A simple classifier is trained on the labeled data to classify the detected peaks as either noise or valid peaks.
1. **Active learning**: The classifier is used to classify all detected peaks and the results are used to improve the model by retraining it on the newly classified data. This can be done in an iterative fashion to improve the model over time. Alternatively, a human can review and label previously unlabeled and hard to classify data points to improve the model.

## Quickstart 🚀

This project is not nicely packaged yet, so you have to install it manually:

```bash
git clone https://github.com/weygoldt/deep_peak_sieve
cd deep_peak_sieve
pip install -r requirements.txt # install the dependencies
pip install -e . # install the package in editable mode
```

A general workflow includes the following steps:

1. Detecting peaks in a signal. This is kept very simple here:
   - Use a low amplitude threshold to detect peaks.
   - Do not distinguish between positive and negative "peaks".
   - Use temporal constraints to filter out false positives (two peaks need to be at least x ms apart).
   - An archetype of each peak (mean across all channels), start and stop index in the dataset, and path to the dataset are saved to a `.npz` file. For each file in the dataset, one `.npz` file is created.

To start peak detection, run:

```bash
collect_peaks --help # show help
collect_peaks /path/to/dataset -vvv
```

2. Sampling a subset of all detected peaks for labeling.
   - Currently, this is simple random sampling. But in the future, I want to
     embed the peaks in a latent space and sample the latent space in a
     structured way (e.g. bias towards high density regions or low density
     regions). To start, we should implement this first via PCA, then UMAP and
     finally a VAE to compare the results.
   - For sampling, the application creates a `<dataset_name>_samples.json`
     file. This file contains the indices for the chosen peaks and the path to
     each `.npz` file it chose the indices from. So this is just a "pointer".

To sample a subset and create the JSON file, run:

```bash
sample_peaks --help # show help
sample_peaks /path/to/dataset -n 100 -vvv
```

3. Labeling peaks to train a classifier

   - During labeling, we will now go through all the samples in the JSON file
     and show the user the archetype of the peak and the raw data of the peak.
   - The user can then label the peak as either noise or valid peak. The labeling
     is done via a `matplotlib` GUI that shows the archetype and the raw data of
     the peak. The user can then press `0` to label the peak as noise or `1` to
     label it as a valid peak (or `-1` as "no label"). The labeled peaks are added
     as an additional array to the `.npz` file under the key `labels`.

1. Training a classifier on the labeled data. To be implemented.

1. Inference

## TODO ✅

- [ ] Make smoothing and resampling configurable over cli
- [ ] Set the smothing window by time not by samples
- [ ] Resample everything to the same sampling rate
- [ ] Generalize to mono- and polyphasic peaks. Currently, only monophonic peaks work well because the sign of them can be easily flipped using the maximum. For polyphasic peaks, we need to consider the order of negative and positive excursions. To be implemented in `deep_peak_sieve/prepro/collect_peaks.py`.

- [x] Fix the logger configurator (does not work with pkgs that have a src/ folder for some reason)

# Other Notes

GITHUB Orga: OpenEfish

- Pulse detection:

  - Low threshold peak detection but we need a hard amplitude threshold in any case
  - Peak sorting (some kind of peak/trough sorting heuristic, see Liz)
  - We need a generalized signal vs noise classifier: Redord recordings without any fish and train a model onto that, keyword "anomaly detection"
    - Write a generalized pulse detector and run it on data without any pulses and train a model that classifies noise yes/no to filter real pulses
  - Bring to same sign (flip if needed)

- Pulse filtering

  - E.g. with powerspectrum, or with the generalized noise detector

- Smooth (e.g. spline) interpolation

- Feature extraction

  - VAE to learn a meaningful representation of the data
  - UMAP/PCA to visualize the latent space or further reduce the dimensionality
  - NOTE: We need to test: Only VAE, Only UMAP, VAE + UMAP, VAE + PCA, VAE + UMAP + PCA and so on

- Clustering

  - HDBSCAN or other neural clustering

## Overpowered Pulsedetector

- Detect pulses with low absolute threshold peak detection
- For each pulse, save head to tail approx, raw pulse on all channels, timepoint, index, file, etc.
- Using the HTT approx, embed pulses in feature space with a VAE
- Regularly sample the latent space for e.g. 1000 points
- Mini MPL gui where you can press 0/1 to classify the pulses as pulses or noise. Should save pulses in a list of some sort that is saved to disk for each label. Should also have the option to correct previous mistakes, e.g. print the index of the pulse to the terminal or in mpl title.
- Use labels to train a simple classifier on the HTTapprox or even the raw pulse on each channel.
- Classify all detected pulses and continue working with them in other analysis steps.
  -> WOW!
