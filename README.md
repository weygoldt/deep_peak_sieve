# Deep Peak Sieve

**Deep Peak Sieve** is a peak detector for 1D signals (on steroids 💊).

---

## Overview 🔎

This project aims to provide a peak detection algorithm that is robust to noise and applicable across various domains. The core idea is to combine the accuracy of supervised deep learning with strategies that counteract biases from training data and avoid hand-tuned heuristics. Instead, the pipeline relies on learned feature extraction directly from the data.

### Pipeline Summary

1. **Preprocessing**  
   Filter and smooth raw data to remove noise and artifacts.

2. **Feature Extraction**  
   Detect peaks on a per-channel basis using a low amplitude threshold and temporal constraints.

3. **Labeling**  
   A latent space is created and sampled (randomly or based on its structure), and a mini GUI is provided to label peaks as either valid or noise.

4. **Classification**  
   Train a classifier on the labeled peaks to distinguish signal from noise.

5. **Active Learning**  
   Use the trained classifier to label all detected peaks and iteratively refine the model. Optionally, a human can relabel ambiguous cases.

> **NOTE:** This project is still in very early stages and is not yet fully
> functional. The goal is to create a robust and flexible peak detection
> pipeline that can be applied to various datasets and domains. At this point,
> the most useful application is to classify electric eel discharges from
> multi-channel electric recordings as those produced by the
> [TeeGrid](https://github.com/janscience/TeeGrid).

---

## Quickstart 🚀

This project is not yet packaged for PyPI, so manual installation is required:

```bash
git clone https://github.com/weygoldt/deep_peak_sieve
cd deep_peak_sieve
pip install -r requirements.txt  # install dependencies
pip install -e .                 # install in editable mode
```

---

## Workflow

### 1. Detect Peaks

- Use a low amplitude threshold.
- Do not distinguish between positive/negative peaks.
- Apply a minimum temporal distance between peaks.
- Save peak archetypes, start/stop indices, and paths in `.npz` files (one per dataset file).

```bash
collect_peaks --help
collect_peaks /path/to/dataset -vvv
```

---

### 2. Sample Peaks

- Currently: random sampling.
- Future plans: sample in latent space using PCA, UMAP, or a VAE.

Creates a `<dataset_name>_samples.json` file containing:
- Peak indices
- Path to each corresponding `.npz` file

```bash
sample_peaks --help
sample_peaks /path/to/dataset -n 100 -vvv
```

---

### 3. Label Peaks

- Review each peak using a simple `matplotlib` GUI.
- Label keys:
  - `t`: valid peak (true)
  - `f`: noise (false)
  - `c`: correct previous label
- Labels are saved under the `labels` key in the `.npz` file.

```bash
label_peaks --help
label_peaks /path/to/json_file.json -vvv
```

You can label ~1000 peaks in ~15–20 minutes, which is usually enough for a decent classifier.

---

### 4. Train a Classifier

- Uses [InceptionTime](https://arxiv.org/abs/1909.04939), a fast and accurate time-series classifier.
- Convolutional layers extract features from the labeled time series.

```bash
train_classifier --help
train_classifier /path/to/json_file.json -vvv
```

Start TensorBoard to monitor training:

```bash
tensorboard --logdir .
# Open http://localhost:6006 in your browser
```

---

### 5. Classify All Detected Peaks

- Classifier labels each peak as signal or noise.
- Saves results in a new `.npz` file under the key `predicted_labels`.

```bash
classify_peaks --help
classify_peaks /path/to/dataset_peaks -vvv
```

---

## TODO ✅

- [ ] Implement active learnign. Currently, we just label -> train -> classify. For that, checkout python package [baal](https://github.com/baal-org/baal?tab=readme-ov-file)
- [ ] Implement the latent space sampling strategy, currently we just do stratified random sampling.
- [ ] Generalize to mono- and polyphasic peaks  
      _Currently, only monophonic peaks are well-supported. Polyphasic peaks require handling the order of positive and negative excursions (see `deep_peak_sieve/prepro/collect_peaks.py`)._

## DONE 🎉

- [x] Make smoothing and resampling configurable via CLI
- [x] Set smoothing window by time instead of samples
- [x] Resample all inputs to the same sampling rate
- [x] Fix logger configurator (issue with `src/` directory structures)

---

## Other Notes

**GitHub Organization**: [OpenEfish](https://github.com/OpenEfish)

### Pulse Detection

- Low-threshold detection with a fixed amplitude threshold.
- Optional peak/trough sorting heuristic (see work by Liz).
- Build a generalized signal-vs-noise classifier:
  - Record "empty" data without fish.
  - Train a model to recognize true pulses vs background noise.
  - Explore anomaly detection techniques.
- Normalize pulse polarity (flip if necessary).

### Pulse Filtering

- Use power spectrum or trained noise classifier to clean the signal.

### Feature Extraction

- Learn representations via:
  - VAE
  - UMAP / PCA
- Experiment with combinations:
  - VAE only
  - VAE + PCA
  - VAE + UMAP
  - VAE + UMAP + PCA

### Clustering

- Use unsupervised methods like HDBSCAN or neural clustering approaches.

> **WOW!** 🎉
