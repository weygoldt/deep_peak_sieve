# Plan

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
