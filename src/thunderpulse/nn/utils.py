import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from thunderpulse.models.base import BaseVAE


def generate_synthetic_peaks(
    num_samples: int = 1000,
    length: int = 1024,
    bullshit_frac: float = 0.2,
    noise_std: float = 1.0,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Generate synthetic 1D signals of shape [num_samples, 1, length].
    A fraction of these signals are "bullshit" random noise, and
    the rest are Gaussian pulses.

    Parameters
    ----------
    num_samples : int, optional
        Total number of signals to generate. Default is 1000.
    length : int, optional
        Length of each 1D signal. Default is 1024.
    bullshit_frac : float, optional
        Fraction of signals that will be pure noise (label=0).
        The remaining signals will be Gaussian pulses (label=1).
        Default is 0.2 (i.e., 20% noise).
    noise_std : float, optional
        Standard deviation for the random noise signals. Larger values
        make it harder to distinguish pulses from noise. Default is 1.0.

    Returns
    -------
    data : torch.Tensor
        A tensor of shape (num_samples, 1, length) containing all generated signals.
    labels : np.ndarray
        An integer array of shape (num_samples,) with 0 or 1 indicating
        whether a sample is noise or a pulse.

    Notes
    -----
    - Pulse amplitude is drawn from [0.5, 1.5].
    - Pulse center is within 20% to 80% of the signal length.
    - Pulse width is between 1% to 5% of the signal length.
    - Noise signals are sampled from N(0, noise_std^2).
    """
    data = torch.zeros(num_samples, 1, length)
    labels = np.zeros(num_samples, dtype=int)
    xs = np.arange(length)

    amplitude_range = (0.5, 1.5)
    center_range = (0.2 * length, 0.8 * length)

    cluster_mean_peak_widths = np.array([0.01, 0.5])
    cluster_std_peak_widths = np.array([0.01, 0.01])

    for i in range(num_samples):
        if np.random.rand() < bullshit_frac:
            # Draw from first cluster (noise)
            amplitude = np.random.uniform(*amplitude_range)
            center = np.random.uniform(*center_range)
            width = np.random.normal(
                loc=cluster_mean_peak_widths[0],
                scale=cluster_std_peak_widths[0],
                size=1,
            )[0]
            width = width * length
            # width = np.clip(width, 0.01 * length, 0.05 * length)
            # Create a Gaussian pulse
            gauss = amplitude * np.exp(-0.5 * ((xs - center) / width) ** 2)
            # Add noise
            noise = np.random.normal(0, noise_std, length)
            gauss += noise
            data[i, 0] = torch.from_numpy(gauss).float()
            labels[i] = 0
        else:
            amplitude = np.random.uniform(0.5, 1.5)
            center = np.random.randint(int(0.2 * length), int(0.8 * length))
            width = np.random.normal(
                loc=cluster_mean_peak_widths[1],
                scale=cluster_std_peak_widths[1],
                size=1,
            )[0]
            width = width * length
            # width = np.clip(width, 0.01 * length, 0.05 * length)
            # Create a Gaussian pulse
            gauss = amplitude * np.exp(-0.5 * ((xs - center) / width) ** 2)
            noise = np.random.normal(0, noise_std, length)
            gauss += noise
            data[i, 0] = torch.from_numpy(gauss).float()
            labels[i] = 1

    return data, labels


def plot_signals(data: torch.Tensor, labels: np.ndarray, num_samples: int = 5):
    """
    Plot a few samples of the generated signals.

    Parameters
    ----------
    data : torch.Tensor
        A tensor of shape (num_samples, 1, length) containing all generated signals.
    labels : np.ndarray
        An integer array of shape (num_samples,) with 0 or 1 indicating
        whether a sample is noise or a pulse.
    num_samples : int, optional
        Number of samples to plot. Default is 5.
    """
    n_classes = len(np.unique(labels))
    colors = sns.color_palette("deep", n_classes)
    fig, axs = plt.subplots(
        2, n_classes, constrained_layout=True, figsize=(10, 5)
    )
    axs = axs.flatten()
    for i in range(num_samples):
        ax = axs[labels[i]]
        ax.plot(data[i, 0].numpy(), color=colors[labels[i]], alpha=0.5)
        ax.set_xlim(0, data.shape[2])
        ax.set_ylim(-3, 3)
        ax.set_title(f"Label: {labels[i]}")
        ax.grid()

    scaler = StandardScaler()
    np_data = data.numpy().reshape(data.shape[0], -1)
    np_data = scaler.fit_transform(np_data)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(np_data)
    umap = UMAP(n_components=2)
    data_umap = umap.fit_transform(np_data)

    axs[2].scatter(
        data_pca[:, 0], data_pca[:, 1], c=labels, cmap="viridis", s=5
    )
    axs[2].set_title("PCA")
    axs[3].scatter(
        data_umap[:, 0], data_umap[:, 1], c=labels, cmap="viridis", s=5
    )
    axs[3].set_title("UMAP")
    plt.show()


if __name__ == "__main__":
    # Generate synthetic data
    num_samples = 5000
    length = 1024
    bullshit_frac = 0.2
    noise_std = 0.01

    data, labels = generate_synthetic_peaks(
        num_samples, length, bullshit_frac, noise_std
    )

    # Plot a few samples
    plot_signals(data, labels, num_samples=100)


def visualize_latent_space(
    vae_model: BaseVAE,
    data: torch.Tensor,
    labels: np.ndarray | torch.Tensor | None = None,
    method: str = "pca",
    use_mean: bool = True,
    n_samples: int = 1000,
) -> None:
    """
    Visualize the latent space of a trained VAE-like model in 2D.

    Parameters
    ----------
    vae_model : BaseVAE
        A trained model implementing the BaseVAE interface (must define .encode()).
    data : torch.Tensor
        Input data of shape (N, in_channels, ...). A batch of samples to embed.
    labels : Optional[Union[np.ndarray, torch.Tensor]], optional
        Labels or classes used for coloring in the scatter plot, by default None.
    method : str, optional
        Reduction method: 'pca' or 'umap'. By default 'pca'.
        If 'umap' is chosen, it requires `pip install umap-learn`.
    use_mean : bool, optional
        If True, use the encoder’s mean (mu) as the latent code.
        If False, sample z via reparameterization (mu, log_var).
        Default is True (common for visualization).
    n_samples : int, optional
        How many data points to visualize (taken from the start of `data`).
        Default is 1000.

    Returns
    -------
    None
        Shows a matplotlib scatter plot of the 2D latent space.
    """
    vae_model.eval()

    # Subsample data if it's large
    data = data[:n_samples]

    # Move data to the same device as vae_model (if it has parameters on GPU)
    device = next(vae_model.parameters()).device
    data = data.to(device)

    with torch.no_grad():
        # Encode to get mu, log_var
        mu, log_var = vae_model.encode(data)
        # Decide whether to use mu or reparameterized z
        if use_mean:
            z = mu
        else:
            z = vae_model.reparameterize(mu, log_var)

    # Convert to CPU numpy for plotting
    z_np = z.cpu().numpy()

    # 2D dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
        coords_2d = reducer.fit_transform(z_np)
    elif method.lower() == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(z_np)
        except ImportError:
            print("UMAP not installed. Falling back to PCA.")
            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(z_np)
    else:
        raise ValueError("Method must be 'pca' or 'umap'.")

    # Plot
    fig, ax = plt.subplots(constrained_layout=True)
    if labels is not None:
        # Convert labels to numpy if it's a tensor
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        labels = labels[: len(coords_2d)]  # in case we truncated data
        sc = ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1], c=labels, s=10, alpha=0.7
        )
    else:
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=10, alpha=0.7)

    ax.set_title(f"Latent Space ({method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.show()


def plot_reconstructions(
    vae_model: BaseVAE,
    data: torch.Tensor,
    n_samples: int = 10,
    random: bool = True,
) -> None:
    """
    Plot original vs reconstructed samples from a trained BaseVAE model.
    For 1D signals, each subplot overlays the original and reconstructed signal.

    Parameters
    ----------
    vae_model : BaseVAE
        A trained model implementing the BaseVAE interface.
    data : torch.Tensor
        Input data of shape (N, C, L) for 1D signals (or shape (N, C, H, W) for images).
    n_samples : int, optional
        Number of samples to plot, by default 5.
    random : bool, optional
        If True, pick samples at random from `data`. Otherwise, use the first `n_samples`.
        Default is True.

    Returns
    -------
    None
        Displays a matplotlib figure with the requested reconstructions.

    Notes
    -----
    - For 1D signals, we assume data in (N, 1, length). The function overlays two line plots.
    - For 2D images, you can adapt the plotting logic to use `imshow()` instead.
    """
    vae_model.eval()

    # Pick which samples to visualize
    num_total = data.size(0)
    if random:
        indices = np.random.choice(
            num_total, size=min(n_samples, num_total), replace=False
        )
    else:
        indices = np.arange(min(n_samples, num_total))

    # Move data/model to same device if needed
    device = next(vae_model.parameters()).device
    selected = data[indices].to(device)

    # Forward pass through VAE
    with torch.no_grad():
        # forward(...) typically returns (recon, input, mu, log_var)
        outputs = vae_model(selected)
        # The reconstruction is often the first element in that tuple
        # Adjust if your model returns something different
        if isinstance(outputs, (tuple, list)):
            recons = outputs[0]
        else:
            # If your model just returns reconstructions
            recons = outputs

    # Move to CPU for plotting
    recons = recons.cpu()
    selected = selected.cpu()

    # Plot each example
    ncols = int(np.ceil(np.sqrt(n_samples)))
    nrows = int(np.ceil(n_samples / ncols))
    figgrid = (nrows, ncols)
    fig, axes = plt.subplots(*figgrid, figsize=(8, 3 * len(indices)))
    axes = axes.flatten()
    # If only 1 sample, axes won't be an array
    if len(indices) == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]
        # For 1D signals: shape = (1, length). Let's flatten for plotting
        original = selected[i].numpy().flatten()
        reconstruction = recons[i].numpy().flatten()

        ax.plot(original, label="Original", alpha=0.7)
        ax.plot(reconstruction, label="Reconstruction", alpha=0.7)
        ax.set_title(f"Sample idx={idx}")
        ax.legend()

    plt.tight_layout()
    plt.show()
