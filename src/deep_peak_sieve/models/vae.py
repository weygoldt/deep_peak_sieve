import matplotlib.pyplot as plt
from typing import Any, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import LowRankMultivariateNormal, Normal
import lightning as L
from lightning.pytorch.callbacks import StochasticWeightAveraging
import numpy as np
from sklearn.preprocessing import StandardScaler
from torchinfo import summary


class BaseLatentDistribution(nn.Module, ABC):
    """
    Abstract base class for a latent distribution module.
    Each subclass implements the forward pass (to get distribution parameters)
    and any sampling (e.g. reparameterization) needed.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, x) -> Any:
        """
        Produce distribution parameters from an encoder representation `x`.
        Returns: parameters in a tuple (or dictionary)
        """
        pass

    @abstractmethod
    def sample(self, dist_params) -> torch.Tensor:
        """
        Sample from the distribution given dist_params.
        Typically returns z (and optionally can return the distribution object).
        """
        pass

    @abstractmethod
    def kl_divergence(self, dist_params: Any) -> torch.Tensor:
        """
        Computes KL(q(z|x) || p(z)), where p(z) = N(0, I) by default.

        Should return a scalar (batch-averaged KL) or a shape [batch_size] you
        can average externally. Conventionally we return a single scalar.
        """
        pass


class DiagonalGaussian(BaseLatentDistribution):
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__(latent_dim)
        # For example, define the linear layers that map
        # from an encoder MLP output to mu and logvar.
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, hidden_dim]
        returns (mu, logvar)
        """
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return (mu, logvar)

    def sample(self, dist_params) -> torch.Tensor:
        """
        dist_params is (mu, logvar)
        We do the usual reparameterization trick:
            std = exp(0.5 * logvar)
            z = mu + std * eps
        """
        (mu, logvar) = dist_params
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def kl_divergence(self, dist_params):
        """
        KL( q(z|x) || p(z) ) for diagonal Gaussian vs. N(0,I).
        Returns a scalar (mean over batch).
        """
        (mu, logvar) = dist_params
        # KL per dimension
        # shape: [batch_size, latent_dim]
        kl_per_dim = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)
        # sum over latent_dim -> [batch_size]
        kl_per_sample = kl_per_dim.sum(dim=1)
        # average over batch -> scalar
        return kl_per_sample.mean()


class LowRankGaussian(BaseLatentDistribution):
    def __init__(self, latent_dim, hidden_dim=128, rank=2):
        super().__init__(latent_dim)
        self.rank = rank

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_u = nn.Linear(hidden_dim, latent_dim * rank)
        self.fc_d = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, hidden_dim]
        returns (mu, u, d)
        """
        mu = self.fc_mu(x)  # shape [N, latent_dim]
        u_vec = self.fc_u(x)  # shape [N, latent_dim*rank]
        batch_size = x.size(0)
        u = u_vec.view(batch_size, self.latent_dim, self.rank)
        d = torch.exp(self.fc_d(x))  # ensure positivity
        return (mu, u, d)

    def sample(self, dist_params) -> torch.Tensor:
        (mu, u, d) = dist_params
        # Build a distribution object for each item in the batch
        # Then reparameterize
        dist = LowRankMultivariateNormal(mu, u, d)
        z = dist.rsample()
        return z

    def kl_divergence(self, dist_params) -> torch.Tensor:
        """
        KL( q(z|x) || p(z) ) for LowRankGaussian vs. N(0, I),
        computed in closed form for each sample. Returns the
        mean KL over the batch.

        We do the standard formula:
            KL = 0.5 * [ trace(Sigma) + mu^T mu - d - log(det(Sigma)) ]
        where Sigma = diag(d) + U U^T.
        log(det(Sigma)) is computed using the matrix determinant lemma.

        dist_params = (mu, u, d) with shapes:
          - mu: [N, latent_dim]
          - u:  [N, latent_dim, rank]
          - d:  [N, latent_dim]
        """

        (mu, u, diag_d) = dist_params
        batch_size, latent_dim = mu.shape
        rank = self.rank

        # We'll accumulate KL for each sample, then average
        kl_vals = []

        for i in range(batch_size):
            mu_i = mu[i]  # [latent_dim]
            u_i = u[i]  # [latent_dim, rank]
            d_i = diag_d[i]  # [latent_dim]

            # 1) trace(Sigma) = sum(d_i) + sum of squares of U
            trace_Sigma = d_i.sum() + (u_i**2).sum()

            # 2) mu^T mu
            mu_sq = (mu_i**2).sum()

            # 3) logdet(Sigma) via matrix determinant lemma
            #    log det(diag(d_i) + U U^T) = sum(log d_i) + log det(I_r + M^T M)
            #    where M = diag(1/sqrt(d_i)) * U, shape [latent_dim, rank]
            logdet_diag = torch.log(d_i).sum()  # sum_j log d_{i,j}

            # Build M = D^(-1/2)*U => shape [latent_dim, rank]
            M = u_i / torch.sqrt(d_i.unsqueeze(1))  # broadcast divide each column

            # M^T M is [rank, rank]
            MtM = M.transpose(0, 1) @ M  # or torch.matmul(M.t(), M)

            # eigenvalues of MtM
            # We'll assume MtM is symmetric/hermitian => use eigvalsh
            eigvals = torch.linalg.eigvalsh(MtM)  # shape [rank]

            # log det(I + MtM) = sum of log(1 + eigvals)
            logdet_I_MtM = torch.log1p(eigvals).sum()

            logdet_Sigma = logdet_diag + logdet_I_MtM

            # 4) Combine
            # KL_i = 0.5 [ trace(Sigma) + mu^T mu - latent_dim - log det(Sigma)]
            kl_i = 0.5 * (trace_Sigma + mu_sq - latent_dim - logdet_Sigma)

            kl_vals.append(kl_i)

        kl_vals = torch.stack(kl_vals, dim=0)  # [batch_size]
        return kl_vals.mean()


class Encoder(nn.Module):
    """
    Convolutional 1D Encoder with configurable layers.

    Parameters:
    -----------
    input_dim : int
        Length of the input signal.
    base_channels : int
        Number of channels in the first conv layer. (e.g. 8)
    num_downsamples : int
        How many times we do stride=2 in the encoder.
    hidden_dim : int
        Dimension of the final encoder output (default=128).
        This output is fed into the LatentDistribution.
    """

    def __init__(self, input_dim, base_channels, num_downsamples, hidden_dim=128):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # We'll do a sequence of Conv1d layers. Each time we "downsample" we use stride=2.
        # For simplicity, we'll multiply the channels by 2 each time we downsample.
        encoder_layers = []
        in_ch = 1
        out_ch = base_channels

        current_dim = input_dim

        for i in range(num_downsamples):
            encoder_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )
            encoder_layers.append(nn.ReLU())  # activation
            current_dim = (
                current_dim // 2
            )  # floor division if input_dim is multiple of 2
            in_ch = out_ch
            out_ch *= 2

        self.conv_encoder = nn.Sequential(*encoder_layers)

        # Flatten for linear layers
        final_channels = in_ch  # after finishing the loop
        self.final_channels = final_channels
        self.current_dim = current_dim  # store so the VAE can pass to Decoder

        # We'll do a small MLP that outputs a "hidden_dim"-size feature vector
        fc_in = final_channels * current_dim
        self.fc1 = nn.Linear(fc_in, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, hidden_dim)

    def forward(self, x):
        """
        Returns a hidden representation of size [batch_size, hidden_dim].
        """
        x = self.conv_encoder(x)  # [N, final_channels, current_dim]
        x = x.view(x.size(0), -1)  # flatten to [N, final_channels * current_dim]
        x = F.silu(self.fc1(x))  # [N, 512]
        x = F.silu(self.fc2(x))  # [N, 256]
        x = F.silu(self.fc3(x))  # [N, hidden_dim]
        return x


class Decoder(nn.Module):
    """
    Convolutional 1D Decoder with configurable layers.

    Parameters:
    -----------
    latent_dim : int
    base_channels : int
        Should match the one used in the encoder for symmetrical design.
    num_downsamples : int
        Must match encoder for symmetrical upsampling (mirroring the layers).
    final_channels : int
        The number of channels in the last conv layer of the encoder.
    final_spatial_dim : int
        The spatial dimension after the encoder (i.e. input_dim // (2^num_downsamples)).
    """

    def __init__(
        self,
        latent_dim,
        base_channels,
        num_downsamples,
        final_channels,
        final_spatial_dim,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_downsamples = num_downsamples
        self.final_channels = final_channels
        self.final_spatial_dim = final_spatial_dim

        hidden_dim = 512
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, final_channels * final_spatial_dim)

        decoder_layers = []
        in_ch = final_channels
        out_ch = in_ch // 2  # if we doubled channels during encoding, we'll halve now

        for i in range(num_downsamples):
            decoder_layers.append(
                nn.ConvTranspose1d(
                    in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            decoder_layers.append(nn.ReLU())

            in_ch = out_ch
            out_ch = max(out_ch // 2, 1)  # avoid going below 1

        self.conv_decoder = nn.Sequential(*decoder_layers)
        self.final_conv = nn.Conv1d(in_ch, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = F.silu(self.fc1(z))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = self.fc4(x)

        # Reshape to [N, final_channels, final_spatial_dim]
        x = x.view(x.size(0), self.final_channels, self.final_spatial_dim)

        x = self.conv_decoder(x)
        x = self.final_conv(x)
        return x


def build_latent_distribution(dist_type, latent_dim, hidden_dim=128, rank=2):
    """
    A simple factory function to build the chosen distribution class.
    """
    if dist_type == "diagonal":
        return DiagonalGaussian(latent_dim, hidden_dim=hidden_dim)
    elif dist_type == "lowrank":
        return LowRankGaussian(latent_dim, hidden_dim=hidden_dim, rank=rank)
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


class VAE(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        base_channels=8,
        num_downsamples=3,
        hidden_dim=128,
        latent_dim=64,
        dist_type="diagonal",
        rank=2,
    ):
        """
        VAE that uses:
          - Encoder to produce [batch_size, hidden_dim]
          - LatentDistribution (diagonal or low-rank) to get z
          - Decoder from z to reconstructed signals

        Args:
            input_dim (int): length of the 1D signal
            base_channels (int): #channels in the first conv layer
            num_downsamples (int): how many stride=2 layers
            hidden_dim (int): final encoder output dimension
            latent_dim (int): dimension of the latent variable
            dist_type (str): 'diagonal' or 'lowrank'
            rank (int): rank for low-rank distribution
        """
        super().__init__()

        # 1) Build the encoder
        self.encoder = Encoder(
            input_dim=input_dim,
            base_channels=base_channels,
            num_downsamples=num_downsamples,
            hidden_dim=hidden_dim,
        )

        # 2) Build the chosen latent distribution
        self.latent_dist = build_latent_distribution(
            dist_type, latent_dim, hidden_dim=hidden_dim, rank=rank
        )

        # We'll store these for use in building the decoder
        self.final_channels = self.encoder.final_channels
        self.final_spatial_dim = self.encoder.current_dim

        # 3) Build the decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_downsamples=num_downsamples,
            final_channels=self.final_channels,
            final_spatial_dim=self.final_spatial_dim,
        )

        # Save for reference
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dist_type = dist_type
        self.rank = rank

    def forward(self, x):
        """
        1) Encoder -> hidden
        2) latent_dist(...) -> distribution params
        3) latent_dist.sample(...) -> z
        4) Decoder(z) -> x_recon
        """
        # 1) Encode
        h = self.encoder(x)  # shape [batch_size, hidden_dim]

        # 2) Get distribution parameters (mu/logvar or mu/u/d, etc.)
        dist_params = self.latent_dist(h)

        # 3) Sample
        z = self.latent_dist.sample(dist_params)

        # 4) Decode
        x_recon = self.decoder(z)

        return x_recon, dist_params


def cyclical_beta(epoch: int, cycle_length: int = 50) -> float:
    """
    Returns a beta value in [0,1], cycling every 'cycle_length' epochs.
    For a triangular wave that goes 0->1->0, you can modify accordingly.
    """
    # current position in cycle: 0 <= phase < 1
    phase = (
        epoch % cycle_length
    ) / cycle_length  # fraction of the way through current cycle

    # Example: just ramp up 0->1 over one cycle, then jump back
    # If you want a triangle 0->1->0, you could do something like:
    # phase = 2 * phase if phase < 0.5 else 2 * (1 - phase)
    # return phase
    return phase  # ramp 0->1, then reset


class LitVAE(L.LightningModule):
    def __init__(
        self,
        vae_model,  # your VAE instance (Encoder + Dist + Decoder)
        learning_rate=1e-3,
        recon_loss_type="mse",  # or "bce"
    ):
        """
        vae_model: an instance of your VAE class that has:
                    - encoder
                    - latent_dist
                    - decoder
        """
        super().__init__()
        self.save_hyperparameters(ignore=["vae_model"])
        self.vae_model = vae_model
        self.lr = learning_rate
        self.recon_loss_type = recon_loss_type

    def forward(self, x):
        # Just calls the VAE's forward
        return self.vae_model(x)

    def _compute_recon_loss(self, x_recon, x):
        if self.recon_loss_type == "mse":
            return F.mse_loss(x_recon, x, reduction="mean")
        elif self.recon_loss_type == "bce":
            # For BCE, you'd typically sigmoid the output or ensure it's in [0,1]
            # Also flatten or keep shape consistent.
            return F.binary_cross_entropy_with_logits(x_recon, x, reduction="mean")
        else:
            raise ValueError(f"Unknown recon loss type: {self.recon_loss_type}")

    def training_step(self, batch, batch_idx):
        (x,) = batch
        x_recon, dist_params = self.vae_model(x)

        recon_loss = self._compute_recon_loss(x_recon, x)
        kl_loss = self.vae_model.latent_dist.kl_divergence(dist_params)
        beta = cyclical_beta(self.current_epoch, cycle_length=100)
        elbo = recon_loss + kl_loss * beta * 0.5

        self.log("train/recon_loss", recon_loss, prog_bar=True)
        self.log("train/kl_loss", kl_loss, prog_bar=True)
        self.log("train/beta", beta, prog_bar=True)
        self.log("train/elbo", elbo, prog_bar=True)

        return elbo

    def validation_step(self, batch, batch_idx):
        (x,) = batch
        x_recon, dist_params = self.forward(x)

        recon_loss = self._compute_recon_loss(x_recon, x)
        kl_loss = self.vae_model.latent_dist.kl_divergence(dist_params)
        elbo = recon_loss + kl_loss

        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_kl_loss", kl_loss, prog_bar=True)
        self.log("val_elbo", elbo, prog_bar=True)
        return elbo

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def generate_synthetic_pulses(num_samples=1000, length=1024, bullshit_frac=0.2):
    """
    Generate synthetic 1D signals [num_samples, 1, length]
    Each signal contains a simple "pulse" of random amplitude/width
    on a zero baseline.
    The "bullshit_frac" is the fraction of samples that are
    completely random noise (normal).
    """
    data = torch.zeros(num_samples, 1, length)

    # For each sample, randomly pick:
    #   - amplitude in [0.5, 1.5]
    #   - center in [200, 800]
    #   - width in [10, 40]
    fig, ax = plt.subplots(constrained_layout=True)
    labels = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        xs = np.arange(length)
        if np.random.rand() < bullshit_frac:
            # Random noise
            noise = np.random.normal(0, 1, length)
            ax.plot(xs, noise, alpha=0.1, color="grey")
            data[i, 0] = torch.from_numpy(noise).float()
            labels[i] = 0
        else:
            amplitude = np.random.uniform(0.5, 1.5)
            center = np.random.randint(int(0.2 * length), int(0.8 * length))
            width = np.random.randint(int(0.01 * length), int(0.05 * length))

            # Create a small pulse using a Gaussian or a simple triangular shape
            # Here, let's do a simple Gaussian:
            gauss = amplitude * np.exp(-0.5 * ((xs - center) / width) ** 2)
            ax.plot(xs, gauss, alpha=0.1, color="k")
            data[i, 0] = torch.from_numpy(gauss).float()
            labels[i] = 1

    print(data.shape)  # [num_samples, 1, length]
    plt.show()

    return data, labels


def visualize_latent_space(vae_model, data, labels, num_points=500, method="pca"):
    """
    Pass a subset of `data` through the encoder, get latent codes, then
    use PCA or UMAP to reduce to 2D for visualization.
    """
    from sklearn.decomposition import PCA

    try:
        import umap

        use_umap = True
    except ImportError:
        use_umap = False
        print("UMAP not installed, will use PCA instead.")

    vae_model.eval()

    # We'll sample a subset of data
    subset = data[:num_points]  # shape [num_points, 1, length]
    with torch.no_grad():
        # Encode to get the distribution params
        hidden = vae_model.encoder(subset)
        dist_params = vae_model.latent_dist(hidden)
        recon, z = vae_model.forward(subset)
        # For visualization, let's just use the mean (mu) if diagonal, or mu if low-rank
        if vae_model.dist_type == "diagonal":
            mu, logvar = dist_params
            latents = mu.cpu().numpy()
        else:
            mu, u, d = dist_params
            latents = mu.cpu().numpy()

    # Now reduce latents to 2D
    if method == "pca" or (method == "umap" and not use_umap):
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(latents)
    else:
        # Use UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(latents)

    # Plot latent space
    fig, ax = plt.subplots(constrained_layout=True)
    colors = ["tab:blue", "tab:orange"]
    for label in np.unique(labels):
        ax.scatter(
            coords_2d[labels == label, 0],
            coords_2d[labels == label, 1],
            color=colors[label],
        )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.show()

    # Plot 25 reconstructions
    plotgrid = (5, 5)
    fig, ax = plt.subplots(*plotgrid, figsize=(10, 10))
    for i in range(plotgrid[0] * plotgrid[1]):
        if i >= num_points:
            break
        ax[i // plotgrid[1], i % plotgrid[1]].plot(
            subset[i, 0].cpu().numpy(),
            c="k",
            alpha=0.5,
        )
        ax[i // plotgrid[1], i % plotgrid[1]].plot(
            recon[i, 0].cpu().numpy(),
            c="r",
        )
    plt.show()


if __name__ == "__main__":
    # 1) Generate structured synthetic data
    num_samples = 500
    batch_size = 100
    input_dim = 128
    synthetic_data, labels = generate_synthetic_pulses(
        num_samples=num_samples, length=input_dim
    )

    # Split into train/val
    split_idx = int(num_samples * 0.8)
    train_data = synthetic_data[:split_idx]
    val_data = synthetic_data[split_idx:]
    val_labels = labels[split_idx:]

    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # 2) Define and train the VAE
    vae_model = VAE(
        input_dim=input_dim,
        base_channels=8,
        num_downsamples=3,
        hidden_dim=64,
        latent_dim=16,
        dist_type="lowrank",  # or "diagonal"
        # dist_type="diagonal",
        rank=2,
    )
    summary(vae_model, input_size=(batch_size, 1, input_dim))

    lightning_module = LitVAE(vae_model, learning_rate=1e-3, recon_loss_type="mse")
    trainer = L.Trainer(
        max_epochs=500,
        callbacks=[
            StochasticWeightAveraging(swa_lrs=1e-3),
        ],
    )
    trainer.fit(lightning_module, train_loader, val_loader)

    # 3) Visualize the latent space
    # We'll reuse the `vae_model` (which has been trained in place).
    visualize_latent_space(
        vae_model, val_data, val_labels, num_points=500, method="pca"
    )
