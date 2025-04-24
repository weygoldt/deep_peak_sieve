import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, Tensor
import lightning as L
from typing import List, Any, Optional
from lightning.pytorch.callbacks import LearningRateMonitor, lr_monitor

from thunderpulse.models.utils import (
    generate_synthetic_peaks,
    visualize_latent_space,
    plot_reconstructions,
)
from thunderpulse.models.base import BaseVAE, LitVAE
from thunderpulse.models.params import batch_size, lr, n_epochs, gamma


class BetaVAE1D(BaseVAE):
    """
    Beta-VAE for 1D signals.
    """

    num_iter = 0  # Track training iterations for capacity scheduling

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: Optional[List[int]] = None,
        input_length: int = 1024,
        beta: float = 4.0,
        gamma: float = 1000.0,
        max_capacity: float = 25.0,
        Capacity_max_iter: float = 1e5,
        loss_type: str = "B",
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        in_channels : int
            Number of channels in the input signal.
        latent_dim : int
            Dimensionality of the latent space.
        hidden_dims : List[int], optional
            Channels for intermediate Conv1d layers, by default [32, 64, 128, 256].
        input_length : int
            Length of the 1D input signal.
        beta : float
            KL weight (if using 'H' loss type).
        gamma : float
            Weight for capacity-based loss (if using 'B' loss type).
        max_capacity : float
            Maximum capacity (C_max) for capacity-based loss.
        Capacity_max_iter : float
            Iterations to reach max capacity (C_stop_iter).
        loss_type : str
            'H' (Higgins) or 'B' (Burgess) variant.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.tensor([max_capacity], dtype=torch.float32)
        self.C_stop_iter = Capacity_max_iter

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # ----------------
        # Encoder
        # ----------------
        modules = []
        last_channels = in_channels

        receptive_field_size = 4

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    # nn.Conv1d(last_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.Conv1d(
                        last_channels,
                        h_dim,
                        kernel_size=receptive_field_size,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            last_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Determine final length after downsampling
        with torch.no_grad():
            test_in = torch.zeros(1, in_channels, input_length)
            test_out = self.encoder(test_in)
        self.enc_length = test_out.shape[-1]  # final spatial size

        # Linear layers to produce mu/logvar
        self.fc_mu = nn.Linear(last_channels * self.enc_length, latent_dim)
        self.fc_var = nn.Linear(last_channels * self.enc_length, latent_dim)

        # ----------------
        # Decoder
        # ----------------
        hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * self.enc_length)

        modules = []
        prev_channels = hidden_dims[0]

        # Build upsampling conv layers
        for i in range(len(hidden_dims) - 1):
            out_channels = hidden_dims[i + 1]
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        prev_channels, out_channels, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(),
                )
            )
            prev_channels = out_channels

        # Final layer to reconstruct original channel count
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(
                prev_channels, in_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),  # or remove if signals can be unbounded
        )

        self.decoder = nn.Sequential(*modules)
        hidden_dims.reverse()  # (optional) revert if you need original order later

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Parameters
        ----------
        input : torch.Tensor
            Input of shape (N, in_channels, length).

        Returns
        -------
        List[torch.Tensor]
            [mu, log_var], each (N, latent_dim).
        """
        x = self.encoder(input)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]

    def decode(self, input: Tensor) -> Any:
        """
        Parameters
        ----------
        input : torch.Tensor
            Latent vector (N, latent_dim).

        Returns
        -------
        Any
            Reconstructed signal (N, in_channels, original_length).
        """
        x = self.decoder_input(input)
        x = x.view(-1, self.decoder[0][0].in_channels, self.enc_length)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mean : torch.Tensor
            (N, latent_dim).
        logvar : torch.Tensor
            (N, latent_dim).

        Returns
        -------
        torch.Tensor
            Reparameterized sample (N, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        current_device : int
            Device index.

        Returns
        -------
        torch.Tensor
            Generated samples (batch_size, in_channels, length).
        """
        z = torch.randn(batch_size, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input to encode and then decode.

        Returns
        -------
        torch.Tensor
            Reconstructed signals (N, in_channels, length).
        """
        return self.forward(x)[0]

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        *inputs : torch.Tensor
            Expect a single Tensor (N, in_channels, length).

        Returns
        -------
        torch.Tensor
            A tuple (recons, original_input, mu, log_var) concatenated in a single tensor or container.
            For compatibility with the ABC, we type as torch.Tensor here.
        """
        input_tensor = inputs[0]
        mu, log_var = self.encode(input_tensor)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        # Return as a tuple. If strictly returning a Tensor is required, you'd adapt accordingly:
        return (recons, input_tensor, mu, log_var)  # type: ignore

    def loss_function(self, *inputs: Any, **kwargs) -> Any:
        """
        Parameters
        ----------
        *inputs : Any
            Typically (recons, input, mu, log_var).
        **kwargs : dict
            Contains e.g. kld_weight (`M_N`).

        Returns
        -------
        torch.Tensor
            The total loss.
        """
        self.num_iter += 1
        recons, input_tensor, mu, log_var = inputs

        kld_weight = kwargs.get("M_N", 1.0)
        recons_loss = F.mse_loss(recons, input_tensor)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        C = None
        if self.loss_type == "H":
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":
            self.C_max = self.C_max.to(input_tensor.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.item()
            )
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss_type. Choose 'H' or 'B'.")

        return {
            "loss": loss,
            "recons_loss": recons_loss,
            "kld_loss": kld_loss,
            "C": C if self.loss_type == "B" else None,
        }


if __name__ == "__main__":
    num_samples = 50_000
    x, y = generate_synthetic_peaks(
        num_samples=num_samples,
        length=512,
        bullshit_frac=0.5,
        noise_std=0.0,
    )

    # Z-score normalization
    x = (x - x.mean()) / x.std()

    # Split into train/val
    split_idx = int(num_samples * 0.8)
    train_data = x[:split_idx]
    val_data = x[split_idx:]
    val_labels = y[split_idx:]

    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize model
    model = BetaVAE1D(
        in_channels=1,
        latent_dim=64,
        hidden_dims=[32, 64, 128, 256, 512],
        input_length=512,
        beta=4.0,
        gamma=gamma,
        max_capacity=25.0,
        Capacity_max_iter=1e5,
        loss_type="B",
    )

    lit_model = LitVAE(model, lr=lr)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(
        max_epochs=n_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        callbacks=[lr_monitor],
    )

    trainer.fit(lit_model, train_loader, val_loader)

    # Visualize latent space
    visualize_latent_space(
        lit_model.model,
        val_data[:1000],
        labels=val_labels[:1000],
        method="pca",
        use_mean=True,
        n_samples=1000,
    )
    # Plot reconstructions
    plot_reconstructions(
        lit_model.model,
        val_data,
        n_samples=10,
    )
