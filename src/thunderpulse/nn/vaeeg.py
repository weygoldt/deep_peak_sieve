# Code is mainly adapted from  https://github.com/ViktorvdValk/Interpretable_VAE_for_ECG

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from thunderpulse.models.base import BaseVAE, LitVAE
from thunderpulse.models.params import batch_size, gamma, lr
from thunderpulse.models.utils import (
    generate_synthetic_peaks,
    plot_reconstructions,
    visualize_latent_space,
)


class Conv1dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation=1,
        bias=True,
    ):
        super(Conv1dLayer, self).__init__()

        total_p = kernel_size + (kernel_size - 1) * (dilation - 1) - 1
        left_p = total_p // 2
        right_p = total_p - left_p

        self.conv = nn.Sequential(
            nn.ConstantPad1d((left_p, right_p), 0),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
            ),
        )

    def forward(self, x):
        return self.conv(x)


class FConv1dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation=1,
        bias=True,
    ):
        super(FConv1dLayer, self).__init__()

        p = (dilation * (kernel_size - 1)) // 2
        op = stride - 1

        self.fconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=p,
            output_padding=op,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        return self.fconv(x)


class LSTMLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout=0.0,
        bidirectional=False,
    ):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return outputs


class GRULayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout=0.0,
        bidirectional=False,
    ):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        outputs, _ = self.gru(x)
        return outputs


class HeadLayer(nn.Module):
    """
    Multiple paths to process input data. Four paths with kernel size 5, 7, 9, 11, respectively.
    Each path has one convolution layer.
    """

    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(HeadLayer, self).__init__()

        if out_channels % 4 != 0:
            raise ValueError(
                "out_channels must be divisible by 4, but got: %d"
                % out_channels
            )

        unit = out_channels // 4

        self.conv1 = nn.Sequential(
            Conv1dLayer(
                in_channels=in_channels,
                out_channels=unit,
                kernel_size=11,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(unit),
            nn.LeakyReLU(negative_slope),
        )

        self.conv2 = nn.Sequential(
            Conv1dLayer(
                in_channels=in_channels,
                out_channels=unit,
                kernel_size=9,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(unit),
            nn.LeakyReLU(negative_slope),
        )

        self.conv3 = nn.Sequential(
            Conv1dLayer(
                in_channels=in_channels,
                out_channels=unit,
                kernel_size=7,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(unit),
            nn.LeakyReLU(negative_slope),
        )

        self.conv4 = nn.Sequential(
            Conv1dLayer(
                in_channels=in_channels,
                out_channels=unit,
                kernel_size=5,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(unit),
            nn.LeakyReLU(negative_slope),
        )

        self.conv5 = nn.Sequential(
            Conv1dLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.conv5(out)
        return out


class ResBlockV1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        negative_slope=0.2,
    ):
        super(ResBlockV1, self).__init__()

        if stride == 1 and in_channels == out_channels:
            self.projection = None
        else:
            self.projection = nn.Sequential(
                Conv1dLayer(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        self.conv1 = nn.Sequential(
            Conv1dLayer(
                in_channels, out_channels, kernel_size, stride, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope),
        )

        self.conv2 = nn.Sequential(
            Conv1dLayer(
                out_channels, out_channels, kernel_size, 1, bias=False
            ),
            nn.BatchNorm1d(out_channels),
        )

        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        if self.projection:
            res = self.projection(x)
        else:
            res = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        out = self.act(out)
        return out


def re_parameterize(mu, log_var):
    """
    Re-parameterize trick to sample from N(mu, var) from N(0,1).

    :param mu: (Tensor) Mean of the latent Gaussian [N, z_dims]
    :param log_var: (Tensor) Standard deviation of the latent Gaussian [N, z_dims]
    :return: (Tensor) [N, z_dims]
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def z_sample(z):
    zp = Variable(torch.randn_like(z))
    return zp


def recon_loss(x, x_bar):
    """
    Reconstruct loss
    :param x:
    :param x_bar:
    :return:
    """
    value = torch.nn.functional.mse_loss(x, x_bar, reduction="mean")
    return value


def kl_loss(mu, log_var):
    """
    Compute KL loss

    𝐾𝐿(𝑁(𝜇,𝜎^2),𝑁(0,1)) = -0.5*(log𝜎^2 - 𝜎^2 - 𝜇^2 + 1), 𝜎 > 0

    using log_var
    t = log𝜎^2

    KL= -0.5 * ( t - e^t - u^2 + 1)

    :param mu:
    :param log_var:
    :return:
    """
    value = torch.mean(-0.5 * (1 + log_var - torch.exp(log_var) - mu**2))
    return value


class Encoder(nn.Module):
    def __init__(self, in_channels, z_dim, negative_slope=0.2):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                HeadLayer(
                    in_channels=in_channels,
                    out_channels=16,
                    negative_slope=negative_slope,
                )
            ]
        )

        in_features = [16, 16, 24, 32]
        out_features = [16, 24, 32, 32]
        n_blocks = [2, 2, 2, 2]

        for in_chan, out_chan, n_block in zip(
            in_features, out_features, n_blocks, strict=False
        ):
            self.layers.append(
                nn.Sequential(
                    Conv1dLayer(in_chan, out_chan, 3, 2, bias=False),
                    nn.BatchNorm1d(out_chan),
                    nn.LeakyReLU(negative_slope),
                )
            )
            for _ in range(n_block):
                self.layers.append(
                    ResBlockV1(out_chan, out_chan, 3, 1, negative_slope)
                )

        self.layers.append(
            nn.Sequential(
                nn.Flatten(1),
                nn.Linear(256, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(negative_slope),
            )
        )

        self.mu = nn.Linear(32, z_dim)
        self.log_var = nn.Linear(32, z_dim)

    def forward(self, x):
        # x: (N, 1, L)
        for m in self.layers:
            x = m(x)

        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, z_dim, negative_slope=0.2, last_lstm=True):
        super(Decoder, self).__init__()
        # (N, 256) to (N, 32, 8)
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope),
        )

        in_features = [32, 32, 24, 16, 16]
        out_features = [32, 24, 16, 16, 8]
        n_blocks = [2, 2, 2, 2, 2]

        self.layers = nn.ModuleList()

        for in_chan, out_chan, n_block in zip(
            in_features, out_features, n_blocks, strict=False
        ):
            self.layers.append(
                nn.Sequential(
                    FConv1dLayer(in_chan, out_chan, 3, 2, bias=False),
                    nn.BatchNorm1d(out_chan),
                    nn.LeakyReLU(negative_slope),
                )
            )
            for _ in range(n_block):
                self.layers.append(
                    ResBlockV1(out_chan, out_chan, 3, 1, negative_slope)
                )

        self.layers.append(
            nn.Sequential(
                Conv1dLayer(
                    out_features[-1], out_features[-1], 3, 1, bias=False
                ),
                nn.BatchNorm1d(out_features[-1]),
                nn.LeakyReLU(negative_slope),
            )
        )
        if last_lstm:
            self.tail = LSTMLayer(out_features[-1], 1, 2)
        else:
            # self.tail = Conv1dLayer(out_features[-1], 1, 1, 1, bias=True)
            self.tail = nn.Sequential(
                Conv1dLayer(
                    out_features[-1], out_features[-1] // 2, 5, 1, bias=True
                ),
                nn.BatchNorm1d(out_features[-1] // 2),
                nn.LeakyReLU(negative_slope),
                Conv1dLayer(out_features[-1] // 2, 1, 3, 1, bias=True),
            )

        self.last_lstm = last_lstm

    def forward(self, x):
        """

        :param x: (N, z_dims)
        :return: (N, 1, L)
        """
        x = self.fc(x)

        n_batch, nf = x.shape
        x = x.view(n_batch, 32, 8)

        for m in self.layers:
            x = m(x)

        if self.last_lstm:
            x = torch.permute(x, (2, 0, 1))
            x = self.tail(x)
            x = torch.permute(x, (1, 2, 0))
        else:
            x = self.tail(x)
        return x


class VAEEGModel(BaseVAE):
    def __init__(
        self,
        in_channels,
        z_dim,
        negative_slope=0.2,
        decoder_last_lstm=True,
        gamma=500.0,
    ):
        super().__init__()  # calls BaseVAE.__init__ (which is typically empty)
        self.z_dim = z_dim
        self.encoder_net = Encoder(
            in_channels=in_channels, z_dim=z_dim, negative_slope=negative_slope
        )
        self.decoder_net = Decoder(
            z_dim=z_dim,
            negative_slope=negative_slope,
            last_lstm=decoder_last_lstm,
        )
        self.gamma = gamma

    def encode(self, input) -> list[torch.Tensor]:
        """
        Returns (mu, log_var).
        """
        mu, log_var = self.encoder_net(input)
        return [mu, log_var]

    def decode(self, input) -> torch.Tensor:
        """
        Returns the decoded/reconstructed x.
        """
        return self.decoder_net(input)

    def reparameterize(self, mean, logvar) -> torch.Tensor:
        """
        Standard reparameterization: z = mu + eps * exp(0.5*log_var).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, inputs) -> tuple[torch.Tensor, ...]:
        """
        1) Encode -> (mu, log_var)
        2) Reparameterize -> z
        3) Decode -> x_bar
        Returns x_bar, (mu, log_var).
        """
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return (recon, inputs, mu, log_var)

    def kl_divergence(self, mu, log_var):
        """
        KL( q(z|x) || p(z) ) with q ~ N(mu, var).
        By your formula:
          KL = -0.5 * mean(1 + log_var - exp(log_var) - mu^2)
        """
        kl_val = torch.mean(-0.5 * (1 + log_var - torch.exp(log_var) - mu**2))
        return kl_val

    def loss_function(self, x, x_bar, mu, log_var):
        """
        Loss function: reconstruction loss + KL divergence.
        """
        recon_loss_val = recon_loss(x, x_bar)
        kl_loss_val = kl_loss(mu, log_var)
        loss = recon_loss_val + self.gamma * kl_loss_val

        return {
            "loss": loss,
            "recons_loss": recon_loss_val,
            "kld_loss": kl_loss_val,
        }


if __name__ == "__main__":
    num_samples = 50_000
    x, y = generate_synthetic_peaks(
        num_samples=num_samples,
        length=256,
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
    model = VAEEGModel(
        in_channels=1,
        z_dim=16,
        negative_slope=0.2,
        decoder_last_lstm=False,
        gamma=gamma,
    )

    lit_model = LitVAE(model, lr=lr)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(
        max_epochs=200,
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
