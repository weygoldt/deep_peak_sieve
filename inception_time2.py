import torch
import lightning as L
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict


class Inception(nn.Module):
    """
    Inception block for use in 1D convolutional neural networks.

    Parameters
    ----------
    input_size : int
        Number of input channels.
    filters : int
        Number of filters for each internal convolution.
    """

    def __init__(self, input_size: int, filters: int) -> None:
        super().__init__()
        self.bottleneck1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding="same",  # PyTorch >= 2.0 supports 'same'
            bias=False,
        )
        self.conv10 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=10,
            stride=1,
            padding="same",
            bias=False,
        )
        self.conv20 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=20,
            stride=1,
            padding="same",
            bias=False,
        )
        self.conv40 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=40,
            stride=1,
            padding="same",
            bias=False,
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.bottleneck2 = nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4 * filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.bottleneck1(x)
        x1 = self.conv10(x0)
        x2 = self.conv20(x0)
        x3 = self.conv40(x0)
        x4 = self.bottleneck2(self.max_pool(x))

        y = torch.concat([x1, x2, x3, x4], dim=1)
        y = nn.functional.relu(self.batch_norm(y))
        return y


class Residual(nn.Module):
    """
    Residual connection to add the original input back to the network output.
    """

    def __init__(self, input_size: int, filters: int) -> None:
        super().__init__()
        self.bottleneck = nn.Conv1d(
            in_channels=input_size,
            out_channels=4 * filters,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4 * filters)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        skip = self.batch_norm(self.bottleneck(x))
        out = F.relu(y + skip)
        return out


class Lambda(nn.Module):
    """
    A simple wrapper that applies a user-defined function in the forward pass.
    """

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class InceptionModel(nn.Module):
    """
    Single InceptionTime model composed of multiple Inception blocks with optional residual connections.

    Parameters
    ----------
    input_size : int
        Number of input channels (dimensions) in the time-series data.
    num_classes : int
        Number of output classes for classification.
    filters : int
        Number of filters in each inception block path.
    depth : int
        How many inception blocks are stacked.
    """

    def __init__(
        self, input_size: int, num_classes: int, filters: int, depth: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.filters = filters
        self.depth = depth

        modules = OrderedDict()
        for d in range(depth):
            # Inception block
            modules[f"inception_{d}"] = Inception(
                input_size=(input_size if d == 0 else 4 * filters),
                filters=filters,
            )
            # Add a residual block every 3 inception blocks
            if d % 3 == 2:
                modules[f"residual_{d}"] = Residual(
                    input_size=(input_size if d == 2 else 4 * filters),
                    filters=filters,
                )

        # Global Average Pool
        modules["avg_pool"] = Lambda(f=lambda x: torch.mean(x, dim=-1))
        # Final linear layer
        modules["linear"] = nn.Linear(in_features=4 * filters, out_features=num_classes)

        self.model = nn.Sequential(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We can't iterate over self.model directly for the residual logic
        y = None
        for d in range(self.depth):
            inc_block = self.model.get_submodule(f"inception_{d}")
            y = inc_block(x if d == 0 else y)

            if d % 3 == 2:
                res_block = self.model.get_submodule(f"residual_{d}")
                y = res_block(x, y)
                x = y

        avg_pool = self.model.get_submodule("avg_pool")
        linear = self.model.get_submodule("linear")

        y = avg_pool(y)
        y = linear(y)
        return y


class InceptionTimeLightning(L.LightningModule):
    """
    LightningModule wrapping a single InceptionTime model for time-series classification.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        filters: int,
        depth: int,
        learning_rate: float = 1e-3,
    ):
        """
        Parameters
        ----------
        input_size : int
            Number of input channels (dimensions) in the time-series data.
        num_classes : int
            Number of output classes for classification.
        filters : int
            Number of filters in each inception block path.
        depth : int
            Number of stacked inception blocks for each model.
        learning_rate : float, optional
            Learning rate for the optimizer, by default 1e-3.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = InceptionModel(
            input_size=input_size,
            num_classes=num_classes,
            filters=filters,
            depth=depth,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the InceptionTime model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size, sequence_length).

        Returns
        -------
        torch.Tensor
            Unnormalized logits of shape (batch_size, num_classes).
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Computes training loss and logs it.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        # Accuracy on training batch
        preds = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Computes validation loss and logs it.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Computes test loss and logs it.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        preds = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Defines the optimizer to be used (Adam by default).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
