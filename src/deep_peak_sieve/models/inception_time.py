import torch
import lightning as L
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from deep_peak_sieve.utils.loggers import get_logger

log = get_logger(__name__)


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

        # Log the learning rate
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True)

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
        acc = accuracy_score(y.cpu(), preds.cpu())
        f1 = f1_score(y.cpu(), preds.cpu(), average="weighted")
        avg_precision = average_precision_score(
            y.cpu(), preds.cpu(), average="weighted"
        )
        balanced_acc = balanced_accuracy_score(y.cpu(), preds.cpu())

        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val_f1",
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val_avg_precision",
            avg_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val_balanced_acc",
            balanced_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

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
        balanced_acc = balanced_accuracy_score(y.cpu(), preds.cpu())
        precision = precision_score(y.cpu(), preds.cpu(), average="weighted")
        recall = recall_score(y.cpu(), preds.cpu(), average="weighted")
        f1 = f1_score(y.cpu(), preds.cpu(), average="weighted")
        roc_auc = roc_auc_score(y.cpu(), preds.cpu(), average="weighted")
        avg_precision = average_precision_score(
            y.cpu(), preds.cpu(), average="weighted"
        )
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_balanced_acc", balanced_acc, on_step=False, on_epoch=True)
        self.log("test_precision", precision, on_step=False, on_epoch=True)
        self.log("test_recall", recall, on_step=False, on_epoch=True)
        self.log("test_f1", f1, on_step=False, on_epoch=True)
        self.log("test_roc_auc", roc_auc, on_step=False, on_epoch=True)
        self.log("test_avg_precision", avg_precision, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> dict:
        """
        Defines the optimizer to be used (Adam by default).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.1,
                    patience=10,
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": "lr_scheduler",
            },
        }


class InceptionTimeEnsemble:
    def __init__(self, n_models=5, input_size=300, num_classes=2, filters=32, depth=6):
        self.n_models = n_models
        self.models = []
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        for i in range(n_models):
            model = InceptionTimeLightning(
                input_size=input_size,
                num_classes=num_classes,
                filters=filters,
                depth=depth,
            )
            model = model.to(self.device)
            self.models.append(model)
        log.info(f"Created ensemble of {n_models} models.")

    def load_model(self, chckpoint_path):
        # Check how many models are in the checkpoint directory
        chckpoint_path = Path(chckpoint_path)
        if not chckpoint_path.exists():
            msg = f"Checkpoint path {chckpoint_path} does not exist."
            raise ValueError(msg)

        if not chckpoint_path.is_dir():
            msg = f"Checkpoint path {chckpoint_path} is not a directory."
            raise ValueError(msg)

        # Check if the number of models in the checkpoint directory matches the number of models in the ensemble
        current_model_checkpoints = list(chckpoint_path.glob("*ckpt"))
        model_handles = [x.stem.split("_")[1] for x in current_model_checkpoints]
        if len(np.unique(model_handles)) != self.n_models:
            msg = f"Number of models in the checkpoint directory ({len(np.unique(model_handles))}) does not match the number of models in the ensemble ({self.n_models})."
            raise ValueError(msg)

        for i in range(self.n_models):
            current_model_checkpoints = list(chckpoint_path.glob(f"model_{i}_*"))
            if len(current_model_checkpoints) == 0:
                msg = f"No checkpoints found for model {i}. Train the model first or set the correct number of models."
                raise ValueError(msg)

            best_checkpoint = sorted(
                current_model_checkpoints,
                key=lambda x: float(x.stem.split("=")[-1]),
                reverse=True,
            )[0]

            log.info(f"Loading model {i} from checkpoint {best_checkpoint}")

            # self.models[i].load_from_checkpoint(
            #     best_checkpoint,
            #     map_location="cpu",
            #     model=self.models[i],
            # )
            self.models[i] = InceptionTimeLightning.load_from_checkpoint(
                best_checkpoint,
                map_location="cpu",
                model=self.models[i],
            )
            self.models[i].to(self.device)
        log.info("Loaded all models from checkpoints.")

    def _create_datasets(self, x, y):
        # Random shuffle the data, this way each model in the ensemble sees a different
        # random sample of the data for train/val/test
        shuffled_indices = np.arange(len(x))
        np.random.shuffle(shuffled_indices)
        x = x[shuffled_indices]
        y = y[shuffled_indices]

        # Split the data
        train_x, val_x, train_y, val_y = train_test_split(
            x, y, stratify=y, test_size=0.3
        )
        # Split the data again for testing
        val_x, test_x, val_y, test_y = train_test_split(
            val_x, val_y, stratify=val_y, test_size=0.5
        )

        # Convert to PyTorch tensors
        train_x = torch.from_numpy(train_x).float()
        train_y = torch.from_numpy(train_y).long()
        val_x = torch.from_numpy(val_x).float()
        val_y = torch.from_numpy(val_y).long()
        test_x = torch.from_numpy(test_x).float()
        test_y = torch.from_numpy(test_y).long()

        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)
        test_dataset = TensorDataset(test_x, test_y)

        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=64, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=4
        )
        return train_loader, val_loader, test_loader

    def fit(self, x, y, n_epochs, batch_size=64):
        for i, model in enumerate(self.models):
            train_loader, val_loader, test_loader = self._create_datasets(x, y)
            chckpoint_filename = f"model_{i}_" + "{epoch:02d}-{val_acc:.2f}"
            chckpoint_path = Path("checkpoints")
            checkpoint_callback = ModelCheckpoint(
                monitor="val_balanced_acc",
                filename=chckpoint_filename,
                save_top_k=3,
                mode="max",
                dirpath=chckpoint_path,
            )
            trainer = Trainer(
                max_epochs=n_epochs,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                log_every_n_steps=1,
                callbacks=[checkpoint_callback],
            )

            # Fit the model on the training set, validating on the val_loader
            trainer.fit(model, train_loader, val_loader)
            trainer.test(model, test_loader, ckpt_path="best", verbose=True)
            model.eval()

    def predict(self, x):
        preds = []
        probs = []

        x = torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x
        x = x.to(self.device)

        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model.forward(x)
                prob = torch.nn.functional.softmax(logits, dim=-1)
            pred = torch.argmax(prob, dim=-1)
            preds.append(pred.cpu().numpy())
            probs.append(prob.cpu().numpy())

        preds = np.mean(preds, axis=0)
        preds = np.round(preds).astype(int)
        probs = np.mean(probs, axis=0)

        return preds, probs
