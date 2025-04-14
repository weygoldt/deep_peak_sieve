from torch import Tensor
from torch import nn
from abc import abstractmethod
from typing import Any, List, Union, Tuple
import lightning as L
from torch import optim

batch_size = 1000
lr = 1e-3
n_epochs = 100
gamma = 1.0


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    @abstractmethod
    def encode(self, input: Tensor) -> List[Tensor]:
        pass

    @abstractmethod
    def decode(self, input: Tensor) -> Any:
        pass

    @abstractmethod
    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        pass

    @abstractmethod
    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tuple[Tensor, ...]:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Any:
        pass


class LitVAE(L.LightningModule):
    """
    Generic LightningModule to train models inheriting from BaseVAE.
    """

    def __init__(
        self,
        model: BaseVAE,
        lr: float = 1e-3,
        kld_weight: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        model : BaseVAE
            A model that implements the BaseVAE interface.
        lr : float
            Learning rate for the optimizer.
        kld_weight : float
            Optional scaling of the KL term in the loss function (M_N).
        """
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])  # saves lr, kld_weight, etc.
        self.lr = lr
        self.kld_weight = kld_weight

    def forward(self, x: Tensor) -> Any:
        """
        Pass input through the VAE model.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.

        Returns
        -------
        Any
            The model's output (structure depends on the specific BaseVAE implementation).
        """
        return self.model(x)

    def training_step(self, batch: Union[Tensor, Any], batch_idx: int) -> Tensor:
        """
        Parameters
        ----------
        batch : Union[torch.Tensor, Any]
            Training batch (e.g. (x,) or just x).
        batch_idx : int
            Index of the batch within the current epoch.

        Returns
        -------
        torch.Tensor
            The loss for this training batch.
        """
        # If using a TensorDataset, batch might be a tuple: (x,)
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        # model.forward(...) -> (recons, input, mu, log_var) or similar
        outputs = self.model(x)
        # Typically the model outputs a tuple, e.g. (recons, input, mu, log_var)
        # data = self.model.loss_function(*outputs, M_N=self.kld_weight)
        data = self.model.loss_function(*outputs)
        # Log the total loss
        for keys, values in data.items():
            key = f"train_{keys}"
            self.log(key, values, on_step=True, on_epoch=True, prog_bar=True)
        loss = data["loss"]
        return loss

    def validation_step(self, batch: Union[Tensor, Any], batch_idx: int) -> Tensor:
        """
        Parameters
        ----------
        batch : Union[torch.Tensor, Any]
            Validation batch.
        batch_idx : int
            Index of the batch within the current epoch.

        Returns
        -------
        torch.Tensor
            The loss for this validation batch.
        """
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        outputs = self.model(x)
        # data = self.model.loss_function(*outputs, M_N=self.kld_weight)
        data = self.model.loss_function(*outputs)
        val_loss = data["loss"]
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """
        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer for training.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        return [optimizer], [
            {
                "scheduler": step_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        ]
