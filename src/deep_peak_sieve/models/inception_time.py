import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn")


class Inception(torch.nn.Module):
    def __init__(self, input_size, filters):
        super(Inception, self).__init__()

        self.bottleneck1 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv10 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=10,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv20 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=20,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv40 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=40,
            stride=1,
            padding="same",
            bias=False,
        )

        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bottleneck2 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.batch_norm = torch.nn.BatchNorm1d(num_features=4 * filters)

    def forward(self, x):
        x0 = self.bottleneck1(x)
        x1 = self.conv10(x0)
        x2 = self.conv20(x0)
        x3 = self.conv40(x0)
        x4 = self.bottleneck2(self.max_pool(x))
        y = torch.concat([x1, x2, x3, x4], dim=1)
        y = torch.nn.functional.relu(self.batch_norm(y))
        return y


class Residual(torch.nn.Module):
    def __init__(self, input_size, filters):
        super(Residual, self).__init__()

        self.bottleneck = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=4 * filters,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.batch_norm = torch.nn.BatchNorm1d(num_features=4 * filters)

    def forward(self, x, y):
        y = y + self.batch_norm(self.bottleneck(x))
        y = torch.nn.functional.relu(y)
        return y


class Lambda(torch.nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class InceptionModel(torch.nn.Module):
    def __init__(self, input_size, num_classes, filters, depth):
        super(InceptionModel, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.filters = filters
        self.depth = depth

        modules = OrderedDict()

        for d in range(depth):
            modules[f"inception_{d}"] = Inception(
                input_size=input_size if d == 0 else 4 * filters,
                filters=filters,
            )
            if d % 3 == 2:
                modules[f"residual_{d}"] = Residual(
                    input_size=input_size if d == 2 else 4 * filters,
                    filters=filters,
                )

        modules["avg_pool"] = Lambda(f=lambda x: torch.mean(x, dim=-1))
        modules["linear"] = torch.nn.Linear(
            in_features=4 * filters, out_features=num_classes
        )

        self.model = torch.nn.Sequential(modules)

    def forward(self, x):
        for d in range(self.depth):
            y = self.model.get_submodule(f"inception_{d}")(x if d == 0 else y)
            if d % 3 == 2:
                y = self.model.get_submodule(f"residual_{d}")(x, y)
                x = y
        y = self.model.get_submodule("avg_pool")(y)
        y = self.model.get_submodule("linear")(y)
        return y


class InceptionTime:
    def __init__(self, x, y, filters, depth, models):
        """
        Implementation of InceptionTime model introduced in Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier,
        C., Schmidt, D.F., Weber, J., Webb, G.I., Idoumghar, L., Muller, P.A. and Petitjean, F., 2020. InceptionTime:
        Finding AlexNet for Time Series Classification. Data Mining and Knowledge Discovery, 34(6), pp.1936-1962.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, channels, length) where samples is the number of time series,
            channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
            is the length of the time series.

        y: np.array.
            Class labels, array with shape (samples,) where samples is the number of time series.

        filters: int.
            The number of filters (or channels) of the convolutional layers of each model.

        depth: int.
            The number of blocks of each model.

        models: int.
            The number of models.
        """

        # Check if GPU is available.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Scale the data.
        self.mu = np.nanmean(x, axis=0, keepdims=True)
        self.sigma = np.nanstd(x, axis=0, keepdims=True)
        x = (x - self.mu) / self.sigma

        # Save the data.
        self.x = torch.from_numpy(x).float().to(self.device)
        self.y = torch.from_numpy(y).long().to(self.device)

        # Build and save the models.
        self.models = [
            InceptionModel(
                input_size=x.shape[1],
                num_classes=len(np.unique(y)),
                filters=filters,
                depth=depth,
            ).to(self.device)
            for _ in range(models)
        ]

    def fit(self, learning_rate, batch_size, epochs, verbose=True):
        """
        Train the models.

        Parameters:
        __________________________________
        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
        """

        # Generate the training dataset.
        dataset = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x, self.y),
            batch_size=batch_size,
            shuffle=True,
        )

        for m in range(len(self.models)):
            # Define the optimizer.
            optimizer = torch.optim.Adam(self.models[m].parameters(), lr=learning_rate)

            # Define the loss function.
            loss_fn = torch.nn.CrossEntropyLoss()

            # Train the model.
            print(f"Training model {m + 1} on {self.device}.")
            self.models[m].train(True)
            for epoch in range(epochs):
                for features, target in dataset:
                    optimizer.zero_grad()
                    output = self.models[m](features.to(self.device))
                    loss = loss_fn(output, target.to(self.device))
                    loss.backward()
                    optimizer.step()
                    accuracy = (
                        torch.argmax(
                            torch.nn.functional.softmax(output, dim=-1), dim=-1
                        )
                        == target
                    ).float().sum() / target.shape[0]
                if verbose:
                    print(
                        "epoch: {}, loss: {:,.6f}, accuracy: {:.6f}".format(
                            1 + epoch, loss, accuracy
                        )
                    )
            self.models[m].train(False)
            print("-----------------------------------------")

    def predict(self, x):
        """
        Predict the class labels.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, channels, length) where samples is the number of time series,
            channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
            is the length of the time series.

        Returns:
        __________________________________
        y: np.array.
            Predicted labels, array with shape (samples,) where samples is the number of time series.
        """

        # Scale the data.
        x = torch.from_numpy((x - self.mu) / self.sigma).float().to(self.device)

        # Get the predicted probabilities.
        with torch.no_grad():
            p = torch.concat(
                [
                    torch.nn.functional.softmax(model(x), dim=-1).unsqueeze(-1)
                    for model in self.models
                ],
                dim=-1,
            ).mean(-1)

        # Get the predicted labels.
        y = p.argmax(-1).detach().cpu().numpy().flatten()

        return y
