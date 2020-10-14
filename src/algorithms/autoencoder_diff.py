import numpy as np
import pandas as pd
from .autoencoder import AutoEncoder


def diff(X: pd.DataFrame):
    X = X.interpolate()
    X.bfill(inplace=True)
    X_diff = X.diff()
    X_diff.bfill(inplace=True)
    return X_diff


class AutoEncoderDiff(AutoEncoder):
    def __init__(
        self,
        name: str = "AutoEncoderDifference",
        num_epochs: int = 10,
        batch_size: int = 20,
        lr: float = 1e-3,
        hidden_size: int = 5,
        sequence_length: int = 30,
        train_gaussian_percentage: float = 0.25,
        seed: int = None,
        gpu: int = None,
        details=True,
        activation="Tanh",
    ):
        AutoEncoder.__init__(
            self,
            name,
            num_epochs,
            batch_size,
            lr,
            hidden_size,
            sequence_length,
            train_gaussian_percentage,
            seed,
            gpu,
            details,
            activation,
        )

    def fit(self, X: pd.DataFrame):
        X_diff = diff(X)
        AutoEncoder.fit(self, X_diff)

    def predict(self, X: pd.DataFrame) -> np.array:
        X_diff = diff(X)
        return AutoEncoder.predict(self, X_diff)
