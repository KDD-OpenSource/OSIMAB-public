import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from .algorithm_utils import Algorithm, PyTorchUtils
from .helpers import make_sequences
from .helpers import split_sequences
from .helpers import average_sequences


class AutoEncoder(Algorithm, PyTorchUtils):
    def __init__(
        self,
        name: str = "AutoEncoder",
        num_epochs: int = 10,
        batch_size: int = 20,
        lr: float = 1e-3,
        hidden_size: int = 5,
        sequence_length: int = 30,
        stride: int = 1,
        train_gaussian_percentage: float = 0.25,
        seed: int = None,
        gpu: int = None,
        details=True,
        train_max=None,
        sensor_specific=False,
    ):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.sensor_specific = sensor_specific
        self.input_size = None
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.train_gaussian_percentage = train_gaussian_percentage
        self.train_max = train_max

        self.aed = None
        self.mean, self.cov = None, None

    def SensorSpecificLoss(self, yhat, y):
        # mse = nn.MSELoss()
        # batch_size = yhat.size()[0]
        subclassLength = self.hidden_size
        yhat = yhat.view((-1, subclassLength))
        y = y.view((-1, subclassLength))
        error = yhat - y
        sqr_err = error ** 2
        sum_sqr_err = sqr_err.sum(1)
        root_sum_sqr_err = torch.sqrt(sum_sqr_err)
        return torch.mean(root_sum_sqr_err)

    def fit(self, X: pd.DataFrame):
        sequences = make_sequences(
            data=X, sequence_length=self.sequence_length, stride=self.stride
        )
        seq_train, seq_val = split_sequences(sequences, self.train_gaussian_percentage)
        train_loader = DataLoader(
            dataset=seq_train,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
        )
        train_gaussian_loader = DataLoader(
            dataset=seq_val, batch_size=self.batch_size, drop_last=True, pin_memory=True
        )

        self.input_size = sequences.shape[2]
        self.aed = AutoEncoderModule(
            self.input_size,
            self.sequence_length,
            self.hidden_size,
            seed=self.seed,
            gpu=self.gpu,
        )
        self.to_device(self.aed)  # .double()
        optimizer = torch.optim.Adam(self.aed.parameters(), lr=self.lr)

        self.aed.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f"Epoch {epoch+1}/{self.num_epochs}.")
            for ts_batch in train_loader:
                output = self.aed(self.to_var(ts_batch))
                if not self.sensor_specific:
                    loss = nn.MSELoss(size_average=False)(
                        output, self.to_var(ts_batch.float())
                    )
                else:
                    loss = self.SensorSpecificLoss(
                        output, self.to_var(ts_batch.float())
                    )
                self.aed.zero_grad()
                loss.backward()
                optimizer.step()

        self.aed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.aed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

    def predict(self, X: pd.DataFrame) -> np.array:
        sequences = make_sequences(
            data=X, sequence_length=self.sequence_length, stride=self.stride
        )
        data_loader = DataLoader(
            dataset=sequences,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.aed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        encodings = []
        errors = []
        for idx, ts in enumerate(data_loader):
            output, enc = self.aed(self.to_var(ts), return_latent=True)
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                encodings.append(enc)
                errors.append(error.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = average_sequences(
            scores, self.sequence_length, output_shape=X.shape[0]
        )

        if self.details:
            outputs = average_sequences(
                sequences=outputs,
                sequence_length=self.sequence_length,
                output_shape=(X.shape[0], X.shape[1]),
            )
            self.prediction_details.update({"reconstructions_mean": outputs})

            errors = average_sequences(
                sequences=errors,
                sequence_length=self.sequence_length,
                output_shape=(X.shape[0], X.shape[1]),
            )
            self.prediction_details.update({"errors_mean": errors})

            encodings = [e.detach().numpy() for e in encodings]
            encodings = np.concatenate(encodings)
            self.prediction_details.update({"encodings": encodings})

        return scores

    def save(self, f):
        torch.save(
            {
                "model_state_dict": self.aed.state_dict(),
                "mean": self.mean,
                "cov": self.cov,
                "input_size": self.input_size,
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "seed": self.seed,
                "gpu": self.gpu,
            },
            f,
        )

    def load(self, f):
        checkpoint = torch.load(f)
        model_state_dict = checkpoint["model_state_dict"]
        del checkpoint["model_state_dict"]
        for key in checkpoint:
            setattr(self, key, checkpoint[key])
        self.aed = AutoEncoderModule(
            self.input_size,
            self.sequence_length,
            self.hidden_size,
            seed=self.seed,
            gpu=self.gpu,
        )
        self.aed.load_state_dict(model_state_dict)


class AutoEncoderModule(nn.Module, PyTorchUtils):
    def __init__(
        self,
        n_features: int,
        sequence_length: int,
        hidden_size: int,
        seed: int,
        gpu: int,
    ):
        # Each point is a flattened window and thus has as many features as sequence_length * features
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        input_length = n_features * sequence_length

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = (
            2
            ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_length))[
                1:
            ]
        )
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]

        layers = np.array(
            [
                [nn.Linear(int(a), int(b)), nn.Tanh()]
                for a, b in enc_setup.reshape(-1, 2)
            ]
        ).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        self.to_device(self._encoder)

        layers = np.array(
            [
                [nn.Linear(int(a), int(b)), nn.Tanh()]
                for a, b in dec_setup.reshape(-1, 2)
            ]
        ).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)
        self.to_device(self._decoder)

    def forward(self, ts_batch, return_latent: bool = False):
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(flattened_sequence.float())
        dec = self._decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return (
            (reconstructed_sequence, enc) if return_latent else reconstructed_sequence
        )
