import logging
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange

from .algorithm_utils import Algorithm, PyTorchUtils
from .helpers import make_sequences
from .helpers import split_sequences


class AdversarialAutoEncoder(Algorithm, PyTorchUtils):
    def __init__(
        self,
        name: str = "AdverserialAutoEncoder",
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
        name = name + activation
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        activation_dict = {"Tanh": nn.Tanh, "ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU}
        self.activation = activation_dict[activation]

        self.aed = None
        self.discriminator = None
        self.mean, self.cov = None, None

    def fit(self, X: pd.DataFrame):
        sequences = make_sequences(data=X, sequence_length=self.sequence_length)
        seq_train, seq_test = split_sequences(
            sequences, self.train_gaussian_percentage
        )
        train_loader = DataLoader(
            dataset=seq_train,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
        )
        train_gaussian_loader = DataLoader(
            dataset=seq_val,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
        )

        self.aed = AutoEncoderModule(
            X.shape[1],
            self.sequence_length,
            self.hidden_size,
            seed=self.seed,
            gpu=self.gpu,
            activation=self.activation,
        )
        self.to_device(self.aed)

        self.discriminator = DiscriminatorModule(
            self.hidden_size, seed=self.seed, gpu=self.gpu, activation=self.activation
        )
        self.to_device(self.discriminator)

        autoencoder_loss = nn.MSELoss(size_average=False)

        optimizer_ae = torch.optim.Adam(self.aed.parameters(), lr=self.lr)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.aed.train()
        self.discriminator.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f"Epoch {epoch + 1}/{self.num_epochs}.")
            for ts_batch in train_loader:
                # Reconstruction phase
                self.aed.train()

                output = self.aed(self.to_var(ts_batch))
                optimizer_ae.zero_grad()
                loss_a = autoencoder_loss(output, self.to_var(ts_batch.float()))
                loss_a.backward()
                optimizer_ae.step()

                # Regularization phase
                self.aed.eval()
                self.discriminator.train()
                z_real = Variable(torch.randn(ts_batch.size()[0], self.hidden_size))
                self.to_device(z_real)
                disc_real = self.discriminator(z_real)

                z_fake = self.aed(ts_batch, return_latent=True)
                disc_fake = self.discriminator(z_fake)

                optimizer_d.zero_grad()
                loss_d = -torch.mean(
                    torch.log(disc_real + 1e-15) + torch.log(1 - disc_fake + 1e-15)
                )
                loss_d.backward()
                optimizer_d.step()

                self.aed.train()
                z_fake = self.aed(ts_batch, return_latent=True)
                disc_fake = self.discriminator(z_fake)

                optimizer_ae.zero_grad()
                loss_g = -torch.mean(torch.log(disc_fake + 1e-15))
                loss_g.backward()
                optimizer_ae.step()

        self.aed.eval()
        self.discriminator.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.aed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

    def predict(self, X: pd.DataFrame) -> np.array:
        sequences = make_sequences(data=X, sequence_length=self.sequence_length)
        data_loader = DataLoader(
            dataset=sequences,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.aed.eval()
        self.discriminator.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        z_normal = multivariate_normal(np.zeros(self.hidden_size), 1)
        scores = []
        outputs = []
        errors = []
        for idx, ts in enumerate(data_loader):
            encoded = self.aed(self.to_var(ts), return_latent=True)
            output = self.aed(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))

            score = torch.norm(encoded, p=2, dim=1).data.cpu().numpy()
            score = score.repeat(30)

            scores.append(score.reshape(ts.size(0), self.sequence_length))

            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, X.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, i : i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)
        lattice.sort(axis=0)
        weight = np.arange(1, lattice.shape[0] + 1)
        weight = np.power(2.0, -weight)
        weight = weight.reshape(-1, 1)
        lattice = lattice * weight
        scores = np.nansum(lattice, axis=0)

        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[
                    i % self.sequence_length, i : i + self.sequence_length, :
                ] = output
            self.prediction_details.update(
                {"reconstructions_mean": np.nanmean(lattice, axis=0).T}
            )

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[
                    i % self.sequence_length, i : i + self.sequence_length, :
                ] = error
            self.prediction_details.update(
                {"errors_mean": np.nanmean(lattice, axis=0).T}
            )

        return scores


class AutoEncoderModule(nn.Module, PyTorchUtils):
    def __init__(
        self,
        n_features: int,
        sequence_length: int,
        hidden_size: int,
        seed: int,
        gpu: int,
        activation: nn.Module,
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
        dec_setup = np.concatenate(
            [[hidden_size], dec_steps.repeat(2), [input_length]]
        )
        enc_setup = dec_setup[::-1]

        layers = np.array(
            [
                [nn.Linear(int(a), int(b)), activation()]
                for a, b in enc_setup.reshape(-1, 2)
            ]
        ).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        self.to_device(self._encoder)

        layers = np.array(
            [
                [nn.Linear(int(a), int(b)), activation()]
                for a, b in dec_setup.reshape(-1, 2)
            ]
        ).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)
        self.to_device(self._decoder)

    def forward(self, ts_batch, return_latent: bool = False):
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(flattened_sequence.float())
        if return_latent:
            return enc
        dec = self._decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return reconstructed_sequence


class EncoderModule(nn.Module, PyTorchUtils):
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
        dec_setup = np.concatenate(
            [[hidden_size], dec_steps.repeat(2), [input_length]]
        )
        enc_setup = dec_setup[::-1]

        layers = np.array(
            [
                [nn.Linear(int(a), int(b)), nn.Tanh()]
                for a, b in enc_setup.reshape(-1, 2)
            ]
        ).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        self.to_device(self._encoder)

    def forward(self, ts_batch, return_latent: bool = False):
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(flattened_sequence.float())
        return enc


class DecoderModule(nn.Module, PyTorchUtils):
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
        dec_setup = np.concatenate(
            [[hidden_size], dec_steps.repeat(2), [input_length]]
        )

        layers = np.array(
            [
                [nn.Linear(int(a), int(b)), nn.Tanh()]
                for a, b in dec_setup.reshape(-1, 2)
            ]
        ).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)
        self.to_device(self._decoder)

    def forward(self, ts_batch, output_size):
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        dec = self._decoder(flattened_sequence)
        reconstructed_sequence = dec.view(output_size)
        return reconstructed_sequence


class DiscriminatorModule(nn.Module, PyTorchUtils):
    def __init__(self, hidden_size: int, seed: int, gpu: int, activation: nn.Module):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        N = 10
        layers = np.array(
            [
                nn.Linear(hidden_size, N),
                activation(),
                nn.Linear(N, N),
                activation(),
                nn.Linear(N, 1),
                nn.Sigmoid(),
            ]
        )
        self._discriminator = nn.Sequential(*layers)
        self.to_device(self._discriminator)

    def forward(self, ts_batch):
        return self._discriminator(ts_batch)
