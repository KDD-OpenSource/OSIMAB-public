import logging

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from scipy.stats import norm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from itertools import product
from .algorithm_utils import Algorithm, PyTorchUtils
from .helpers import make_sequences
from .helpers import split_sequences

# from .helpers import average_sequences


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
        self.used_error_vects = 0

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

    @Algorithm.measure_train_time
    def fit(self, X: pd.DataFrame):
        X, sequences = self.data_preprocessing(X)
        train_ind, train_gaussian_ind = self.get_random_indices(len(sequences))
        train_loader, train_gaussian_loader = self.get_dataloaders(
            sequences, train_ind, train_gaussian_ind
        )
        self.input_size = X.shape[1]

        self.set_autoencoder_module(X)
        self.num_params = (sum(p.numel() for p in self.aed.parameters() if p.requires_grad))

        self.to_device(self.aed)  # .double()
        optimizer = torch.optim.Adam(self.aed.parameters(), lr=self.lr)

        self.aed.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f"Epoch {epoch+1}/{self.num_epochs}.")
            epochloss = 0
            for ts_batch in train_loader:
                output = self.aed(self.to_var(ts_batch))
                loss = torch.mean(self.calc_errors(ts_batch, output))
                epochloss += loss
                self.aed.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch Loss: {epochloss}")

        self.train_gaussians(train_gaussian_loader)

    def train_gaussians(self, train_gaussian_loader):
        self.aed.eval()
        for ts_batch in train_gaussian_loader:
            output = self.aed(self.to_var(ts_batch))
            # error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error = self.calc_errors(ts_batch, output)
            self.update_gaussians(error)

        if len(train_gaussian_loader) == 0:
            if self.window_anomaly:
                self.mean = [0]
                self.cov = np.identity(1)* 0.00000001

        self.anomaly_thresholds = []
        for mean, var in zip(self.mean, np.diagonal(self.cov)):
            normdist = norm(loc=mean, scale=np.sqrt(var))
            self.anomaly_thresholds.append(-normdist.logpdf(mean + 3 * np.sqrt(var)))

    def update_gaussians(self, error):
        (self.mean, self.cov) = self.update_gaussians_one_side(
            error, self.mean, self.cov
        )
        self.used_error_vects += error.size()[0]

    def update_gaussians_one_side(self, errors, mean, cov):
        # errors = errors.reshape(-1, self.input_size)
        errors = errors.data.cpu().numpy()
        if mean is None or cov is None:
            mean = np.mean(errors)
            cov = np.cov(errors)
        else:
            localErrorCount = 0
            try:
                mean.shape[0]
            except:
                mean = mean[np.newaxis]
            try:
                cov_dim = cov.shape[0]
            except:
                cov = cov[np.newaxis][np.newaxis]
                cov_dim = cov.shape[0]
            summedcov_new = np.empty(shape=(cov.shape))
            for error in errors:
                try:
                    error.shape[0]
                except:
                    error = error[np.newaxis]
                # Mean Calculation
                mean_old = mean
                numErrorsAfterUpdate = self.used_error_vects + localErrorCount + 1
                mean_new = (
                    1
                    / (numErrorsAfterUpdate)
                    * (error + (numErrorsAfterUpdate - 1) * mean_old)
                )
                localErrorCount += 1
                # Cov Calculation
                cov_old = cov
                numErrorsBeforeUpdate = numErrorsAfterUpdate - 1
                summedcov_old = cov_old * numErrorsBeforeUpdate
                for i, j in product(range(cov_dim), range(cov_dim)):
                    summedcov_new[i, j] = summedcov_old[i, j] + (
                        error[i] - mean_old[i]
                    ) * (error[j] - mean_new[j])
                cov_new = summedcov_new / numErrorsAfterUpdate

                mean = mean_new
                cov = cov_new
        return mean, cov

    def calc_errors(self, ts, output):
        error = nn.MSELoss(reduction="none")(output, self.to_var(ts.float()))
        error = torch.mean(error, dim=(1, 2))
        return error

    def data_preprocessing(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [
            data[i : i + self.sequence_length]
            for i in range(data.shape[0] - self.sequence_length + 1)
        ]
        return X, sequences

    def get_random_indices(self, num_sequences):
        indices = np.random.permutation(num_sequences)
        if self.train_max is None:
            self.train_max = 1
        train_max_point = int(self.train_max * num_sequences)
        split_point = train_max_point - int(
            self.train_gaussian_percentage * train_max_point
        )
        train_ace_indices = indices[:split_point]
        train_gaussian_indices = indices[split_point:train_max_point]
        return train_ace_indices, train_gaussian_indices

    def get_dataloaders(self, sequences, train_ace_ind, train_gaussian_ind):
        train_ace_loader = DataLoader(
            dataset=sequences,
            batch_size=self.batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_ace_ind),
            pin_memory=True,
        )
        train_gaussian_loader = DataLoader(
            dataset=sequences,
            batch_size=self.batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_gaussian_ind),
            pin_memory=True,
        )
        return train_ace_loader, train_gaussian_loader

    @Algorithm.measure_test_time
    def predict(self, X: pd.DataFrame) -> np.array:
        X, sequences = self.data_preprocessing(X)
        # train_ind, train_gaussian_ind = self.get_random_indices(len(sequences))
        # train_loader, train_gaussian_loader = self.get_dataloaders(
        # sequences, train_ind, train_gaussian_ind
        # )
        self.input_size = X.shape[1]

        # sequences = make_sequences(
        #    data=X, sequence_length=self.sequence_length, stride=self.stride
        # )
        data_loader = DataLoader(
            dataset=sequences,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.aed.eval()
        normal = norm(self.mean[0], self.cov.flatten()[0])
        # mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        encodings = []
        errors = []

        data_loader_len = len(data_loader)
        for idx, ts in enumerate(data_loader):
            output, enc = self.aed(self.to_var(ts), return_latent=True)
            error = self.calc_errors(ts, output)
            score = -normal.logpdf(error.data.cpu().numpy())
            # scores.append(score.reshape(ts.size(0), self.sequence_length))
            scores.append(score)
            print(idx/data_loader_len)
            if self.details:
                outputs.append(output.data.numpy())
                encodings.append(enc)
                errors.append(error.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        # import pdb; pdb.set_trace()
        scores = self.spread_seq_over_time_and_aggr(
            scores, self.sequence_length, X, aggr="mean"
        )
        self.anomaly_values = self.get_anomaly_values(X, scores)

        if self.details:
            output_lattice = np.full(
                (self.sequence_length, X.shape[0], X.shape[1]), np.nan
            )
            outputs = self.calc_lattice(X, outputs, output_lattice)

            # outputs = self.spread_seq_over_time_and_aggr(
            # sequence=outputs,
            # time_length=self.sequence_length,
            # X=X,
            # a#ggr='mean',
            # )
            self.prediction_details.update({"reconstructions_mean": outputs})

            errors = self.spread_seq_over_time_and_aggr(
                sequence=errors,
                time_length=self.sequence_length,
                X=X,
                aggr="mean",
            )
            self.prediction_details.update({"errors_mean": errors})

            encodings = [e.detach().numpy() for e in encodings]
            encodings = np.concatenate(encodings)
            # self.prediction_details.update({"encodings": encodings})

        return scores

    def get_anomaly_values(self, X, scoresSensors):
        anomalyValues = scoresSensors.T > self.anomaly_thresholds
        anomalyValues_ints = np.zeros(shape=X.shape)
        anomalyValues_ints[anomalyValues == True] = 1
        anomaly_values = pd.DataFrame(columns=X.columns, data=anomalyValues_ints)
        return anomaly_values

    def set_autoencoder_module(self, X: pd.DataFrame):
        if self.aed == None:
            self.input_size = X.shape[1]
            self.sensor_list = list(X.columns)
            self.aed = AutoEncoderModule(
                n_features=self.input_size,
                sequence_length=self.sequence_length,
                hidden_size=self.hidden_size,
                seed=self.seed,
                gpu=self.gpu,
            )
            self.to_device(self.aed)  # .double()
        elif list(X.columns) != self.sensor_list:
            raise ValueError(
                "You predict on other attributes than you trained on.\n"
                f"The model was trained using attributes {self.sensor_list}"
                f"The prediction data contains attributes {list(X.columns)}"
            )

    def save(self, path):
        os.makedirs(os.path.join("./results", self.name), exist_ok=True)
        torch.save(
            {
                "input_size": self.input_size,
                "sensor_list": self.sensor_list,
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "seed": self.seed,
                "gpu": self.gpu,
            },
            os.path.join(path, "model_detailed.pth"),
        )

        torch.save(
            {
                "input_size": self.input_size,
                "sensor_list": self.sensor_list,
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "seed": self.seed,
                "gpu": self.gpu,
            },
            os.path.join("./results", self.name, "model_detailed.pth"),
        )

        torch.save(self.aed.state_dict(), os.path.join(path, "model.pth"))
        torch.save(
            self.aed.state_dict(), os.path.join("./results", self.name, "model.pth")
        )

        with open(os.path.join(path, "gaussian_param.npy"), "wb") as f:
            np.save(f, self.mean)
            np.save(f, self.cov)
            np.save(f, self.anomaly_thresholds)

        with open(os.path.join(path, "anomalyThresholds.txt"), "w") as f:
            np.savetxt(f, self.anomaly_thresholds)

        with open(
            os.path.join("./results", self.name, "gaussian_param.npy"), "wb"
        ) as f:
            np.save(f, self.mean)
            np.save(f, self.cov)
            np.save(f, self.anomaly_thresholds)

    #    def save(self, f):
    #        torch.save(
    #            {
    #                "model_state_dict": self.aed.state_dict(),
    #                "mean": self.mean,
    #                "cov": self.cov,
    #                "input_size": self.input_size,
    #                "sequence_length": self.sequence_length,
    #                "hidden_size": self.hidden_size,
    #                "seed": self.seed,
    #                "gpu": self.gpu,
    #            },
    #            f,
    #        )

    def load(self, path):
        model_details = torch.load(os.path.join(path, "model_detailed.pth"))

        self.input_size = model_details["input_size"]
        self.sensor_list = model_details["sensor_list"]
        self.sequence_length = model_details["sequence_length"]
        self.hidden_size = model_details["hidden_size"]
        self.seed = model_details["seed"]
        self.gpu = model_details["gpu"]

        self.aed = AutoEncoderModule(
            n_features=model_details["input_size"],
            sequence_length=model_details["sequence_length"],
            hidden_size=model_details["hidden_size"],
            seed=model_details["seed"],
            gpu=model_details["gpu"],
        )
        self.aed.load_state_dict(torch.load(os.path.join(path, "model.pth")))
        with open(os.path.join(path, "gaussian_param.npy"), "rb") as f:
            self.mean = np.load(f)
            self.cov = np.load(f)
            self.anomaly_thresholds = np.load(f)

    #    def load(self, f):
    #        checkpoint = torch.load(f)
    #        model_state_dict = checkpoint["model_state_dict"]
    #        del checkpoint["model_state_dict"]
    #        for key in checkpoint:
    #            setattr(self, key, checkpoint[key])
    #        self.aed = AutoEncoderModule(
    #            self.input_size,
    #            self.sequence_length,
    #            self.hidden_size,
    #            seed=self.seed,
    #            gpu=self.gpu,
    #        )
    #        self.aed.load_state_dict(model_state_dict)
    def spread_seq_over_time_and_aggr(self, sequence, time_length, X, aggr):
        # sequence is a list of np arrays of shape (20,)
        sequence = np.hstack(sequence)
        sequence_repeated = np.repeat(sequence, time_length).reshape(-1, time_length)
        # sequence_stacked = np.dstack(sequence_repeated)
        lattice = np.full((time_length, X.shape[0]), np.nan)
        for i, elem in enumerate(sequence_repeated):
            lattice[i % time_length, i : i + time_length] = elem
        if aggr == "mean":
            aggr_lattice = np.nanmean(lattice, axis=0).T
        if aggr == "max":
            aggr_lattice = np.nanmax(lattice, axis=0).T
        if aggr == "min":
            aggr_lattice = np.nanmin(lattice, axis=0).T
        return aggr_lattice

    def calc_lattice(self, X, data, lattice, aggregate="mean"):
        data = np.concatenate(data)
        for i, elem in enumerate(data):
            lattice[i % self.sequence_length, i : i + self.sequence_length] = elem
        if aggregate == "mean":
            return np.nanmean(lattice, axis=0).T
        elif aggregate == "median":
            return np.nanmedian(lattice, axis=0).T
        else:
            raise Exception("You must specify an aggregation method")


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
