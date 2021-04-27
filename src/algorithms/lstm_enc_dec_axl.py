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


class LSTMED(Algorithm, PyTorchUtils):
    def __init__(
        self,
        name: str = "LSTM-ED",
        num_epochs: int = 10,
        batch_size: int = 20,
        lr: float = 1e-3,
        hidden_size: int = 5,
        sequence_length: int = 30,
        window_anomaly: bool = True,
        stride: int = 1,
        train_gaussian_percentage: float = 0.25,
        train_max: float = 1.0,
        n_layers: tuple = (1, 1),
        use_bias: tuple = (True, True),
        dropout: tuple = (0, 0),
        seed: int = None,
        gpu: int = None,
        details=True,
    ):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.window_anomaly = window_anomaly
        self.stride = stride
        self.train_gaussian_percentage = train_gaussian_percentage
        self.train_max = train_max
        self.used_error_vects = 0

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstmed = None
        self.mean, self.cov = None, None

    @Algorithm.measure_train_time
    def fit(self, X: pd.DataFrame):
        # torch.set_num_threads(1)
        self.input_size = X.shape[1]
        sequences = make_sequences(data=X, sequence_length=self.sequence_length)
        sequences = sequences[: int(self.train_max * len(sequences))]
        seq_train, seq_val = split_sequences(sequences, self.train_gaussian_percentage)
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

        self.set_lstmed_module(X)
        self.num_params = (sum(p.numel() for p in self.lstmed.parameters() if p.requires_grad))
        self.to_device(self.lstmed)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)

        self.lstmed.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f"Epoch {epoch+1}/{self.num_epochs}.")
            for ts_batch in train_loader:
                output = self.lstmed(self.to_var(ts_batch))
                loss = nn.MSELoss(reduction="sum")(
                    output, self.to_var(ts_batch.float())
                )
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()

        self.train_gaussians(train_gaussian_loader)

    def train_gaussians(self, train_gaussian_loader):
        self.lstmed.eval()
        for ts_batch in train_gaussian_loader:
            output = self.lstmed(self.to_var(ts_batch))
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
        if len(errors.shape) > 1:
            errors = errors.reshape(-1, self.input_size)
        errors = errors.data.cpu().numpy()
        if mean is None or cov is None:
            mean = np.mean(errors, axis=0)
            cov = np.cov(errors, rowvar=False)
        else:
            localErrorCount = 0
            try:
                cov_dim = cov.shape[0]
            except:
                cov = cov[np.newaxis][np.newaxis]
                cov_dim = cov.shape[0]
            summedcov_new = np.empty(shape=(cov.shape))
            for error in errors:
                try:
                    error[0]
                except:
                    error = error[np.newaxis]
                try:
                    mean[0]
                    mean_old = mean
                except:
                    mean_old = mean[np.newaxis]
                # Mean Calculation
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

    def set_lstmed_module(self, X: pd.DataFrame):
        if self.lstmed == None:
            self.input_size = X.shape[1]
            self.sensor_list = list(X.columns)
            self.lstmed = LSTMEDModule(
                n_features=self.input_size,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                use_bias=self.use_bias,
                dropout=self.dropout,
                seed=self.seed,
                gpu=self.gpu,
            )
            self.to_device(self.lstmed)  # .double()
        #        elif len(self.aed._encoder) != X.shape[1]:
        #            raise Exception(
        #                "You cannot continue training the autoencoder,"
        #                "because the autoencoders structure does not match the"
        #                "structuro of the data."
        #            )
        elif list(X.columns) != self.sensor_list:
            raise ValueError(
                "You predict on other attributes than you trained on.\n"
                f"The model was trained using attributes {self.sensor_list}"
                f"The prediction data contains attributes {list(X.columns)}"
            )

    @Algorithm.measure_test_time
    def predict(self, X: pd.DataFrame) -> np.array:
        self.lstmed.eval()
        X, sequences = self.data_preprocessing(X)
        if list(X.columns) != self.sensor_list:
            raise ValueError(
                "You predict on other attributes than you trained on.\n"
                f"The model was trained using attributes {self.sensor_list}"
                f"The prediction data contains attributes {list(X.columns)}"
            )
        data_loader = DataLoader(
            dataset=sequences,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        if self.window_anomaly == False:
            scores = self.predict_sensor_anomaly(X, data_loader)
            return scores
        elif self.window_anomaly == True:
            scores = self.predict_window_anomaly(X, data_loader)
            return scores

    def predict_window_anomaly(self, X: pd.DataFrame, data_loader):
        sensorNormals = []
        for mean, var in zip(self.mean, np.diagonal(self.cov)):
            sensorNormals.append(norm(loc=mean, scale=np.sqrt(var)))
        normal = norm(loc = self.mean, scale = np.sqrt(self.cov[0]))

        scores = []
        outputs = []
        errors = []
        data_loader_len = len(data_loader)
        for idx, ts in enumerate(data_loader):
            print(idx/data_loader_len)
            output = self.lstmed(self.to_var(ts))
            error = self.calc_errors(ts, output)
            score = -normal.logpdf(error.data.cpu().numpy())
            scores.append(score)
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        scores = self.concat_batches(scores)
        scores = self.spread_seq_over_time_and_aggr(
            scores, self.sequence_length, X, "mean"
        )

        lattice_sensors = np.full(
            (self.sequence_length, X.shape[0], X.shape[1]), np.nan
        )
        # check how to best get anomaly_values
        self.anomaly_values = self.get_window_anomaly_values(X, scores)

        if self.details:

            self.prediction_details.update(
                {"anomaly_values": self.anomaly_values.values.T}
            )

            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for idx, output in enumerate(outputs):
                i = idx * self.stride
                lattice[
                    i % self.sequence_length, i : i + self.sequence_length, :
                ] = output
            self.prediction_details.update(
                {"reconstructions_mean": np.nanmean(lattice, axis=0).T}
            )

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for idx, error in enumerate(errors):
                i = idx * self.stride
                lattice[
                    i % self.sequence_length, i : i + self.sequence_length, :
                ] = error
            self.prediction_details.update(
                {"errors_mean": np.nanmean(lattice, axis=0).T}
            )

        return scores


    def predict_sensor_anomaly(self, X: pd.DataFrame, data_loader):
#        X.interpolate(inplace=True)
#        X.bfill(inplace=True)
#        data = X.values
#        sequences = make_sequences(data=X, sequence_length=self.sequence_length)
#        data_loader = DataLoader(
#            dataset=sequences,
#            batch_size=self.batch_size,
#            shuffle=False,
#            drop_last=False,
#        )
#
#        self.lstmed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        # add normals here
        sensorNormals = []
        for mean, var in zip(self.mean, np.diagonal(self.cov)):
            sensorNormals.append(norm(loc=mean, scale=np.sqrt(var)))

        scores = []
        scoresSensors = []
        outputs = []
        errors = []
        data_loader_len = len(data_loader)
        for idx, ts in enumerate(data_loader):
            print(idx/data_loader_len)
            output = self.lstmed(self.to_var(ts))
            error = self.calc_errors(ts, output)
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())

            scoreSensors = []
            for sensorInd in range(self.input_size):
                scoreSensors.append(
                    -sensorNormals[sensorInd].logpdf(error[:, :, sensorInd].data.cpu())
                )
                # scoreSensors[-1] = np.repeat(
                #    scoreSensors[-1], self.sequence_length
                # ).reshape(ts.size(0), self.sequence_length)
            scoreSensors = np.dstack(scoreSensors)
            scoresSensors.append(scoreSensors)

            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, data.shape[0]), np.nan)
        for idx, score in enumerate(scores):
            i = idx * self.stride
            lattice[i % self.sequence_length, i : i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)

        lattice_sensors = np.full(
            (self.sequence_length, X.shape[0], X.shape[1]), np.nan
        )
        # scoresSensors_lhs = self.calc_lattice(X, scoresSensors_lhs,
        # lattice_sensors_lhs, 'median')
        scoresSensors = self.calc_lattice(X, scoresSensors, lattice_sensors)

        self.anomaly_values = self.get_sensor_anomaly_values(X, scoresSensors)

        if self.details:

            self.prediction_details.update(
                {"anomaly_values": self.anomaly_values.values.T}
            )

            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for idx, output in enumerate(outputs):
                i = idx * self.stride
                lattice[
                    i % self.sequence_length, i : i + self.sequence_length, :
                ] = output
            self.prediction_details.update(
                {"reconstructions_mean": np.nanmean(lattice, axis=0).T}
            )

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for idx, error in enumerate(errors):
                i = idx * self.stride
                lattice[
                    i % self.sequence_length, i : i + self.sequence_length, :
                ] = error
            self.prediction_details.update(
                {"errors_mean": np.nanmean(lattice, axis=0).T}
            )

        return scores

    def get_sensor_anomaly_values(self, X, scoresSensors):
        anomalyValues = scoresSensors.T > self.anomaly_thresholds
        anomalyValues_ints = np.zeros(shape=X.shape)
        anomalyValues_ints[anomalyValues == True] = 1
        anomaly_values = pd.DataFrame(columns=X.columns, data=anomalyValues_ints)
        return anomaly_values

    def get_window_anomaly_values(self, X, scores):
        anomalyValues = scores > self.anomaly_thresholds
        anomalyValues_ints = np.zeros(shape=scores.shape)
        anomalyValues_ints[anomalyValues == True] = 1
        anomaly_values = pd.DataFrame(data=anomalyValues_ints)
        return anomaly_values

    def calc_errors(self, ts, output):
        #error = nn.L1Loss(reduction="none")(output, self.to_var(ts.float()))
        # changed to MSE as in training MSE has been used as well
        error = nn.MSELoss(reduction="none")(output, self.to_var(ts.float()))
        if self.window_anomaly:
            error = torch.mean(error, dim=(1,2))
        return error

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

    def save(self, path):
        os.makedirs(os.path.join("./results", self.name), exist_ok=True)
        torch.save(
            {
                "input_size": self.input_size,
                "sensor_list": self.sensor_list,
                "n_layers": self.n_layers,
                "use_bias": self.use_bias,
                "dropout": self.dropout,
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
                "n_layers": self.n_layers,
                "use_bias": self.use_bias,
                "dropout": self.dropout,
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "seed": self.seed,
                "gpu": self.gpu,
            },
            os.path.join("./results", self.name, "model_detailed.pth"),
        )

        torch.save(self.lstmed.state_dict(), os.path.join(path, "model.pth"))
        torch.save(
            self.lstmed.state_dict(), os.path.join("./results", self.name, "model.pth")
        )

        with open(os.path.join(path, "gaussian_param.npy"), "wb") as f:
            np.save(f, self.mean)
            np.save(f, self.cov)
            np.save(f, self.anomaly_thresholds)

        with open(os.path.join(path, "anomalyThresholds.txt"), "w") as f:
            np.savetxt(f, self.anomaly_thresholds)
        #        with open(os.path.join(path, "anomalyThresholds_rhs.txt"), "w") as f:
        #            np.savetxt(f, self.anomaly_thresholds_rhs)

        with open(
            os.path.join("./results", self.name, "gaussian_param.npy"), "wb"
        ) as f:
            np.save(f, self.mean)
            np.save(f, self.cov)
            #            np.save(f, self.mean_lhs)
            #            np.save(f, self.mean_rhs)
            #            np.save(f, self.cov_lhs)
            #            np.save(f, self.cov_rhs)
            np.save(f, self.anomaly_thresholds)

    #            np.save(f, self.anomaly_thresholds_rhs)

    def load(self, path):
        model_details = torch.load(os.path.join(path, "model_detailed.pth"))

        self.input_size = model_details["input_size"]
        self.sensor_list = model_details["sensor_list"]
        self.sequence_length = model_details["sequence_length"]
        self.hidden_size = model_details["hidden_size"]
        self.seed = model_details["seed"]
        self.gpu = model_details["gpu"]

        self.lstmed = LSTMEDModule(
            n_features=model_details["input_size"],
            hidden_size=model_details["hidden_size"],
            n_layers=model_details["n_layers"],
            use_bias=model_details["use_bias"],
            dropout=model_details["dropout"],
            seed=model_details["seed"],
            gpu=model_details["gpu"],
        )
        self.lstmed.load_state_dict(torch.load(os.path.join(path, "model.pth")))
        with open(os.path.join(path, "gaussian_param.npy"), "rb") as f:
            self.mean = np.load(f)
            self.cov = np.load(f)
            self.anomaly_thresholds = np.load(f)

    def data_preprocessing(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [
            data[i : i + self.sequence_length]
            for i in range(data.shape[0] - self.sequence_length + 1)
        ]
        return X, sequences

class LSTMEDModule(nn.Module, PyTorchUtils):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        n_layers: tuple,
        use_bias: tuple,
        dropout: tuple,
        seed: int,
        gpu: int,
    ):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[0],
            bias=self.use_bias[0],
            dropout=self.dropout[0],
        )
        self.to_device(self.encoder)
        self.decoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[1],
            bias=self.use_bias[1],
            dropout=self.dropout[1],
        )
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        return (
            self.to_var(
                torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()
            ),
            self.to_var(
                torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()
            ),
        )

    def forward(self, ts_batch, return_latent: bool = False):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(
            ts_batch.float(), enc_hidden
        )  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(
                    ts_batch[:, i].unsqueeze(1).float(), dec_hidden
                )
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output
