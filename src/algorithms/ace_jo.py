import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
from scipy.stats import norm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from matplotlib.animation import FuncAnimation
from itertools import product

from .algorithm_utils import Algorithm, PyTorchUtils


class AutoEncoderJO(Algorithm, PyTorchUtils):
    def __init__(
        self,
        name: str = "AutoEncoderJO",
        num_epochs: int = 10,
        batch_size: int = 20,
        lr: float = 1e-4,
        hidden_size1: int = 5,
        hidden_size2: int = 2,
        sequence_length: int = 30,
        train_gaussian_percentage: float = 0.25,
        seed: int = 123,
        gpu: int = None,
        details=True,
        latentVideo=True,
        train_max=None,
        sensor_specific=True,
        corr_loss=True,
    ):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.sensor_specific = sensor_specific
        self.compute_corr_loss = corr_loss
        self.input_size = None
        self.sensor_list = None
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage
        self.train_max = train_max
        self.latentVideo = latentVideo
        self.anomaly_thresholds_lhs = []
        self.anomaly_thresholds_rhs = []
        self.anomaly_values_or = None
        self.anomaly_values_xor = None

        self.encoding_details = {}

        self.aed = None
        self.mean_lhs, self.var_lhs, self.cov_lhs = None, None, None
        self.mean_rhs, self.var_rhs, self.cov_rhs = None, None, None
        self.used_error_vects = 0

    def fit(self, X: pd.DataFrame, path):
        # Data preprocessing
        X, sequences = self.data_preprocessing(X)
        train_ace_ind, train_gaussian_ind = self.get_random_indices(len(sequences))
        train_ace_loader, train_gaussian_loader = self.get_dataloaders(
            sequences, train_ace_ind, train_gaussian_ind
        )
        self.input_size = X.shape[1]

        # Train
        self.set_ace_module(X)
        optimizer = torch.optim.Adam(self.aed.parameters(), lr=self.lr)
        self.train_ace(train_ace_loader, optimizer)

        # Validate (train gaussians)
        self.train_gaussians(train_gaussian_loader)

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

    def set_ace_module(self, X: pd.DataFrame):
        if self.aed == None:
            self.input_size = X.shape[1]
            self.sensor_list = list(X.columns)
            self.aed = ACEModule(
                self.input_size,
                self.sequence_length,
                self.hidden_size1,
                self.hidden_size2,
                seed=self.seed,
                gpu=self.gpu,
            )
            self.to_device(self.aed)  # .double()
        elif len(self.aed._encoder) != X.shape[1]:
            raise Exception(
                "You cannot continue training the autoencoder,"
                "because the autoencoders structure does not match the"
                "structuro of the data."
            )
        elif list(X.columns) != self.sensor_list:
            raise ValueError(
                "You predict on other attributes than you trained on.\n"
                f"The model was trained using attributes {self.sensor_list}"
                f"The prediction data contains attributes {list(X.columns)}"
            )

    def train_ace(self, train_ace_loader, optimizer):
        self.aed.train()
        alpha, beta = self.initTradeoff()
        for epoch in trange(self.num_epochs):
            print(f"Alpha: {alpha}")
            print(f"Beta: {beta}")
            epochLossLhs = 0
            epochLossRhs = 0
            latentSpace = []
            logging.debug(f"Epoch {epoch+1}/{self.num_epochs}.")
            for ts_batch in train_ace_loader:
                self.aed.zero_grad()
                output = self.aed(self.to_var(ts_batch), return_latent=True)
                latentSpace.append(output[2])
                loss1, loss2 = self.calcLosses(ts_batch, output)
                epochLossLhs += loss1
                epochLossRhs += loss2
                (alpha * loss1 + beta * loss2).backward()
                optimizer.step()
            alpha, beta = self.updateTradeoff(alpha, beta, epoch + 1)
            latentSpace = np.vstack(
                list(map(lambda x: x.detach().numpy(), latentSpace))
            )
            self.printIntermedResults(epoch, epochLossLhs, epochLossRhs, latentSpace)

    def initTradeoff(self):
        alpha = 1
        beta = 0
        return alpha, beta

    def updateTradeoff(self, alpha, beta, epoch):
        alpha = 1 - epoch / (self.num_epochs - 1)
        beta = epoch / (self.num_epochs - 1)
        return alpha, beta

    def calcLosses(self, ts_batch, output):
        loss1 = nn.MSELoss(reduction="mean")(output[0], self.to_var(ts_batch.float()))
        loss2 = 0
        if not self.sensor_specific and not self.compute_corr_loss:
            loss2 += nn.MSELoss(reduction="mean")(
                output[1], output[2].view((ts_batch.size()[0], -1)).data
            )

        if self.sensor_specific:
            loss2 += torch.mean(
                self.sensor_specific_loss(
                    output[1], output[2].view((ts_batch.size()[0], -1)).data
                )
            )

        if self.compute_corr_loss:
            loss2 += torch.mean(
                self.corr_loss(
                    output[1], output[2].view((ts_batch.size()[0], -1)).data
                )
            )
        return loss1, loss2

    def sensor_specific_loss(self, yhat, y):
        subclassLength = self.hidden_size1
        yhat = yhat.view((-1, subclassLength))
        y = y.view((-1, subclassLength))
        error = yhat - y
        sqr_err = error ** 2
        sum_sqr_err = sqr_err.sum(1)
        root_sum_sqr_err = torch.sqrt(sum_sqr_err)
        sqr_sum_sqr_err = sum_sqr_err ** 2
        return root_sum_sqr_err
        # return sqr_sum_sqr_err
        # return sum_sqr_err

    def corr_loss(self, yhat, y):
        subclassLength = self.hidden_size1
        yhat = yhat.view((-1, subclassLength))
        y = y.view((-1, subclassLength))
        vhat = yhat - torch.mean(yhat, 0)
        vy = y - torch.mean(y, 0)
        cost = torch.sum(vhat * vy, 1)
        cost1 = torch.rsqrt(torch.sum(vhat ** 2, 1))
        cost2 = torch.rsqrt(torch.sum(vy ** 2, 1))
        cost = 1.0 - torch.abs(torch.mean(cost * cost1 * cost2))
        return cost

    def printIntermedResults(self, epoch, epochLossLhs, epochLossRhs, latentSpace):
        print(f"Epoch {epoch}")
        print(f"Epoch Loss Lhs: {epochLossLhs}")
        print(f"Epoch Loss Rhs: {epochLossRhs}")
        # print("Mean of Latent Space is:")
        print(latentSpace.mean(axis=0))
        # print("Standard Deviation of Latent Space is:")
        print(latentSpace.std(axis=0))

    def train_gaussians(self, train_gaussian_loader):
        self.aed.eval()
        for ts_batch in train_gaussian_loader:
            output = self.aed(self.to_var(ts_batch), return_latent=True)
            error_lhs, error_rhs = self.calc_errors(ts_batch, output)
            self.update_gaussians(error_lhs, error_rhs)

        self.anomaly_thresholds_lhs = []
        self.anomaly_thresholds_rhs = []
        for mean, var in zip(self.mean_lhs, np.diagonal(self.cov_lhs)):
            normdist = norm(loc=mean, scale=np.sqrt(var))
            self.anomaly_thresholds_lhs.append(
                -normdist.logpdf(mean + 3 * np.sqrt(var))
            )
        for mean, var in zip(self.mean_rhs, np.diagonal(self.cov_rhs)):
            normdist = norm(loc=mean, scale=np.sqrt(var))
            self.anomaly_thresholds_rhs.append(
                -normdist.logpdf(mean + 3 * np.sqrt(var))
            )

    def update_gaussians(self, error_lhs, error_rhs):
        (self.mean_lhs, self.cov_lhs) = self.update_gaussians_one_side(
            error_lhs, self.mean_lhs, self.cov_lhs
        )
        (self.mean_rhs, self.cov_rhs) = self.update_gaussians_one_side(
            error_rhs, self.mean_rhs, self.cov_rhs
        )
        self.used_error_vects += error_lhs.size()[0]

    def update_gaussians_one_side(self, errors, mean, cov):
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

    def predict(self, X: pd.DataFrame) -> np.array:
        self.aed.eval()
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

        (
            mvnormal,
            mvnormal_rhs,
            sensorNormals_lhs,
            sensorNormals_rhs,
        ) = self.set_gaussians()

        scores_lhs = []
        scores_rhs = []
        scoresSensors_lhs = []
        scoresSensors_rhs = []
        outputs_lhs = []
        encodings_lhs = []
        encodings_rhs = []
        outputs_rhs = []
        errors_lhs = []
        errors_rhs = []

        for idx, ts in enumerate(data_loader):
            output = self.aed(self.to_var(ts), return_latent=True)
            error_lhs, error_rhs = self.calc_errors(ts, output)

            (
                score_lhs,
                score_rhs,
                scoreSensors_lhs,
                scoreSensors_rhs,
            ) = self.calc_score(
                X,
                ts,
                output,
                error_lhs,
                error_rhs,
                mvnormal,
                mvnormal_rhs,
                sensorNormals_lhs,
                sensorNormals_rhs,
            )
            scoresSensors_lhs.append(scoreSensors_lhs)
            scores_lhs.append(score_lhs.reshape(ts.size(0), self.sequence_length))
            scoresSensors_rhs.append(scoreSensors_rhs)
            scores_rhs.append(score_rhs)

            if self.details:
                outputs_lhs.append(output[0].data.numpy())
                encodings_lhs.append(output[2].data.numpy())
                encodings_rhs.append(output[3].data.numpy())
                outputs_rhs.append(output[1].data.numpy())
                errors_lhs.append(error_lhs.data.numpy())
                errors_rhs.append(error_rhs.data.numpy())

        (
            scores_lhs,
            scores_rhs,
            scoresSensors_lhs,
            scoresSensors_rhs,
        ) = self.calc_score_lattices(
            X, scores_lhs, scores_rhs, scoresSensors_lhs, scoresSensors_rhs
        )

        self.anomaly_values_or = self.get_anomaly_values_or(
            X, scoresSensors_lhs, scoresSensors_rhs
        )
        self.anomaly_values_xor = self.get_anomaly_values_xor(
            X, scoresSensors_lhs, scoresSensors_rhs
        )
        anomaly_values_lhs = (scoresSensors_lhs.T > self.anomaly_thresholds_lhs).T
        anomaly_values_rhs = (scoresSensors_rhs.T > self.anomaly_thresholds_rhs).T
        # self.anomaly_values_xor = anomaly_values_lhs ^ anomaly_values_rhs

        if self.details:
            self.prediction_details.update(
                {"anomaly_values": self.anomaly_values_or.values.T}
            )

            self.prediction_details.update({"anomaly_values_lhs": anomaly_values_lhs})
            self.prediction_details.update({"anomaly_values_rhs": anomaly_values_rhs})

            # New
            self.prediction_details.update(
                {"anomaly_values_diff": self.anomaly_values_xor.values.T}
            )

            self.prediction_details.update({"scoresSensors_lhs": scoresSensors_lhs})
            self.prediction_details.update({"scoresSensors_rhs": scoresSensors_rhs})
            self.prediction_details.update({"scores_lhs": scores_lhs})
            self.prediction_details.update({"scores_rhs": scores_rhs})

            lattice_reconstr = np.full(
                (self.sequence_length, X.shape[0], X.shape[1]), np.nan
            )
            reconstructions_mean = self.calc_lattice(X, outputs_lhs, lattice_reconstr)
            self.prediction_details.update(
                {"reconstructions_mean": reconstructions_mean}
            )

            lattice_error_lhs = np.full(
                (self.sequence_length, X.shape[0], X.shape[1]), np.nan
            )
            errors_mean_lhs = self.calc_lattice(X, errors_lhs, lattice_error_lhs)
            self.prediction_details.update({"errors_mean_lhs": errors_mean_lhs})

            lattice_error_rhs = np.full(
                (self.sequence_length, X.shape[0], X.shape[1]), np.nan
            )
            errors_mean_rhs = self.calc_lattice(X, errors_rhs, lattice_error_rhs)
            self.prediction_details.update({"errors_mean_rhs": errors_mean_rhs})

            # animation
            if self.latentVideo:
                self.createLatentVideo(
                    encodings_lhs, encodings_rhs, outputs_rhs, sequences
                )

        return scores_lhs + scores_rhs

    def set_gaussians(self):
        mvnormal = multivariate_normal(
            self.mean_lhs, self.cov_lhs, allow_singular=True
        )
        mvnormal_rhs = multivariate_normal(
            self.mean_rhs, self.cov_rhs, allow_singular=True
        )
        sensorNormals_lhs = []
        for mean, var in zip(self.mean_lhs, np.diagonal(self.cov_lhs)):
            sensorNormals_lhs.append(norm(loc=mean, scale=np.sqrt(var)))
        sensorNormals_rhs = []
        for mean, var in zip(self.mean_rhs, np.diagonal(self.cov_rhs)):
            sensorNormals_rhs.append(norm(loc=mean, scale=np.sqrt(var)))
        return mvnormal, mvnormal_rhs, sensorNormals_lhs, sensorNormals_rhs

    def calc_errors(self, ts, output):
        error_lhs = nn.L1Loss(reduction="none")(
            output[0], self.to_var(ts.float())
        ).mean(axis=1)
        error_rhs = nn.L1Loss(reduction="none")(
            output[1].view(output[2].shape), output[2]
        ).mean(axis=2)

        return error_lhs, error_rhs

    def calc_score(
        self,
        X,
        ts,
        output,
        error_lhs,
        error_rhs,
        mvnormal,
        mvnormal_rhs,
        sensorNormals_lhs,
        sensorNormals_rhs,
    ):
        scoreSensors_lhs = []
        for sensorInd in range(self.input_size):
            scoreSensors_lhs.append(
                -sensorNormals_lhs[sensorInd].logpdf(
                    error_lhs[:, sensorInd].data.cpu()
                )
            )
            scoreSensors_lhs[-1] = np.repeat(
                scoreSensors_lhs[-1], self.sequence_length
            ).reshape(ts.size(0), self.sequence_length)
        scoreSensors_lhs = np.dstack(scoreSensors_lhs)

        score_lhs = -mvnormal.logpdf(
            error_lhs.reshape(-1, X.shape[1]).data.cpu().numpy()
        )
        score_lhs = np.repeat(score_lhs, self.sequence_length).reshape(
            ts.size(0), self.sequence_length
        )

        scoreSensors_rhs = []
        for sensorInd in range(self.input_size):
            scoreSensors_rhs.append(
                -sensorNormals_rhs[sensorInd].logpdf(
                    error_rhs[:, sensorInd].data.cpu()
                )
            )
            scoreSensors_rhs[-1] = np.repeat(
                scoreSensors_rhs[-1], self.sequence_length
            ).reshape(ts.size(0), self.sequence_length)
        scoreSensors_rhs = np.dstack(scoreSensors_rhs)

        score_rhs = -mvnormal_rhs.logpdf(
            error_rhs.view(-1, output[2].shape[1]).data.cpu().numpy()
        )
        score_rhs = np.repeat(score_rhs, self.sequence_length).reshape(
            ts.size(0), self.sequence_length
        )
        return score_lhs, score_rhs, scoreSensors_lhs, scoreSensors_rhs

    def calc_score_lattices(
        self, X, scores_lhs, scores_rhs, scoresSensors_lhs, scoresSensors_rhs
    ):

        lattice_lhs = np.full((self.sequence_length, X.shape[0]), np.nan)
        # scores_lhs = self.calc_lattice(X, scores_lhs, lattice_lhs, 'median')
        scores_lhs = self.calc_lattice(X, scores_lhs, lattice_lhs)

        lattice_rhs = np.full((self.sequence_length, X.shape[0]), np.nan)
        # scores_rhs = self.calc_lattice(X, scores_rhs, lattice_rhs, 'median')
        scores_rhs = self.calc_lattice(X, scores_rhs, lattice_rhs)

        lattice_sensors_lhs = np.full(
            (self.sequence_length, X.shape[0], X.shape[1]), np.nan
        )
        # scoresSensors_lhs = self.calc_lattice(X, scoresSensors_lhs,
        # lattice_sensors_lhs, 'median')
        scoresSensors_lhs = self.calc_lattice(
            X, scoresSensors_lhs, lattice_sensors_lhs
        )

        lattice_sensors_rhs = np.full(
            (self.sequence_length, X.shape[0], X.shape[1]), np.nan
        )
        # scoresSensors_rhs = self.calc_lattice(X, scoresSensors_rhs,
        # lattice_sensors_rhs, 'median')
        scoresSensors_rhs = self.calc_lattice(
            X, scoresSensors_rhs, lattice_sensors_rhs
        )

        return scores_lhs, scores_rhs, scoresSensors_lhs, scoresSensors_rhs

    def get_anomaly_values_or(self, X, scoresSensors_lhs, scoresSensors_rhs):
        anomalyValues_lhs = scoresSensors_lhs.T > self.anomaly_thresholds_lhs
        anomalyValues_rhs = scoresSensors_rhs.T > self.anomaly_thresholds_rhs

        combinedAnomalyValues = np.logical_or(anomalyValues_lhs, anomalyValues_rhs)
        combinedAnomalyValues_Ints = np.zeros(shape=X.shape)
        combinedAnomalyValues_Ints[combinedAnomalyValues == True] = 1
        anomaly_values = pd.DataFrame(
            columns=X.columns, data=combinedAnomalyValues_Ints
        )
        return anomaly_values

    def get_anomaly_values_xor(self, X, scoresSensors_lhs, scoresSensors_rhs):
        anomalyValues_lhs = scoresSensors_lhs.T > self.anomaly_thresholds_lhs
        anomalyValues_rhs = scoresSensors_rhs.T > self.anomaly_thresholds_rhs

        combinedAnomalyValues = np.logical_xor(anomalyValues_lhs, anomalyValues_rhs)
        combinedAnomalyValues_Ints = np.zeros(shape=X.shape)
        combinedAnomalyValues_Ints[combinedAnomalyValues == True] = 1
        anomaly_values = pd.DataFrame(
            columns=X.columns, data=combinedAnomalyValues_Ints
        )
        return anomaly_values

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
                "sequence_length": self.sequence_length,
                "hidden_size1": self.hidden_size1,
                "hidden_size2": self.hidden_size2,
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
                "hidden_size1": self.hidden_size1,
                "hidden_size2": self.hidden_size2,
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
            np.save(f, self.mean_lhs)
            np.save(f, self.mean_rhs)
            np.save(f, self.cov_lhs)
            np.save(f, self.cov_rhs)
            np.save(f, self.anomaly_thresholds_lhs)
            np.save(f, self.anomaly_thresholds_rhs)

        with open(os.path.join(path, "anomalyThresholds_lhs.txt"), "w") as f:
            np.savetxt(f, self.anomaly_thresholds_lhs)
        with open(os.path.join(path, "anomalyThresholds_rhs.txt"), "w") as f:
            np.savetxt(f, self.anomaly_thresholds_rhs)

        with open(
            os.path.join("./results", self.name, "gaussian_param.npy"), "wb"
        ) as f:
            np.save(f, self.mean_lhs)
            np.save(f, self.mean_rhs)
            np.save(f, self.cov_lhs)
            np.save(f, self.cov_rhs)
            np.save(f, self.anomaly_thresholds_lhs)
            np.save(f, self.anomaly_thresholds_rhs)

    def save_gaussian_tmp(self):
        joined_path = f"tmp/{self.name}"
        os.makedirs(os.path.join(os.getcwd(), joined_path), exist_ok=True)
        # os.mkdir(os.path.join(os.getcwd(), joined_path))
        with open(
            os.path.join(os.getcwd(), joined_path, "gaussian_param.npy"), "wb"
        ) as f:
            np.save(f, self.mean_lhs)
            np.save(f, self.mean_rhs)
            np.save(f, self.cov_lhs)
            np.save(f, self.cov_rhs)
            np.save(f, self.anomaly_thresholds_lhs)
            np.save(f, self.anomaly_thresholds_rhs)

        with open(
            os.path.join(os.getcwd(), joined_path, "anomalyThresholds_lhs.txt"), "w"
        ) as f:
            np.savetxt(f, self.anomaly_thresholds_lhs)
        with open(
            os.path.join(os.getcwd(), joined_path, "anomalyThresholds_rhs.txt"), "w"
        ) as f:
            np.savetxt(f, self.anomaly_thresholds_rhs)

    def load(self, path):
        model_details = torch.load(os.path.join(path, "model_detailed.pth"))

        self.input_size = model_details["input_size"]
        self.sensor_list = model_details["sensor_list"]
        self.sequence_length = model_details["sequence_length"]
        self.hidden_size1 = model_details["hidden_size1"]
        self.hidden_size2 = model_details["hidden_size2"]
        self.seed = model_details["seed"]
        self.gpu = model_details["gpu"]

        self.aed = ACEModule(
            model_details["input_size"],
            model_details["sequence_length"],
            model_details["hidden_size1"],
            model_details["hidden_size2"],
            seed=model_details["seed"],
            gpu=model_details["gpu"],
        )
        self.aed.load_state_dict(torch.load(os.path.join(path, "model.pth")))
        with open(os.path.join(path, "gaussian_param.npy"), "rb") as f:
            self.mean_lhs = np.load(f)
            self.mean_rhs = np.load(f)
            self.cov_lhs = np.load(f)
            self.cov_rhs = np.load(f)
            self.anomaly_thresholds_lhs = np.load(f)
            self.anomaly_thresholds_rhs = np.load(f)

        # self.anomaly_thresholds_lhs = []
        # self.anomaly_thresholds_rhs = []
        # for mean, var in zip(self.mean_lhs, np.diagonal(self.cov_lhs)):
        #    normdist = norm(loc=mean, scale=np.sqrt(var))
        #    self.anomaly_thresholds_lhs.append(
        #        -normdist.logpdf(mean + 2.5 * np.sqrt(var))
        #    )
        # for mean, var in zip(self.mean_rhs, np.diagonal(self.cov_rhs)):
        #    normdist = norm(loc=mean, scale=np.sqrt(var))
        #    self.anomaly_thresholds_rhs.append(
        #        -normdist.logpdf(mean + 2.5 * np.sqrt(var))
        #        #-normdist.logpdf(mean + 3.0 * np.sqrt(var))
        #    )
        # self.anomaly_thresholds_lhs = []
        # self.anomaly_thresholds_rhs = []
        # for mean, var in zip(self.mean_lhs, np.diagonal(self.cov_lhs)):
        #    normdist = norm(loc=mean, scale=np.sqrt(var))
        #    self.anomaly_thresholds_lhs.append(
        #        -normdist.logpdf(mean + 2.5 * np.sqrt(var))
        #    )
        # for mean, var in zip(self.mean_rhs, np.diagonal(self.cov_rhs)):
        #    normdist = norm(loc=mean, scale=np.sqrt(var))
        #    self.anomaly_thresholds_rhs.append(
        #        -normdist.logpdf(mean + 1.6 * np.sqrt(var))
        #    )
        # import pdb; pdb.set_trace()

    def createLatentVideo(self, encodings_lhs, encodings_rhs, outputs_rhs, sequences):
        # save in folder 'latentVideos' with timestamp?
        encodings_lhs = np.concatenate(encodings_lhs)
        encodings_rhs = np.concatenate(encodings_rhs)
        outputs_rhs = np.concatenate(outputs_rhs)
        for channel in range(self.input_size):
            self.encoding_details.update(
                {f"channel_{channel}": encodings_lhs[:, channel]}
            )

        numPlots = 50
        encodings_lhs = encodings_lhs.reshape((encodings_lhs.shape[0], -1))
        outputs_rhs = outputs_rhs.reshape((encodings_lhs.shape[0], -1))
        origDataTmp = np.array(sequences[: 10 * numPlots : 10])
        numChannels = origDataTmp.shape[2]

        ylim = []
        ylimEnc = [
            encodings_lhs.min() - 0.1 * abs(encodings_lhs.min()),
            encodings_lhs.max() + 0.1 * abs(encodings_lhs.max()),
        ]
        ylimOutRhs = [
            outputs_rhs.min() - 0.1 * abs(outputs_rhs.min()),
            outputs_rhs.max() + 0.1 * abs(outputs_rhs.max()),
        ]
        ylimEncRhs = [
            encodings_rhs.min() - 0.1 * abs(encodings_rhs.min()),
            encodings_rhs.max() + 0.1 * abs(encodings_rhs.max()),
        ]
        ylimLatent = [
            min(list([*ylimEnc, *ylimOutRhs])),
            max(list([*ylimEnc, *ylimOutRhs])),
        ]
        for channelInd in range(numChannels):
            channelMin = origDataTmp[:, :, channelInd].min()
            channelMax = origDataTmp[:, :, channelInd].max()
            ylim.append(
                [
                    channelMin - 0.1 * abs(channelMin),
                    channelMax + 0.1 * abs(channelMax),
                ]
            )

        fig, ax = plt.subplots(numChannels + 3, 1, figsize=(15, 10))
        lns = []
        for i in range(numChannels + 3):
            lns.append(ax[i].plot([], []))
        lns.append(ax[0].plot([], []))

        def init():
            ax[0].set_ylim(ylimLatent)
            ax[0].set_xlim(0, encodings_lhs.shape[1] + 1)
            ax[1].set_ylim(ylimLatent)
            ax[1].set_xlim(0, encodings_lhs.shape[1] + 1)
            ax[2].set_ylim(ylimEncRhs)
            ax[2].set_xlim(0, encodings_rhs.shape[1] + 1)
            for channelInd in range(numChannels):
                ax[channelInd + 3].set_ylim(ylim[channelInd])
                ax[channelInd + 3].set_xlim(
                    0, origDataTmp[0, :, channelInd].shape[0] - 1
                )

        def update(frame):
            print(frame)
            xdata = np.linspace(1, encodings_lhs.shape[1], encodings_lhs.shape[1])
            lns[0][0].set_data(xdata, encodings_lhs[int(frame) * 10])
            xdata = np.linspace(1, outputs_rhs.shape[1], outputs_rhs.shape[1])
            lns[-1][0].set_data(xdata, outputs_rhs[int(frame) * 10])
            xdata = np.linspace(1, outputs_rhs.shape[1], outputs_rhs.shape[1])
            lns[1][0].set_data(xdata, outputs_rhs[int(frame) * 10])
            xdata = np.linspace(1, encodings_rhs.shape[1], encodings_rhs.shape[1])
            lns[2][0].set_data(xdata, encodings_rhs[int(frame) * 10])
            for channelInd in range(numChannels):
                length = origDataTmp[int(frame), :, channelInd].shape[0]
                xdata = range(length)
                lns[channelInd + 3][0].set_data(
                    xdata, origDataTmp[int(frame), :, channelInd]
                )

        ani = FuncAnimation(fig, update, frames=range(numPlots), init_func=init)
        os.chdir("tmp")
        ani.save("test.mp4")
        os.chdir("../")


class ACEModule(nn.Module, PyTorchUtils):
    def __init__(
        self,
        n_features: int,
        sequence_length: int,
        hidden_size1: int,
        hidden_size2: int,
        seed: int,
        gpu: int,
    ):
        # Each point is a flattened window and thus has as many features as sequence_length * features
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        input_length = sequence_length
        self.channels = n_features

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = (
            2
            ** np.arange(
                max(np.ceil(np.log2(hidden_size1)), 2), np.log2(input_length)
            )[1:]
        )
        dec_setup = np.concatenate(
            [[hidden_size1], dec_steps.repeat(2), [input_length]]
        )
        enc_setup = dec_setup[::-1]

        # self._encoder = []
        self._encoder = nn.ModuleList()
        # self._decoder = []
        self._decoder = nn.ModuleList()
        for k in range(self.channels):
            layers = np.array(
                [
                    [nn.Linear(int(a), int(b)), nn.Tanh()]
                    for a, b in enc_setup.reshape(-1, 2)
                ]
            ).flatten()[:-1]
            _encoder_tmp = nn.Sequential(*layers)
            # _encoder_tmp = nn.Parameter(nn.Sequential(*layers))
            self.to_device(_encoder_tmp)
            self._encoder.append(_encoder_tmp)

            layers = np.array(
                [
                    [nn.Linear(int(a), int(b)), nn.Tanh()]
                    for a, b in dec_setup.reshape(-1, 2)
                ]
            ).flatten()[:-1]
            _decoder_tmp = nn.Sequential(*layers)
            self.to_device(_decoder_tmp)
            self._decoder.append(_decoder_tmp)

        input_length = n_features * hidden_size1

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = (
            2
            ** np.arange(
                max(np.ceil(np.log2(hidden_size2)), 2), np.log2(input_length)
            )[1:]
        )
        dec_setup = np.concatenate(
            [[hidden_size2], dec_steps.repeat(2), [input_length]]
        )
        enc_setup = dec_setup[::-1]

        layers_rhs = np.array(
            [
                [nn.Linear(int(a), int(b)), nn.Tanh()]
                for a, b in enc_setup.reshape(-1, 2)
            ]
        ).flatten()[:-1]
        self._encoder_rhs = nn.Sequential(*layers_rhs)
        self.to_device(self._encoder_rhs)

        layers_rhs = np.array(
            [
                [nn.Linear(int(a), int(b)), nn.Tanh()]
                for a, b in dec_setup.reshape(-1, 2)
            ]
        ).flatten()[:-1]
        self._decoder_rhs = nn.Sequential(*layers_rhs)
        self.to_device(self._decoder_rhs)

    def forward(self, ts_batch, return_latent: bool = False):
        enc = []
        dec = []
        for k in range(self.channels):
            enc.append(self._encoder[k](ts_batch[:, :, k].float()).unsqueeze(1))
            dec.append(self._decoder[k](enc[k]).unsqueeze(1))
        enc = torch.cat(enc, dim=1)
        dec = torch.cat(dec, dim=1)
        reconstructed_sequence = dec.transpose(1, 3).view(ts_batch.size())

        enc_rhs = self._encoder_rhs(enc.view((ts_batch.size()[0], -1)))
        dec_rhs = self._decoder_rhs(enc_rhs)
        reconstructed_latent = dec_rhs
        return (
            (reconstructed_sequence, reconstructed_latent, enc, enc_rhs)
            if return_latent
            else (reconstructed_sequence, reconstructed_latent)
        )
