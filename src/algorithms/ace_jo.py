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


class ACE(Algorithm, PyTorchUtils):
    def __init__(
        self,
        loss_func: list = ["lhs_mse", "rhs_sens_spec"],
        name: str = "ACE",
        num_epochs: int = 10,
        batch_size: int = 20,
        lr: float = 1e-4,
        hidden_size1: int = 5,
        hidden_size2: int = 2,
        sequence_length: int = 30,
        window_anomaly: bool = True,
        anom_based_on_avg_score: bool = False,
        train_gaussian_percentage: float = 0.25,
        seed: int = 123,
        gpu: int = None,
        details=True,
        latentVideo=True,
        train_max=None,
        aggr_func="xor",
    ):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.loss_func = loss_func
        self.aggr_func = aggr_func
        self.input_size = None
        self.sensor_list = None
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.sequence_length = sequence_length
        self.window_anomaly = window_anomaly
        self.train_gaussian_percentage = train_gaussian_percentage
        self.train_max = train_max
        self.latentVideo = latentVideo
        self.anomaly_thresholds_lhs = []
        self.anomaly_thresholds_rhs = []
        self.anomaly_values_or = None
        self.anomaly_values_xor = None

        self.anom_based_on_avg_score = anom_based_on_avg_score

        self.encoding_details = {}

        self.aed = None
        self.mean_lhs, self.var_lhs, self.cov_lhs = None, None, None
        self.mean_rhs, self.var_rhs, self.cov_rhs = None, None, None
        self.used_error_vects = 0

    @Algorithm.measure_train_time
    def fit(self, X: pd.DataFrame, path=None):
        # Data preprocessing
        X, sequences = self.data_preprocessing(X)
        # add higher variance sequences as well
        #new_sequences = self.get_higherVar_sequences(sequences, var_mult=3,
                #seq_mult=3)
        #sequences = new_sequences

        train_ace_ind, train_gaussian_ind = self.get_random_indices(len(sequences))
        train_ace_loader, train_gaussian_loader = self.get_dataloaders(
            sequences, train_ace_ind, train_gaussian_ind
        )
        self.input_size = X.shape[1]

        # Train
        self.set_ace_module(X)
        self.num_params = (sum(p.numel() for p in self.aed.parameters() if p.requires_grad))
        optimizer = torch.optim.Adam(self.aed.parameters(), lr=self.lr)
        self.train_ace(train_ace_loader, optimizer)

        # Validate (train gaussians)
        self.train_gaussians(train_gaussian_loader, X)

    def get_higherVar_sequences(self, sequences, var_mult, seq_mult):
        sequences_vars = np.var(np.array(sequences), axis=(0,1))
        new_sequences = []
        for sequence in sequences:
            if (np.var(np.array(sequence), axis=0)>var_mult*sequences_vars).sum() > 0:
                new_sequences.extend(sequence for _ in range(seq_mult))
        return new_sequences


    def data_preprocessing(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [
            data[i : i + self.sequence_length]
            for i in range(data.shape[0] - self.sequence_length + 1)
        ]
        # add higher variance sequences as well
        #sequences_vars = np.var(np.array(sequences), axis=(0,1))
        #var_mult = 2
        #seq_mult = 2
        #new_sequences = []
        #for sequence in sequences:
        #    if (np.var(np.array(sequence), axis=0)>var_mult*sequences_vars).sum() > 0:
        #        new_sequences.extend(sequence for _ in range(seq_mult))
        #return X, sequences + new_sequences
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
            epochLoss_dict = {}
            for loss_func in self.loss_func:
                epochLoss_dict.update({loss_func: 0})
            latentSpace = []
            logging.debug(f"Epoch {epoch+1}/{self.num_epochs}.")
            for ts_batch in train_ace_loader:
                self.aed.zero_grad()
                output = self.aed(self.to_var(ts_batch), return_latent=True)
                latentSpace.append(output[2])
                # new Code
                loss_dict = self.calcLosses(ts_batch, output)
                loss1_batch, loss2_batch = self.chooseLosses(loss_dict)
                loss1 = torch.mean(loss1_batch)
                loss2 = torch.mean(loss2_batch)
                epochLossLhs += loss1
                epochLossRhs += loss2

                for key in loss_dict.keys():
                    epochLoss_dict[key] += torch.mean(loss_dict[key])

                (alpha * loss1 + beta * loss2).backward()
                optimizer.step()
            alpha, beta = self.updateTradeoff(alpha, beta, epoch + 1)
            latentSpace = np.vstack(
                list(map(lambda x: x.detach().numpy(), latentSpace))
            )
            self.printIntermedResults(
                epoch,
                epochLossLhs,
                epochLossRhs,
                epochLoss_dict,
                latentSpace,
            )

    def initTradeoff(self):
        alpha = 1
        beta = 1
        # beta = 0
        return alpha, beta

    def updateTradeoff(self, alpha, beta, epoch):
        # alpha = 1 - epoch / (self.num_epochs - 1)
        # beta = epoch / (self.num_epochs - 1)
        alpha = 1
        beta = 1
        return alpha, beta

    def calcLosses(self, ts_batch, output):
        loss_dict = {}
        if "lhs_mse" in self.loss_func:
            loss_dict.update(
                {
                    "lhs_mse": torch.mean(
                        nn.MSELoss(reduction="none")(
                            output[0], self.to_var(ts_batch.float())
                        ),
                        dim=(1, 2),
                    )
                }
            )
        if "rhs_mse" in self.loss_func:
            loss_dict.update(
                {
                    "rhs_mse": torch.mean(
                        nn.MSELoss(reduction="none")(
                            output[1], output[2].view((ts_batch.size()[0], -1)).data
                        ),
                        dim=(1),
                    )
                }
            )
        if "rhs_sens_spec" in self.loss_func:
            loss_dict.update(
                {
                    "rhs_sens_spec": self.sensor_specific_loss(
                        output[1], output[2].view((ts_batch.size()[0], -1)).data
                    )
                }
            )
        if "rhs_corr" in self.loss_func:
            loss_dict.update(
                {
                    "rhs_corr": self.corr_loss(
                        output[1], output[2].view((ts_batch.size()[0], -1)).data
                    )
                }
            )
        if "rhs_var" in self.loss_func:
            loss_dict.update(
                {
                    "rhs_var": self.var_loss(
                        output[1], output[2].view((ts_batch.size()[0], -1)).data
                    )
                }
            )
        return loss_dict

    def chooseLosses(self, loss_dict):
        # change self.batch_size to something from loss_dict
        batch_size = loss_dict[list(loss_dict)[0]].shape[0]
        error_lhs = torch.zeros(batch_size)
        for loss_dict_key in loss_dict.keys():
            if "lhs" in loss_dict_key and loss_dict_key in self.loss_func:
                error_lhs += loss_dict[loss_dict_key]

        error_rhs = torch.zeros(batch_size)
        for loss_dict_key in loss_dict.keys():
            if "rhs" in loss_dict_key and loss_dict_key in self.loss_func:
                error_rhs += loss_dict[loss_dict_key]
        return error_lhs, error_rhs

    def sensor_specific_loss(self, yhat, y):
        batch_size = yhat.shape[0]
        subclassLength = self.hidden_size1
        yhat = yhat.view((batch_size, -1, subclassLength))
        y = y.view((batch_size, -1, subclassLength))
        error = yhat - y
        sqr_err = error ** 2
        sum_sqr_err = sqr_err.sum(dim=(1, 2))
        root_sum_sqr_err = torch.sqrt(sum_sqr_err)
        sqr_sum_sqr_err = sum_sqr_err ** 2
        return root_sum_sqr_err

    def corr_loss(self, yhat, y):
        # yhat_tmp = yhat
        # y_tmp = y
        # subclassLength = self.hidden_size1
        # yhat = yhat.view((-1, subclassLength))
        # y = y.view((-1, subclassLength))
        # vhat = yhat - torch.mean(yhat, 0)
        # #gives mean of sensor_dims (shape = 5)
        # vy = y - torch.mean(y, 0)
        # #substract the mean of all sensor values from each 5 dimensional array
        # cost = torch.sum(vhat * vy, 1)
        # #sum the errors over the 5 dimensions (return shape 360)
        # cost1 = torch.rsqrt(torch.sum(vhat ** 2, 1))
        # cost2 = torch.rsqrt(torch.sum(vy ** 2, 1))
        # cost = 1.0 - torch.abs(torch.mean(cost * cost1 * cost2))
        subclassLength = self.hidden_size1
        batch_size = yhat.shape[0]
        yhat = yhat.view((batch_size, -1, subclassLength))
        y = y.view((batch_size, -1, subclassLength))
        vhat = yhat - torch.mean(yhat, 1).unsqueeze(1).repeat(1, yhat.shape[1], 1)
        vy = y - torch.mean(y, 1).unsqueeze(1).repeat(1, y.shape[1], 1)
        cost = torch.sum(vhat * vy, 2)
        cost1 = torch.rsqrt(torch.sum(vhat ** 2, 2))
        cost2 = torch.rsqrt(torch.sum(vy ** 2, 2))
        cost = 1.0 - torch.abs(torch.mean(cost * cost1 * cost2, 1))
        return cost

    def var_loss(self, y, yhat):
        subclassLength = self.hidden_size1
        batch_size = y.shape[0]
        y = y.view((batch_size, -1, subclassLength))
        vy = y - torch.mean(y, 1).unsqueeze(1).repeat(1, y.shape[1], 1)
        # cost1 = torch.sum(torch.mean(vy ** 2, 1), axis=1)
        cost1 = torch.sum(torch.sqrt(torch.mean(vy ** 2, 1)), axis=1)

        subclassLength = self.hidden_size2
        batch_size = yhat.shape[0]
        yhat = yhat.view((batch_size, -1, subclassLength))
        vyhat = yhat - torch.mean(yhat, 1).unsqueeze(1).repeat(1, yhat.shape[1], 1)
        # cost2 = torch.sum(torch.mean(vyhat ** 2, 1), axis=1)
        cost2 = torch.sum(torch.sqrt(torch.mean(vyhat ** 2, 1)), axis=1)
        return cost1 + cost2

    def printIntermedResults(
        self,
        epoch,
        epochLossLhs,
        epochLossRhs,
        loss_dict,
        latentSpace,
    ):
        print(f"Epoch {epoch}")
        print(f"Epoch Loss Lhs: {epochLossLhs}")
        print(f"Epoch Loss Rhs: {epochLossRhs}")
        for key, value in loss_dict.items():
            print(f"Epoch {key}: {value}")

    def train_gaussians(self, train_gaussian_loader, X):
        self.aed.eval()
        for ts_batch in train_gaussian_loader:
            output = self.aed(self.to_var(ts_batch), return_latent=True)
            error_lhs, error_rhs = self.calc_errors(ts_batch, output)
            self.update_gaussians(error_lhs, error_rhs)

        if len(train_gaussian_loader) == 0:
            if self.window_anomaly:
                self.mean_lhs = [0]
                self.cov_lhs = np.identity(1)* 0.00000001
                self.mean_rhs = [0]
                self.cov_rhs = np.identity(1)* 0.00000001
            else:
                self.mean_lhs = np.repeat(0, X.shape[1])
                self.cov_lhs = np.identity(X.shape[1])* 0.00000001
                self.mean_rhs = np.repeat(0, X.shape[1])
                self.cov_rhs = np.identity(X.shape[1])* 0.00000001

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

    @Algorithm.measure_test_time
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

        if self.window_anomaly == False:
            scores = self.predict_sensor_anomaly(X, data_loader)
            return scores
        elif self.window_anomaly == True:
            scores = self.predict_window_anomaly(X, data_loader)
            return scores

    def predict_window_anomaly(self, X, data_loader):
        normal_lhs, normal_rhs = map(lambda x: x[0], self.set_gaussians())

        scores_lhs = []
        scores_rhs = []
        outputs = []
        encodings_lhs = []
        errors_lhs = []
        errors_rhs = []
        error_loss_dict = {}
        for loss_func in self.loss_func:
            error_loss_dict.update({loss_func: []})

        data_loader_len = len(data_loader)
        for idx, ts in enumerate(data_loader):
            output = self.aed(self.to_var(ts), return_latent=True)
            loss_dict = self.calcLosses(ts, output)
            error_lhs, error_rhs = self.chooseLosses(loss_dict)
            score_lhs, score_rhs = self.calc_score_window(
                ts, error_lhs, error_rhs, normal_lhs, normal_rhs
            )

            scores_lhs.append(score_lhs)
            scores_rhs.append(score_rhs)
            print(idx/data_loader_len)

            if self.details:
                encodings_lhs.append(output[2].data.numpy())
                outputs.append(output[0].data.numpy())
                errors_lhs.append(error_lhs.data.numpy())
                errors_rhs.append(error_rhs.data.numpy())
                for key in loss_dict.keys():
                    error_loss_dict[key].append(loss_dict[key].data.numpy())

        (
            scores_lhs,
            scores_rhs,
            outputs,
            encodings_lhs,
            errors_lhs,
            errors_rhs,
        ) = self.concat_batches(
            scores_lhs, scores_rhs, outputs, encodings_lhs,errors_lhs, errors_rhs
        )
        for key in error_loss_dict.keys():
            error_loss_dict[key] = self.concat_batches(error_loss_dict[key])[0]

        scores_lhs_avg = self.spread_seq_over_time_and_aggr(
            scores_lhs, self.sequence_length, X, "mean"
        )
        scores_rhs_avg = self.spread_seq_over_time_and_aggr(
            scores_rhs, self.sequence_length, X, "mean"
        )

        if self.anom_based_on_avg_score == True:
            # test if the same
            self.get_anomaly_values(X, scores_lhs_avg, scores_rhs_avg, aggr='or')
            #anomaly_values_lhs = scores_lhs_avg > self.anomaly_thresholds_lhs
            #anomaly_values_rhs = scores_rhs_avg > self.anomaly_thresholds_rhs
        else:
            anomaly_values_lhs_raw = (
                np.hstack(scores_lhs) > self.anomaly_thresholds_lhs
            )
            anomaly_values_rhs_raw = (
                np.hstack(scores_rhs) > self.anomaly_thresholds_rhs
            )
            anomaly_values_lhs = self.spread_seq_over_time_and_aggr(
                anomaly_values_lhs_raw, self.sequence_length, X, "mean"
            )
            anomaly_values_rhs = self.spread_seq_over_time_and_aggr(
                anomaly_values_rhs_raw, self.sequence_length, X, "mean"
            )
            combinedAnomalyValues = np.logical_or(anomaly_values_lhs, anomaly_values_rhs)
            combinedAnomalyValues_Ints = np.zeros(shape=X.shape[0])
            combinedAnomalyValues_Ints[combinedAnomalyValues == True] = 1
            self.anomaly_values = combinedAnomalyValues_Ints

        if self.details:

            self.prediction_details.update({"latent_mean":
                encodings_lhs.mean(axis=0)})

            self.update_details(
                X,
                anomaly_values_lhs,
                anomaly_values_rhs,
                scores_lhs_avg,
                scores_rhs_avg,
                outputs,
                errors_lhs,
                errors_rhs,
                error_loss_dict)

        return scores_lhs_avg + scores_rhs_avg

    def update_details(
            self,
            X,
            anomaly_values_lhs,
            anomaly_values_rhs,
            scores_lhs_avg,
            scores_rhs_avg,
            outputs,
            errors_lhs,
            errors_rhs,
            error_loss_dict):
        self.prediction_details.update({"anomaly_values_lhs": anomaly_values_lhs})
        self.prediction_details.update({"anomaly_values_rhs": anomaly_values_rhs})
        self.prediction_details.update({"scores_lhs_avg": scores_lhs_avg})
        self.prediction_details.update({"scores_rhs_avg": scores_rhs_avg})

        lattice_reconstr = np.full(
            (self.sequence_length, X.shape[0], X.shape[1]), np.nan
        )
        reconstructions_mean = self.calc_lattice(outputs, lattice_reconstr)
        self.prediction_details.update(
            {"reconstructions_mean": reconstructions_mean}
        )

        lattice_error_lhs = np.full((self.sequence_length, X.shape[0]), np.nan)
        errors_mean_lhs = self.calc_lattice(errors_lhs, lattice_error_lhs)
        self.prediction_details.update({"errors_mean_lhs": errors_mean_lhs})

        lattice_error_rhs = np.full((self.sequence_length, X.shape[0]), np.nan)
        errors_mean_rhs = self.calc_lattice(errors_rhs, lattice_error_rhs)
        self.prediction_details.update({"errors_mean_rhs": errors_mean_rhs})


        for key, value in error_loss_dict.items():
            lattice_error = np.full((self.sequence_length, X.shape[0]), np.nan)
            error = self.calc_lattice(value, lattice_error)
            self.prediction_details.update({key: error})

    def predict_sensor_anomaly(X, data_loader):
        (
            sensorNormals_lhs,
            sensorNormals_rhs,
        ) = self.set_gaussians()

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

            (scoreSensors_lhs, scoreSensors_rhs,) = self.calc_score_sensors(
                ts,
                error_lhs,
                error_rhs,
                sensorNormals_lhs,
                sensorNormals_rhs,
            )
            scoresSensors_lhs.append(scoreSensors_lhs)
            scoresSensors_rhs.append(scoreSensors_rhs)

            if self.details:
                outputs_lhs.append(output[0].data.numpy())
                encodings_lhs.append(output[2].data.numpy())
                encodings_rhs.append(output[3].data.numpy())
                outputs_rhs.append(output[1].data.numpy())
                errors_lhs.append(error_lhs.data.numpy())
                errors_rhs.append(error_rhs.data.numpy())

        (
            scoresSensors_lhs,
            scoresSensors_rhs,
        ) = self.calc_score_lattices(X, scoresSensors_lhs, scoresSensors_rhs)

        self.anomaly_values_or = self.get_anomaly_values(
            X, scoresSensors_lhs, scoresSensors_rhs, aggr='or'
        )
        self.anomaly_values_xor = self.get_anomaly_values(
            X, scoresSensors_lhs, scoresSensors_rhs, aggr='xor'
        )

        if self.aggr_func == "or":
            self.anomaly_values = self.anomaly_values_or
        elif self.aggr_func == "xor":
            self.anomaly_values = self.anomaly_values_xor
        else:
            raise Exception("No aggr function for ace model defined.")

        anomaly_values_lhs = (scoresSensors_lhs.T > self.anomaly_thresholds_lhs).T
        anomaly_values_rhs = (scoresSensors_rhs.T > self.anomaly_thresholds_rhs).T
        # self.anomaly_values_xor = anomaly_values_lhs ^ anomaly_values_rhs

        if self.details:
            # self.prediction_details.update(
            #    {"anomaly_values": self.anomaly_values.values.T}
            # )
            self.prediction_details.update(
                {"anomaly_values_or": self.anomaly_values_or.values.T}
            )

            # New
            self.prediction_details.update(
                {"anomaly_values_xor": self.anomaly_values_xor.values.T}
            )

            self.prediction_details.update({"anomaly_values_lhs": anomaly_values_lhs})
            self.prediction_details.update({"anomaly_values_rhs": anomaly_values_rhs})

            self.prediction_details.update({"scoresSensors_lhs": scoresSensors_lhs})
            self.prediction_details.update({"scoresSensors_rhs": scoresSensors_rhs})

            lattice_reconstr = np.full(
                (self.sequence_length, X.shape[0], X.shape[1]), np.nan
            )
            reconstructions_mean = self.calc_lattice(outputs_lhs, lattice_reconstr)
            self.prediction_details.update(
                {"reconstructions_mean": reconstructions_mean}
            )

            lattice_error_lhs = np.full(
                (self.sequence_length, X.shape[0], X.shape[1]), np.nan
            )
            errors_mean_lhs = self.calc_lattice(errors_lhs, lattice_error_lhs)
            self.prediction_details.update({"errors_mean_lhs": errors_mean_lhs})

            lattice_error_rhs = np.full(
                (self.sequence_length, X.shape[0], X.shape[1]), np.nan
            )
            errors_mean_rhs = self.calc_lattice(errors_rhs, lattice_error_rhs)
            self.prediction_details.update({"errors_mean_rhs": errors_mean_rhs})

            # animation
            if self.latentVideo:
                self.createLatentVideo(
                    encodings_lhs, encodings_rhs, outputs_rhs, sequences
                )

        # return the sum over the sensors or something similar (dummy result
        # for mv case)
        return scores_lhs + scores_rhs

    def set_gaussians(self):
        sensorNormals_lhs = []
        for mean, var in zip(self.mean_lhs, np.diagonal(self.cov_lhs)):
            sensorNormals_lhs.append(norm(loc=mean, scale=np.sqrt(var)))
        sensorNormals_rhs = []
        for mean, var in zip(self.mean_rhs, np.diagonal(self.cov_rhs)):
            sensorNormals_rhs.append(norm(loc=mean, scale=np.sqrt(var)))
        return sensorNormals_lhs, sensorNormals_rhs

    def calc_errors(self, ts, output, loss_type="MSE"):
        losses = self.calcLosses(ts, output)
        error_lhs, error_rhs = self.chooseLosses(losses)
        return error_lhs, error_rhs

    def calc_score_window(
        self,
        ts,
        error_lhs,
        error_rhs,
        normal_lhs,
        normal_rhs,
    ):
        # check dimensions
        score_lhs = -normal_lhs.logpdf(error_lhs.data.cpu())
        score_rhs = -normal_rhs.logpdf(error_rhs.data.cpu())
        return score_lhs, score_rhs

    def calc_score_sensors(
        self,
        ts,
        error_lhs,
        error_rhs,
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

        return scoreSensors_lhs, scoreSensors_rhs

#    def spread_seq_over_time_and_aggr(self, sequence, time_length, X, aggr):
#        sequence_repeated = np.repeat(sequence, time_length).reshape(-1, time_length)
#        lattice = np.full((time_length, X.shape[0]), np.nan)
#        for i, elem in enumerate(sequence_repeated):
#            lattice[i % time_length, i : i + time_length] = elem
#        if aggr == "mean":
#            aggr_lattice = np.nanmean(lattice, axis=0).T
#        if aggr == "max":
#            aggr_lattice = np.nanmax(lattice, axis=0).T
#        if aggr == "min":
#            aggr_lattice = np.nanmin(lattice, axis=0).T
#        return aggr_lattice

    def calc_score_lattices(self, X, scoresSensors_lhs, scoresSensors_rhs):

        lattice_lhs = np.full((self.sequence_length, X.shape[0]), np.nan)
        # scores_lhs = self.calc_lattice(scores_lhs, lattice_lhs, 'median')

        lattice_rhs = np.full((self.sequence_length, X.shape[0]), np.nan)
        # scores_rhs = self.calc_lattice(scores_rhs, lattice_rhs, 'median')

        lattice_sensors_lhs = np.full(
            (self.sequence_length, X.shape[0], X.shape[1]), np.nan
        )
        # scoresSensors_lhs = self.calc_lattice(scoresSensors_lhs,
        # lattice_sensors_lhs, 'median')
        scoresSensors_lhs = self.calc_lattice(scoresSensors_lhs, lattice_sensors_lhs)

        lattice_sensors_rhs = np.full(
            (self.sequence_length, X.shape[0], X.shape[1]), np.nan
        )
        # scoresSensors_rhs = self.calc_lattice(scoresSensors_rhs,
        # lattice_sensors_rhs, 'median')
        scoresSensors_rhs = self.calc_lattice(scoresSensors_rhs, lattice_sensors_rhs)

        return scoresSensors_lhs, scoresSensors_rhs

    def get_anomaly_values(self, X, scoresSensors_lhs, scoresSensors_rhs,
            aggr='or'):
        anomalyValues_lhs = scoresSensors_lhs.T > self.anomaly_thresholds_lhs
        anomalyValues_rhs = scoresSensors_rhs.T > self.anomaly_thresholds_rhs

        if aggr == 'or':
            combinedAnomalyValues = np.logical_or(anomalyValues_lhs, anomalyValues_rhs)
        elif aggr == 'xor':
            combinedAnomalyValues = np.logical_xor(anomalyValues_lhs, anomalyValues_rhs)
        else:
            raise Exception('Aggregation type not implemented')

        combinedAnomalyValues_Ints = np.zeros(shape=X.shape)
        combinedAnomalyValues_Ints[combinedAnomalyValues == True] = 1
        anomaly_values = pd.DataFrame(
            columns=X.columns, data=combinedAnomalyValues_Ints
        )
        return anomaly_values

#    def concat_batches(self, *args):
#        result_list = []
#        for data in args:
#            if type(data) == list and data[0].shape[0] == self.batch_size:
#                result_list.append(np.concatenate(data))
#            else:
#                raise Exception("Case not implemented")
#        return result_list

    def calc_lattice(self, data, lattice, aggregate="mean"):
        for i, elem in enumerate(data):
            lattice[i % self.sequence_length, i : i + self.sequence_length] = elem
        if aggregate == "mean":
            result = np.nanmean(lattice, axis=0).T
        elif aggregate == "median":
            result = np.nanmedian(lattice, axis=0).T
        else:
            raise Exception("You must specify an aggregation method")
        return result

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
                "loss_func": self.loss_func,
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
                "loss_func": self.loss_func,
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

    def load(self, path):
        model_details = torch.load(os.path.join(path, "model_detailed.pth"))

        self.input_size = model_details["input_size"]
        self.sensor_list = model_details["sensor_list"]
        self.sequence_length = model_details["sequence_length"]
        self.hidden_size1 = model_details["hidden_size1"]
        self.hidden_size2 = model_details["hidden_size2"]
        self.seed = model_details["seed"]
        self.gpu = model_details["gpu"]
        self.loss_func = model_details["loss_func"]

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
