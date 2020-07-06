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


class AutoEncoderJO(Algorithm, PyTorchUtils):
    def __init__(self, name: str='AutoEncoderJO', num_epochs: int=10, batch_size: int=20, lr: float=1e-3,
                 hidden_size1: int=5, hidden_size2: int=2, sequence_length: int=30, train_gaussian_percentage: float=0.25,
                 seed: int=123, gpu: int=None, details=True, train_max=None, sensor_specific = True):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.sensor_specific = sensor_specific
        self.input_size = None
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage
        self.train_max = train_max

        self.aed = None
        self.mean, self.cov = None, None
        self.mean_rhs, self.cov_rhs = None, None

    def SensorSpecificLoss(self, yhat, y):
        # mse = nn.MSELoss()
        # batch_size = yhat.size()[0]
        subclassLength=self.hidden_size1
        yhat = yhat.view((-1, subclassLength))
        y = y.view((-1, subclassLength))
        error = yhat - y
        sqr_err = error ** 2
        sum_sqr_err = sqr_err.sum(1)
        root_sum_sqr_err = torch.sqrt(sum_sqr_err)
        return torch.mean(root_sum_sqr_err)

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        indices = np.random.permutation(len(sequences))
        split_point = len(sequences) - int(self.train_gaussian_percentage * len(sequences))
        if self.train_max is not None:
            split_point = min(self.train_max, split_point)
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[split_point:]), pin_memory=True)

        self.input_size = X.shape[1]
        self.aed = ACEModule(self.input_size, self.sequence_length, self.hidden_size1, self.hidden_size2, seed=self.seed,
                                     gpu=self.gpu)
        self.to_device(self.aed)  # .double()
        optimizer = torch.optim.Adam(self.aed.parameters(), lr=self.lr)

        self.aed.train()
        alpha = 1
        beta = 1e-3
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                output = self.aed(self.to_var(ts_batch), return_latent=True)
                loss1 = nn.MSELoss(size_average=False)(output[0], self.to_var(ts_batch.float()))
                if not self.sensor_specific:
                    loss2 = nn.MSELoss(size_average=False)(output[1], output[2].view((ts_batch.size()[0], -1)).data)
                else:
                    loss2 = self.SensorSpecificLoss(output[1], output[2].view((ts_batch.size()[0], -1)).data)
                self.aed.zero_grad()
                (alpha*loss1 + beta*loss2).backward()
                optimizer.step()
            alpha/=2
            beta*=2

        self.aed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.aed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output[0], self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.aed(self.to_var(ts_batch), return_latent = True)
            error = nn.L1Loss(reduce=False)(output[1], output[2].view((ts_batch.size()[0], -1)).data)
            error_vectors += list(error.view(-1, output[2].shape[1]).data.cpu().numpy())

        self.mean_rhs = np.mean(error_vectors, axis=0)
        self.cov_rhs = np.cov(error_vectors, rowvar=False)

    def predict(self, X: pd.DataFrame) -> np.array:
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.aed.eval()
        # For LHS
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        # For RHS
        mvnormal_rhs = multivariate_normal(self.mean_rhs, self.cov_rhs, allow_singular=True)

        scores = []
        scores_rhs = []
        outputs = []
        outputs_rhs = []
        errors = []
        errors_rhs = []
        for idx, ts in enumerate(data_loader):
            output = self.aed(self.to_var(ts), return_latent=True)
            error = nn.L1Loss(reduce=False)(output[0], self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            scores.append(score.reshape(ts.size(0), self.sequence_length))


            error_rhs = nn.L1Loss(reduce=False)(output[1], output[2].view((ts.size()[0], -1)).data)
            score_rhs = -mvnormal_rhs.logpdf(error_rhs.view(-1, output[2].shape[1]).data.cpu().numpy())
            scores_rhs.append(score_rhs.reshape(ts.size(0), -1))

            if self.details:
                outputs.append(output[0].data.numpy())
                outputs_rhs.append(output[1].data.numpy())
                errors.append(error.data.numpy())
                errors_rhs.append(error_rhs.data.numpy())
        

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, X.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, i:i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)

        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})

        scores_rhs = np.concatenate(scores_rhs)
        return scores+scores_rhs

    

    def save(self, f):
        torch.save({
            'model_state_dict': self.aed.state_dict(),
            'mean': self.mean,
            'cov': self.cov,
            'input_size': self.input_size,
            'sequence_length': self.sequence_length,
            'hidden_size1': self.hidden_size1,
            'hidden_size2': self.hidden_size2,
            'seed': self.seed,
            'gpu': self.gpu
        }, f)

    def load(self, f):
        checkpoint = torch.load(f)
        model_state_dict = checkpoint['model_state_dict']
        del checkpoint['model_state_dict']
        for key in checkpoint:
            setattr(self, key, checkpoint[key])
        self.aed = ACEModule(self.input_size, self.sequence_length, self.hidden_size1, self.hidden_size2, seed=self.seed,
                                     gpu=self.gpu)
        self.aed.load_state_dict(model_state_dict)

class ACEModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, sequence_length: int, hidden_size1: int, hidden_size2: int, seed: int, gpu: int):
        # Each point is a flattened window and thus has as many features as sequence_length * features
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        input_length = sequence_length
        self.channels = n_features

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size1)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size1], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]

        self._encoder = []
        self._decoder = []
        for k in range(self.channels):
            layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
            _encoder_tmp = nn.Sequential(*layers)
            self.to_device(_encoder_tmp)
            self._encoder.append(_encoder_tmp)

            layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
            _decoder_tmp = nn.Sequential(*layers)
            self.to_device(_decoder_tmp)
            self._decoder.append(_decoder_tmp)

        input_length = n_features * hidden_size1

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size2)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size2], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]

        layers_rhs = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder_rhs = nn.Sequential(*layers_rhs)
        self.to_device(self._encoder_rhs)

        layers_rhs = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._decoder_rhs = nn.Sequential(*layers_rhs)
        self.to_device(self._decoder_rhs)
    
    def forward(self, ts_batch, return_latent: bool=False):
        enc = []
        dec = []
        for k in range(self.channels):
            enc.append(self._encoder[k](ts_batch[:, :, k].float()).unsqueeze(1))
            dec.append(self._decoder[k](enc[k]).unsqueeze(1))
        enc = torch.cat(enc, dim=1)
        dec = torch.cat(dec, dim=1)
        reconstructed_sequence = dec.view(ts_batch.size())

        enc_rhs = self._encoder_rhs(enc.view((ts_batch.size()[0], -1)))
        dec_rhs = self._decoder_rhs(enc_rhs)
        reconstructed_latent = dec_rhs
        return (reconstructed_sequence, reconstructed_latent, enc, enc_rhs) if return_latent else (reconstructed_sequence, reconstructed_latent)
