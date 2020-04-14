import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange

from .algorithm_utils import Algorithm, PyTorchUtils
from .autoencoder import AutoEncoderModule, AutoEncoder


class AutoCorrelationEncoder(Algorithm, PyTorchUtils):
    def __init__(self, name: str = 'AutoEncoderLeftRight',
                 num_epochs: int = 10, batch_size: int = 20, lr: float = 1e-3, hidden_size: int = 5,
                 sequence_length: int = 30, stride: int=1, train_gaussian_percentage: float = 0.25,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0), seed: int = None,
                 gpu: int = None, details=True, train_max=math.inf):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.train_gaussian_percentage = train_gaussian_percentage
        self.train_max = train_max

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.ae_list = []
        self.ae_rhs = None
        self.mean, self.cov = None, None

    def fit(self, X):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        indices = np.random.permutation(len(sequences))
        split_point = len(sequences) - int(self.train_gaussian_percentage * len(sequences))
        split_point = min(self.train_max, split_point)
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[split_point:]), pin_memory=True)

        self.ae_list = [AutoEncoderModule(1, self.sequence_length, self.hidden_size, seed=self.seed,
                                          gpu=self.gpu)
                        for _ in X.columns]
        for ae in self.ae_list:
            self.to_device(ae)
        optimizers = {ae: torch.optim.Adam(ae.parameters(), lr=self.lr) for ae in self.ae_list}

        for ae in self.ae_list:
            ae.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                for channel in range(ts_batch.size(-1)):
                    ae = self.ae_list[channel]
                    ts_batch_channel = ts_batch[:, :, channel]
                    output_channel = ae(self.to_var(ts_batch_channel))
                    loss = nn.MSELoss(size_average=False)(output_channel, self.to_var(ts_batch_channel.float()))
                    ae.zero_grad()
                    loss.backward()
                    optimizers[ae].step()
        for ae in self.ae_list:
            ae.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = [ae(self.to_var(ts_batch[:, :, channel])) for channel, ae in enumerate(self.ae_list)]
            output = torch.cat([o.unsqueeze(-1) for o in output], axis=2)
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

        self.ae_rhs = AutoEncoder(name='AutoEncoderRHS', num_epochs=self.num_epochs, batch_size=self.batch_size,
                                  lr=self.lr, hidden_size=self.hidden_size, sequence_length=1, seed=self.seed,
                                  gpu=self.gpu, details=self.details, sensor_specific=True)
        df_enc = self.generate_enc(data_loader=train_loader)
        self.ae_rhs.fit(df_enc)

    def predict(self, X, return_subscores=False):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length]
                     for i in range(0, data.shape[0] - self.sequence_length + 1, self.stride)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        for ae in self.ae_list:
            ae.train()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores_lhs = []
        outputs = []
        errors = []
        for idx, ts in enumerate(data_loader):
            output = [ae(self.to_var(ts[:, :, channel])) for channel, ae in enumerate(self.ae_list)]
            output = torch.cat([o.unsqueeze(-1) for o in output], axis=2)
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            scores_lhs.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        df_enc = self.generate_enc(data_loader)
        scores_rhs = self.ae_rhs.predict(df_enc).reshape(-1, 1)

        # stores seq_len-many scores per timestamp and averages them
        scores_lhs = np.concatenate(scores_lhs)
        lattice_lhs = np.full((self.sequence_length, X.shape[0]), np.nan)
        lattice_rhs = np.full((self.sequence_length, X.shape[0]), np.nan)
        for idx, score in enumerate(zip(scores_lhs, scores_rhs)):
            i = idx * self.stride
            score_lhs, score_rhs = score
            lattice_lhs[i % self.sequence_length, i:i + self.sequence_length] = score_lhs
            lattice_rhs[i % self.sequence_length, i:i + self.sequence_length] = score_rhs
        scores_lhs = np.nanmean(lattice_lhs, axis=0)
        scores_rhs = np.nanmean(lattice_rhs, axis=0)
        scores = scores_lhs + scores_rhs

        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for idx, output in enumerate(outputs):
                i = idx * self.stride
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for idx, error in enumerate(errors):
                i = idx * self.stride
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})

        return (scores, scores_lhs, scores_rhs) if return_subscores else scores

    def generate_enc(self, data_loader):
        encodings = []
        for ae in self.ae_list:
            ae.eval()
        for ts_batch in data_loader:
            enc = [ae(self.to_var(ts_batch[:, :, channel]), return_latent=True)[1]
                   for channel, ae in enumerate(self.ae_list)]
            enc = torch.cat([e for e in enc], axis=1)
            encodings.append(enc)
        encodings = torch.cat([o for o in encodings], axis=0)
        df_enc = pd.DataFrame(encodings.detach().numpy())
        return df_enc
