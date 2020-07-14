import logging
import math
import threading

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
from .helpers import average_sequences
from .helpers import split_sequences
from .autoencoder import AutoEncoderModule, AutoEncoder


class AutoCorrelationEncoder(Algorithm, PyTorchUtils):
    def __init__(self, name: str = 'AutoCorrelationEncoder',
                 num_epochs: int = 10, batch_size: int = 20, lr: float = 1e-3, hidden_size: int = 5,
                 sequence_length: int = 30, stride: int=1, train_gaussian_percentage: float = 0.25,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0), seed: int = None,
                 gpu: int = None, details=True, train_max=math.inf, use_threading=False):
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

        self.use_threading = use_threading

    def fit(self, X):
        sequences = make_sequences(data=X, sequence_length=self.sequence_length, stride = self.stride)
        seq_train, seq_val = split_sequences(sequences, self.train_gaussian_percentage)
        indices = np.random.permutation(len(sequences))
        split_point = len(sequences) - int(self.train_gaussian_percentage * len(sequences))
        split_point = min(self.train_max, split_point)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[split_point:]), pin_memory=True)

        self.ae_list = [AutoEncoderModule(
            1,
            self.sequence_length,
            self.hidden_size,
            seed=self.seed,
            gpu=self.gpu)
            for _ in range(X.shape[1])]

        def train_ae(ae, sequences, channel, lr):
            self.to_device(ae)
            optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
            loss = nn.MSELoss(size_average=False)
            ae.train()
            train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                      pin_memory=True)
            for epoch in trange(self.num_epochs):
                logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
                for ts_batch in train_loader:
                    output_channel = ae(self.to_var(ts_batch))
                    loss_channel = loss(output_channel, self.to_var(ts_batch.float()))
                    ae.zero_grad()
                    loss_channel.backward()
                    optimizer.step()

        if self.use_threading:
            thread_dict = {}
            for channel, ae in enumerate(self.ae_list):
                seq_channel = seq_train[:, :, channel]
                thread = threading.Thread(target=train_ae, args=(ae, seq_channel,
                    channel, self.lr))
                thread.start()
                thread_dict[ae] = thread
            for ae in self.ae_list:
                thread_dict[ae].join()
        else:
            for channel, ae in enumerate(self.ae_list):
                seq_channel = seq_train[:, :, channel]
                train_ae(ae, seq_channel, channel, self.lr)

        for ae in self.ae_list:
            ae.eval()

        def train_gaussian(ae, sequences, channel):
            ae.eval()
            train_gaussian_loader = DataLoader(dataset=sequences,
                    batch_size=self.batch_size, drop_last=True,
                    pin_memory=True)
            
            error_vectors = []
            error_loss = nn.L1Loss(reduce=False)
            for ts_batch in train_gaussian_loader:
                output = ae(self.to_var(ts_batch))
                error = error_loss(output, self.to_var(ts_batch.float()))
                error_vector = error.view(-1, 1).data.cpu().numpy()
                error_vector = list(error_vector)
                error_vectors += error_vector
            return error_vectors

        # error_vectors = []
        # for channel, ae in enumerate(self.ae_list):
            # error_vectors = train_gaussian(ae, seq_val[:, :, channel], channel)
            # error_vectors.append(error_vectors)
        # error_vectors = np.hstack(error_vectors)

        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = [ae(self.to_var(ts_batch[:, :, channel]))
                      for channel, ae in enumerate(self.ae_list)]
            output = torch.cat([o.unsqueeze(-1) for o in output], axis=2)
            error = nn.L1Loss(reduce=False)(output,
                                            self.to_var(ts_batch.float()))
            error_vector = error.view(-1, X.shape[1]).data.cpu().numpy()
            error_vector = list(error_vector)
            error_vectors += error_vector

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

        self.ae_rhs = AutoEncoder(name='AutoEncoderRHS',
                                  num_epochs=self.num_epochs,
                                  batch_size=self.batch_size,
                                  lr=self.lr, hidden_size=self.hidden_size,
                                  sequence_length=1, seed=self.seed,
                                  gpu=self.gpu, details=self.details)
        predict_loader = DataLoader(dataset=sequences,
                batch_size=self.batch_size, drop_last=True, pin_memory=True)
        df_enc = self.generate_enc(data_loader=predict_loader)
        self.ae_rhs.fit(df_enc)

    def predict(self, X, return_subscores=False):
        sequences = make_sequences(data=X, sequence_length=self.sequence_length, stride = self.stride)
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size,
                                 shuffle=False, drop_last=False)

        for ae in self.ae_list:
            ae.eval()
        mvnormal = multivariate_normal(self.mean, self.cov,
                                       allow_singular=True)
        scores_lhs = []
        outputs = []
        errors = []
        for idx, ts in enumerate(data_loader):
            output = [ae(self.to_var(ts[:, :, channel]))
                      for channel, ae in enumerate(self.ae_list)]
            output = torch.cat([o.unsqueeze(-1) for o in output], axis=2)
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = error.view(-1, X.shape[1]).data.cpu().numpy()
            score = -mvnormal.logpdf(score)
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
            lattice_start = i % self.sequence_length
            lattice_lhs[lattice_start, i:i + self.sequence_length] = score_lhs
            lattice_rhs[lattice_start, i:i + self.sequence_length] = score_rhs
        scores_lhs = np.nanmean(lattice_lhs, axis=0)
        scores_rhs = np.nanmean(lattice_rhs, axis=0)
        scores = scores_lhs + scores_rhs

        if self.details:
            outputs_avg = average_sequences(
                outputs,
                self.sequence_length,
                (X.shape[0], X.shape[1]),
                stride = self.stride
                )
            self.prediction_details.update({'reconstructions_mean': outputs_avg})

            errors_avg = average_sequences(
                errors,
                self.sequence_length,
                (X.shape[0], X.shape[1]),
                stride = self.stride
                )
            self.prediction_details.update({'errors_mean': errors_avg})

            ae_rhs_errors = self.ae_rhs.prediction_details['errors_mean'].T
            err_shape = (ae_rhs_errors.shape[0], len(self.ae_list),-1)
            ae_rhs_errors = ae_rhs_errors.reshape(err_shape).transpose(0,2,1)
            # ae_rhs_errors_summed must be a list, because otherwise 'average_
            # sequences' destroys its shape
            #ae_rhs_errors = [np.square(ae_rhs_errors.sum(axis = 1))]
            ae_rhs_errors = [ae_rhs_errors.sum(axis = 1)]
            ae_rhs_errors_avg = average_sequences(
                ae_rhs_errors,
                self.sequence_length,
                (X.shape[0], X.shape[1]),
                stride = self.stride
                )
            self.prediction_details.update({'errors_rhs_mean': ae_rhs_errors_avg})

            encodings = self.ae_rhs.prediction_details['encodings']
            self.prediction_details.update({'encodings': encodings})

        return (scores, scores_lhs, scores_rhs) if return_subscores else scores


    def generate_enc(self, data_loader):
        encodings = []
        for ae in self.ae_list:
            ae.eval()
        for ts_batch in data_loader:
            enc = [ae(self.to_var(ts_batch[:, :, channel]),
                      return_latent=True)[1]
                   for channel, ae in enumerate(self.ae_list)]
            enc = torch.cat([e for e in enc], axis=1)
            encodings.append(enc)
        encodings = torch.cat([o for o in encodings], axis=0)
        df_enc = pd.DataFrame(encodings.detach().numpy())
        return df_enc
