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
    def __init__(self, name: str='AutoEncoderJO', num_epochs: int=10, batch_size: int=20, lr: float=1e-4,
                 hidden_size1: int=5, hidden_size2: int=2, sequence_length: int=30, train_gaussian_percentage: float=0.25,
                 seed: int=123, gpu: int=None, details=True, train_max=None, sensor_specific = True, corr_loss = False):
        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.sensor_specific = sensor_specific
        self.compute_corr_loss = corr_loss
        self.input_size = None
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage
        self.train_max = train_max

        self.encoding_details = {}

        self.aed = None
        self.mean, self.cov = None, None
        self.mean_rhs, self.cov_rhs = None, None

    def sensor_specific_loss(self, yhat, y):
        # mse = nn.MSELoss()
        # batch_size = yhat.size()[0]
        subclassLength=self.hidden_size1
        yhat = yhat.view((-1, subclassLength))
        y = y.view((-1, subclassLength))
        error = yhat - y
        sqr_err = error ** 2
        sum_sqr_err = sqr_err.sum(1)
        root_sum_sqr_err = torch.sqrt(sum_sqr_err)
        return root_sum_sqr_err

    def corr_loss(self, yhat, y):
        # mse = nn.MSELoss()
        # batch_size = yhat.size()[0]
        subclassLength=self.hidden_size1
        yhat = yhat.view((-1, subclassLength))
        y = y.view((-1, subclassLength))
        vhat = yhat - torch.mean(yhat, 0)
        vy = y - torch.mean(y, 0)
        cost = torch.sum(vhat*vy, 1)
        cost1 = torch.rsqrt(torch.sum(vhat ** 2, 1))
        cost2 = torch.rsqrt(torch.sum(vy ** 2, 1))
        cost = 1.0-torch.abs(torch.mean(cost*cost1*cost2))
        #print(cost)
        return cost

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
        #alpha = 1
        #beta = 1e-3
        alpha = 1
        beta = 0
        #beta = 0
        for epoch in trange(self.num_epochs):
            latentSpace = []
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                self.aed.zero_grad()
                output = self.aed(self.to_var(ts_batch), return_latent=True)
                latentSpace.append(output[2])
                #loss1 = nn.MSELoss(size_average=False)(output[0], self.to_var(ts_batch.float()))
                #loss1 = nn.MSELoss(reduction = 'sum')(output[0], self.to_var(ts_batch.float()))
                loss1 = nn.MSELoss(reduction = 'mean')(output[0], self.to_var(ts_batch.float()))
                loss2 = 0
                if not self.sensor_specific and not self.compute_corr_loss:
                    #loss2 = nn.MSELoss(size_average=False)(output[1], output[2].view((ts_batch.size()[0], -1)).data)
                    # loss2 = nn.MSELoss(reduction = 'sum')(output[1], output[2].view((ts_batch.size()[0], -1)).data)
                    loss2 += nn.MSELoss(reduction = 'mean')(output[1], output[2].view((ts_batch.size()[0], -1)).data)

                if self.sensor_specific:
                    loss2 += torch.mean(self.sensor_specific_loss(output[1],
                        output[2].view((ts_batch.size()[0], -1)).data))

                if self.compute_corr_loss:
                    loss2 += torch.mean(self.corr_loss(output[1],
                        output[2].view((ts_batch.size()[0], -1)).data))

                (alpha*loss1 + beta*loss2).backward()
                optimizer.step()
            #alpha/=2
            #beta*=2
            alpha = 1 - epoch/self.num_epochs
            beta = epoch/self.num_epochs
            #alpha = 1
            #beta = 0
            latentSpace = np.vstack(list(map(lambda x:x.detach().numpy(),
                latentSpace)))
            print(f'Epoch {epoch}')
            print('Mean of Latent Space is:')
            print(latentSpace.mean(axis = 0))
            print('Standard Deviation of Latent Space is:')
            print(latentSpace.std(axis = 0))

        self.aed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.aed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output[0], self.to_var(ts_batch.float()))
            error_vectors += list(error.reshape(-1, X.shape[1]).data.cpu().numpy())
            #error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

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
        range_ = data.shape[0] - self.sequence_length + 1
        sequences = [data[i:i + self.sequence_length] for i in range(range_)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.aed.eval()
        # For LHS
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        # For RHS
        mvnormal_rhs = multivariate_normal(self.mean_rhs, self.cov_rhs, allow_singular=True)

        scores_lhs = []
        scores_rhs = []
        outputs = []

        encodings = []
        encodings_rhs = []

        outputs_rhs = []
        errors = []
        errors_rhs = []
        for idx, ts in enumerate(data_loader):
            output = self.aed(self.to_var(ts), return_latent=True)
            error = nn.L1Loss(reduce=False)(output[0], self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.reshape(-1, X.shape[1]).data.cpu().numpy())
            scores_lhs.append(score.reshape(ts.size(0), self.sequence_length))

            error_rhs = nn.L1Loss(reduce=False)(output[1], output[2].view((ts.size()[0], -1)).data)
            score_rhs = -mvnormal_rhs.logpdf(error_rhs.view(-1, output[2].shape[1]).data.cpu().numpy()).reshape(ts.size(0), -1).mean(axis=1)
            scores_rhs.append(score_rhs.reshape(ts.size(0), -1))

            if self.details:
                outputs.append(output[0].data.numpy())
                encodings.append(output[2].data.numpy())
                encodings_rhs.append(output[3].data.numpy())
                outputs_rhs.append(output[1].data.numpy())
                errors.append(error.data.numpy())
                errors_rhs.append(error_rhs.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores_lhs = np.concatenate(scores_lhs)
        scores_rhs = np.concatenate(scores_rhs)
        scores_rhs = scores_rhs.mean(axis=1)
        scores_rhs = scores_rhs.reshape(-1, 1)
        lattice_lhs = np.full((self.sequence_length, X.shape[0]), np.nan)
        lattice_rhs = np.full((self.sequence_length, X.shape[0]), np.nan)
        for i, score in enumerate(zip(scores_lhs, scores_rhs)):
            score_lhs, score_rhs = score
            lattice_start = i % self.sequence_length
            lattice_lhs[lattice_start, i:i + self.sequence_length] = score_lhs
            lattice_rhs[lattice_start, i:i + self.sequence_length] = score_rhs
        scores_lhs = np.nanmean(lattice_lhs, axis=0)
        scores_rhs = np.nanmean(lattice_rhs, axis=0)

        # stores seq_len-many scores per timestamp and averages them
        # scores_rhs = np.concatenate(scores_rhs)
        lattice = np.full((self.sequence_length, X.shape[0]), np.nan)
        for i, score in enumerate(scores_rhs):
            lattice[i % self.sequence_length, i:i + self.sequence_length] = score
        scores_rhs = np.nanmean(lattice, axis=0)

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

            # Adding the rhs error by summing it for each sensor and then
            # spreading it over the length of the timeseries
            errors_rhs = np.concatenate(errors_rhs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error_rhs in enumerate(errors_rhs):
                error_rhs = error_rhs.reshape(self.hidden_size1,
                        -1).sum(axis=0)
                error_rhs = error_rhs.repeat(
                        self.sequence_length).reshape(X.shape[1],
                                self.sequence_length).transpose()
                lattice[i % self.sequence_length, i:i + self.sequence_length,
                        :] = error_rhs
            self.prediction_details.update({'errors_mean_rhs': np.nanmean(lattice, axis=0).T})

            import os
            import matplotlib.pyplot as plt
            os.chdir('tmp')
            encodings = np.concatenate(encodings)
            encodings_rhs = np.concatenate(encodings_rhs)
            outputs_rhs = np.concatenate(outputs_rhs)
            for channel in range(self.input_size):
                self.encoding_details.update({f'channel_{channel}':
                    encodings[:,channel]})

            num_plots = 19
            encodings = encodings.reshape((encodings.shape[0],-1))
            outputs_rhs = outputs_rhs.reshape((encodings.shape[0],-1))
            origDataTmp = np.array(sequences[:10*num_plots:10])
            numChannels = origDataTmp.shape[2]
            ylim = []
            ylimEnc = [1.1*encodings.min(), 1.1*encodings.max()]
            ylimOutRhs = [1.1*outputs_rhs.min(), 1.1*outputs_rhs.max()]
            ylimEncRhs = [1.1*encodings_rhs.min(), 1.1*encodings_rhs.max()]
            ylimLatent = [min(list([*ylimEnc,*ylimOutRhs])),max(list([*ylimEnc,*ylimOutRhs]))]
            for channelInd in range(numChannels):
                channelMin = origDataTmp[:,:,channelInd].min()
                channelMax = origDataTmp[:,:,channelInd].max()
                ylim.append([1.1*channelMin, 1.1*channelMax])

            for i in range(num_plots):
                fig, ax = plt.subplots(numChannels+3,1, figsize =(15,10))
                ax[0].plot(encodings[i*10])
                ax[0].set_ylim(ylimLatent)
                ax[1].plot(outputs_rhs[i*10])
                ax[1].set_ylim(ylimLatent)
                ax[2].plot(encodings_rhs[i*10])
                ax[2].set_ylim(ylimEncRhs)
                for channelInd in range(numChannels):
                    ax[channelInd+3].plot(origDataTmp[i,:,channelInd])
                    ax[channelInd+3].set_ylim(ylim[channelInd])

                fig.savefig(f'enc_{i}')
                plt.close('all')
            os.chdir('../')

            self.encoding_details.update({'encodings': encodings})

        return scores_lhs + scores_rhs

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

        #self._encoder = []
        self._encoder = nn.ModuleList()
        #self._decoder = []
        self._decoder = nn.ModuleList()
        for k in range(self.channels):
            layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
            _encoder_tmp = nn.Sequential(*layers)
            #_encoder_tmp = nn.Parameter(nn.Sequential(*layers))
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
        reconstructed_sequence = dec.transpose(1,3).view(ts_batch.size())
        #reconstructed_sequence = dec.view(ts_batch.size())

        enc_rhs = self._encoder_rhs(enc.view((ts_batch.size()[0], -1)))
        dec_rhs = self._decoder_rhs(enc_rhs)
        reconstructed_latent = dec_rhs
        return (reconstructed_sequence, reconstructed_latent, enc, enc_rhs) if return_latent else (reconstructed_sequence, reconstructed_latent)
