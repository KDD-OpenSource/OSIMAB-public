import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .lstm_enc_dec_axl import LSTMED


class LSTMEDLatent(LSTMED):
    def predict_latent(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.lstmed.eval()
        latent_vectors = []
        for idx, ts in enumerate(data_loader):
            _, latent = self.lstmed(self.to_var(ts), return_latent=True)
            latent_vectors.append(latent)
        latent_vectors = np.concatenate(latent_vectors)
        return latent_vectors
