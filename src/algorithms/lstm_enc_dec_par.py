import numpy as np

from .lstm_enc_dec_latent import LSTMEDLatent
from .algorithm_utils import Algorithm, PyTorchUtils


class LSTMEDP(Algorithm, PyTorchUtils):
    def __init__(
        self,
        name: str = "LSTM-ED-P",
        num_epochs: int = 10,
        batch_size: int = 20,
        lr: float = 1e-3,
        hidden_size: int = 5,
        sequence_length: int = 30,
        train_gaussian_percentage: float = 0.25,
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
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstmed_list = None

    def fit(self, X):
        self.lstmed_list = [
            LSTMEDLatent(
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                hidden_size=self.hidden_size,
                sequence_length=self.sequence_length,
                train_gaussian_percentage=self.train_gaussian_percentage,
                n_layers=self.n_layers,
                use_bias=self.use_bias,
                dropout=self.dropout,
                seed=self.seed,
                gpu=self.gpu,
                details=self.details,
            )
            for _ in range(X.shape[1])
        ]
        for lstmed, channel in zip(self.lstmed_list, X.columns):
            lstmed.fit(X[[channel]])

    def predict(self, X):
        scores = []
        for lstmed, channel in zip(self.lstmed_list, X.columns):
            scores.append(lstmed.predict(X[[channel]]))
        scores = np.array(scores)
        scores = scores.mean(axis=0)
        return scores
