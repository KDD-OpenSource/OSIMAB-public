import numpy as np
import pandas as pd

from .aae import AdversarialAutoEncoder


def diff(X: pd.DataFrame):
    X = X.interpolate()
    X.bfill(inplace=True)
    X_diff = X.diff()
    X_diff.bfill(inplace=True)
    return X_diff


class AdversarialAutoEncoderDiff(AdversarialAutoEncoder):
    def __init__(
        self,
        name: str = "AdverserialAutoEncoderDiff",
        num_epochs: int = 10,
        batch_size: int = 20,
        lr: float = 1e-3,
        hidden_size: int = 5,
        sequence_length: int = 30,
        train_gaussian_percentage: float = 0.25,
        seed: int = None,
        gpu: int = None,
        details=True,
        activation="Tanh",
    ):
        AdversarialAutoEncoder.__init__(
            self,
            name,
            num_epochs,
            batch_size,
            lr,
            hidden_size,
            sequence_length,
            train_gaussian_percentage,
            seed,
            gpu,
            details,
            activation,
        )

    def fit(self, X: pd.DataFrame):
        X_diff = diff(X)
        AdversarialAutoEncoder.fit(self, X_diff)

    def predict(self, X: pd.DataFrame) -> np.array:
        X_diff = diff(X)
        return AdversarialAutoEncoder.predict(self, X_diff)
        # data = X_diff.values
        # sequences = [data[i:i + self.sequence_length] for i in
        #              range(data.shape[0] - self.sequence_length + 1)]
        # data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size,
        #                          shuffle=False, drop_last=False)
        #
        # self.encoder.eval()
        # self.decoder.eval()
        # self.discriminator.eval()
        # mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        # z_normal = multivariate_normal(np.zeros(self.hidden_size), 1)
        # scores = []
        # outputs = []
        # errors = []
        # for idx, ts in enumerate(data_loader):
        #     encoded = self.encoder(self.to_var(ts))
        #     output = self.decoder(encoded, ts.size())
        #     error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
        #
        #     # error = torch.norm(encoded, p=2, dim=1)
        #     # score = -z_normal.logpdf(encoded.data.cpu().numpy())
        #     score = torch.norm(encoded, p=2, dim=1).data.cpu().numpy()
        #     # score = -mvnormal.logpdf(error.data.cpu().numpy())
        #     score = score.repeat(30)
        #
        #     scores.append(score.reshape(ts.size(0), self.sequence_length))
        #
        #     if self.details:
        #         outputs.append(output.data.numpy())
        #         errors.append(error.data.numpy())
        #
        # scores = np.concatenate(scores)
        # lattice = np.full((self.sequence_length, X.shape[0]), np.nan)
        # for i, score in enumerate(scores):
        #     lattice[i % self.sequence_length, i:i + self.sequence_length] = score
        # scores = np.nanmean(lattice, axis=0)
        #
        # if self.details:
        #     outputs = np.concatenate(outputs)
        #     lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]),
        #                       np.nan)
        #     for i, output in enumerate(outputs):
        #         lattice[i % self.sequence_length, i:i + self.sequence_length,
        #         :] = output
        #     outputs = np.nanmean(lattice, axis=0).T
        #     outputs = outputs.cumsum(axis=0)
        #     self.prediction_details.update({'reconstructions_mean': outputs})
        #
        #     errors = np.concatenate(errors)
        #     lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]),
        #                       np.nan)
        #     for i, error in enumerate(errors):
        #         lattice[i % self.sequence_length, i:i + self.sequence_length,
        #         :] = error
        #     self.prediction_details.update(
        #         {'errors_mean': np.nanmean(lattice, axis=0).T})
        #
        # return scores
