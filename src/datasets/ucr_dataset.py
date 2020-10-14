import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from .real_datasets import RealDataset
import tslearn.datasets as dataset


class UCRDataset(RealDataset):
    def __init__(self):
        file_name = "ucr-data.csv"
        super().__init__(
            name="UCR Dataset", raw_path="osimab-data", file_name=file_name
        )

    def load(self):
        (a, b), (c, d) = self.get_data()
        self._data = (a, b, c, d)

    # Sample data
    def get_data(self):

        data_loader = dataset.UCR_UEA_datasets()
        X_train, y_train, X_test, y_test = data_loader.load_dataset("PenDigits")
        X_test = X_test[:100]
        y_test = y_test[:100]
        y_train = pd.Series(np.repeat(y_train, np.shape(X_train)[1], axis=0))
        y_test = pd.Series(np.repeat(y_test, np.shape(X_test)[1], axis=0))

        p = y_train == "0"  # Anomaly
        y_train[p] = 1
        y_train[~p] = 0

        p = y_test == "0"  # Anomaly
        y_test[p] = 1
        y_test[~p] = 0
        y_test = y_test.values.tolist()

        X_train = pd.DataFrame(X_train.reshape(-1, np.shape(X_train)[-1]))
        X_test = pd.DataFrame(X_test.reshape(-1, np.shape(X_test)[-1]))

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

        return (X_train, y_train), (X_test, y_test)
