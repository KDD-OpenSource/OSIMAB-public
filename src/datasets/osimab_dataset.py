import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from .real_datasets import RealDataset


class OSIMABDataset(RealDataset):
    def __init__(self, file_name=None):
        if file_name is None:
            file_name = 'osimab-data.csv'
        super().__init__(
            name='OSIMAB Dataset', raw_path='osimab-data', file_name=file_name
        )

    def load(self):
        (a, b), (c, d) = self.get_data_osimab()
        self._data = (a, b, c, d)

    def get_data_osimab(self):
        df = pd.read_csv(self.processed_path)

        n_train = int(df.shape[0] * 0.7)
        train = df.iloc[:n_train]
        test = df.iloc[n_train:]

        scaler = StandardScaler()
        scaler.fit(train)
        train = standardize(train, scaler)
        test = standardize(test, scaler)

        train_label = pd.Series(np.zeros(train.shape[0]))
        test_label = pd.Series(np.zeros(test.shape[0]))

        return (train, train_label), (test, test_label)


def standardize(df, scaler):
    columns = df.columns
    data = scaler.transform(df)
    return pd.DataFrame(data, columns=columns)

