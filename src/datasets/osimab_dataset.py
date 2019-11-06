import pandas as pd
import numpy as np

from .real_datasets import RealDataset


class OSIMABDataset(RealDataset):
    def __init__(self):
        super().__init__(
            name='OSIMAB Dataset', raw_path='osimab-data', file_name='osimab-data.csv'
        )

    def load(self):
        (a, b), (c, d) = self.get_data_osimab()
        self._data = (a, b, c, d)

    def get_data_osimab(self):
        df = pd.read_csv(self.processed_path)
        n_train = int(df.shape[0] * 0.7)
        train = df.iloc[:n_train]
        test = df.iloc[n_train:]
        train_label = pd.Series(np.zeros(train.shape[0]))
        test_label = pd.Series(np.zeros(test.shape[0]))
        return (train, train_label), (test, test_label)
