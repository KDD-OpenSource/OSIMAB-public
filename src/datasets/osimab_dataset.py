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
        (a, b), (c, d) = self.get_data_osimab_with_anomaly()
        self._data = (a, b, c, d)

    def get_data_osimab(self):
        df = pd.read_csv(self.processed_path)
        n_train = int(df.shape[0] * 0.7)
        train = df.iloc[:n_train, 2:7]
        scaler = StandardScaler()
        train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        test = df.iloc[n_train:, 2:7]
        test = pd.DataFrame(scaler.transform(test), columns=test.columns)
        train_label = pd.Series(np.zeros(train.shape[0]))
        test_label = pd.Series(np.zeros(test.shape[0]))
        return (train, train_label), (test, test_label)

    def get_data_osimab_with_anomaly(self):
        df = pd.read_csv(self.processed_path)
        n_train = int(df.shape[0] * 0.7)
        train = df.iloc[:n_train, 2:7]
        scaler = StandardScaler()
        train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        test = df.iloc[n_train:, 2:7]

        train_label = pd.Series(np.zeros(train.shape[0]))
        test_label = pd.Series(np.zeros(test.shape[0]))

        idxs = np.random.choice(1500, 5)
        dur = 700
        for idx in idxs:
            channel = np.random.choice(5,1)[0]
            tmp = test.iloc[dur*idx:dur*(idx+1),channel]
            tmp = tmp.shift(periods=np.random.choice(29,1)[0]+1, fill_value=np.mean(tmp))
            test.iloc[dur*idx:dur*(idx+1),channel] = tmp
            test_label.iloc[dur*idx:dur*(idx+1)] = 1
        test = pd.DataFrame(scaler.transform(test), columns=test.columns)
        return (train, train_label), (test, test_label)
