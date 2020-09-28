import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
from .real_datasets import RealDataset


class OSIMABDataset(RealDataset):
    def __init__(self, cfg, file_name=None):
        if file_name is None:
            file_name = 'osimab-data.csv'
        super().__init__(
            #name='OSIMAB Dataset', raw_path='osimab-data', file_name=file_name
            name=file_name, raw_path='osimab-data', file_name=file_name
        )
        self.cfg = cfg

    def load(self):
        if self.cfg.dataset.impute_anomaly:
            (a, b), (c, d) = self.get_data_osimab_with_anomaly(test_len = self.cfg.testSize)
        else:
            (a, b), (c, d) = self.get_data_osimab(test_len = self.cfg.testSize)
        self._data = (a, b, c, d)

    def get_data_osimab(self, test_len):
        df = pd.read_csv(self.processed_path)

        n_train = int(df.shape[0] * self.cfg.ace.train_per)
        train = df.iloc[:n_train]

        # take a random index
        # rand_idx = random.randint(n_train, df.shape[0]-1-test_len)
        # take the last indices
        rand_idx = df.shape[0]-1-test_len
        test = df.iloc[rand_idx:rand_idx+test_len]

        scaler = StandardScaler()
        #scaler = MinMaxScaler()
        scaler.fit(train)
        train = standardize(train, scaler)
        test = standardize(test, scaler)

        train_label = pd.Series(np.zeros(train.shape[0]))
        test_label = pd.Series(np.zeros(test.shape[0]))

        return (train, train_label), (test, test_label)

    def get_data_osimab_with_anomaly(self, test_len):
        df = pd.read_csv(self.processed_path)

        n_train = int(df.shape[0] * self.cfg.ace.train_per)
        #train = df.iloc[:n_train, 2:7]
        train = df.iloc[:n_train]
        scaler = StandardScaler()
        #scaler = MinMaxScaler()
        train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        #test = df.iloc[n_train:, 2:7]
        # take a random index
        #rand_idx = random.randint(n_train, df.shape[0]-1-test_len)
        # take the last indices
        rand_idx = df.shape[0]-1-test_len
        test = df.iloc[rand_idx:rand_idx+test_len]

        train_label = pd.Series(np.zeros(train.shape[0]))
        test_label = pd.Series(np.zeros(test.shape[0]))

        dur = int(test.shape[0]/10)
        num_channels = df.shape[1]
        idxs = test.shape[0] - dur
        idxs = np.random.choice(idxs, 1*num_channels)

        for idx in idxs:
            channel = np.random.choice(num_channels,1)[0]
            #tmp = test.iloc[dur*idx:dur*(idx+1),channel]
            tmp = test.iloc[idx:idx+dur,channel]
            #tmp = tmp.shift(periods=np.random.choice(100,1)[0]+1, fill_value=np.mean(tmp))
            tmp = tmp.shift(int(dur/2), fill_value=np.mean(tmp))
            #test.iloc[dur*idx:dur*(idx+1),channel] = tmp
            test.iloc[idx:idx+dur,channel] = tmp
            #test_label.iloc[dur*idx:dur*(idx+1)] = 1
            test_label.iloc[idx:idx+dur] = 1
        test = pd.DataFrame(scaler.transform(test), columns=test.columns)
        return (train, train_label), (test, test_label)


def standardize(df, scaler):
    columns = df.columns
    data = scaler.transform(df)
    return pd.DataFrame(data, columns=columns)

