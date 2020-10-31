import pandas as pd
from pprint import pprint
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
from .real_datasets import RealDataset
from .catman_data import catman_to_df
import os


class OSIMABDataset(RealDataset):
    def __init__(self, cfg, file_name=None):
        if file_name is None:
            file_name = "osimab-data.csv"
        super().__init__(
            name=os.path.basename(file_name),
            raw_path="osimab-data",
            file_name=file_name,
        )
        self.processed_path = os.path.abspath(file_name)
        self.name = os.path.basename(file_name)
        self.cfg = cfg

    def load(self):
        (a, b), (c, d) = self.get_data_osimab(test_len=self.cfg.testSize)
        self._data = (a, b, c, d)
        # if self.cfg.dataset.impute_anomaly:
        #    (a, b), (c, d) = self.get_data_osimab_with_anomaly(
        #        test_len=self.cfg.testSize
        #    )
        # else:
        #    (a, b), (c, d) = self.get_data_osimab(test_len=self.cfg.testSize)
        # self._data = (a, b, c, d)

    # def get_data_osimab(self, test_len):
    #    # must be replaced by the procedure of reading the zip file and
    #    df = catman_to_df(self.processed_path)[0]
    #    df = filterSensors(df, self.cfg.dataset.regexp_sensor)
    #    n_train = int(df.shape[0] * self.cfg.ace.train_per)
    #    train = df.iloc[:n_train]
    #    # take a random index
    #    # rand_idx = random.randint(n_train, df.shape[0]-1-test_len)
    #    # take the last indices
    #    rand_idx = df.shape[0] - 1 - test_len
    #    test = df.iloc[rand_idx : rand_idx + test_len]

    #    scaler = StandardScaler()
    #    scaler.fit(df)
    #    train = standardize(train, scaler)
    #    test = standardize(test, scaler)

    #    train_label = pd.Series(np.zeros(train.shape[0]))
    #    test_label = pd.Series(np.zeros(test.shape[0]))

    #    return (train, train_label), (test, test_label)

    # def get_data_osimab_with_anomaly(self, test_len):
    def get_data_osimab(self, test_len):
        # must be replaced by the procedure of reading the zip file and
        df = catman_to_df(self.processed_path)[0]
        df = filterSensors(df, self.cfg.dataset.regexp_sensor)
        scaler = StandardScaler()
        scaler.fit(df)

        # train dataframe
        n_train = int(df.shape[0] * self.cfg.ace.train_per)
        train = df.iloc[:n_train]
        train = pd.DataFrame(scaler.transform(train), columns=train.columns)

        # take the last indices
        rand_idx = df.shape[0] - 1 - test_len
        test = df.iloc[rand_idx : rand_idx + test_len]

        train_label = pd.Series(np.zeros(train.shape[0]))
        test_label = pd.Series(np.zeros(test.shape[0]))

        # anomalous part
        if self.cfg.dataset.anomalies is not None:
            num_anomalies = sum(self.cfg.dataset.anomalies.values())
            anomaly_list = []
            for item in self.cfg.dataset.anomalies.items():
                anomaly_list.extend(np.repeat(item[0], item[1]))

            dur = int(test.shape[0] / (2 * num_anomalies))
            # num_channels = df.shape[1]
            idxs = int((test.shape[0] - dur) / dur)
            # idxs = np.random.choice(idxs, max(1 * num_channels, 5))
            idxs = np.random.choice(idxs, num_anomalies)
            idxs = idxs * dur

            # for idx in idxs:
            for idx, anomaly in zip(idxs, anomaly_list):
                test, test_label = impute_anomaly(test, test_label, dur, idx, anomaly)
            # for

        #                channel = np.random.choice(num_channels, 1)[0]
        #                # tmp = test.iloc[dur*idx:dur*(idx+1),channel]
        #                tmp = test.iloc[idx : idx + dur, channel]
        #                # tmp = tmp.shift(periods=np.random.choice(100,1)[0]+1, fill_value=np.mean(tmp))
        #                tmp = tmp.shift(int(dur / 2), fill_value=np.mean(tmp) + 2)
        #                # test.iloc[dur*idx:dur*(idx+1),channel] = tmp
        #                test.iloc[idx : idx + dur, channel] = tmp
        #                # test_label.iloc[dur*idx:dur*(idx+1)] = 1
        #                test_label.iloc[idx : idx + dur] = 1

        # after (optionally) imputing anomalies we rescale the dataset
        test = pd.DataFrame(scaler.transform(test), columns=test.columns)
        return (train, train_label), (test, test_label)


def impute_anomaly(test, test_label, dur, idx, anomaly):
    pd.options.mode.chained_assignment = None
    num_channels = test.shape[1]
    if anomaly == "shift":
        channels = np.random.choice(num_channels, int(num_channels / 2))
        for channel in channels:
            tmp = test.iloc[idx : idx + dur, channel]
            tmp = tmp + 2
            test.iloc[idx : idx + dur, channel] = tmp
            test_label.iloc[idx : idx + dur] = 1
    elif anomaly == "variance":
        channels = np.random.choice(num_channels, int(num_channels / 2))
        for channel in channels:
            tmp = test.iloc[idx : idx + dur, channel]
            tmp = tmp * 2
            test.iloc[idx : idx + dur, channel] = tmp
            test_label.iloc[idx : idx + dur] = 1
    elif anomaly == "peak":
        channels = np.random.choice(num_channels, int(num_channels / 2))
        for channel in channels:
            tmp = test.iloc[idx : idx + int(dur / 10), channel]
            tmp = tmp + 10
            test.iloc[idx : idx + int(dur / 10), channel] = tmp
            test_label.iloc[idx : idx + int(dur / 10)] = 1
    elif anomaly == "timeshift":
        channels = np.random.choice(num_channels, int(num_channels / 2))
        for channel in channels:
            tmp = test.iloc[idx : idx + dur, channel]
            tmp = tmp.shift(int(dur / 2), fill_value=np.mean(tmp) + 2)
            test.iloc[idx : idx + dur, channel] = tmp
            test_label.iloc[idx : idx + dur] = 1
    elif anomaly == "trend":
        channels = np.random.choice(num_channels, int(num_channels / 2))
        for channel in channels:
            tmp = test.iloc[idx : idx + dur, channel]
            tmp = tmp + np.linspace(0, 3, tmp.shape[0])
            test.iloc[idx : idx + dur, channel] = tmp
            test_label.iloc[idx : idx + dur] = 1
    else:
        raise Exception("Unknown Anomaly Type")
    pd.options.mode.chained_assignment = "warn"
    return test, test_label


def standardize(df, scaler):
    columns = df.columns
    data = scaler.transform(df)
    return pd.DataFrame(data, columns=columns)


def filterSensors(sensorData, regexs):
    sensorDataFiltered = []
    for df in [sensorData]:
        filteredDF = pd.DataFrame()
        for regex in regexs:
            tmp = df.filter(regex=regex)
            filteredDF = pd.concat([filteredDF, tmp], axis=1)
            print("Used columns:")
            pprint(list(tmp.columns))
        sensorDataFiltered.append(filteredDF)
    return sensorDataFiltered[0]
