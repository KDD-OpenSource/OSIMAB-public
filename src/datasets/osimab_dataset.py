import pandas as pd
from pprint import pprint
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
from .real_datasets import RealDataset
from .catman_data import catman_to_df
from pycatmanread.catmanread import CatmanRead
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

    def get_sensor_list(self):
        cmreader = CatmanRead()
        cmreader.open_sevenzip(self.processed_path)
        cmreader.read_all_header_data()
        info_df = cmreader.channel_info_to_df()
        sensor_list = []
        print(f"Processing {self.processed_path}")
        for regex in self.cfg.dataset.regexp_sensor:
            tmp = info_df[info_df["Channel Name"].str.contains(regex)]
            tmp = tmp["Channel Name"]
            sensor_list.extend(list(tmp))
        return sensor_list



    def load(self, sensor_list=None):
        # when we use the function .data() we must give it the optional
        # parameter of 'sensor_list' in order to be able to give it the sensor
        # list
        (a, b), (c, d) = self.get_data_osimab(
            test_len=self.cfg.testSize, sensor_list=sensor_list
        )
        self._data = (a, b, c, d)

    def get_data_osimab(self, test_len, sensor_list=None):
        # must be replaced by the procedure of reading the zip file and
        df = catman_to_df(self.processed_path)[0]
        if sensor_list is not None:
            df = filterSensors(df, sensor_list)
        else:
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
            idxs = int((test.shape[0] - dur) / dur)
            idxs = np.random.choice(idxs, num_anomalies)
            idxs = idxs * dur
            channels = []
            num_channels = test.shape[1]
            for anomaly in anomaly_list:
                channels.append(np.random.choice(num_channels, 1)[0])

            for idx, anomaly, channel in zip(idxs, anomaly_list, channels):
                test, test_label = impute_anomaly(
                    test, test_label, dur, idx, anomaly, channel
                )

        # after (optionally) imputing anomalies we rescale the dataset
        test = pd.DataFrame(scaler.transform(test), columns=test.columns)
        return (train, train_label), (test, test_label)


def impute_anomaly(test, test_label, dur, idx, anomaly, channel):
    pd.options.mode.chained_assignment = None
    num_channels = test.shape[1]
    if anomaly == "nullSensor":
        test.iloc[:, channel] = 0
        test_label.iloc[:] = 1
    elif anomaly == "shift":
        tmp = test.iloc[idx : idx + dur, channel]
        tmp = tmp + 4
        test.iloc[idx : idx + dur, channel] = tmp
        test_label.iloc[idx : idx + dur] = 1
    elif anomaly == "variance":
        tmp = test.iloc[idx : idx + dur, channel]
        tmp = tmp * 3
        test.iloc[idx : idx + dur, channel] = tmp
        test_label.iloc[idx : idx + dur] = 1
    elif anomaly == "peak":
        tmp = test.iloc[idx : idx + 3, channel]
        tmp = tmp + 12
        test.iloc[idx : idx + 3, channel] = tmp
        test_label.iloc[idx : idx + 3] = 1
    elif anomaly == "timeshift":
        tmp = test.iloc[idx : idx + dur, channel]
        tmp = tmp.shift(int(dur / 2), fill_value=np.mean(tmp) + 2)
        test.iloc[idx : idx + dur, channel] = tmp
        test_label.iloc[idx : idx + dur] = 1
    elif anomaly == "trend":
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
