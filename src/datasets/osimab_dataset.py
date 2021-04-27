import pandas as pd
import librosa
import re
from pprint import pprint
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
from .real_datasets import RealDataset
from .catman_data import catman_to_df
from pycatmanread.catmanread import CatmanRead
import os


class OSIMABDataset(RealDataset):
    def __init__(self, cfg, file_name=None, shifted_sensors = None,
            shift_length = 100, scaler=None, preproc=None):
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
        self.shifted_sensors = shifted_sensors
        self.shift_length=shift_length
        self.scaler = scaler
        self.preproc = preproc

    def shift_sensors(self, df, sensor_list, length):
        res_df = pd.DataFrame()
        df_length = df.shape[0]
        for sensor in df.columns:
            if sensor in sensor_list:
                res_df[sensor] = df[sensor].iloc[:df_length-length].values
                # shift
            else:
                res_df[sensor] = df[sensor].iloc[length:].values
                # no shift
        return res_df

    def get_abs_value(self, train, test):
        return abs(train), abs(test)

    def get_squared_value(self, train, test):
        return train**2, test**2

    def get_mfcc(self, train, test):
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for elem in train.values.transpose():
            train_df = pd.concat([train_df, pd.DataFrame(librosa.feature.mfcc(elem,
                sr=100, hop_length=1)).transpose()], axis=1)
        for elem in test.values.transpose():
            test_df = pd.concat([test_df, pd.DataFrame(librosa.feature.mfcc(elem,
                sr=100, hop_length=1)).transpose()], axis=1)
        return train_df, test_df


    def check_validity(self, cm_reader):
        try:
            valid = True
            info_df = cm_reader.channel_info_to_df()
            # filter out measuring rates
            info_df = info_df.loc[
                -(info_df["Channel Name"].str.contains(r".*essrate.*"))
            ]
            info_df = info_df.loc[-(info_df["Channel Name"].str.contains(r".*time.*"))]
            info_df = info_df.loc[-(info_df["Channel Name"].str.contains(r".*Time.*"))]
            info_df = info_df.loc[
                -(info_df["Channel Name"].str.contains(r".*Watchdog.*"))
            ]
            info_df = info_df.loc[-(info_df["Channel Name"].str.contains(r".*WIM.*"))]
            channel_names = info_df["Channel Name"]
            trim_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}_"
            # channel_names = [re.sub(trim_pattern, '', channel)
            #                 for channel in channel_names]
            data_dict = {}

            # target_values = info_df["Number of Values"].max()
            target_values = info_df[info_df["Sampling Frequency"] == 100][
                "Number of Values"
            ].unique()[0]
            # target_frequency = info_df["Sampling Frequency"].max()
            if target_values < 100000:
                valid = False
            target_frequency = 100
            measure_time = target_values / target_frequency
            target_index = np.arange(0, measure_time, 1 / target_frequency)
            # here must be something like enumerate(channel_names)
            # for n in range(cm_reader.no_of_channels):
            for n in info_df.index:
                channel_name = re.sub(trim_pattern, "", channel_names[n])
                channel_data = cm_reader.return_channel_data_n(n)
                frequency = info_df["Sampling Frequency"][n]
                channel_index = np.arange(0, measure_time, 1 / frequency)
                try:
                    channel_interp = np.interp(
                        target_index, channel_index, channel_data
                    )
                except:
                    print(f"found a culprit: {info_df.loc[n]}")
                    valid = False
            return valid
        except:
            return False

    def get_sensor_list(self):
        cmreader = CatmanRead()
        cmreader.open_sevenzip(self.processed_path)
        cmreader.read_all_header_data()
        info_df = cmreader.channel_info_to_df()
        validity = self.check_validity(cmreader)
        if validity == False:
            return None
        else:
            target_values = info_df[info_df["Sampling Frequency"] == 100][
                "Number of Values"
            ].unique()[0]
            sensor_list = []
            print(f"Processing {self.processed_path}")
            for regex in self.cfg.dataset.osimabLarge.regexp_sensor:
                tmp = info_df[info_df["Channel Name"].str.contains(regex)]
                tmp = tmp["Channel Name"]
                sensor_list.extend(list(tmp))
            pd.DataFrame(sensor_list).to_csv(f"files_sensors/{self.name}.csv")
            return sensor_list

    def free_space(self):
        self._data = None

    def load(self, sensor_list=None):
        self.sensor_list = sensor_list
        # when we use the function .data() we must give it the optional
        # parameter of 'sensor_list' in order to be able to give it the sensor
        # list
        if "osimabSmall" in self.cfg.dataset_type:
            test_len = self.cfg.dataset.osimabSmall.testSize
        elif "osimabLarge" in self.cfg.dataset_type:
            test_len = self.cfg.dataset.osimabLarge.testSize
        elif "osimabSmall_6Sensors" in self.cfg.dataset_type:
            test_len = self.cfg.dataset.osimabSmall_6Sensors.testSize
        elif "osimabSmall_South" in self.cfg.dataset_type:
            test_len = self.cfg.dataset.osimabSmall_South.testSize
        else:
            raise Exception("No osimabdataset has been defined in type")
        (a, b), (c, d) = self.get_data_osimab(
            test_len=test_len, sensor_list=sensor_list
        )
        self._data = (a, b, c, d)

    def get_data_osimab(self, test_len, sensor_list=None):
        # must be replaced by the procedure of reading the zip file and
        if self.processed_path[-3:] == "csv":
            df = pd.read_csv(self.processed_path)
        else:
            try:
                df = catman_to_df(self.processed_path)[0]
            except:
                raise Exception("Could not open file")
        if sensor_list is not None:
            df = filterSensors(df, sensor_list)
        else:
            df = filterSensors(df, self.cfg.datasets.regexp_sensor)
        if self.shifted_sensors != None:
            df = self.shift_sensors(df, self.shifted_sensors, self.shift_length)

        if test_len == -1:
            test = df
            n_train = 0
        else:
            test = df.iloc[df.shape[0]-test_len:]
            n_train = int(df.shape[0])-test_len
        train = df.iloc[:n_train]

        train_label = pd.Series(np.zeros(train.shape[0]))
        test_label = pd.Series(np.zeros(test.shape[0]))

        # anomalous part
        dataset_type = self.cfg.dataset_type[0]

        if self.cfg.dataset[dataset_type].anomalies is not None:
            num_anomalies = sum(self.cfg.dataset[dataset_type].anomalies.values())
            anomaly_list = []
            for item in self.cfg.dataset[dataset_type].anomalies.items():
                anomaly_list.extend(np.repeat(item[0], item[1]))

            dur = int(test.shape[0] / (2 * num_anomalies))
            idxs = int((test.shape[0] - dur) / dur)
            idxs = np.random.choice(idxs, num_anomalies)
            idxs = idxs * dur
            num_channels = test.shape[1]
            channels = np.random.choice(
                num_channels, size=len(anomaly_list), replace=False
            )

            for idx, anomaly, channel in zip(idxs, anomaly_list, channels):
                test, test_label = impute_anomaly(
                    test, test_label, dur, idx, anomaly, channel
                )

        train, test = self.scale_data(train, test, scaler=self.scaler)
        if self.preproc == 'abs_value':
            train, test = self.get_abs_value(train, test)
        if self.preproc == 'squared':
            train, test = self.get_squared_value(train, test)
        if self.preproc == 'mfcc':
            train, test = self.get_mfcc(train, test)


        return (train, train_label), (test, test_label)


def impute_anomaly(test, test_label, dur, idx, anomaly, channel):
    pd.options.mode.chained_assignment = None
    num_channels = test.shape[1]
    if anomaly == "nullSensor":
        test.iloc[:, channel] = 0
        test_label.iloc[:] = 1
    elif anomaly == "shift":
        tmp = test.iloc[idx : idx + dur, channel]
        tmp = tmp + 1 * test.iloc[:, channel].max()
        # tmp = tmp + 4
        test.iloc[idx : idx + dur, channel] = tmp
        test_label.iloc[idx : idx + dur] = 1
    elif anomaly == "variance":
        tmp = test.iloc[idx : idx + dur, channel]
        tmp = tmp * 3
        test.iloc[idx : idx + dur, channel] = tmp
        test_label.iloc[idx : idx + dur] = 1
    elif anomaly == "peak":
        tmp = test.iloc[idx : idx + 3, channel]
        tmp = tmp + 3 * test.iloc[:, channel].max()
        test.iloc[idx : idx + 3, channel] = tmp
        test_label.iloc[idx : idx + 3] = 1
    elif anomaly == "timeshift_part":
        tmp = test.iloc[idx : idx + dur, channel]
        tmp = tmp.shift(int(dur / 2), fill_value=np.mean(tmp))
        test.iloc[idx : idx + dur, channel] = tmp
        test_label.iloc[idx : idx + dur] = 1
    elif anomaly == "timeshift":
        tmp = test.iloc[:, channel]
        tmp = tmp.shift(dur, fill_value=0)
        local_vars = calc_var_values(tmp)
        test.iloc[:, channel] = tmp
        test_label[local_vars > local_vars.quantile(0.80)] = 1
    elif anomaly == "trend":
        tmp = test.iloc[idx : idx + dur, channel]
        goal_value = 2 * test.iloc[:, channel].max()
        tmp = tmp + np.linspace(0, goal_value, tmp.shape[0])
        test.iloc[idx : idx + dur, channel] = tmp
        test_label.iloc[idx : idx + dur] = 1
    elif anomaly == "shuffle":
        num_splits = 10
        split_length = int(test.shape[0] / num_splits)
        print(split_length)
        splits = [
            test.iloc[i * split_length : (i + 1) * split_length]
            for i in range(num_splits)
        ]
        print(splits)
        permutation = np.random.permutation(range(num_splits))
        print(permutation)
        for i in range(10):
            test.iloc[i * split_length : (i + 1) * split_length] = splits[
                permutation[i]
            ]
        print(test)
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


def calc_var_values(tmp):
    var_list = [0 for i in range(50)]
    for i in range(50, 9950):
        var_list.append(tmp.iloc[i - 50 : i + 50].var())
    for i in range(50):
        var_list.append(0)
    return pd.Series(var_list)
