import re
import glob
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from pycatmanread.catmanread import CatmanRead


def catman_to_df(files, columns=None, standardize=True):
    """
    convert file or list of files to list of pd.DataFrames

    :param files: List of file names or single file name
    :param columns: List of column regex.  If None, all columns are selected
        Example: ['F1', 'F2'] selects ['N_F1_WA', 'S_F1_ACC', 'N_F2_WA', ...]
        Note: '.*' at beginning and end can be omitted
    :param standardize: If True, mean and variance of all files is standardized
    """

    if not isinstance(files, (str, list)):
        raise TypeError('files must be str or list. '
                'Was {}'.format(type(files)))

    if isinstance(files, str):
        files = [files]

    df_list = []
    for cm_file in files:
        df = read_catman_file(cm_file)
        if columns is not None:
            df = df.filter(regex='|'.join(columns))
        df_list.append(df)

    if standardize:
        scaler = StandardScaler()
        all_data = np.vstack([df.values for df in df_list])
        scaler.fit(all_data)
        df_list = [standardize_df(df, scaler) for df in df_list]

    return df_list


def read_catman_file(file_name):
    cm_reader = CatmanRead()
    if re.match('.*\.bin$', file_name):
        cm_reader.open_file(file_name)
    else:
        cm_reader.open_sevenzip(file_name)
    cm_reader.read_all_header_data()

    info_df = cm_reader.channel_info_to_df()
    channel_names = info_df['Channel Name']
    trim_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}_'
    channel_names = [re.sub(trim_pattern, '', channel)
                     for channel in channel_names]
    data_dict = {}

    target_values = info_df['Number of Values'].max()
    target_frequency = info_df['Sampling Frequency'].max()
    measure_time = target_values / target_frequency
    target_index = np.arange(0, measure_time, 1/target_frequency)
    for n in range(cm_reader.no_of_channels):
        channel_name = re.sub(trim_pattern, '', channel_names[n])
        channel_data = cm_reader.return_channel_data_n(n)
        frequency = info_df['Sampling Frequency'][n]
        channel_index = np.arange(0, measure_time, 1/frequency)
        channel_interp = np.interp(target_index, channel_index, channel_data)
        data_dict[channel_name] = channel_interp
    data_df = pd.DataFrame(data_dict)
    return data_df


def standardize_df(df, scaler):
    columns = df.columns
    data = scaler.transform(df)
    return pd.DataFrame(data, columns=columns)


def main():
    #files = glob.glob('./OSIMABData_03_12/*')
    #files = glob.glob('/osimab/data/itc-prod2.com/*03_12*.zip')
    files = glob.glob('/osimab/data/itc-prod2.com/*04_01_19*.zip')
    sensorData = catman_to_df(files)
    sensorDataFiltered = []
    regexs = ['N_F3_INC_1']
    #numrows = 10000
    #numrows = 100000
    numrows = 360000
    #temperature = True
    temperature = False
    if temperature:
        regexs.append('N_F1_T_1$')
    # regexs = ['F'+str(i) for i in range(1,7)]
    for df in sensorData:
        filteredDF = pd.DataFrame()
        for regex in regexs:
            tmp = df.filter(regex = regex)[:numrows]
            filteredDF = pd.concat([filteredDF, tmp], axis = 1)
        sensorDataFiltered.append(filteredDF)
    import pdb; pdb.set_trace()
    for index in range(len(files)):
        #sensorDataFiltered[index].to_csv('OSIMABData_03_12_'+str(index)+'.csv', index = False)
        #sensorDataFiltered[index].to_csv('OSIMABData_04_01_19_F6_SG.csv', index = False)
        #sensorDataFiltered[index].to_csv('OSIMAB_04_01_19_F6_WA_SO.csv', index = False)
        sensorDataFiltered[index].to_csv('OSIMAB_full_NT_INC_1.csv', index = False)


if __name__ == '__main__':
    main()
