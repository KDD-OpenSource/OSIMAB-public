import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from .real_datasets import RealDataset
from .dataset import Dataset
from .catman_data import catman_to_df


class CatmanDataset(RealDataset):
    def __init__(self, file_name=None):
        if file_name is None:
            file_name = 'osimab-data.csv'
        if isinstance(file_name, str):
            file_name = [file_name]
        super().__init__(
            name='OSIMAB Dataset', raw_path='osimab-data', file_name=file_name[0]
        )
        
        self.processed_path = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "../../data/processed/", f)
                for f in file_name
        ]

    def load(self):
        (a, b), (c, d) = self.get_data_osimab()
        self._data = (a, b, c, d)

    def get_data_osimab(self):
        df_list = catman_to_df(self.processed_path)

        if len(df_list) == 1:
            df = df_list[0]
            n_train = int(df.shape[0] * 0.7)
            train = df.iloc[:n_train]
            test = df.iloc[n_train:]
            train_label = pd.Series(np.zeros(train.shape[0]))
            test_label = pd.Series(np.zeros(test.shape[0]))
        elif len(df_list) == 2:
            train = df_list[0]
            test = df_list[1]
            train_label = pd.Series(np.zeros(train.shape[0]))
            test_label = pd.Series(np.zeros(test.shape[0]))
        else:
            train = df_list[:-1]
            test = df_list[-1]
            train_label = [pd.Series(np.zeros(df.shape[0])) for df in train]
            test_label = pd.Series(np.zeros(test.shape[0]))

        return (train, train_label), (test, test_label)

