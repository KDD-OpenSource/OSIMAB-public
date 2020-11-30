import abc
import os
import pickle
import logging

import pandas as pd


class Dataset:
    def __init__(self, name: str, file_name: str):
        self.name = name
        self.processed_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../data/processed/",
                file_name,
            )
        )

        self._data = None
        self.logger = logging.getLogger(__name__)

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def load(self):
        """Load data"""

    def data(
        self, sensor_list=None
    ) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        """Return data, load if necessary"""
        if self._data is None:
            self.load(sensor_list)
        return self._data

    def save(self):
        pickle.dump(self._data, open(self.processed_path, "wb"))
