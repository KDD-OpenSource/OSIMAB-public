import abc
import os
import time
import logging
import random

import numpy as np
import torch

import tensorflow as tf
from tensorflow.python.client import device_lib
from torch.autograd import Variable


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, module_name, name, seed, details=False):
        self.logger = logging.getLogger(module_name)
        self.name = name
        self.seed = seed
        self.details = details
        self.prediction_details = {}
        self.train_time = 0
        self.test_time = 0

        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def fit(self, X):
        """
        Train the algorithm on the given dataset
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        :return anomaly score
        """

    def save_train_time(self, model_dir):
        path = os.path.join(model_dir, "train_time.txt")
        with open(path, 'w') as train_time_file:
            train_time_file.write(str(self.train_time))

    def save_test_time(self, model_dir):
        path = os.path.join(model_dir, "test_time.txt")
        with open(path, 'w') as test_time_file:
            test_time_file.write(str(self.test_time))

    def save_num_params(self, model_dir):
        path = os.path.join(model_dir, "num_params.txt")
        with open(path, 'w') as param_file:
            param_file.write(str(self.num_params))

    def spread_seq_over_time_and_aggr(self, sequence, time_length, X, aggr):
        sequence_repeated = np.repeat(sequence, time_length).reshape(-1, time_length)
        lattice = np.full((time_length, X.shape[0]), np.nan)
        for i, elem in enumerate(sequence_repeated):
            lattice[i % time_length, i : i + time_length] = elem
        if aggr == "mean":
            aggr_lattice = np.nanmean(lattice, axis=0).T
        if aggr == "max":
            aggr_lattice = np.nanmax(lattice, axis=0).T
        if aggr == "min":
            aggr_lattice = np.nanmin(lattice, axis=0).T
        return aggr_lattice

    def concat_batches(self, *args):
        result_list = []
        for data in args:
            if type(data) == list and data[0].shape[0] == self.batch_size:
                result_list.append(np.concatenate(data))
            else:
                raise Exception("Case not implemented")
        return result_list

    def measure_train_time(func):
        def inner(*args):
            start = time.time()
            result = func(*args)
            end = time.time()
            args[0].train_time += end-start
            return result
        return inner

    def measure_test_time(func):
        def inner(*args):
            start = time.time()
            result = func(*args)
            end = time.time()
            args[0].test_time += end-start
            return result
        return inner


class PyTorchUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        self.framework = 0

    @property
    def device(self):
        return torch.device(
            f"cuda:{self.gpu}"
            if torch.cuda.is_available() and self.gpu is not None
            else "cpu"
        )

    def to_var(self, t, **kwargs):
        # ToDo: check whether cuda Variable.
        t = t.to(self.device)
        return Variable(t, **kwargs)

    def to_device(self, model):
        model.to(self.device)


class TensorflowUtils(metaclass=abc.ABCMeta):
    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            tf.set_random_seed(seed)
        self.framework = 1

    @property
    def device(self):
        local_device_protos = device_lib.list_local_devices()
        gpus = [x.name for x in local_device_protos if x.device_type == "GPU"]
        return tf.device(gpus[self.gpu] if gpus and self.gpu is not None else "/cpu:0")
