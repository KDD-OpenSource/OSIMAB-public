import numpy as np

from src.datasets import OSIMABDataset
from src.evaluation import Evaluator
from src.algorithms import AutoEncoder
from src.algorithms import LSTMEDP
from src.algorithms import LSTMED
from src.algorithms import AutoEncoderJO
from src.algorithms import AutoCorrelationEncoder
from config import config
import random
import glob
import os
from pprint import pprint

import random


def detectors(seed, cfg, name=None):
    # Reading config
    if name is None:
        regexpName = str(cfg.dataset.regexp_sensor).replace("'", "")
        name = f"AutoencoderJO_{regexpName}"
    dets = [
        AutoEncoderJO(
            name=name,
            num_epochs=cfg.epoch,
            hidden_size1=cfg.ace.hiddenSize1,
            hidden_size2=cfg.ace.hiddenSize2,
            lr=cfg.ace.LR,
            sequence_length=cfg.ace.seq_len,
            latentVideo=False,
            train_max=cfg.ace.train_max,
            sensor_specific=cfg.ace.sensor_spec_loss,
            corr_loss=cfg.ace.corr_loss,
            seed=seed,
        )
    ]

    return sorted(dets, key=lambda x: x.framework)


def main():
    evaluate_osimab_jo()


def evaluate_osimab_jo():
    seed = np.random.randint(1000)
    # seed = 42
    cfgs = []
    for file_name in os.listdir("./configs"):
        if file_name.endswith(".yaml"):
            cfgs.append(config(external_path=os.path.join("./configs/", file_name)))
    for cfg in cfgs:
        # Load data
        pathnames_train = []
        for regexp_bin in cfg.dataset.regexp_bin_train:
            pathnamesRegExp = os.path.join(cfg.dataset.data_dir, regexp_bin)
            pathnames_train += glob.glob(pathnamesRegExp)
        filenames_train = [os.path.basename(pathname) for pathname in pathnames_train]
        print("Used binfiles for training:")
        pprint(filenames_train)
        datasets_train = [
            OSIMABDataset(cfg, file_name=filename) for filename in pathnames_train
        ]

        pathnames_test = []
        for regexp_bin in cfg.dataset.regexp_bin_test:
            pathnamesRegExp = os.path.join(cfg.dataset.data_dir, regexp_bin)
            pathnames_test += glob.glob(pathnamesRegExp)
        filenames_test = [os.path.basename(pathname) for pathname in pathnames_test]
        print("Used binfiles for testing:")
        pprint(filenames_test)
        datasets_test = [
            OSIMABDataset(cfg, file_name=filename) for filename in pathnames_test
        ]

        # Load or train model
        models = []
        if cfg.ace.load_file is not None:
            total_sensors = []
            single_sensors = []
            for path in cfg.ace.load_file:
                modelName = path[path.rfind("/") + 1 :]
                models.append(detectors(seed, cfg, name=modelName)[0])
                models[-1].load(path)
                total_sensors.extend(models[-1].sensor_list)
                single_sensors.append(models[-1].sensor_list)
                # check if different models have the same sensor
            if sum(map(lambda x: len(set(x)), single_sensors)) != len(
                set(total_sensors)
            ):
                raise Exception("You try to combine models which share sensors.")
        else:
            models.append(detectors(seed, cfg)[0])
            for dataset in datasets_train:
                X_train = dataset.data()[0]
                models[-1].fit(X_train, path=None)

        # Evaluate model
        dets = models
        evaluator = Evaluator(datasets_test, dets, seed=seed, cfg=cfg)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        evaluator.plot_roc_curves()
        evaluator.plot_threshold_comparison()
        evaluator.plot_scores()


if __name__ == "__main__":
    main()
