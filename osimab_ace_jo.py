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


def detectors(seed, cfg):
    # Reading config
    dets = [
        AutoEncoderJO(
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
    cfgs = []
    for file_name in os.listdir("./configs"):
        if file_name.endswith(".yaml"):
            cfgs.append(config(external_path=os.path.join("./configs/", file_name)))
    for cfg in cfgs:
        # Load data
        pathnames = []
        for regexp_bin in cfg.dataset.regexp_bin_train:
            pathnamesRegExp = os.path.join(cfg.dataset.data_dir, regexp_bin)
            pathnames += glob.glob(pathnamesRegExp)
        filenames = [os.path.basename(pathname) for pathname in pathnames]
        print("Used binfiles:")
        pprint(filenames)
        datasets = [OSIMABDataset(cfg, file_name=filename) for filename in pathnames]

        # Load or train model
        model = detectors(np.random.randint(1000), cfg)[0]
        if cfg.ace.load_file is not None:
            model.load(cfg.ace.load_file)
        else:
            X_train = datasets[0].data()[0]
            model.fit(X_train)

        # Evaluate model
        dets = [model]
        evaluator = Evaluator(datasets, dets, seed=seed, cfg=cfg)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        evaluator.plot_roc_curves()
        evaluator.plot_threshold_comparison()
        evaluator.plot_scores()


if __name__ == "__main__":
    main()
