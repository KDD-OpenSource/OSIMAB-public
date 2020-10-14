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
            num_error_vects=cfg.ace.num_error_vects,
            seed=seed,
        )
    ]

    return sorted(dets, key=lambda x: x.framework)


def main():
    evaluate_osimab_jo()


def evaluate_osimab_jo():
    seed = 42
    cfgs = []
    for elem in os.listdir("./configs"):
        if elem[-1] == "l":
            cfgs.append(config(external_path="./configs/" + elem))
    for cfg in cfgs:
        pathnames = []
        for regexp_bin in cfg.dataset.regexp_bin:
            pathnamesRegExp = (
                os.path.join(
                    os.path.dirname(os.path.dirname(os.getcwd())), "data/itc-prod2.com/"
                )
                + regexp_bin
            )
            pathnames.append(glob.glob(pathnamesRegExp))
        pathnames = [path for paths in pathnames for path in paths]
        filenames = [os.path.basename(pathname) for pathname in pathnames]
        print("Used binfiles:")
        pprint(filenames)
        datasets = [OSIMABDataset(cfg, file_name=filename) for filename in filenames]
        evaluator = Evaluator(datasets, detectors, seed=seed, cfg=cfg)
        evaluator.evaluate()
        result = evaluator.benchmarks()
        evaluator.plot_roc_curves()
        evaluator.plot_threshold_comparison()
        evaluator.plot_scores()


if __name__ == "__main__":
    main()
