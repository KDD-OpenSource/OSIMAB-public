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

def detectors(seed):
    # Reading config
    cfg = config(external_path="config.yaml")
    dets = [AutoEncoderJO(num_epochs=cfg.epoch,
        hidden_size1 = cfg.ace.hiddenSize1,
        hidden_size2 = cfg.ace.hiddenSize2,
        lr = cfg.ace.LR,
        sequence_length = cfg.ace.seq_len,
        latentVideo = False,
        sensor_specific = cfg.ace.sensor_spec_loss,
        corr_loss=cfg.ace.corr_loss,
        seed=seed)]

    return sorted(dets, key=lambda x: x.framework)


def main():
    evaluate_osimab_jo()


def evaluate_osimab_jo():
    seed = 42
    cfg = config(external_path="config.yaml")
    pathnamesRegExp = '/osimab/data/itc-prod2.com/' + cfg.dataset.regexp_bin
    pathnames = glob.glob(pathnamesRegExp)
    filenames = [os.path.basename(pathname) for pathname in pathnames]
    print('Used binfiles:')
    pprint(filenames)
    datasets = [OSIMABDataset(cfg, file_name = filename) for filename in
        filenames]
    evaluator = Evaluator(datasets, detectors, seed=seed, cfg = cfg)
    evaluator.evaluate()
    result = evaluator.benchmarks()
    evaluator.plot_roc_curves()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()


if __name__ == '__main__':
    main()
