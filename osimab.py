import numpy as np

from src.datasets import OSIMABDataset
from src.evaluation import Evaluator
from src.algorithms import AutoEncoder
from src.algorithms import LSTMEDP
from src.algorithms import LSTMED


def detectors(seed):
    standard_epochs = 40
    dets = [LSTMED(num_epochs=standard_epochs, seed=seed)]

    return sorted(dets, key=lambda x: x.framework)


def main():
    evaluate_osimab()


def evaluate_osimab():
    seed = 0
    datasets = [OSIMABDataset(file_name='N_F2_INC_5_30min.csv')]
    evaluator = Evaluator(datasets, detectors, seed=seed)
    evaluator.evaluate()
    result = evaluator.benchmarks()
    evaluator.plot_roc_curves()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()


if __name__ == '__main__':
    main()
