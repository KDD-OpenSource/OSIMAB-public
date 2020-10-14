import numpy as np

from src.datasets import OSIMABDataset
from src.evaluation import Evaluator
from src.algorithms import AutoEncoder
from src.algorithms import LSTMEDP
from src.algorithms import LSTMED
from src.algorithms import AutoCorrelationEncoder


def detectors(seed):
    standard_epochs = 1
    dets = [
        AutoCorrelationEncoder(
            num_epochs=standard_epochs, seed=seed, stride=50, sequence_length=100
        )
    ]

    return sorted(dets, key=lambda x: x.framework)


def main():
    evaluate_osimab()


def evaluate_osimab():
    seed = 0
    # datasets = [OSIMABDataset(file_name='OSIMAB_2020_04_01_19_15_01.csv')]
    datasets = [OSIMABDataset(file_name="OSIMAB_2020_04_01_19_15_01_F6_ACC_N1.csv")]
    # filenames in itcprod -> gets me datasets without .csv imtermediate step
    # maybe even better per regex -> one for files, another for grouping of sensors regex = '.*F1.*'
    evaluator = Evaluator(datasets, detectors, seed=seed)
    evaluator.evaluate()
    result = evaluator.benchmarks()
    evaluator.plot_roc_curves()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()


if __name__ == "__main__":
    main()
