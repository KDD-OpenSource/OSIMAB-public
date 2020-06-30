import numpy as np

from src.datasets import OSIMABDataset
from src.evaluation import Evaluator
from src.algorithms import AutoEncoder
from src.algorithms import LSTMEDP
from src.algorithms import LSTMED
from src.algorithms import AutoEncoderJO
from src.algorithms import AutoCorrelationEncoder


def detectors(seed):
    standard_epochs = 4
    dets = [AutoEncoderJO(num_epochs=standard_epochs, seed=seed)]

    return sorted(dets, key=lambda x: x.framework)


def main():
    evaluate_osimab_jo()


def evaluate_osimab_jo():
    seed = 0
    datasets = [OSIMABDataset(file_name='tuesday_test_30min.csv')]
    evaluator = Evaluator(datasets, detectors, seed=seed)
    evaluator.evaluate()
    result = evaluator.benchmarks()
    evaluator.plot_roc_curves()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()


if __name__ == '__main__':
    main()
