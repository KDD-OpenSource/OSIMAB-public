import numpy as np

from src.datasets import OSIMABDataset
from src.evaluation import Evaluator
from main import detectors


def main():
    evaluate_osimab()


def evaluate_osimab():
    seed = np.random.randint(np.iinfo(np.uint32).max, size=1, dtype=np.uint32)[0]
    datasets = [OSIMABDataset()]
    evaluator = Evaluator(datasets, detectors, seed=seed)
    evaluator.evaluate()
    result = evaluator.benchmarks()
    evaluator.plot_roc_curves()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()


if __name__ == '__main__':
    main()
