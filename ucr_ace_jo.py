from src.datasets import UCRDataset
from src.evaluation import Evaluator
from src.algorithms import AutoEncoderJO


def detectors(seed):
    standard_epochs = 20
    dets = [AutoEncoderJO(num_epochs=standard_epochs, seed=seed)]

    return sorted(dets, key=lambda x: x.framework)


def main():
    evaluate_ucr_jo()


def evaluate_ucr_jo():
    seed = 0

    datasets = [UCRDataset()]
    evaluator = Evaluator(datasets, detectors, seed=seed)
    evaluator.evaluate()
    result = evaluator.benchmarks()
    evaluator.plot_roc_curves()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()


if __name__ == '__main__':
    main()
