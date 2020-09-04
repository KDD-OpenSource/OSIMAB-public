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


def detectors(seed):
    # Reading config
    cfg = config(external_path="config.yaml")
    dets = [AutoEncoderJO(num_epochs=cfg.epoch,
        hidden_size1 = cfg.ace.hiddenSize1,
        hidden_size2 = cfg.ace.hiddenSize2,
        lr = cfg.ace.LR,
        sequence_length = cfg.ace.seq_len,
        #batch_size = 2,
        sensor_specific = False,
        #train_max = 1000,
        seed=seed)]

    return sorted(dets, key=lambda x: x.framework)


def main():
    evaluate_osimab_jo()


def evaluate_osimab_jo():
    #seed = random.randint(0,100)
    seed = 4
    cfg = config(external_path="config.yaml")
    datasets = [
            #OSIMABDataset(file_name='OSIMAB_04_01_19_F6_ACC_S1.csv')
            #OSIMABDataset(file_name='OSIMAB_04_01_19_F6_INC_1.csv')
            #OSIMABDataset(file_name='OSIMAB_04_01_19_F6_SG_1_NU.csv')
            #OSIMABDataset(file_name='OSIMAB_04_01_19_F6_WA_SO.csv')
            #OSIMABDataset(file_name='OSIMABData_04_01_19_F6_SG.csv')
            #OSIMABDataset(file_name='OSIMAB_full_NT_INC.csv')
            #OSIMABDataset(file_name='OSIMAB_full_NT_INC_1.csv')
            #OSIMABDataset(file_name='OSIMAB_full_NT_WA.csv'),
            #OSIMABDataset(file_name='OSIMAB_full_NT_ACC.csv'),
            #OSIMABDataset(file_name='OSIMAB_full_NT_SG4.csv'),
            #OSIMABDataset(file_name='OSIMAB_full_NT_SG8.csv')
            OSIMABDataset(cfg, file_name='OSIMAB_mid_NT_INC.csv')
            #OSIMABDataset(file_name='OSIMAB_mid_NT_INC_1.csv')
            #OSIMABDataset(file_name='OSIMAB_mid_NT_WA.csv'),
            #OSIMABDataset(file_name='OSIMAB_mid_NT_ACC.csv'),
            #OSIMABDataset(file_name='OSIMAB_mid_NT_SG4.csv'),
            #OSIMABDataset(file_name='OSIMAB_mid_NT_SG8.csv')
            #OSIMABDataset(file_name='OSIMAB_small_NT_INC.csv')
            #OSIMABDataset(file_name='OSIMAB_small_NT_INC_1.csv')
            #OSIMABDataset(file_name='OSIMAB_small_NT_WA.csv'),
            #OSIMABDataset(file_name='OSIMAB_small_NT_ACC.csv'),
            #OSIMABDataset(file_name='OSIMAB_small_NT_SG4.csv'),
            #OSIMABDataset(file_name='OSIMAB_small_NT_SG8.csv')
            ]
    evaluator = Evaluator(datasets, detectors, seed=seed)
    evaluator.evaluate()
    result = evaluator.benchmarks()
    evaluator.plot_roc_curves()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()


if __name__ == '__main__':
    main()