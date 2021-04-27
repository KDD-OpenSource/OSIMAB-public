import numpy as np
import time
import yaml
from box import Box
import logging
import copy

from src.datasets import OSIMABDataset
from src.datasets import OSIMABDatasetSmall
from src.datasets import OSIMABDatasetSmall_6Sensors
from src.datasets import OSIMABDatasetSmall_South
from src.datasets import SWaTDatasets
from src.datasets import WADIDatasets
from src.datasets import BATADALDatasets
from src.evaluation import Evaluator
from src.algorithms import AutoEncoder
from src.algorithms import LSTMEDP
from src.algorithms import LSTMED
from src.algorithms import ACE
from src.algorithms import AutoCorrelationEncoder
from src.evaluation.config import init_logging
from src.utils import (
    detectors,
    load_different_sensor_models,
    load_same_sensor_models,
    load_sensor_models,
    clean_trainsets_by_whitelist,
    getIntersectedSensors,
)
from config import config
from functools import reduce
import pathos.multiprocessing as mp
import random
import pandas as pd
import gc
import glob
import sys
import os
from pprint import pprint


def main():
    cfgs = get_configs()
    for cfg in cfgs:
        evaluators = evaluate_osimab_jo(cfg)


def evaluate_osimab_jo(cfg):
    seed = np.random.randint(1000)
    # Load data
    output_dir, logger = get_logger(cfg)

    if cfg.osimab_crossval == True:
        evaluators = eval_osimab_crossval(seed, cfg, output_dir)
        return evaluators

    # train and/or get models
    model_types = cfg.models
    models = []
    if "train" in cfg.modeFlg:
        models.extend(train_models(seed, cfg, output_dir, logger))
    for model_type in model_types:
        if model_type in cfg.model and cfg.model[model_type].load_file is not None:
            if cfg.model[model_type].same_sensor_model == True:
                models.extend(load_same_sensor_models(cfg, seed))
            else:
                models.extend(load_different_sensor_models(cfg, seed))

    evaluators = []
    if "test" in cfg.modeFlg:
        evaluators.append(test_models(seed, cfg, output_dir, models))


def eval_osimab_crossval(seed, cfg, output_dir):
    datasets_tot = get_datasets_train_osimab_small(cfg)
    intersected_sensors = getIntersectedSensors(cfg, datasets_tot)

    # get arguments
    arg_list = []
    for dataset_test_ind in range(len(datasets_tot)):
        dataset_test = datasets_tot[dataset_test_ind]
        datasets_train = (
            datasets_tot[:dataset_test_ind] + datasets_tot[dataset_test_ind + 1 :]
        )
        arg_list.append((cfg, seed, datasets_train, dataset_test, output_dir))

    # process parallel
    results = []
    evaluators = []
    pool = mp.Pool(int(mp.cpu_count()))
    for arguments in arg_list:
        results.append(pool.apply_async(cross_val_async, args=arguments))
    pool.close()
    pool.join()
    for result in results:
        evaluators.append(result.get())
    return evaluators


def train_models(seed, cfg, output_dir, logger):
    models = []
    datasets_train = get_datasets_train(cfg)
    intersected_sensors = getIntersectedSensors(cfg, datasets_train)
    logger.info(
        f"""
            Training on sensors {intersected_sensors} with datasets
            given by {datasets_train}"""
    )
    model_name = cfg.models[0] + "_" + cfg.dataset_type[0]
    models.extend(
        add_train_models(
            cfg,
            datasets_train,
            seed,
            output_dir,
            intersected_sensors,
            logger=logger,
            name=model_name,
        )
    )
    return models


def test_models(seed, cfg, output_dir, models):
    if len(models) == 0:
        raise Exception("No models were specified (neither trained nor loaded)")

    if "osimabSmall" in cfg.dataset_type:
        datasets_test = get_datasets_test_osimab_small(cfg)
    elif "osimabSmall_6Sensors" in cfg.dataset_type:
        datasets_test = get_datasets_test_osimab_small_6Sensors(cfg)
    elif "osimabSmall_South" in cfg.dataset_type:
        datasets_test = get_datasets_test_osimab_small_South(cfg)
    elif "SWaT" in cfg.dataset_type:
        datasets_test = get_datasets_test_swat(cfg)
    elif "WADI" in cfg.dataset_type:
        datasets_test = get_datasets_test_wadi(cfg)
    elif "BATADAL" in cfg.dataset_type:
        datasets_test = get_datasets_test_batadal(cfg)
    else:
        datasets_test = get_datasets_test(cfg)
    evaluator_res = predict_testsets(datasets_test, models, output_dir, seed, cfg)
    return evaluator_res


def cross_val_async(cfg, seed, datasets_train, dataset_test, output_dir):
    models = []
    if "train" in cfg.modeFlg:
        intersected_sensors = getIntersectedSensors(cfg, datasets_train)
        name = cfg.model.type[0] + "_" + str(dataset_test)[-6:-4]
        models.extend(
            add_train_models(
                cfg, datasets_train, seed, output_dir, intersected_sensors, name=name
            )
        )

    if cfg.model.load_file is not None:
        models.extend(load_same_sensor_models(cfg, seed))

    if "test" in cfg.modeFlg:
        evaluator_res = test_cross_val_async(
            seed, cfg, output_dir, models, dataset_test
        )
        return evaluator_res
    return None


def test_cross_val_async(seed, cfg, output_dir, models, dataset_test):
    if len(models) == 0:
        raise Exception("No models were specified (neither trained nor loaded)")
    datasets_test = [dataset_test]
    # choose right model
    for model in models:
        if model.name[-2:] in dataset_test.name[-7:]:
            right_model = model
    models = [right_model]
    evaluator_res = predict_testsets(datasets_test, models, output_dir, seed, cfg)
    return evaluator_res


def get_configs():
    cfgs = []
    if sys.argv[1] == "configs/configlist.txt":
        with open(
            os.path.join(os.getcwd(), "configs/configlist.txt"), "r"
        ) as configlist:
            for line in configlist.readlines():
                print(line.strip())
                cfg_box = Box(config(external_path=line.strip()).config_dict)
                cfgs.append(cfg_box)
    else:
        for cfg in sys.argv[1:]:
            cfg_box = Box(config(external_path=cfg).config_dict)
            cfgs.append(cfg_box)
    return cfgs


def get_logger(cfg):
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    run_dir = timestamp + "_" + cfg.ctx + "_train"
    output_dir = os.path.join("reports", run_dir)
    init_logging(os.path.join(output_dir, "logs"))
    logger = logging.getLogger(__name__)
    return output_dir, logger


def get_datasets_train(cfg):
    train_datasets_type = cfg.dataset_type
    if "osimabSmall" in cfg.dataset_type:
        return get_datasets_train_osimab_small(cfg)
    if "osimabSmall_6Sensors" in cfg.dataset_type:
        return get_datasets_train_osimab_small_6Sensors(cfg)
    if "osimabSmall_South" in cfg.dataset_type:
        return get_datasets_train_osimab_small_South(cfg)
    if "osimabLarge" in cfg.dataset_type:
        return get_datasets_train_osimab_large(cfg)
    if "SWaT" in cfg.dataset_type:
        return get_datasets_train_swat(cfg)
    if "WADI" in cfg.dataset_type:
        return get_datasets_train_wadi(cfg)
    if "BATADAL" in cfg.dataset_type:
        return get_datasets_train_batadal(cfg)


def get_datasets_train_osimab_large(cfg):
    if cfg.dataset.osimabLarge.regexp_bin_train == None:
        raise Exception("You specified to train yet provided no regexp dataset")

    datasets_train = []
    pathnames_train = []
    for regexp_bin in cfg.dataset.osimabLarge.regexp_bin_train:
        pathnamesRegExp = os.path.join(cfg.dataset.osimabLarge.data_dir, regexp_bin)
        pathnames_train += glob.glob(pathnamesRegExp)
    filenames_train = [os.path.basename(pathname) for pathname in pathnames_train]

    datasets_train = [
        OSIMABDataset(cfg, file_name=filename) for filename in pathnames_train
    ]

    datasets_train = clear_trainsets_by_whitelist(datasets_train)
    print("Used binfiles for training:")
    pprint(filenames_train)
    return datasets_train


def get_datasets_train_osimab_small(cfg):
    osimabDatasetSmall = OSIMABDatasetSmall(cfg)
    datasets_train = osimabDatasetSmall.datasets
    return datasets_train


def get_datasets_train_osimab_small_6Sensors(cfg):
    osimabDatasetSmall = OSIMABDatasetSmall_6Sensors(cfg)
    datasets_train = osimabDatasetSmall.datasets
    return datasets_train


def get_datasets_train_osimab_small_South(cfg):
    osimabDatasetSmall = OSIMABDatasetSmall_South(cfg)
    datasets_train = osimabDatasetSmall.datasets
    return datasets_train


def get_datasets_train_swat(cfg):
    swatDataset = SWaTDatasets(cfg)
    datasets_train = swatDataset.datasets
    return datasets_train


def get_datasets_train_wadi(cfg):
    swatDataset = WADIDatasets(cfg)
    datasets_train = swatDataset.datasets
    return datasets_train


def get_datasets_train_batadal(cfg):
    swatDataset = BATADALDatasets(cfg)
    datasets_train = swatDataset.datasets
    return datasets_train


def get_datasets_test_by_day(cfg):
    datasets_test = []
    pathnames_test = []
    for regexp_bin in cfg.dataset.osimabLarge.regexp_bin_test:
        pathnamesRegExp = os.path.join(cfg.dataset.osimabLarge.data_dir, regexp_bin)
        pathnames_test += glob.glob(pathnamesRegExp)
    filenames_test = [os.path.basename(pathname) for pathname in pathnames_test]
    all_days = list(set(map(lambda x: x[:17], filenames_test)))
    test_days = []
    output_dirs = []
    for predictionDate in all_days:
        if predictionDate not in os.listdir(
            os.path.join(os.getcwd(), "results", "PredictionResults")
        ):
            output_dir = os.path.join(
                os.getcwd(), "results", "PredictionResults", predictionDate
            )
            output_dirs.append(output_dir)
            os.mkdir(output_dir)
            test_days.append(
                [
                    OSIMABDataset(cfg, file_name=filename)
                    for filename in pathnames_test
                    if predictionDate in filename
                ]
            )
    return output_dirs, test_days


def get_datasets_test(cfg):

    pathnames_test = []
    for regexp_bin in cfg.dataset.osimabLarge.regexp_bin_test:
        pathnamesRegExp = os.path.join(cfg.dataset.osimabLarge.data_dir, regexp_bin)
        pathnames_test += glob.glob(pathnamesRegExp)
    filenames_test = [os.path.basename(pathname) for pathname in pathnames_test]

    datasets_test = [
        OSIMABDataset(cfg, file_name=filename) for filename in pathnames_test
    ]

    print("used binfiles for testing:")
    pprint(filenames_test)
    return datasets_test


def get_datasets_test_osimab_small(cfg):
    osimabDatasetSmall = OSIMABDatasetSmall(cfg)
    datasets_test = osimabDatasetSmall.datasets
    return datasets_test


def get_datasets_test_osimab_small_6Sensors(cfg):
    osimabDatasetSmall = OSIMABDatasetSmall_6Sensors(cfg)
    datasets_test = osimabDatasetSmall.datasets
    return datasets_test


def get_datasets_test_osimab_small_South(cfg):
    osimabDatasetSmall = OSIMABDatasetSmall_South(cfg)
    datasets_test = osimabDatasetSmall.datasets
    return datasets_test


def get_datasets_test_swat(cfg):
    swatDataset = SWaTDatasets(cfg)
    datasets_test = swatDataset.datasets
    return datasets_test


def get_datasets_test_wadi(cfg):
    swatDataset = WADIDatasets(cfg)
    datasets_test = swatDataset.datasets
    return datasets_test


def get_datasets_test_batadal(cfg):
    swatDataset = BATADALDatasets(cfg)
    datasets_test = swatDataset.datasets
    return datasets_test


def add_train_models(
    cfg, datasets_train, seed, output_dir, intersected_sensors, name=None, logger=None
):
    models = []
    models.append(detectors(seed, cfg, name)[0])
    while len(datasets_train) != 0:
        # for dataset_train in datasets_train:
        if logger is not None:
            logger.info(
                f"Training {models[-1].name} on {datasets_train[0].name} with seed {seed}"
            )
        X_train = datasets_train[0].data(sensor_list=intersected_sensors)[0]
        models[-1].fit(X_train)
        datasets_train[0]._data = None
        gc.collect()
        datasets_train.pop(0)

        for model in models:
            model_dir = os.path.join(output_dir, f"model_{model.name}")
            os.makedirs(model_dir, exist_ok=True)
            model.save(model_dir)
            model.save_train_time(model_dir)
            model.save_num_params(model_dir)
    return models


def predict_test_days(test_days, dets, output_dirs, seed, cfg):
    for testDayCounter in range(len(test_days)):
        if "output_dirs" in locals():
            evaluator = Evaluator(
                test_days[testDayCounter],
                dets,
                seed=seed,
                cfg=cfg,
                output_dir=output_dirs[testDayCounter],
                create_log_file=cfg.log_file,
            )
        else:
            raise Exception("You should specify an output folder")
        evaluator.evaluate()
        evaluator.plot_roc_curves()


def predict_testsets(datasets_test, dets, output_dir, seed, cfg):
    evaluator = Evaluator(
        datasets_test,
        dets,
        seed=seed,
        cfg=cfg,
        output_dir=output_dir,
        create_log_file=cfg.log_file,
    )
    evaluator.evaluate()
    evaluator.plot_roc_curves()
    return evaluator


if __name__ == "__main__":
    main()
