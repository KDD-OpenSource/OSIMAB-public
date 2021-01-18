import numpy as np
import time
import logging

from src.datasets import OSIMABDataset
from src.evaluation import Evaluator
from src.algorithms import AutoEncoder
from src.algorithms import LSTMEDP
from src.algorithms import LSTMED
from src.algorithms import AutoEncoderJO
from src.algorithms import AutoCorrelationEncoder
from src.evaluation.config import init_logging
from config import config
from functools import reduce
import random
import pandas as pd
import gc
import glob
import sys
import os
from pprint import pprint


import random


def detectors(seed, cfg, name=None):
    # Reading config
    if name is None:
        regexpName = str(cfg.dataset.regexp_sensor).replace("'", "")
        name = f"AutoencoderJO_{regexpName}"
    dets = [
        AutoEncoderJO(
            name=name,
            num_epochs=cfg.epoch,
            hidden_size1=cfg.ace.hiddenSize1,
            hidden_size2=cfg.ace.hiddenSize2,
            lr=cfg.ace.LR,
            sequence_length=cfg.ace.seq_len,
            details=cfg.ace.details,
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


def remove_files(files_blocklist, filenames, pathnames, cfg):
    for filename in files_blocklist:
        if filename in filenames:
            filenames.remove(filename)
        if os.path.join(cfg.dataset.data_dir, filename) in pathnames:
            pathnames.remove(os.path.join(cfg.dataset.data_dir, filename))
    return filenames, pathnames


def evaluate_osimab_jo():
    seed = np.random.randint(1000)
    files_blocklist = get_blocklist()
    cfgs = get_configs()

    for cfg in cfgs:
        # Load data
        if "DailyPrediction" not in cfg.ctx:
            output_dir, logger = get_logger(cfg)

        if cfg.dataset.regexp_bin_train is not None:
            datasets_train = get_datasets_train(cfg, files_blocklist)
            # check lengths
            intersected_sensors = getIntersectedSensors(
                [dataset.name + ".csv" for dataset in datasets_train],
                cfg.dataset.regexp_sensor,
            )
            logger.info(
                f"""
                    Training on sensors {intersected_sensors} with datasets
                    given by {datasets_train}"""
            )

        if cfg.dataset.regexp_bin_test is not None:
            output_dirs, test_days = get_datasets_test_by_day(cfg)

        # Load or train model
        if cfg.ace.load_file is not None:
            models = load_models(cfg, seed)
        else:
            models = add_train_models(
                cfg, datasets_train, seed, output_dir, logger, intersected_sensors
            )

        # Evaluate model
        dets = models
        predict_test_days(test_days, dets, output_dirs, seed, cfg)


def getIntersectedSensors(filenames_train, regexp_sensor):
    cwd = os.getcwd()

    sensor_lists = []
    os.chdir("files_sensors")
    for sensor_list_file in filenames_train:
        sensor_list_pd = pd.read_csv(sensor_list_file)
        sensor_list_pd = sensor_list_pd[
            -sensor_list_pd.iloc[:, 1].str.contains(r".*essrate.*")
        ].iloc[:, 1]
        sensor_list_pd = sensor_list_pd[-sensor_list_pd.str.contains(r".*WIM.*")]
        sensor_list_pd = sensor_list_pd[-sensor_list_pd.str.contains(r".*Kanal.*")]
        if sensor_list_pd.shape[0] < 50:
            print(sensor_list_file)
            import pdb

            pdb.set_trace()
        for regexp_s in regexp_sensor:
            sensor_list_pd = sensor_list_pd[sensor_list_pd.str.contains(regexp_s)]
        sensor_lists.append(list(sensor_list_pd))

    intersected_sensors = []

    for sensor_list in sensor_lists:
        intersected_sensors.append(sensor_list)

    intersected_sensors = list(
        reduce(set.intersection, [set(item) for item in intersected_sensors])
    )
    os.chdir(cwd)

    if len(intersected_sensors) == 0:
        raise Exception("You have no sensors to train on")

    return intersected_sensors


def get_blocklist():
    files_blocklist = [
        "OSIMAB_2020_08_18_13_49_51.bin.zip",
        "OSIMAB_2020_12_09_18_55_19.bin.zip",
        "OSIMAB_2020_12_08_12_39_54.bin.zip",
        "OSIMAB_2020_12_09_19_16_10.bin.zip",
        "OSIMAB_2020_12_09_19_11_27.bin.zip",
        "OSIMAB_2020_10_07_09_37_37.bin.zip",
    ]
    return files_blocklist


def get_configs():
    cfgs = []

    cfgs.append(
        config(
            external_path=os.path.join(os.getcwd(), "dailyconfig", "config_daily.yaml")
        )
    )
    cfgs[0].config_dict["ctx"] += time.strftime("%Y-%m-%d-%H%M%S")
    cfgs[0].ctx += time.strftime("%Y-%m-%d-%H%M%S")
    return cfgs


def get_logger(cfg):
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    run_dir = timestamp + "_" + cfg.ctx + "_train"
    output_dir = os.path.join("reports", run_dir)
    init_logging(os.path.join(output_dir, "logs"))
    logger = logging.getLogger(__name__)
    return output_dir, logger


def get_datasets_train(cfg, files_blocklist):
    datasets_train = []
    pathnames_train = []
    for regexp_bin in cfg.dataset.regexp_bin_train:
        pathnamesRegExp = os.path.join(cfg.dataset.data_dir, regexp_bin)
        pathnames_train += glob.glob(pathnamesRegExp)
    filenames_train = [os.path.basename(pathname) for pathname in pathnames_train]
    filenames_train, pathnames_train = remove_files(
        files_blocklist, filenames_train, pathnames_train, cfg
    )
    print("Used binfiles for training:")
    pprint(filenames_train)
    datasets_train = [
        OSIMABDataset(cfg, file_name=filename) for filename in pathnames_train
    ]

    to_be_removed = []
    with open(
        os.path.join(os.getcwd(), "tmp", "whiteList.txt"), "r"
    ) as whitelist_file:
        whitelist = whitelist_file.readlines()
        whitelistStripped = [name.strip() for name in whitelist]
        for elem in datasets_train:
            if elem.name not in whitelistStripped:
                to_be_removed.append(elem)

    # to_be_removed = []
    # intersected_sensors = []
    # for dataset_train in datasets_train:
    #    sensor_list = dataset_train.get_sensor_list()
    #    if sensor_list is not None:
    #        intersected_sensors.append(sensor_list)
    #        logger.info(
    #                f"""Dataset: {dataset_train}, sensor_list:
    #                {intersected_sensors[-1]}"""
    #        )
    #        logger.info(
    #            f"""
    #            Intersected List: {list(reduce(set.intersection, [set(item)
    #            for item in intersected_sensors]))}
    #            """
    #            )
    #    else:
    #        to_be_removed.append(dataset_train)

    for dataset_train in to_be_removed:
        datasets_train.remove(dataset_train)
    return datasets_train


def get_datasets_test_by_day(cfg):
    datasets_test = []
    pathnames_test = []
    for regexp_bin in cfg.dataset.regexp_bin_test:
        pathnamesRegExp = os.path.join(cfg.dataset.data_dir, regexp_bin)
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


def get_datasets(cfg):
    for regexp_bin in cfg.dataset.regexp_bin_test:
        pathnamesRegExp = os.path.join(cfg.dataset.data_dir, regexp_bin)
        pathnames_test += glob.glob(pathnamesRegExp)
    filenames_test = [os.path.basename(pathname) for pathname in pathnames_test]

    datasets_test = [
        OSIMABDataset(cfg, file_name=filename) for filename in pathnames_test
    ]

    print("used binfiles for testing:")
    pprint(filenames_test)


def load_models(cfg, seed):
    models = []
    total_sensors = []
    single_sensors = []
    for path in cfg.ace.load_file:
        modelName = path[path.rfind("/") + 1 :]
        models.append(detectors(seed, cfg, name=modelName)[0])
        models[-1].load(path)
        total_sensors.extend(models[-1].sensor_list)
        single_sensors.append(models[-1].sensor_list)
        # check if different models have the same sensor
    if sum(map(lambda x: len(set(x)), single_sensors)) != len(set(total_sensors)):
        raise Exception("You try to combine models which share sensors.")
    return models


def add_train_models(
    cfg, datasets_train, seed, output_dir, logger, intersected_sensors
):
    models = []
    save_counter = 1
    models.append(detectors(seed, cfg)[0])
    while len(datasets_train) != 0:
        # for dataset_train in datasets_train:
        logger.info(
            f"Training {models[-1].name} on {datasets_train[0].name} with seed {seed}"
        )
        X_train = datasets_train[0].data(sensor_list=intersected_sensors)[0]
        models[-1].fit(X_train, path=None)
        # dataset_train.free_space()I
        datasets_train[0]._data = None
        # here I am not sure whether we freed the space everywhere!
        gc.collect()
        datasets_train.pop(0)

        if save_counter % 3 == 0:
            print("saving model")
            for model in models:
                model_dir = os.path.join(output_dir, f"model_{model.name}")
                os.makedirs(model_dir, exist_ok=True)
                model.save(model_dir)
        save_counter = save_counter + 1
    return models


def predict_test_days(test_days, dets, output_dirs, seed, cfg):
    for testDayCounter in range(len(test_days)):
        evaluator = Evaluator(
            test_days[testDayCounter],
            dets,
            seed=seed,
            cfg=cfg,
            output_dir=output_dirs[testDayCounter],
            create_log_file=cfg.log_file,
        )
        evaluator.evaluate()
        # result = evaluator.benchmarks()
        # evaluator.plot_roc_curves()
        # evaluator.plot_threshold_comparison()
        # evaluator.plot_scores()


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
    result = evaluator.benchmarks()
    evaluator.plot_roc_curves()
    evaluator.plot_threshold_comparison()
    evaluator.plot_scores()


if __name__ == "__main__":
    main()
