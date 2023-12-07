#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Written by H.Turbé, June 2022.
    Modified by J. Wei May 2023.

"""
import math
import os
import random
import sys
import warnings

import numpy as np
from petastorm import make_reader
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row
from src.generate_synthetic.components.schema import ShapeSchema

warnings.simplefilter(action="ignore", category=FutureWarning)


class PreprocessSynthetic:
    """
    Main class to generate the synthetic dataset
    """
    def __init__(
            self, 
            save_path=None, 
            data_config=None,
    ):

        self.save_path = save_path
        self.nb_simulation = data_config["nb_simulation"]
        self.n_points = data_config["n_points"]
        self.n_support = data_config["n_support"]
        self.n_feature = data_config["n_feature"]
        self.f_min_support, self.f_max_support = data_config["f_sin"]
        self.f_min_base, self.f_max_base = data_config["f_base"]
        self.dataset_split = data_config["dataset_split"]
        # we create a distribution for the sum of sine frequency to
        # have control over class distribution
        f_sine_1 = np.random.randint(self.f_min_support, (self.f_max_support + 1), self.nb_simulation)
        f_base_1 = np.random.randint(self.f_min_base, (self.f_max_base + 1), self.nb_simulation)
        f_sine_2 = np.random.randint(self.f_min_support, (self.f_max_support + 1), self.nb_simulation)
        f_base_2 = np.random.randint(self.f_min_base, (self.f_max_base + 1), self.nb_simulation)
        f_sum = f_sine_1 + f_sine_2
        self.quantile_class_sum = np.quantile(f_sum, data_config["quantile_class"])
        f_ratio = (f_sine_1 / f_base_1) + (f_sine_2 / f_base_2)
        self.quantile_class_ratio = np.quantile(f_ratio, data_config["quantile_class"])
        self.bool_snr, self.target_snr = data_config["target_snr"]


    def format_to_parquet(self, spark):
        """
        Method to create the dataset in parquet format
        Parameters
        ----------
        spark: SparkSession
        Returns
        -------
        None
        """
        sc = spark.sparkContext

        indices = range(self.nb_simulation)
        ROWGROUP_SIZE_MB = 256

        TOTAL_PARQUET = os.path.join(self.save_path, "total")
        output_URL_total = f"file:///{TOTAL_PARQUET}"

        TRAIN_PARQUET = os.path.join(self.save_path, "train")
        self.output_URL_train = f"file:///{TRAIN_PARQUET}"

        VAL_PARQUET = os.path.join(self.save_path, "val")
        self.output_URL_val = f"file:///{VAL_PARQUET}"

        TEST_PARQUET = os.path.join(self.save_path, "test")
        self.output_URL_test = f"file:///{TEST_PARQUET}"

        self.dict_all, self.dict_train, self.dict_val, self.dict_test = self.create_split(list_split=self.dataset_split)

        # Entire dataset
        with materialize_dataset(spark, output_URL_total, ShapeSchema, ROWGROUP_SIZE_MB):
            data_list = sc.parallelize(indices).map(lambda index: (self.dict_all[index]))

            rows_rdd_total = data_list.map(lambda r: dict_to_spark_row(ShapeSchema, r))

            spark.createDataFrame(
                rows_rdd_total, ShapeSchema.as_spark_schema()
            ).write.mode("overwrite").option("compression", "none").parquet(output_URL_total)

        # Train set
        with materialize_dataset(spark, self.output_URL_train, ShapeSchema, ROWGROUP_SIZE_MB):
            train_list = sc.parallelize(range(len(self.dict_train))).map(lambda index: (self.dict_train[index]))

            rows_rdd_train = train_list.map(lambda r: dict_to_spark_row(ShapeSchema, r))

            spark.createDataFrame(
                rows_rdd_train, ShapeSchema.as_spark_schema()
            ).write.mode("overwrite").option("compression", "none").parquet(self.output_URL_train)

        # Validation set
        with materialize_dataset(spark, self.output_URL_val, ShapeSchema, ROWGROUP_SIZE_MB):
            val_list = sc.parallelize(range(len(self.dict_val))).map(lambda index: (self.dict_val[index]))

            rows_rdd_val = val_list.map(lambda r: dict_to_spark_row(ShapeSchema, r))

            spark.createDataFrame(
                rows_rdd_val, ShapeSchema.as_spark_schema()
            ).write.mode("overwrite").option("compression", "none").parquet(self.output_URL_val)

        # Test set
        with materialize_dataset(spark, self.output_URL_test, ShapeSchema, ROWGROUP_SIZE_MB):
            test_list = sc.parallelize(range(len(self.dict_test))).map(lambda index: (self.dict_test[index]))

            rows_rdd_test = test_list.map(lambda r: dict_to_spark_row(ShapeSchema, r))
            
            spark.createDataFrame(
                rows_rdd_test, ShapeSchema.as_spark_schema()
            ).write.mode("overwrite").option("compression", "none").parquet(self.output_URL_test)


    def create_split(self, list_split):
        """
        Method to create the train, val and test split
        Parameters
        ----------
        list_split: list
            list of the percentage of the split

        Returns
        ----------
        dict_all: list (entire dataset)
        dict_train: list (train set)
        dict_val: list (validation set)
        dict_test: list (test set)
        """
        split = {"train": list_split[0], "val": list_split[1], "test": list_split[2]}
        counter = 0
        start_set = 0
        namefield = ["noun_id"]
        dict_all = list(map(lambda index: self._generate_sample(index), range(self.nb_simulation)))

        random.shuffle(dict_all)
        nb_samples = len(dict_all)
        for key, val in split.items():
            nb_set = math.ceil(val * nb_samples)
            dict_set = dict_all[start_set : start_set + nb_set]
            if key == "train":
                dict_train = dict_set
                train_names = [idx['noun_id'] for idx in dict_train]
                np.save(os.path.join(self.save_path, key), np.array(train_names))
            elif key == "val":
                dict_val = dict_set
                val_names = [idx['noun_id'] for idx in dict_val]
                np.save(os.path.join(self.save_path, key), np.array(val_names))
            else:
                dict_test = dict_set
                test_names = [idx['noun_id'] for idx in dict_test]
                np.save(os.path.join(self.save_path, key), np.array(test_names))
            counter += nb_set
            start_set += nb_set

        print("Total sample", counter)
        return dict_all, dict_train, dict_val, dict_test


    def _add_noise(self, signal):
        """
        Add noise to original signals
        ----------
        signal: np.array (original signal)

        Returns
        ----------
        signal_: np.array (signal with noise)
        """
        target_snr_db = self.target_snr
        signal_db = 10 * np.log10(np.mean(signal**2))
        noise_db = signal_db - target_snr_db
        noise = np.random.normal(0, np.sqrt(10 ** (noise_db / 10)), len(signal))
        signal_ = signal + noise

        return signal_


    def _generate_feature(self, wave: str, f_support: int, f_base: float):
        """
        Generate a given feature of a sample with a support of a given type and frequency
        overlaid over a sine wave of a given frequency
        Parameters
        ----------
        wave: str
            type of the support wave
        f_support: int
            frequency of the support wave
        f_base: float
            frequency of the base wave

        Returns
        -------
        x_feature: np.array
            feature of the sample
        start_idx: int
            idx where the support starts
        """
        x_feature = np.sin(np.linspace(0, 2 * np.pi * f_base, self.n_points)).reshape(-1, 1)
        x_feature *= 0.5
        start_idx = np.random.randint(0, self.n_points - self.n_support)

        if wave == "sine":
            x_tmp = np.sin(np.linspace(0, 2 * np.pi * f_support, self.n_points)).reshape(-1, 1)
            start_tmp = 0
            x_feature[start_idx : start_idx + self.n_support, 0] += x_tmp[
                start_tmp : start_tmp + self.n_support, 0
            ]

        elif wave == "square":
            x_tmp = np.sign(np.sin(np.linspace(0, 2 * np.pi * f_support, self.n_points))).reshape(-1, 1)
            start_tmp = 0
            x_feature[start_idx : start_idx + self.n_support, 0] += x_tmp[
                start_tmp : start_tmp + self.n_support, 0
            ]

        elif wave == "line":
            x_feature[start_idx : start_idx + self.n_support, 0] += [0] * self.n_support
        else:
            raise ValueError("wave must be one of sine, square, line")
            # raise ValueError("wave must be one of sine, square, sawtooth, line")

        return x_feature, start_idx


    def _generate_sample(self, index):
        """
        Generate a sample
        Parameters
        ----------
        index: int
            index of the sample

        Returns
        -------
        dict_all: dict
            dictionary containing the sample with the information saved
            in parquet format
        """
        dict_all = {}
        dict_all["noun_id"] = f"sample_{index}"
        idx_features = np.random.permutation(np.arange(self.n_feature))
        x_sample = np.zeros((self.n_points, self.n_feature))
        f_sine_sum = 0
        f_ratio = 0
        pos_sin_wave = []
        for enum, idx_feature in enumerate(idx_features):
            # f_base = np.random.randint(self.f_min_base, (self.f_max_base + 1), 1)[0]
            # f_support = np.random.randint(
            #     self.f_min_support, (self.f_max_support + 1), 1
            # )[0]
            f_base = np.random.uniform(self.f_min_base, self.f_max_base, 1)[0]
            f_support = np.random.randint(self.f_min_support, self.f_max_support, 1)[0]

            if enum < 2:
                f_sine_sum += f_support
                f_ratio += f_support / f_base
                x_tmp, start_idx = self._generate_feature(
                    wave="sine", f_support=f_support, f_base=f_base
                )
                x_sample[:, idx_feature] = x_tmp.squeeze()
                pos_sin_wave.append(start_idx)
            else:
                wave = random.choice(["line", "square"])
                x_tmp, _ = self._generate_feature(
                    wave=wave, f_support=f_support, f_base=f_base
                )
                x_sample[:, idx_feature] = x_tmp.squeeze()

            if self.bool_snr:
                x_sample[:, idx_feature] = self._add_noise(signal=x_sample[:, idx_feature])

        dict_all["signal"] = x_sample.astype(np.float32)

        dict_all["target_sum"] = (
            np.argwhere(f_sine_sum <= self.quantile_class_sum).min().astype(str)
        )

        bool_class_ratio = f_ratio <= self.quantile_class_ratio
        if np.sum(bool_class_ratio) == 0:
            dict_all["target_ratio"] = str(len(self.quantile_class_ratio) - 1)
        else:
            dict_all["target_ratio"] = np.argwhere(bool_class_ratio).min().astype(str)

        dict_all["signal_length"] = x_sample.shape[0]
        dict_all["signal_names"] = np.array(
            [f"feature_{str(x)}" for x in range(x_sample.shape[1])]
        ).astype(np.string_)
        dict_all["pos_sin_wave"] = np.array(pos_sin_wave).astype(int)
        dict_all["feature_idx_sin_wave"] = idx_features[:2].astype(int)
        dict_all["f_sine_sum"] = f_sine_sum.astype(int)
        dict_all["f_sine_ratio"] = f_ratio.astype(np.float32)

        return dict_all
        