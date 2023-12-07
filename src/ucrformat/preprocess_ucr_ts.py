#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Written by jiawen, September 2023.
"""
import math
import os
import random
import sys
import warnings
from aeon.datasets import load_classification

import numpy as np
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row
from src.ucrformat.components.schema import ShapeSchema

warnings.simplefilter(action="ignore", category=FutureWarning)


class PreprocessTS:
    """
    Main class to generate the synthetic dataset
    """
    def __init__(
            self, 
            save_path=None, 
            data_config=None,
    ):
        
        self.save_path = save_path

        self.dataset_name = data_config["dataset_name"]
        self.dataset_split = data_config["dataset_split"]

        self.signals, self.targets, self.meta = load_classification(self.dataset_name)
        # self.signals = np.array(self.signals)
        self.signals = self.signals.transpose(0, 2, 1) # (num, timestep, feature)

        
        self.num = self.signals.shape[0]


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
        self.dict_all, self.dict_train, self.dict_val, self.dict_test = self.create_split(list_split=self.dataset_split)

        indices = range(len(self.dict_all))
        ROWGROUP_SIZE_MB = 256

        TOTAL_PARQUET = os.path.join(self.save_path, "total")
        output_URL_total = f"file:///{TOTAL_PARQUET}"
        if not os.path.exists(TOTAL_PARQUET):
            os.makedirs(TOTAL_PARQUET)

        TRAIN_PARQUET = os.path.join(self.save_path, "train")
        output_URL_train = f"file:///{TRAIN_PARQUET}"
        if not os.path.exists(TRAIN_PARQUET):
            os.makedirs(TRAIN_PARQUET)

        VAL_PARQUET = os.path.join(self.save_path, "val")
        output_URL_val = f"file:///{VAL_PARQUET}"
        if not os.path.exists(VAL_PARQUET):
            os.makedirs(VAL_PARQUET)

        TEST_PARQUET = os.path.join(self.save_path, "test")
        output_URL_test = f"file:///{TEST_PARQUET}"
        if not os.path.exists(TEST_PARQUET):
            os.makedirs(TEST_PARQUET)


        # Entire dataset
        with materialize_dataset(spark, output_URL_total, ShapeSchema, ROWGROUP_SIZE_MB):
            data_list = sc.parallelize(indices).map(lambda index: (self.dict_all[index]))

            rows_rdd_total = data_list.map(lambda r: dict_to_spark_row(ShapeSchema, r))

            spark.createDataFrame(
                rows_rdd_total, ShapeSchema.as_spark_schema()
            ).write.mode("overwrite").option("compression", "none").parquet(output_URL_total)

        # Train set
        with materialize_dataset(spark, output_URL_train, ShapeSchema, ROWGROUP_SIZE_MB):
            train_list = sc.parallelize(range(len(self.dict_train))).map(lambda index: (self.dict_train[index]))

            rows_rdd_train = train_list.map(lambda r: dict_to_spark_row(ShapeSchema, r))

            spark.createDataFrame(
                rows_rdd_train, ShapeSchema.as_spark_schema()
            ).write.mode("overwrite").option("compression", "none").parquet(output_URL_train)

        # Validation set
        with materialize_dataset(spark, output_URL_val, ShapeSchema, ROWGROUP_SIZE_MB):
            val_list = sc.parallelize(range(len(self.dict_val))).map(lambda index: (self.dict_val[index]))

            rows_rdd_val = val_list.map(lambda r: dict_to_spark_row(ShapeSchema, r))

            spark.createDataFrame(
                rows_rdd_val, ShapeSchema.as_spark_schema()
            ).write.mode("overwrite").option("compression", "none").parquet(output_URL_val)

        # Test set
        with materialize_dataset(spark, output_URL_test, ShapeSchema, ROWGROUP_SIZE_MB):
            test_list = sc.parallelize(range(len(self.dict_test))).map(lambda index: (self.dict_test[index]))

            rows_rdd_test = test_list.map(lambda r: dict_to_spark_row(ShapeSchema, r))
            
            spark.createDataFrame(
                rows_rdd_test, ShapeSchema.as_spark_schema()
            ).write.mode("overwrite").option("compression", "none").parquet(output_URL_test)


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

        # list_split = [0.85, 0.15]
        split = {"train": list_split[0], "val": list_split[1], "test": list_split[2]}
        counter = 0
        start_set = 0
        dict_all = list(map(lambda index: self._sample(index), range(self.num)))

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


    def _sample(self, index):
        dict_all = {}
        dict_all["noun_id"] = f"sample_{index}"
        signal = self.signals[index]
        dict_all["signal"] = signal.astype(np.float32)
        dict_all["target"] = self.targets[index].astype(str)
        dict_all["signal_length"] = signal.shape[0]
        dict_all["signal_names"] = np.array([f"feature_{str(i)}" for i in range(signal.shape[1])]).astype(np.string_)
        return dict_all
