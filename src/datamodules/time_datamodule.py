from __future__ import annotations
import math
import os
import warnings
from typing import Any, Dict, Optional, Tuple
from functools import reduce

import numpy as np
import pandas as pd
# import pytorch_lightning as pl
import lightning.pytorch as pl
from petastorm import TransformSpec, make_batch_reader, make_reader
from petastorm.codecs import NdarrayCodec, ScalarCodec
from petastorm.predicates import in_lambda, in_pseudorandom_split, in_reduce
from petastorm.pytorch import DataLoader
from petastorm.unischema import UnischemaField
from pyspark.sql.types import IntegerType, StringType
from sklearn import preprocessing
from pickle import dump, load

from datamodules.components import PytorchDataLoader, TS_Scaler
from utils import get_dataset_info

warnings.simplefilter(action="ignore", category=FutureWarning)


class TimeDataModule(pl.LightningDataModule):
    """DataModule focussed on time series for Lightning Estimator"""

    def __init__(
        self,
        data_dir: str,
        save_dir: str,
        feature_name: str,
        target_name: str,
        id_name: str,
        has_val: bool = True,
        num_train_epochs: int = 1,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        shuffle: bool = True,
        num_reader_epochs=None,
        reader_pool_type: str = "thread",
        reader_worker_count: int = 2,
        transformation=None,
        transformation_edit_fields=None,
        transformation_removed_fields=None,
        inmemory_cache_all=False,
        cur_shard: int = 0,
        shard_count: int = 1,
        schema_fields=None,
        storage_options=None,
        verbose=True,
        debug_data_loader: bool = False,
        seed: Optional[int] = None,
        resizing: Optional[int | float] = None,
        feature_scaling_method: Optional[str] = None,
        target_encoding_method: Optional[str] = None,
        select_class=None,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.feature_name = feature_name
        self.target_name = target_name
        self.id_name = id_name
        self.num_train_epochs = num_train_epochs
        self.has_val = has_val
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_reader_epochs = num_reader_epochs
        self.reader_pool_type = reader_pool_type
        self.reader_worker_count = reader_worker_count
        self.transformation = transformation
        self.transformation_edit_fields = transformation_edit_fields
        self.transformation_removed_fields = transformation_removed_fields
        self.inmemory_cache_all = inmemory_cache_all
        self.cur_shard = cur_shard
        self.shard_count = shard_count
        self.schema_fields = schema_fields
        self.storage_options = storage_options
        self.verbose = verbose
        self.debug_data_loader = debug_data_loader
        self.seed = seed
        self.resizing = resizing
        self.target_encoding_method = target_encoding_method
        self.select_class = select_class
        self.feature_scaling_method = feature_scaling_method
        self.__dict__.update(kwargs)  # Store all the extra variables

        self._feature_shape = None
        self._target_shape = None

        self.target_encoder = None
        self.feature_scaler = None
        self.prepared_data = False
        self.transform_dict = {}
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if debug_data_loader:
            print("Creating data_module")

    def prepare_data(self):
        if self.prepared_data:
            print("Data already prepared")
            self.transform_dict = load(open(f"{self.save_dir}/transform_dict.pkl", "rb"))
        else:
            self._prepare_transform()
            self.prepared_data = True


    def setup(self, stage=None):

        if self.target_encoding_method is not None:
            self.target_encoder = load(open(f"{self.save_dir}/target_encoder.pkl", "rb"))
        if self.feature_scaling_method is not None:
            df_stats = pd.read_csv(f"{self.save_dir}/stats_scaler.csv")
            self.feature_scaler = TS_Scaler(df_stats=df_stats)
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "test":
            predicate_expr, transform_spec = self.generate_ML_reader_predicate_transform()

            reader_factory_kwargs = dict()
            reader_factory = make_reader
            reader_factory_kwargs["pyarrow_serialize"] = True

        if stage == "fit" or stage is None:

            train_rows, _ = get_dataset_info(f"{self.data_dir}/train")

            self.steps_per_epoch_train = int(
                math.floor(float(train_rows) / self.train_batch_size / self.shard_count)
            )
            self.train_reader = reader_factory(
                f"file://{self.data_dir}/train",
                num_epochs=self.num_reader_epochs,
                reader_pool_type=self.reader_pool_type,
                workers_count=self.reader_worker_count,
                cur_shard=self.cur_shard,
                shard_count=self.shard_count,
                schema_fields=self.schema_fields,
                storage_options=self.storage_options,
                transform_spec=transform_spec,
                shuffle_rows=self.shuffle,
                shuffle_row_groups=self.shuffle,
                seed=self.seed,
                predicate = predicate_expr,
                **reader_factory_kwargs,
            )

            if self.has_val:
                val_rows, _ = get_dataset_info(f"{self.data_dir}/val")

                self.steps_per_epoch_val = int(
                    math.floor(float(val_rows) / self.val_batch_size / self.shard_count)
                )

                self.val_reader = reader_factory(
                    f"file://{self.data_dir}/val",
                    num_epochs=self.num_reader_epochs,
                    reader_pool_type=self.reader_pool_type,
                    workers_count=self.reader_worker_count,
                    cur_shard=self.cur_shard,
                    shard_count=self.shard_count,
                    schema_fields=self.schema_fields,
                    storage_options=self.storage_options,
                    transform_spec=transform_spec,
                    shuffle_rows=False,
                    shuffle_row_groups=False,
                    predicate = predicate_expr,
                    **reader_factory_kwargs,
                )

        elif stage == "test":
            test_rows, _ = get_dataset_info(f"{self.data_dir}/test")
            self.steps_per_epoch_test = int(
                math.floor(float(test_rows) / self.val_batch_size / self.shard_count)
            )

            self.test_reader = reader_factory(
                f"file://{self.data_dir}/test",
                num_epochs=self.num_reader_epochs,
                reader_pool_type=self.reader_pool_type,
                workers_count=self.reader_worker_count,
                cur_shard=self.cur_shard,
                shard_count=self.shard_count,
                schema_fields=self.schema_fields,
                storage_options=self.storage_options,
                transform_spec=transform_spec,
                shuffle_rows=False,
                shuffle_row_groups=False,
                predicate = predicate_expr,
                **reader_factory_kwargs,
            )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def train_dataloader(self):
        if self.verbose:
            print("Setup train dataloader")
        kwargs = dict(
            reader=self.train_reader,
            batch_size=self.train_batch_size,
            # name="train dataloader",
            limit_step_per_epoch=self.steps_per_epoch_train,
            # verbose=self.verbose,
        )

        self.train_dl = PytorchDataLoader(**kwargs)
        return self.train_dl

    def val_dataloader(self):
        # if not self.has_val:
        #     return None
        if self.verbose:
            print("setup val dataloader")
        kwargs = dict(
            reader=self.val_reader,
            batch_size=self.val_batch_size,
            # name="val dataloader",
            limit_step_per_epoch=self.steps_per_epoch_val,
            # verbose=self.verbose,
        )
        self.val_dl = PytorchDataLoader(**kwargs)
        return self.val_dl

    def test_dataloader(self):
        if self.verbose:
            print("setup val dataloader")
        kwargs = dict(
            reader=self.test_reader,
            batch_size=self.val_batch_size,
            # name="val dataloader",
            limit_step_per_epoch=self.steps_per_epoch_test,
            verbose=self.verbose,
        )

        self.test_dl = PytorchDataLoader(**kwargs)
        return self.test_dl

    def _extract_min_size(self, q):
        """Determine min size to retain qth quantile of the data

        Args:
            q ([float]): [quantile of the data which should be retained]

        Returns:
            [int]: [min size to retain qth quantile of the data]
        """
        with self.create_train_reader(schema_fields=["signal_length"]) as reader:
            l_tot = [l.signal_length for l in reader]
        min_size = np.quantile(l_tot, 1 - q)
        return int(min_size)

    def _prepare_transform(self):
        """[Function to list transform which have to be performed
        when loading dataset as well as create the feature scaler
        and target encoder]"""

        if isinstance(self.resizing, float):
            self.transform_dict["min_size"] = self._extract_min_size(self.resizing)
        elif isinstance(self.resizing, int):
            self.transform_dict["min_size"] = self.resizing

        if self.target_encoding_method is not None:
            self.transform_dict["target_encoding"] = self.target_encoding_method
            self.generate_encoder()

        if self.feature_scaling_method is not None:
            self.transform_dict["feature_scaler"] = self.feature_scaling_method
            self.generate_scaler()
        
        dump(self.transform_dict, open(os.path.join(self.save_dir, "transform_dict.pkl"), "wb"))


    def generate_ML_reader_predicate_transform(self):

        # all_fieds = [self.feature_name, self.target_name, "noun_id", "signal_names"]
        all_fieds = [self.feature_name, self.target_name, self.id_name]
        fields_mod = tuple()
        if self.transform_dict.get("target_encoding", None) == "label_encoding":
            # if target is encoded scheme need to be changed from string to integet
            fields_mod = (
                *fields_mod,
                (self.target_name, np.int32, (None,), NdarrayCodec(), False),
            )

        elif self.transform_dict.get("target_encoding", None) == "binary_encoding":
            # if target is encoded scheme need to be changed from string to integer
            fields_mod = (
                *fields_mod,
                (self.target_name, np.int_, (), ScalarCodec(IntegerType()), False),
            )

        elif self.transform_dict.get("target_encoding", None) == "onehot":
            # if target is encoded scheme need to be changed from string to integer
            fields_mod = (
                *fields_mod,
                (self.target_name, np.int_, (None,), NdarrayCodec(), False),
            )

        if len(self.transform_dict) > 0:
            transform = TransformSpec(
                self._transform_row, edit_fields=fields_mod, selected_fields=all_fieds
            )
        else:
            transform = None

        if self.transform_dict.get("min_size") is not None:
            min_size = self.transform_dict.get("min_size")
            # predicate_expr = in_reduce([in_pseudorandom_split(self.partition_split, partition_num, 'noun_id'),
            # in_lambda(['signal_length'], lambda signal_length: signal_length >= min_size)], all)
            predicate_expr = in_reduce(
                [
                    # in_lambda(
                    #     ["noun_id"],
                    #     lambda noun_id: noun_id.astype("str") in names_sample,
                    # ),
                    in_lambda(
                        ["signal_length"],
                        lambda signal_length: signal_length >= min_size,
                    ),
                ],
                all,
            )
        else:
            predicate_expr = None

        return predicate_expr, transform

    def transform_features(self, data_row, **kwargs):
        """[Function to add the various steps which should be performed on the features]

        Args:
            data_val ([type]): [description]

        Returns:
            [type]: [description]
        """

        data_val = data_row[self.feature_name]

        if self.transform_dict.get("min_size") is not None:
            min_size = self.transform_dict.get("min_size")
            data_val = resize_len(data_val, min_size, cropping="end")

        if self.transform_dict.get("feature_scaler") is not None:
            data_val = self.feature_scaler.transform(data_val)

        # we always convert the data to [features, sequence]
        data_val = np.transpose(data_val, (1, 0))

        return data_val

    def transform_targets(self, data_val):
        """Perform transformation on the target

        Args:
            data_val ([type]): [description]

        Returns:
            [type]: [description]
        """

        transform_target = self.transform_dict.get("target_encoding")

        if self.select_class != None:
            data_val = np.setdiff1d(data_val, self.list_remove).astype(int)

        if transform_target == "label_encoding":
            data_val = self.target_encoder.transform(
                np.expand_dims(data_val, axis=0)
            ).flatten()

        elif transform_target == "binary_encoding":
            if data_val.all() == 0:
                data_val = 0
            else:
                data_val = 1
        elif transform_target == "onehot":
            if np.array(data_val).dtype.type is np.string_:
                data_val = np.array(data_val).astype(str)

            if data_val.size > 1:
                # If we one hot encode a list of classes, we join the diagnostics
                # a string and one hot encode it
                data_val = np.array(",".join(data_val.astype(str).tolist()))
            if data_val.size == 0:
                data_val = np.array(["0"])
            data_val = self.target_encoder.transform(
                data_val.reshape(1, -1).astype(str)
            ).todense()
            # target_dataset = np.array(target_dataset).reshape(-1, 1)
            data_val = np.array(data_val).squeeze()
        # print(data_val.shape)
        return data_val

    def _transform_row(self, data_row):
        result_row = {
            self.feature_name: self.transform_features(data_row),
            self.target_name: self.transform_targets(data_row[self.target_name]),
            self.id_name: data_row[self.id_name],
            # "signal_names": data_row["signal_names"],
        }
        return result_row

    def generate_encoder(self):
        with self.create_train_reader(schema_fields=[self.target_name]) as reader:
            target_dataset = [l[0] for l in reader]
        classes = None
        if np.array(target_dataset).dtype.type is np.string_:
            target_dataset = np.array(target_dataset).astype(str)
        
        if self.select_class != None:
            print(f"Selecting class {self.select_class}")
            tmp = [reduce(lambda i, j: np.concatenate((i, j)), target_dataset)]
            unique = set(tmp[0])
            self.list_remove = [x for x in unique if x not in self.select_class]
            target_dataset = [np.setdiff1d(x, self.list_remove) for x in target_dataset]
            classes = self.select_class

        # if self.config["MANIPULATION"]["target_encoding"] == "label_encoding":
        #     encoder = preprocessing.MultiLabelBinarizer(classes=classes)
        #     encoder.fit(target_dataset)
        #     tmp = encoder.transform(target_dataset)
        #     classes_name = encoder.classes_

        if self.target_encoding_method == "onehot":
            classes = "auto"
            encoder = preprocessing.OneHotEncoder(categories=classes)
            if isinstance(target_dataset, list):
                # we want to deal with case where we onehot encode a list of labels
                target_dataset = np.array(
                    [",".join(x.astype(str).tolist()) for x in target_dataset]
                )

            target_dataset = np.array(["0" if x == "" else x for x in target_dataset])
            target_dataset = np.array(target_dataset).reshape(-1, 1)
            encoder.fit(target_dataset)
            tmp = encoder.transform(target_dataset)
            # classes_name = encoder.get_feature_names_out()

            # self._target_shape = len(classes_name)
            samples_per_cls = tmp.sum(axis=0).A

            np.save(
                os.path.join(self.save_dir, "samples_per_cls.npy"),
                samples_per_cls,
            )

            dump(encoder, open(os.path.join(self.save_dir, "target_encoder.pkl"), "wb"))

        else:
            raise ValueError("Encoder not recognised")

        return True

    def generate_scaler(self):
        # if "stats_scaler.csv" in os.listdir(self.results_path):
        #     df_stats_saved = pd.read_csv(
        #         os.path.join(self.results_path, "stats_scaler.csv")
        #     )
        # else:
        #     df_stats_saved = None

        # df_stats = self.ml_data.generate_scaler(df_stats_saved)
        feature_scaler = TS_Scaler(self.feature_scaling_method.lower())
        save_path_scaler = os.path.join(self.save_dir, "stats_scaler.csv")
        with self.create_train_reader(schema_fields=[self.feature_name]) as reader:
            print("Computing train dataset stats for scaler")
            feature_scaler.fit(reader, save_path_scaler)

        dump(
            feature_scaler,
            open(os.path.join(self.save_dir, "feature_scaler.pkl"), "wb"),
        )
        return True

    def evaluate_features_shape(self):
        with self.create_train_reader(schema_fields=[self.feature_name]) as reader:
            sample = reader.next()
        self._feature_shape = sample[0].shape
        if self.transform_dict.get("min_size") is not None:
            self._feature_shape[0] = self.transform_dict["min_size"]
        return True

    def evaluate_target_shape(self):
        try:
            np_classes =np.load(os.path.join(self.save_dir,"samples_per_cls.npy"))
        except:
            raise ValueError("No target encoder found")

        self._target_shape = np_classes.shape[1]

        return True


    def create_train_reader(
        self, schema_fields=None, predicate=None, transform_spec=None, num_epochs=1
    ):
        return make_reader(
            f"file://{self.data_dir}/train",
            schema_fields=schema_fields,
            predicate=predicate,
            num_epochs=num_epochs,
            transform_spec=transform_spec,
        )

    @property
    def feature_shape(self):
        if self._feature_shape is None:
            self.evaluate_features_shape()
        return self._feature_shape

    @property
    def target_shape(self):
        if self._target_shape is None:
            self.evaluate_target_shape()
        return self._target_shape


def resize_len(np_signal, new_len, cropping="end"):

    if cropping.lower() == "end":
        np_new = np_signal[0:new_len, :]

    elif cropping.lower() == "start":
        tmp_length = np_signal.shape[0]
        difference = tmp_length - new_len
        np_new = np_signal[difference:, :]

    elif cropping.lower() == "both":
        tmp_length = np_signal.shape[0]
        difference = tmp_length - new_len
        splits = difference // 2
        res = difference % 2
        lb = splits
        ub = tmp_length - splits
        if res != 0:
            lb = splits + 1
        np_new = np_signal[lb:ub, :]

    return np_new
