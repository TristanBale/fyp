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
import yaml

import numpy as np
import pandas as pd

# FILEPATH = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(FILEPATH)

warnings.simplefilter(action="ignore", category=FutureWarning)


class MultiSynthetic:
    """
    Main class to generate the synthetic dataset
    """
    def __init__(
            self, 
            save_path=None, 
            data_config=None,
    ):

        self.save_path = save_path
        # self.data_config = data_config
        # print("Generated data save in: ", self.save_path)

        self.nb_simulation = data_config["nb_simulation"]
        self.n_points = data_config["n_points"]
        self.n_support = data_config["n_support"]
        self.n_feature = data_config["n_feature"]
        self.f_min_support, self.f_max_support = data_config["f_sin"]
        self.f_min_base, self.f_max_base = data_config["f_base"]
        self.dataset_split = data_config["dataset_split"]
        # we create a distribution for the sum of sine frequency to
        # have control over class distribution
        f_sine_1 = np.random.randint(self.f_min_support, (self.f_max_support + 1), 15000)
        f_base_1 = np.random.randint(self.f_min_base, (self.f_max_base + 1), 15000)
        f_sine_2 = np.random.randint(self.f_min_support, (self.f_max_support + 1), 15000)
        f_base_2 = np.random.randint(self.f_min_base, (self.f_max_base + 1), 15000)
        f_sum = f_sine_1 + f_sine_2
        self.quantile_class_sum = np.quantile(f_sum, data_config["quantile_class"])
        f_ratio = (f_sine_1 / f_base_1) + (f_sine_2 / f_base_2)
        self.quantile_class_ratio = np.quantile(f_ratio, data_config["quantile_class"])
        self.bool_snr, self.target_snr = data_config["target_snr"]

    # Abstract methods
    # ----------------------------------------------------------------------------

    def format_to_DataFrame(self):
        """
        Method to create the dataset in pandas DataFrame format
        """

        num_instances = range(self.nb_simulation)
        data_list = []
        nounid_list = []
        cls_list = []
        columnsname = []
        df = pd.DataFrame()
        for idx in num_instances:
            sample = self._generate_sample(idx)
            data_list.append(sample)
            # column names for pandas DataFrame
            column_list = list(map(lambda a: sample['noun_id']+'_'+sample['signal_names'].astype(np.str_)[a], range(self.n_feature)))
            nounid_list.append(sample['noun_id'])
            cls_list.extend(sample['target_sum'])
            columnsname.extend(column_list)
            if idx == 0:
                np_signals = sample['signal']
            else:
                np_signals = np.concatenate((np_signals, sample['signal']), axis=1)

            # DataFrame for each instances
            # df_instances = pd.DataFrame(sample['signal'], columns=column_list)
            # df = pd.concat([df, df_instances], axis=1)
        
        # df.to_csv(os.path.join(self.save_path, 'multisynthetic_signals.csv'))
        # DataFrame will spend long time to generate dataset
        df_signal = pd.DataFrame(np_signals, columns=columnsname)
        dict_target = {'instance_id': nounid_list, 'classes': cls_list}
        df_cls = pd.DataFrame(dict_target)

        # save files
        df_signal.to_csv(os.path.join(self.save_path, 'multi_signals.csv'))
        df_cls.to_csv(os.path.join(self.save_path, 'multi_classes.csv'))
        print('Already Done!')



    # ----------------------------------------------------------------------------

    # Private methods
    # ----------------------------------------------------------------------------

    def _add_noise(self, signal):
        """
        Add noise to original signals
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
            x_tmp = np.sin(
                np.linspace(0, 2 * np.pi * f_support, self.n_points)
            ).reshape(-1, 1)
            start_tmp = 0
            x_feature[start_idx : start_idx + self.n_support, 0] += x_tmp[
                start_tmp : start_tmp + self.n_support, 0
            ]

        elif wave == "square":
            x_tmp = np.sign(
                np.sin(np.linspace(0, 2 * np.pi * f_support, self.n_points))
            ).reshape(-1, 1)
            start_tmp = 0
            x_feature[start_idx : start_idx + self.n_support, 0] += x_tmp[
                start_tmp : start_tmp + self.n_support, 0
            ]

        elif wave == "line":
            x_feature[start_idx : start_idx + self.n_support, 0] += [0] * self.n_support
        else:
            raise ValueError("wave must be one of sine, square, sawtooth, line")

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
            f_base = np.random.randint(self.f_min_base, (self.f_max_base + 1), 1)[0]
            f_support = np.random.randint(
                self.f_min_support, (self.f_max_support + 1), 1
            )[0]
            if enum < 2:
                f_sine_sum += f_support
                f_ratio += f_support / f_base
                x_tmp, start_idx = self._generate_feature(
                    wave="sine", f_support=f_support, f_base=f_base
                )
                x_sample[:, idx_feature] = x_tmp.squeeze()
                pos_sin_wave.append(start_idx)
            else:

                # "line": no support
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
            [f"feature{str(x)}" for x in range(x_sample.shape[1])]
        ).astype(np.string_)
        dict_all["pos_sin_wave"] = np.array(pos_sin_wave).astype(int)
        dict_all["feature_idx_sin_wave"] = idx_features[:2].astype(int)
        dict_all["f_sine_sum"] = f_sine_sum.astype(int)
        dict_all["f_sine_ratio"] = f_ratio.astype(np.float32)

        return dict_all


if __name__ == "__main__":
    save_path = "/home/TimeSeries-Interpretability-Robustness/src/generate_synthetic/MultivariateSynthetic/dataset_addnoise"
    config_path = "/home/TimeSeries-Interpretability-Robustness/configs/synthetic_data.yaml"
    with open(config_path) as f:
        data_config = yaml.safe_load(f)
    dict_pre = MultiSynthetic(save_path=save_path, data_config=data_config['properties'])
    dict_pre.format_to_DataFrame()