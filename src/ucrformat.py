from typing import List, Tuple

import hydra
import pyrootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import numpy as np
import os
import sys
import time

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils
from src.ucrformat.preprocess_ucr import PreprocessUCR
from src.ucrformat.preprocess_ucr_ts import PreprocessTS
from pyspark import SparkFiles
from pyspark.sql import SparkSession


log = utils.get_pylogger(__name__)

@utils.task_wrapper
def format_ucr(cfg: DictConfig) -> Tuple[dict, dict]:
    """Format ucr dataset using the config in configs/ucrformat_data.yaml

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """


    log.info(f"Loading config file for formatting ucr dataset...")
    if cfg.properties.arff:
        arff_path = os.path.join(cfg.paths.data_dir, 'ucrdata', cfg.save_name)
    else:
        arff_path = cfg.properties.arff
    
    save_path = os.path.join(cfg.paths.data_dir, cfg.save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log.info(f"Formatted ucr dataset will be saved in: <{save_path}>")

    data_config = cfg.properties

    log.info("Creating Spark Session...")
    session_builder = (
        SparkSession.builder.appName("Dataset Creation")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.driver.memory", "15g")
    )
    session_builder.master("local[2]")

    log.info("Setting Spark Environments...")
    # python_interpeter = sys.executable
    # os.environ["PYSPARK_PYTHON"] = python_interpeter
    # os.environ["PYSPARK_DRIVER_PYTHON"] = python_interpeter
    spark = session_builder.getOrCreate()
    sc = spark.sparkContext
    sc.addFile(
        os.path.join(cfg.paths.root_dir, "src", "ucrformat"), recursive = True
    )
    # sc.addFile(os.path.join(cfg.paths.root_dir, "src", "generate_synthetic", "components", "schema.py"), recursive = True)
    # sys.path.insert(0, SparkFiles.getRootDirectory())

    log.info("Starting UCR dataset formation...")
    if arff_path:
        dict_pre = PreprocessUCR(
            arff_path=arff_path,
            save_path=save_path,
            data_config=data_config,
        )
    else:
        dict_pre = PreprocessTS(
            save_path=save_path,
            data_config=data_config,
        )

    log.info("Formatting to Parquet...")
    t_start = time.time()
    dict_pre.format_to_parquet(spark)
    t_diff = time.time() - t_start
    log.info(f"Done Elapsed time {t_diff}s.")

    log.info("Saving sample names to files...")

    log.info("All Done for ucr dataset format!")

    object_dict = {
        "cfg": cfg,
    }
    metric_dict = True

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="ucrformat_data.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    format_ucr(cfg)


if __name__ == "__main__":
    main()
