from typing import List, Tuple

import hydra
import pyrootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import numpy as np
import os

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
from src.interpretability.manipulation_results import ScoreComputation

log = utils.get_pylogger(__name__)

methods_relevance = [
        "integrated_gradients",
        "deeplift",
        "deepliftshap",
        "gradshap",
        "shapleyvalue",
        "kernelshap",
    ]


@utils.task_wrapper
def evaluate_interpretability(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates interpretability methods for given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # transform data
    datamodule.prepare_data()
    cfg.model.net.input_size = datamodule.feature_shape
    cfg.model.net.output_size = datamodule.target_shape

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    signal_list = []
    target_list = []
    nounid_list = []
    for batch in test_loader:
        if isinstance(batch, dict):
            #Assume that the first key is the signal and the second is the target
            signal_key, target_key, nounid_key = batch.keys()
            signal_tmp = batch[signal_key].detach().numpy()
            target_tmp = batch[target_key].detach().numpy()
            nounid_tmp = batch[nounid_key]
        else:
            raise ValueError("Only batch dictionaries are supported")
        signal_list.extend(signal_tmp)
        target_list.extend(target_tmp)
        nounid_list.extend(nounid_tmp)
    np_signal = np.array(signal_list)
    np_target = np.array(target_list)
    np_nounid = np.array(nounid_list).astype(np.str_)

    with open(os.path.join(cfg.paths.output_dir,"signal.npy"), 'wb') as f:
        np.save(f, np_signal)
    with open(os.path.join(cfg.paths.output_dir,"target.npy"), 'wb') as f:
        np.save(f, np_target)
    with open(os.path.join(cfg.paths.output_dir,"nounid.npy"), 'wb') as f:
        np.save(f, np_nounid)


    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    # evaluate interpretability methods
    signal = np.load(os.path.join(cfg.paths.output_dir,"signal.npy"))
    target = np.load(os.path.join(cfg.paths.output_dir,"target.npy"))
    nounid = np.load(os.path.join(cfg.paths.output_dir,"nounid.npy"))
    all_qfeatures = np.arange(0.05, 1.05, 0.10)
    all_qfeatures = np.round(all_qfeatures, 2).tolist()
    all_qfeatures.append(1.0)
    
    path_results = cfg.paths.output_dir

    score_computation = ScoreComputation(model, signal, target, nounid, path_results)

    for method in methods_relevance:
        score_computation.compute_relevance(method)
    for method in methods_relevance:
        _ = score_computation.compute_scores_wrapper(
            all_qfeatures, method
        )
        score_computation.create_summary(method)
    score_computation.summarise_results()

    save_results_path = score_computation.save_results
    # plot_DeltaS_results(save_results_path)
    # plot_additional_results(save_results_path)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_interpret.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate_interpretability(cfg)


if __name__ == "__main__":
    main()
