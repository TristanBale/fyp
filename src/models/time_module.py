# Edit this file to change label class. Binary vs Multiclass, if mulitclass, uncomment score_x0_2 for the 3 instances it appears

from typing import Any, List

import torch
import lightning.pytorch as pl
# from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import models.augmentation.augment_signal as T

import pandas as pd
import numpy as np


class TSModule(pl.LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        feature_name: str,
        target_name:   str,
        id_name: str,
        save_dir: str,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        **kwargs: Any,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.feature_name = feature_name
        self.target_name = target_name

        self.augmenter = T.RandAugment(magnitude = 10, augmentation_operations = 'Random_block')

        self.id_name = id_name
        self.save_dir = save_dir
        self.net = net
        print("Model createed")
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=3)
        # self.val_acc = Accuracy(task="multiclass", num_classes=3)
        # self.test_acc = Accuracy(task="multiclass", num_classes=3)
        self.train_acc = Accuracy(task="binary", num_classes=2)
        self.val_acc = Accuracy(task="binary", num_classes=2)
        self.test_acc = Accuracy(task="binary", num_classes=2)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.all_nounid = []
        self.all_preds = []
        self.all_targets = []
        self.score_x0_0 = []
        self.score_x0_1 = []
        # self.score_x0_2 = []
        # self.score_x0_3 = []
        # self.score_x0_4 = []
        # self.score_x0_5 = []
        # self.score_x0_6 = []
        # self.score_x0_7 = []
        # self.score_x0_8 = []
        # self.score_x0_9 = []

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x = batch[self.feature_name]
        y = batch[self.target_name]
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, torch.argmax(y,dim=1), torch.nn.functional.softmax(logits, dim=-1)

    def training_step(self, batch: Any, batch_idx: int):
        # add augmentation to original time series (adversarial training)
        batch[self.feature_name] = self.augmenter(batch[self.feature_name], nb_epoch=10)
        loss, preds, targets, scores = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, scores = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, scores = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # print({"loss": loss, "preds": preds, "targets": targets})
        self.all_nounid.extend(np.array(batch['noun_id']).astype('str'))
        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        self.score_x0_0.extend(scores[0][0].cpu().numpy().reshape(1,))
        self.score_x0_1.extend(scores[0][1].cpu().numpy().reshape(1,))
        # self.score_x0_2.extend(scores[0][2].cpu().numpy().reshape(1,))
        # self.score_x0_3.extend(scores[0][3].cpu().numpy().reshape(1,))
        # self.score_x0_4.extend(scores[0][4].cpu().numpy().reshape(1,))
        # self.score_x0_5.extend(scores[0][5].cpu().numpy().reshape(1,))
        # self.score_x0_6.extend(scores[0][6].cpu().numpy().reshape(1,))
        # self.score_x0_7.extend(scores[0][7].cpu().numpy().reshape(1,))
        # self.score_x0_8.extend(scores[0][8].cpu().numpy().reshape(1,))
        # self.score_x0_9.extend(scores[0][9].cpu().numpy().reshape(1,))

        
        # save prediction results for testing samples
        pd_results = pd.DataFrame({
            'noun_id':self.all_nounid, 
            'label':self.all_targets, 
            'pred':self.all_preds, 
            'score_x0_0': self.score_x0_0, 
            'score_x0_1': self.score_x0_1,
            # 'score_x0_2': self.score_x0_2,
            # 'score_x0_3': self.score_x0_3,
            # 'score_x0_4': self.score_x0_4,
            # 'score_x0_5': self.score_x0_5,
            # 'score_x0_6': self.score_x0_6,
            # 'score_x0_7': self.score_x0_7,
            # 'score_x0_8': self.score_x0_8,
            # 'score_x0_9': self.score_x0_9,
            })
        pd_results.to_csv(f"{self.save_dir}/results_test.csv", index=None)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # import hydra
    # import omegaconf
    # import pyrootutils

    # root = pyrootutils.setup_root(__file__, pythonpath=True)
    # cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    # _ = hydra.utils.instantiate(cfg)
    _ = TSModule(None, None, None, None, None, None)
    