import torch
import torchmetrics
import random
import numpy as np
from typing import List, Dict
from torch.functional import Tensor
from torch.optim.optimizer import Optimizer
import pytorch_lightning as pl
from hydra.utils import instantiate, get_class
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class XrayModel(pl.LightningModule):
    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.network = network
        self.lr = cfg.lr
        self.map_metric = MeanAveragePrecision() # out metric for validating training process

    def configure_optimizers(self):
        """Instatiate optimizers and shedulers here

        Returns:
            Dict[Optimizer]: Optimizer, sheduler and some parameters
        """
        optimizer = get_class(self.cfg.optimizer)(filter(lambda p: p.requires_grad, self.network.parameters()),
                                                  lr=self.lr, weight_decay=self.cfg.w_d)
        scheduler = get_class(self.cfg.scheduler)(optimizer, **self.cfg.scheduler_kwargs, )
        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.cfg.scheduler_monitor,
                "interval": self.cfg.scheduler_interval}

    def forward(self, images, targets):
        return self.network(images, targets)

    def training_step(self, batch, batch_idx):
        images, annotations = batch
        # make predict and count losses
        losses = self.network(images, annotations)
        self.log(
            "train_loss",
            losses["loss"],
            logger=True
        )
        self.log(
            "train_class_loss",
            losses["class_loss"],
            logger=True,
            prog_bar=True
        )
        self.log(
            "train_box_loss",
            losses["box_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True)

        return losses['loss']
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations = batch
        outputs = self.network(images, annotations)
        detections = outputs["detections"]
        preds = []
        batch_size = images.shape[0]
        
        # aggregate information for metric computing
        for i in range(batch_size):
            scores = detections[i, ..., 4]
            non_zero_indices = scores.nonzero()
            boxes = detections[i, non_zero_indices, 0:4]
            labels = detections[i, non_zero_indices, 5]
            preds.append(
                dict(
                    boxes=boxes.squeeze(dim=1),
                    scores=scores[non_zero_indices].squeeze(dim=1),
                    labels=labels.squeeze(dim=1).int(),
                )
            )
        targets = []
        for i in range(batch_size):
            targets.append(
                dict(
                    boxes=annotations["bbox"][i][:, [1, 0, 3, 2]],
                    labels=annotations["cls"][i].int(),
                )
            )
        
        self.map_metric.update(preds=preds, target=targets)

        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }

        self.log(
            "valid_loss",
            outputs["loss"],
            logger=True,
            sync_dist=True
        )
        self.log(
            "valid_class_loss",
            logging_losses["class_loss"],
            logger=True,
            sync_dist=True
        )
        self.log(
            "valid_box_loss",
            logging_losses["box_loss"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        
        
        return {'loss': outputs["loss"], 'batch_predictions': preds}
    
    def on_validation_epoch_end(self):
        computed_metrics = self.map_metric.compute()
        # Extract the overall map metric
        overall_map = computed_metrics.get('map', None)
        if overall_map is not None:
            self.log("val_map", overall_map.item(), sync_dist=True)
        # Reset the metric
        self.map_metric.reset()
