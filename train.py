import hydra
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy

from hydra.utils import instantiate
from omegaconf import DictConfig

from src.lightning_module import XrayModel
from src.models.networks import create_model
from src import utils


def train(
    network,
    current_time,
    cfg,
    logger,
    ckpt=None,
    ):
    checkpoints_dir = os.path.join(cfg.checkpoints_root, current_time)
    model_cls = XrayModel
    lightning_model = model_cls(network, cfg)
    # if we want to continue from old checkpoint
    if cfg.from_ckpt and ckpt:
        sd = torch.load(ckpt)["state_dict"]
        lightning_model.load_state_dict(sd)
    # if we want to freeze some layers to tune only head
    if cfg.freeze:
        print(f"Freeze first {cfg.freeze} layers")
        for layer in range(cfg.freeze):
            for param in lightning_model.network.model.features[layer].parameters():
                param.requires_grad = False
    
    dm = instantiate(cfg.datamodule)
    model_checkpoint = instantiate(
        cfg.callbacks.model_checkpoint,
        dirpath=checkpoints_dir,
        )
    early_stopping = instantiate(cfg.callbacks.early_stopping)
    lr_monitor = instantiate(cfg.callbacks.lr_monitor)
    callbacks = [model_checkpoint, early_stopping, lr_monitor]
    # main trainer
    trainer = pl.Trainer(
        default_root_dir=checkpoints_dir,
        devices=cfg.gpus, # num and numbers of gpus
        strategy=DDPStrategy(), # DDPS strategy to train on 2 gpus 
        min_epochs=cfg.min_epochs,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision, # fp16 presicion to train faster 
        callbacks=callbacks, 
        logger=logger,
    )

    dm.get_dataset()

    if cfg.auto_lr_find:
        trainer.tune(
            lightning_model,
            datamodule=dm,
        )
    # start training!
    trainer.fit(
        lightning_model,
        datamodule=dm,
    )

    if utils.is_rank_zero():
        lightning_model = lightning_model.load_from_checkpoint(model_checkpoint.best_model_path)
        torch.save(lightning_model.network, os.path.join(checkpoints_dir, 'model_best.pt'))
    return lightning_model.network

# load config file and start training
@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    current_time = utils.get_current_time()
    # logging everything to the mlflow
    logger = instantiate(cfg.logger, tags={'mlflow.runName': current_time})
    # if we want to start training from our checkpoint
    if cfg.from_ckpt:
        ckpt = cfg.ckpt
    else:
        ckpt = None
    # creating model from out config
    network = create_model(cfg.num_classes, cfg.image_size, cfg.architecture, cfg.backbone)
    # Start training!
    train(
        network=network,
        current_time=current_time,
        cfg=cfg,
        logger=logger,
        ckpt=ckpt,
    )


if __name__ == '__main__':
    main()
