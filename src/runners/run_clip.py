import torch
import numpy as np
from matplotlib import cm
import copy
from torchvision.utils import make_grid

from opt import get_opts

# dataset
from dataset import ImageDataset, TrainTransform, ValTransform
from torch.utils.data import DataLoader

# model
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger



from ..modules.clip_module import ClipModel





if __name__ == '__main__':
    hparams = get_opts()
    system = DINOModule(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_top_k=-1)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      precision=16 if hparams.fp16 else 32,
                      accelerator='auto',
                      devices=hparams.num_gpus,
                      strategy=DDPStrategy(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)