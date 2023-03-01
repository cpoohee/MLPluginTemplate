import hydra
import os
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig
from src.datamodule.audio_datamodule import AudioDataModule
from src.model.wavenet import WaveNet_PL
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cur_path = Path(os.path.abspath(hydra.utils.get_original_cwd()))
    data_path = cfg.dataset.data_path

    batch_size = cfg.training.batch_size
    dm_train = AudioDataModule(data_dir=(cur_path / data_path),
                               cfg=cfg,
                               batch_size=batch_size)

    wavenet_model = WaveNet_PL(cfg)

    if cfg.training.use_checkpoint_callback:
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.training.model_checkpoint_path,
            filename=cfg.training.experiment_name + ' {epoch}'
        )
    else:
        checkpoint_callback = None

    if cfg.training.use_mlflow:
        mlf_logger = MLFlowLogger(experiment_name=cfg.training.experiment_name,
                                  tracking_uri=cfg.training.tracking_uri)
    else:
        mlf_logger = None

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        accelerator=cfg.training.accelerator,
        callbacks=[checkpoint_callback],
        logger=mlf_logger,
        log_every_n_steps=1,
    )

    if cfg.training.resume_checkpoint:
        ckpt_path = cfg.training.checkpoint_file
    else:
        ckpt_path = None

    trainer.fit(wavenet_model,
                train_dataloaders=dm_train,
                ckpt_path=ckpt_path,
    )

    save_model_path = Path(cfg.training.model_checkpoint_path) / (cfg.training.experiment_name + 'ckpt')
    trainer.save_checkpoint(save_model_path)


if __name__ == "__main__":
    main()
