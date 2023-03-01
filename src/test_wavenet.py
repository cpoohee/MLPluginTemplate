import hydra
import os
import mlflow
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig
from src.datamodule.audio_datamodule import AudioDataModule
from src.model.wavenet import WaveNet_PL
from pytorch_lightning.loggers import MLFlowLogger


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cur_path = Path(os.path.abspath(hydra.utils.get_original_cwd()))
    data_path = cfg.dataset.data_path

    batch_size = cfg.training.batch_size
    dm_test = AudioDataModule(data_dir=(cur_path / data_path),
                               cfg=cfg,
                               batch_size=batch_size)

    mlflow.pytorch.autolog()

    if cfg.training.use_mlflow:
        mlf_logger = MLFlowLogger(experiment_name=cfg.training.experiment_name,
                                  tracking_uri=cfg.training.tracking_uri)
    else:
        mlf_logger = None

    ckpt_path = cfg.training.checkpoint_file
    assert (ckpt_path is not None)

    wavenet_model = WaveNet_PL.load_from_checkpoint(cur_path/Path(ckpt_path))

    trainer = pl.Trainer(
        logger=mlf_logger,
    )

    trainer.test(wavenet_model, dataloaders=dm_test)


if __name__ == "__main__":
    main()
