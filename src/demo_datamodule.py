import hydra
import os
from pathlib import Path
from omegaconf import DictConfig

from src.datamodule.audio_datamodule import AudioDataModule


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cur_path = Path(os.path.abspath(hydra.utils.get_original_cwd()))
    data_path = cfg.dataset.data_path

    batch_size = cfg.training.batch_size
    dm_train = AudioDataModule(data_dir=(cur_path/data_path),
                               cfg=cfg,
                               batch_size=batch_size)

    dm_train.setup(stage='fit')

    for i, batch in enumerate(dm_train.train_dataloader()):
        print(i)


if __name__ == "__main__":
    main()