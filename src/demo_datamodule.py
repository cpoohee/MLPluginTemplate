import hydra
import os
import simpleaudio as sa
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
        x, y = batch
        play_tensor(y[0])
        print(i)


def play_tensor(tensor_sample, sample_rate=44100):
    numpy_sample = tensor_sample.numpy()
    play_obj = sa.play_buffer(numpy_sample, 1, 4, sample_rate=sample_rate)
    play_obj.wait_done()


if __name__ == "__main__":
    main()