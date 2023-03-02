import hydra
import os
import mlflow
import simpleaudio as sa
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig
from src.datamodule.audio_datamodule import AudioDataModule
from src.model.wavenet import WaveNet_PL


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cur_path = Path(os.path.abspath(hydra.utils.get_original_cwd()))
    data_path = cfg.dataset.data_path

    batch_size = cfg.training.batch_size
    dm_test = AudioDataModule(data_dir=(cur_path / data_path),
                               cfg=cfg,
                               batch_size=batch_size)
    dm_pred = AudioDataModule(data_dir=(cur_path / data_path),
                              cfg=cfg,
                              batch_size=1)

    mlflow.pytorch.autolog()


    ckpt_path = cfg.testing.checkpoint_file
    assert (ckpt_path is not None)

    wavenet_model = WaveNet_PL.load_from_checkpoint(cur_path/Path(ckpt_path))

    trainer = pl.Trainer(accelerator=cfg.testing.accelerator,)

    trainer.test(wavenet_model, dataloaders=dm_test)

    results = trainer.predict(wavenet_model, dataloaders=dm_pred)

    for y, pred in results:
        print('pred')
        play_tensor(pred[0])
        print('original')
        play_tensor(y[0])

def play_tensor(tensor_sample, sample_rate=44100):
    numpy_sample = tensor_sample.numpy()
    play_obj = sa.play_buffer(numpy_sample, 1, 4, sample_rate=sample_rate)
    play_obj.wait_done()


if __name__ == "__main__":
    main()
