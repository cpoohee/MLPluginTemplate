import hydra
import os
import torch
import mlflow
import simpleaudio as sa
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig
from src.datamodule.audio_datamodule import AudioDataModule
from src.model.wavenet import WaveNet_PL
from tqdm import tqdm


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
                              do_aug_in_predict=True, # allow low passed input
                              batch_size=1)

    mlflow.pytorch.autolog()


    ckpt_path = cfg.testing.checkpoint_file
    assert (ckpt_path is not None)

    wavenet_model = WaveNet_PL.load_from_checkpoint(cur_path/Path(ckpt_path))

    trainer = pl.Trainer(accelerator=cfg.testing.accelerator,)

    trainer.test(wavenet_model, dataloaders=dm_test)

    dm_pred.setup('predict')
    dm_pred = dm_pred.predict_dataloader()
    with torch.no_grad():
        for batch in tqdm(dm_pred):
            x, y = batch
            resized_samples = (x.size()[2] // cfg.dataset.block_size) * cfg.dataset.block_size
            x = x[:, :, 0:resized_samples]
            y = y[:, :, 0:resized_samples]

            x = torch.reshape(x, (-1, 1, cfg.dataset.block_size))
            y = torch.reshape(y, (-1, 1, cfg.dataset.block_size))

            y_pred = wavenet_model(x)

            ## reshape block sized batches into one single wav
            y_pred = torch.reshape(y_pred, (1, 1, -1))
            y = torch.reshape(y, (1, 1, -1))
            x = torch.reshape(x, (1, 1, -1))

            print('pred')
            play_tensor(y_pred[0])
            print('original input')
            play_tensor(x[0])
            print('original target')
            play_tensor(y[0])

def play_tensor(tensor_sample, sample_rate=44100):
    try:
        numpy_sample = tensor_sample.numpy()
        play_obj = sa.play_buffer(numpy_sample, 1, 4, sample_rate=sample_rate)
        play_obj.wait_done()
    except KeyboardInterrupt:
        print("next")
        sa.stop_all()


if __name__ == "__main__":
    main()
