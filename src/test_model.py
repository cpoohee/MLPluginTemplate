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
from src.model.waveUnet import WaveUNet_PL
from src.model.autoencoder import AutoEncoder_PL
from tqdm import tqdm


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cur_path = Path(os.path.abspath(hydra.utils.get_original_cwd()))
    data_path = cfg.dataset.data_path

    batch_size = cfg.training.batch_size
    dm_test = AudioDataModule(data_dir=(cur_path / data_path),
                               cfg=cfg,
                               batch_size=batch_size,
                               do_aug_in_test=cfg.testing.do_aug_in_test)
    dm_pred = AudioDataModule(data_dir=(cur_path / data_path),
                              cfg=cfg,
                              do_aug_in_predict=cfg.testing.do_aug_in_predict, # allow low passed input
                              batch_size=1)

    mlflow.pytorch.autolog()

    ckpt_path = cfg.testing.checkpoint_file
    assert (ckpt_path is not None)

    if cfg.testing.model_name == 'WaveNet_PL':
        model = WaveNet_PL.load_from_checkpoint(cur_path / Path(ckpt_path))
    elif cfg.testing.model_name == 'WaveUNet_PL':
        model = WaveUNet_PL.load_from_checkpoint(cur_path / Path(ckpt_path))
    elif cfg.testing.model_name == 'AutoEncoder_PL':
        model = AutoEncoder_PL.load_from_checkpoint(cur_path / Path(ckpt_path))
    else:
        assert False, " model name is invalid!"

    trainer = pl.Trainer(accelerator=cfg.testing.accelerator)

    trainer.test(model, dataloaders=dm_test)

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

            y_pred = model(x)

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
