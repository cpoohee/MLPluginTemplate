import numpy as np
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import librosa
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from src.datamodule.audio_dataloader import AudioDataset
from torch.utils.data import DataLoader

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, cfg: DictConfig, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.ext = cfg.process_data.ext
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.cfg = cfg

    def setup(self, stage: str):
        if stage == "fit":
            self.df_train = self.form_dataframe(self.data_dir / 'train')
            self.df_val = self.form_dataframe(self.data_dir / 'val')
        if stage == "test":
            self.df_test = self.form_dataframe(self.data_dir / 'test')
        if stage == "predict":
            self.df_predict = self.form_dataframe(self.data_dir / 'test_full')

    def form_dataframe(self, data_path):
        data_path_x = data_path / 'x'
        data_path_y = data_path / 'y'

        x_files = librosa.util.find_files(data_path_x, ext=self.ext)
        y_files_unsorted = librosa.util.find_files(data_path_y, ext=self.ext)

        # ensure files are unique
        x_files_unique = np.unique(x_files)
        y_files_unsorted_unique = np.unique(y_files_unsorted)
        assert (len(x_files) == len(x_files_unique))
        assert (len(y_files_unsorted) == len(y_files_unsorted_unique))

        # do a proper search to assign y based on filename
        y_files = []
        for x_file in x_files:
            train_filename = Path(x_file).name
            found = False
            for i, y_file in enumerate(y_files_unsorted):
                if Path(y_file).name == train_filename:
                    y_files.append(y_file)
                    y_files_unsorted.pop(i)
                    found = True
                    break

            assert (found is True)

        data = {'x': x_files, 'y': y_files}
        df = pd.DataFrame(data=data)
        return df

    def train_dataloader(self):
        assert (self.df_train is not None)
        train_set = AudioDataset(self.df_train, cfg=self.cfg, do_augmentation=True)
        return DataLoader(train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        assert (self.df_val is not None)
        val_set = AudioDataset(self.df_val, cfg=self.cfg)
        return AudioDataset(val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        assert (self.df_test is not None)
        test_set = AudioDataset(self.df_test, cfg=self.cfg)
        return AudioDataset(test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        assert (self.df_predict is not None)
        pred_set = AudioDataset(self.df_predict, cfg=self.cfg)
        return AudioDataset(pred_set, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
