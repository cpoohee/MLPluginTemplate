import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from omegaconf import DictConfig

class AudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DictConfig, do_augmentation: bool = False):
        self.df = df
        self.aug_timeshift = cfg.augmentations.do_timeshift
        self.aug_random_block = cfg.augmentations.do_random_block
        self.aug_polarity_inv = cfg.augmentations.do_polarity_inv
        self.aug_gain = cfg.augmentations.do_gain
        self.do_augmentation = do_augmentation
        self.sample_length = int (cfg.process_data.sr * cfg.process_data.clip_interval_ms/1000.0)
        self.block_size = cfg.dataset.block_size

    def __len__(self):
        return len(self.df)

    def __process_augmenations(self, waveform):
        # TODO: conduct more augmentations
        return waveform

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        x_path = data['x']
        y_path = data['y']

        waveform_x, _ = torchaudio.load(x_path)
        waveform_y, _ = torchaudio.load(y_path)

        # do padding if file is too small
        length_x = waveform_x.size(dim=1)
        if length_x < self.sample_length:
            waveform_x = torch.nn.functional.pad(waveform_x,
                                                 (1, self.sample_length-length_x-1),
                                                 "constant", 0)

        length_y = waveform_y.size(dim=1)
        if length_y < self.sample_length:
            waveform_y = torch.nn.functional.pad(waveform_y,
                                                 (1, self.sample_length-length_y-1),
                                                 "constant", 0)


        # TODO: select some block

        if self.do_augmentation:
            waveform_x = self.__process_augmenations(waveform_x)

        return waveform_x, waveform_y
