import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from omegaconf import DictConfig
from torch_audiomentations import Compose, Identity, Gain, PolarityInversion, \
    Shift, AddColoredNoise, PitchShift
from src.datamodule.augmentations.custom_pitchshift import PitchShift_Slow
from src.datamodule.augmentations.random_crop import RandomCrop


class AudioDatasetPred(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DictConfig):
        self.df = df
        self.sample_length = int(
            cfg.dataset.sample_rate * cfg.process_data.clip_interval_ms / 1000.0)
        self.block_size = cfg.dataset.block_size
        self.sample_rate = cfg.dataset.sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        x_path = data['x']
        y_path = data['y']

        waveform_x, _ = torchaudio.load(x_path)
        waveform_y, _ = torchaudio.load(y_path)

        # waveform_x = torch.cat((waveform_x, waveform_x), dim=0)  # fake stereo
        # waveform_y = torch.cat((waveform_y, waveform_y), dim=0)

        return waveform_x, waveform_y
