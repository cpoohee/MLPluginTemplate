import torchaudio
import random
import pandas as pd
from torch.utils.data import Dataset
from omegaconf import DictConfig


class AudioDatasetPred(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DictConfig):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        x_path = data['x']
        speaker_name = data['speaker_name']
        related_speakers = data['related_speakers']
        id_other = random.choice(related_speakers)
        speaker_path = self.df.iloc[id_other].x

        waveform_x, _ = torchaudio.load(x_path)
        waveform_speaker, _ = torchaudio.load(speaker_path)
        waveform_y = waveform_x

        # waveform_x = torch.cat((waveform_x, waveform_x), dim=0)  # fake stereo
        # waveform_y = torch.cat((waveform_y, waveform_y), dim=0)

        return waveform_x, waveform_y, waveform_speaker, speaker_name
