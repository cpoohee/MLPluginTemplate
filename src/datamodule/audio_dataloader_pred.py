import torch
import torchaudio
import random
import pandas as pd
import torchaudio.transforms as T
from torch.utils.data import Dataset
from omegaconf import DictConfig
from src.model.speaker_encoder.speaker_embedder import SpeechEmbedder, AudioHelper

class AudioDatasetPred(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DictConfig):
        self.df = df
        self.model_name = cfg.model.model_name
        self.block_size_speaker = cfg.dataset.block_size_speaker

        if self.model_name == 'AutoEncoder_Speaker_PL':
            # speech embedder
            self.embedder = SpeechEmbedder()
            chkpt_embed = torch.load(cfg.model.embedder_path, map_location=cfg.training.accelerator)
            self.embedder.load_state_dict(chkpt_embed)
            self.embedder.eval()
            self.audio_helper = AudioHelper()

            # try to be as close as librosa's resampling
            self.resampler = T.Resample(orig_freq=cfg.dataset.block_size_speaker,
                                        new_freq=self.embedder.get_target_sample_rate(),
                                        lowpass_filter_width=64,
                                        rolloff=0.9475937167399596,
                                        resampling_method="sinc_interp_kaiser",
                                        beta=14.769656459379492,
                                        )

    def __get_embedding_vec(self, waveform_speaker):
        # embedding d vec
        waveform_speaker = self.resampler(waveform_speaker)  # resample to 16kHz
        waveform_speaker = waveform_speaker.squeeze()  # [16000]
        dvec_mel, _, _ = self.audio_helper.get_mel_torch(waveform_speaker)
        with torch.no_grad():
            dvec = self.embedder(dvec_mel)
            return dvec

    def __len__(self):
        return len(self.df)

    def __padding(self, waveform, target_size):
        # do padding if file is too small
        length_x = waveform.size(dim=1)
        if length_x < target_size:
            waveform = torch.nn.functional.pad(waveform,
                                               (1, target_size - length_x - 1),
                                               "constant", 0)
        return waveform

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        x_path = data['x']
        speaker_name = data['speaker_name']
        related_speakers = data['related_speakers']
        id_other_related = random.choice(related_speakers)
        speaker_path = self.df.iloc[id_other_related].x

        unrelated_speakers = [i for i in range(0, len(self.df)) if i not in related_speakers]
        unrelated_speakers.remove(idx)

        id_other_unrelated = random.choice(unrelated_speakers)
        unrelated_speaker_path = self.df.iloc[id_other_unrelated].x
        unrelated_speakers_name = self.df.iloc[id_other_unrelated].speaker_name

        waveform_x, _ = torchaudio.load(x_path)
        waveform_speaker, _ = torchaudio.load(speaker_path)
        waveform_unrelated_speaker, _ = torchaudio.load(unrelated_speaker_path)
        waveform_y = waveform_x

        waveform_speaker = self.__padding(waveform_speaker, self.block_size_speaker)

        # get speaker embeddings
        if self.model_name == 'AutoEncoder_Speaker_PL':
            dvec_related = self.__get_embedding_vec(waveform_speaker)
            dvec_unrelated = self.__get_embedding_vec(waveform_unrelated_speaker)

            dvec = (dvec_related, dvec_unrelated)
            speaker_names = (speaker_name, unrelated_speakers_name)
        else:
            dvec = (waveform_speaker, None)
            speaker_names = (speaker_name, None)

        # waveform_x = torch.cat((waveform_x, waveform_x), dim=0)  # fake stereo
        # waveform_y = torch.cat((waveform_y, waveform_y), dim=0)

        return waveform_x, waveform_y, dvec, speaker_names
