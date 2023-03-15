import numpy as np
import torch
import torchaudio
import random
import pandas as pd
import torchaudio.transforms as T
from torch.utils.data import Dataset
from omegaconf import DictConfig
from torch_audiomentations import (
    Compose, Identity, Gain, PolarityInversion,
    Shift, AddColoredNoise, PitchShift, LowPassFilter)
from src.datamodule.augmentations.custom_pitchshift import PitchShift_Slow
from src.datamodule.augmentations.random_crop import RandomCrop
from src.model.speaker_encoder.speaker_embedder import SpeechEmbedder, AudioHelper


class AudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DictConfig, do_augmentation: bool = False):
        self.df = df
        self.sample_length = int(
            cfg.dataset.sample_rate * cfg.process_data.clip_interval_ms / 1000.0)
        self.block_size = cfg.dataset.block_size
        self.block_size_speaker = cfg.dataset.block_size_speaker
        self.sample_rate = cfg.dataset.sample_rate

        # for cropping audio length
        self.do_random_block = cfg.dataset.do_random_block

        # augmentations params
        self.do_augmentation = do_augmentation

        self.aug_pitchshift = cfg.augmentations.do_pitchshift
        self.min_transpose_semitones = cfg.augmentations.min_transpose_semitones
        self.max_transpose_semitones = cfg.augmentations.max_transpose_semitones
        self.pitchshift_p = cfg.augmentations.pitchshift_p

        self.aug_colored_noise = cfg.augmentations.do_colored_noise
        self.min_snr_in_db = cfg.augmentations.min_snr_in_db
        self.max_snr_in_db = cfg.augmentations.max_snr_in_db
        self.min_f_decay = cfg.augmentations.min_f_decay
        self.max_f_decay = cfg.augmentations.max_f_decay
        self.colored_noise_p = cfg.augmentations.colored_noise_p

        self.aug_polarity_inv = cfg.augmentations.do_polarity_inv
        self.polarity_p = cfg.augmentations.polarity_p

        self.aug_gain = cfg.augmentations.do_gain
        self.gain_p = cfg.augmentations.gain_p
        self.min_gain_in_db = cfg.augmentations.min_gain_in_db
        self.max_gain_in_db = cfg.augmentations.max_gain_in_db

        self.aug_gain_indep = cfg.augmentations.do_gain_indep
        self.gain_p_indep = cfg.augmentations.gain_p_indep
        self.min_gain_in_db_indep = cfg.augmentations.min_gain_in_db_indep
        self.max_gain_in_db_indep = cfg.augmentations.max_gain_in_db_indep

        self.aug_timeshift_indep = cfg.augmentations.do_timeshift_indep
        self.min_shift_indep = cfg.augmentations.min_shift_indep
        self.max_shift_indep = cfg.augmentations.max_shift_indep
        self.timeshift_p_indep = cfg.augmentations.timeshift_p_indep

        self.aug_pitchshift_indep = cfg.augmentations.do_pitchshift_indep
        self.min_transpose_semitones_indep = cfg.augmentations.min_transpose_semitones_indep
        self.max_transpose_semitones_indep = cfg.augmentations.max_transpose_semitones_indep
        self.pitchshift_p_indep = cfg.augmentations.pitchshift_p_indep

        self.aug_low_pass_x = cfg.augmentations.do_low_pass_x
        self.min_cutoff_freq_x = cfg.augmentations.min_cutoff_freq_x
        self.max_cutoff_freq_x = cfg.augmentations.max_cutoff_freq_x
        self.low_pass_p_x = cfg.augmentations.low_pass_p_x

        self.__initialise_augmentations()

        self.model_name = cfg.model.model_name

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

    def set_random_crop(self, set):
        self.do_random_block = set

    def __initialise_augmentations(self):
        # optimise augmentation time by doing time sensitive augmentation first,
        # crop and augment on smaller blocks later
        transforms = [Identity()]

        if self.aug_timeshift_indep:
            transforms.append(Shift(min_shift=self.min_shift_indep,
                                    max_shift=self.max_shift_indep,
                                    p=self.timeshift_p_indep,
                                    p_mode='per_example',
                                    shift_unit='seconds'))

        self.apply_augmentation_before_crop = Compose(transforms)

        # include Identity in case no augmentation done, and apply_augmentation will still be valid
        transforms = [Identity()]

        if self.aug_gain:
            transforms.append(Gain(min_gain_in_db=self.min_gain_in_db,
                                   max_gain_in_db=self.max_gain_in_db,
                                   p=self.gain_p,
                                   p_mode='per_batch'))

        if self.aug_polarity_inv:
            transforms.append(PolarityInversion(p=self.polarity_p,
                                                p_mode='per_batch'))

        if self.aug_pitchshift:
            transforms.append(PitchShift(min_transpose_semitones=self.min_transpose_semitones,
                                         max_transpose_semitones=self.max_transpose_semitones,
                                         p=self.pitchshift_p,
                                         sample_rate=self.sample_rate,
                                         p_mode='per_batch'))

        if self.aug_colored_noise:
            transforms.append(AddColoredNoise(min_snr_in_db=self.min_snr_in_db,
                                              max_snr_in_db=self.max_snr_in_db,
                                              min_f_decay=self.min_f_decay,
                                              max_f_decay=self.max_f_decay,
                                              p=self.colored_noise_p,
                                              p_mode='per_batch'))

        self.apply_augmentation_combo = Compose(transforms)

        ## for independent augmentations
        transforms = [Identity()]

        if self.aug_gain_indep:
            transforms.append(Gain(min_gain_in_db=self.min_gain_in_db_indep,
                                   max_gain_in_db=self.max_gain_in_db_indep,
                                   p=self.gain_p_indep,
                                   p_mode='per_example'))

        # micro pitch cannot be done on torch_audiomentation, revert to original audiomentation
        if self.aug_pitchshift_indep:
            transforms.append(
                PitchShift_Slow(min_transpose_semitones=self.min_transpose_semitones_indep,
                                max_transpose_semitones=self.max_transpose_semitones_indep,
                                p=self.pitchshift_p_indep,
                                sample_rate=self.sample_rate,
                                p_mode='per_example'))

        self.apply_augmentation_indep = Compose(transforms)

        self.apply_augmentation_crop = \
            Compose([RandomCrop(max_length=self.block_size,
                                sampling_rate=self.sample_rate,
                                max_length_unit='samples'),
                     ])

        self.apply_augmentation_crop_speaker = \
            Compose([RandomCrop(max_length=self.block_size_speaker,
                                sampling_rate=self.sample_rate,
                                max_length_unit='samples'),
                     ])

        transforms = [Identity()]
        if self.aug_low_pass_x:
            transforms.append(
                LowPassFilter(
                    min_cutoff_freq=self.min_cutoff_freq_x,
                    max_cutoff_freq=self.max_cutoff_freq_x,
                    p=self.low_pass_p_x,
                    p_mode='per_example'
                )
            )
        self.apply_augmentation_x = Compose(transforms)

    def __len__(self):
        return len(self.df)

    def __process_augmentations_input_only(self, waveform):
        waveform = self.apply_augmentation_x(waveform, sample_rate=self.sample_rate)
        return waveform

    def __process_augmentations_before_crop(self, waveform):
        waveform = self.apply_augmentation_before_crop(waveform, sample_rate=self.sample_rate)
        return waveform

    def __process_augmentations_combo(self, waveform):
        # conduct more augmentations,
        waveform = self.apply_augmentation_combo(waveform, sample_rate=self.sample_rate)
        return waveform

    def __process_augmentations_independent(self, waveform):
        waveform = self.apply_augmentation_indep(waveform, sample_rate=self.sample_rate)
        return waveform

    def __random_block(self, waveform):
        if waveform.size(dim=2) > self.block_size:
            waveform = self.apply_augmentation_crop(waveform, sample_rate=self.sample_rate)
        return waveform

    def __random_block_speaker(self, waveform):
        if waveform.size(dim=2) > self.block_size_speaker:
            waveform = self.apply_augmentation_crop_speaker(waveform, sample_rate=self.sample_rate)
        return waveform

    def __padding(self, waveform, target_size):
        # do padding if file is too small
        length_x = waveform.size(dim=1)
        if length_x < self.sample_length:
            waveform = torch.nn.functional.pad(waveform,
                                               (1, target_size - length_x - 1),
                                               "constant", 0)
        return waveform

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        x_path = data['x']
        speaker_name = data['speaker_name']
        related_speakers = data['related_speakers']
        id_other = random.choice(related_speakers)
        speaker_path = self.df.iloc[id_other].x

        waveform_x, _ = torchaudio.load(x_path)
        waveform_speaker, _ = torchaudio.load(speaker_path)

        waveform_x = self.__padding(waveform_x, self.block_size)
        waveform_speaker = self.__padding(waveform_speaker, self.block_size_speaker)

        # 'batch' up waveforms x and y together
        waveform_x = torch.unsqueeze(waveform_x, dim=0)
        waveform_y = waveform_x  # copy input as target
        waveform_speaker = torch.unsqueeze(waveform_speaker, dim=0)
        waveform = torch.cat((waveform_x, waveform_y), 0)

        # decided to do time shift earlier then do block cropping, to prevent too many zero pads
        if self.do_augmentation:
            waveform = self.__process_augmentations_before_crop(waveform)

        if self.do_random_block:
            waveform = self.__random_block(waveform)
            waveform_speaker = self.__random_block_speaker(waveform_speaker)

        # do the rest of the augmentations with smaller block for faster processing

        if self.do_augmentation:
            # do augmentations that we want to affect on both x and y equally
            waveform = self.__process_augmentations_combo(waveform)

            # do augmentations that we want to affect on x and y independently
            waveform = self.__process_augmentations_independent(waveform)

        waveform_x = waveform[0]
        waveform_y = waveform[1]

        if self.do_augmentation:
            waveform_x = torch.unsqueeze(waveform_x, dim=0)
            waveform_x = self.__process_augmentations_input_only(waveform_x)
            waveform_x = waveform_x[0]

        waveform_x = self.__padding(waveform_x, self.block_size)
        waveform_y = self.__padding(waveform_y, self.block_size)
        waveform_speaker = self.__padding(waveform_speaker[0], self.block_size_speaker)

        # get speaker embeddings
        if self.self.model_name == 'AutoEncoder_Speaker_PL':
            dvec = self.__get_embedding_vec(waveform_speaker)
        else:
            dvec = waveform_speaker

        # waveform_x = torch.cat((waveform_x, waveform_x), dim=0) # fake stereo
        # waveform_y = torch.cat((waveform_y, waveform_y), dim=0)

        return waveform_x, waveform_y, dvec, speaker_name
