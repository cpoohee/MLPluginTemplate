import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from omegaconf import DictConfig
from torch_audiomentations import (
    Compose, Identity, Gain, PolarityInversion, \
    Shift, AddColoredNoise, PitchShift, LowPassFilter)
from src.datamodule.augmentations.custom_pitchshift import PitchShift_Slow
from src.datamodule.augmentations.random_crop import RandomCrop


class AudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DictConfig, do_augmentation: bool = False):
        self.df = df
        self.sample_length = int(
            cfg.dataset.sample_rate * cfg.process_data.clip_interval_ms / 1000.0)
        self.block_size = cfg.dataset.block_size
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

        # self.device = cfg.training.accelerator

        self.__initialise_augmentations()

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

        self.apply_augmentation_crop = Compose([RandomCrop(max_length=self.block_size,
                                                           sampling_rate=self.sample_rate,
                                                           max_length_unit='samples'),
                                                ])

        if self.aug_low_pass_x:
            transforms = [Identity()]
            transforms.append(
                LowPassFilter(
                    min_cutoff_freq=self.min_cutoff_freq_x,
                    max_cutoff_freq=self.max_cutoff_freq_x,
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
        waveform = self.apply_augmentation_crop(waveform, sample_rate=self.sample_rate)
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
                                                 (1, self.sample_length - length_x - 1),
                                                 "constant", 0)

        length_y = waveform_y.size(dim=1)
        if length_y < self.sample_length:
            waveform_y = torch.nn.functional.pad(waveform_y,
                                                 (1, self.sample_length - length_y - 1),
                                                 "constant", 0)

        # 'batch' up waveforms x and y together
        waveform_x = torch.unsqueeze(waveform_x, dim=0)
        waveform_y = torch.unsqueeze(waveform_y, dim=0)
        waveform = torch.cat((waveform_x, waveform_y), 0)

        # still faster in cpu than accelerator.. so we disable it, furthermore, mps still do not work for many augmentations.
        # if self.device == 'mps':
        #     mpsdevice = torch.device('mps')
        #     waveform = waveform.to(mpsdevice)
        # elif self.device == 'gpu':
        #     gpudevice = torch.device('gpu')
        #     waveform = waveform.to(gpudevice)

        # decided to do time shift earlier then do block cropping, to prevent too many zero pads
        if self.do_augmentation:
            waveform = self.__process_augmentations_before_crop(waveform)

        if self.do_random_block:
            waveform = self.__random_block(waveform)

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

        # do padding if random block cuts the sample length
        length_x = waveform_x.size(dim=1)
        if length_x < self.block_size:
            waveform_x = torch.nn.functional.pad(waveform_x,
                                                 (1, self.block_size - length_x - 1),
                                                 "constant", 0)

        length_y = waveform_y.size(dim=1)
        if length_y < self.block_size:
            waveform_y = torch.nn.functional.pad(waveform_y,
                                                 (1, self.block_size - length_y - 1),
                                                 "constant", 0)

        return waveform_x, waveform_y
