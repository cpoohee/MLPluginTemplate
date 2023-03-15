import torch
import torchaudio
import librosa
import numpy as np
import pytest
import os
import torchaudio.functional as F
import torchaudio.transforms as T
from src.model.speaker_encoder.speaker_embedder import AudioHelper


def test_librosa_torch_resample():
    # create a random audio file
    org_sample_rate = 44100
    rand_audio = torch.randn(1, org_sample_rate)
    temp_path = './temp_audio.wav'
    torchaudio.save(temp_path, rand_audio, org_sample_rate)

    # check resample is close enough
    target_resample_rate = 16000
    librosa_wav, _ = librosa.load(temp_path,
                                  sr=target_resample_rate)
    librosa_wav = torch.from_numpy(librosa_wav)
    librosa_wav = librosa_wav.unsqueeze(0)

    torch_wav, torch_org_sr = torchaudio.load(temp_path)

    # delete after loaded to mem
    os.remove(temp_path)

    # kaiser best algo
    # see https://pytorch.org/audio/main/tutorials/audio_resampling_tutorial.html#kaiser-best
    torch_resampled_wav = F.resample(
                            torch_wav,
                            torch_org_sr,
                            target_resample_rate,
                            lowpass_filter_width=64,
                            rolloff=0.9475937167399596,
                            resampling_method="sinc_interp_kaiser",
                            beta=14.769656459379492,
                        )

    mse = torch.square(librosa_wav - torch_resampled_wav).mean().item()

    assert mse < 1e-2


def test_librosa_torch_mel_basis():
    audio_helper = AudioHelper()

    mel_np = audio_helper.mel_basis
    mel_np = torch.from_numpy(mel_np)
    mel_torch = audio_helper.mel_basis_np
    mse = torch.square(mel_np - mel_torch).mean().item()
    assert mse < 1e-6

def test_librosa_torch_stft():
    audio_helper = AudioHelper()
    target_resample_rate = 16000
    rand_wav_torch = torch.randn(target_resample_rate)
    rand_wav_np = rand_wav_torch.numpy()

    stft_np = audio_helper.stft(rand_wav_np)
    stft_np = torch.from_numpy(stft_np)

    rand_wav_torch = rand_wav_torch.unsqueeze(0)
    stft_torch = audio_helper.stft_torch(rand_wav_torch)
    stft_torch = stft_torch.squeeze()

    mse = torch.square(stft_np.real - stft_torch.real).mean().item()
    assert mse < 1e-6
    mse = torch.square(stft_np.imag - stft_torch.imag).mean().item()
    assert mse < 1e-6


def test_librosa_torch_melspec():
    audio_helper = AudioHelper()
    target_resample_rate = 16000
    rand_wav_torch = torch.randn(target_resample_rate)
    rand_wav_np = rand_wav_torch.numpy()

    mel_np, mag_np, dot_np = audio_helper.get_mel(rand_wav_np)
    mel_np = torch.from_numpy(mel_np)
    mag_np = torch.from_numpy(mag_np)
    dot_np = torch.from_numpy(dot_np)

    mel_torch, mag_torch, dot_torch = audio_helper.get_mel_torch(rand_wav_torch)

    mse = torch.square(mag_np - mag_torch).mean().item()
    assert mse < 1e-6
    mse = torch.square(dot_np - dot_torch).mean().item()
    assert mse < 1e-6
    mse = torch.square(mel_np - mel_torch).mean().item()
    assert mse < 1e-6