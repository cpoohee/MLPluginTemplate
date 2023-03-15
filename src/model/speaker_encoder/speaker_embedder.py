import torch
import librosa
import torch.nn as nn
import numpy as np


class LinearNorm(nn.Module):
    def __init__(self, lstm_hidden, emb_dim):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(lstm_hidden, emb_dim)

    def forward(self, x):
        return self.linear_layer(x)


class SpeechEmbedder(nn.Module):
    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        self.init_hp()
        self.lstm = nn.LSTM(self.num_mels,
                            self.lstm_hidden,
                            num_layers=self.lstm_layers,
                            batch_first=True)
        self.proj = LinearNorm(self.lstm_hidden, self.emb_dim)

    def get_target_sample_rate(self):
        return self.sample_rate

    def init_hp(self):
        self.num_mels = 40
        self.n_fft = 512
        self.emb_dim = 256
        self.lstm_hidden = 768
        self.lstm_layers = 3
        self.window = 80
        self.stride = 40

        self.n_fft = 1200
        self.num_freq = 601  # n_fft//2 + 1
        self.sample_rate = 16000
        self.hop_length = 160
        self.win_length = 400
        self.min_level_db = -100.0
        self.ref_level_db = 20.0
        self.preemphasis = 0.97
        self.power = 0.30

    def forward(self, mel):
        # (num_mels, T)
        mels = mel.unfold(1, self.window,
                              self.stride)  # (num_mels, T', window)
        mels = mels.permute(1, 2, 0)  # (T', window, num_mels)
        x, _ = self.lstm(mels)  # (T', window, lstm_hidden)
        x = x[:, -1, :]  # (T', lstm_hidden), use last frame only
        x = self.proj(x)  # (T', emb_dim)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)  # (T', emb_dim)
        x = x.sum(0) / x.size(0)  # (emb_dim), average pooling over time frames
        return x


# adapted from Keith Ito's tacotron implementation
# https://github.com/keithito/tacotron/blob/master/util/audio.py


class AudioHelper:
    def __init__(self):
        self.init_hp()
        self.mel_basis = librosa.filters.mel(sr=self.sample_rate,
                                             n_fft=self.n_fft,
                                             n_mels=self.num_mels)

    def init_hp(self):
        self.num_mels = 40
        self.n_fft = 512
        self.emb_dim = 256
        self.lstm_hidden = 768
        self.lstm_layers = 3
        self.window = 80
        self.stride = 40

        self.n_fft = 1200
        self.num_freq = 601  # n_fft//2 + 1
        self.sample_rate = 16000
        self.hop_length = 160
        self.win_length = 400
        self.min_level_db = -100.0
        self.ref_level_db = 20.0
        self.preemphasis = 0.97
        self.power = 0.30

    def get_mel(self, y):
        y = librosa.core.stft(y=y, n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              win_length=self.win_length,
                              window='hann')
        magnitudes = np.abs(y) ** 2
        mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
        return mel

    def wav2spec(self, y):
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - self.ref_level_db
        S, D = self.normalize(S), np.angle(D)
        S, D = S.T, D.T  # to make [time, freq]
        return S, D

    def spec2wav(self, spectrogram, phase):
        spectrogram, phase = spectrogram.T, phase.T
        # used during inference only
        # spectrogram: enhanced output
        # phase: use noisy input's phase, so no GLA is required
        S = self.db_to_amp(self.denormalize(spectrogram) + self.ref_level_db)
        return self.istft(S, phase)

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length)

    def istft(self, mag, phase):
        stft_matrix = mag * np.exp(1j * phase)
        return librosa.istft(stft_matrix,
                             hop_length=self.hop_length,
                             win_length=self.win_length)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def normalize(self, S):
        return np.clip(S / -self.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -self.min_level_db
