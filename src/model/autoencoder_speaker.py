import torch
import librosa
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from src.utils.losses import Losses, PreEmphasisFilter
from archisound import ArchiSound


class Dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class LinearNorm(nn.Module):
    def __init__(self, hp):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(hp.embedder.lstm_hidden, hp.embedder.emb_dim)

    def forward(self, x):
        return self.linear_layer(x)


class SpeechEmbedder(nn.Module):
    def __init__(self, hp):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(hp.embedder.num_mels,
                            hp.embedder.lstm_hidden,
                            num_layers=hp.embedder.lstm_layers,
                            batch_first=True)
        self.proj = LinearNorm(hp)
        self.hp = hp

    def forward(self, mel):
        # (num_mels, T)
        if len(mel.size()) == 4:
            mels = mel.unfold(3, self.hp.embedder.window,
                              self.hp.embedder.stride)
            ### TODO: FIX THIS, the command to unfold is not working in with batch dimension
        else:
            mels = mel.unfold(1, self.hp.embedder.window,
                              self.hp.embedder.stride)  # (num_mels, T', window)
        mels = mels.permute(1, 2, 0)  # (T', window, num_mels)
        x, _ = self.lstm(mels)  # (T', window, lstm_hidden)
        x = x[:, -1, :]  # (T', lstm_hidden), use last frame only
        x = self.proj(x)  # (T', emb_dim)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)  # (T', emb_dim)
        x = x.sum(0) / x.size(0)  # (emb_dim), average pooling over time frames
        return x


# adapted from Keith Ito's tacotron implementation
# https://github.com/keithito/tacotron/blob/master/util/audio.py


class AudioHelper():
    def __init__(self, hp):
        self.hp = hp
        self.mel_basis = librosa.filters.mel(sr=hp.audio.sample_rate,
                                             n_fft=hp.embedder.n_fft,
                                             n_mels=hp.embedder.num_mels)

        # self.torch_mel_filters = F.melscale_fbanks(
        #     int(self.hp.embedder.n_fft // 2 + 1),
        #     n_mels=hp.embedder.num_mels,
        #     f_min=0.0,
        #     f_max=hp.audio.sample_rate / 2.0,
        #     sample_rate=hp.audio.sample_rate,
        #     norm="slaney",
        # )
        #
        # self.window = torch.hann_window(self.hp.audio.win_length)
        #
        # self.torch_mel_spectrogram = T.MelSpectrogram(
        #     sample_rate=hp.audio.sample_rate,
        #     n_fft=self.hp.embedder.n_fft,
        #     win_length=self.hp.audio.win_length,
        #     hop_length=self.hp.audio.hop_length,
        #     center=True,
        #     pad_mode="reflect",
        #     power=2.0,
        #     norm="slaney",
        #     onesided=True,
        #     n_mels=hp.embedder.num_mels,
        #     mel_scale="htk",
        # )

    def get_mel(self, y):
        if isinstance(y, torch.Tensor):

            y_np = y.detach().cpu().numpy()
            batch_size = np.shape(y_np)[0]

            mels = []

            # not very efficient!
            for b in range(0, batch_size):
                y_b = y_np[b]
                y = librosa.core.stft(y=y_b, n_fft=self.hp.embedder.n_fft,
                                      hop_length=self.hp.audio.hop_length,
                                      win_length=self.hp.audio.win_length,
                                      window='hann')
                magnitudes = np.abs(y) ** 2
                mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
                mels.append(mel)
            mels = np.asarray(mels)

            mels = torch.Tensor(mels)

            return mels


        else:
            y = librosa.core.stft(y=y, n_fft=self.hp.embedder.n_fft,
                                  hop_length=self.hp.audio.hop_length,
                                  win_length=self.hp.audio.win_length,
                                  window='hann')
            magnitudes = np.abs(y) ** 2
            mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
            return mel

    def wav2spec(self, y):
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - self.hp.audio.ref_level_db
        S, D = self.normalize(S), np.angle(D)
        S, D = S.T, D.T  # to make [time, freq]
        return S, D

    def spec2wav(self, spectrogram, phase):
        spectrogram, phase = spectrogram.T, phase.T
        # used during inference only
        # spectrogram: enhanced output
        # phase: use noisy input's phase, so no GLA is required
        S = self.db_to_amp(self.denormalize(spectrogram) + self.hp.audio.ref_level_db)
        return self.istft(S, phase)

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.hp.audio.n_fft,
                            hop_length=self.hp.audio.hop_length,
                            win_length=self.hp.audio.win_length)

    def istft(self, mag, phase):
        stft_matrix = mag * np.exp(1j * phase)
        return librosa.istft(stft_matrix,
                             hop_length=self.hp.audio.hop_length,
                             win_length=self.hp.audio.win_length)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def normalize(self, S):
        return np.clip(S / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db


class AutoEncoder_Speaker(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder_Speaker, self).__init__()
        # Speaker embedder
        self.__init_speaker_params()  # self.hp is initialised here
        self.audio_helper = AudioHelper(self.hp)
        self.embedder = SpeechEmbedder(self.hp)
        chkpt_embed = torch.load(cfg.model.embedder_path, map_location=cfg.training.accelerator)
        self.embedder.load_state_dict(chkpt_embed)

        self.resampler = T.Resample(orig_freq=cfg.dataset.block_size_speaker,
                                    new_freq=self.hp.audio.sample_rate)

        # auto encoder
        self.autoencoder = ArchiSound.from_pretrained("autoencoder1d-AT-v1")

        self.bottleneck_dropout = nn.Dropout(p=cfg.model.bottleneck_dropout)

        # the autoencoder's encoder output size is [1, 32, 8192]
        # this linear will learn from embedding sized input and will be added into the bottleneck
        self.linear = nn.Linear(in_features=self.hp.embedder.emb_dim,
                                out_features=8192)

    def __init_speaker_params(self):
        self.hp = Dotdict()
        embedder = Dotdict()

        embedder.num_mels = 40
        embedder.n_fft = 512
        embedder.emb_dim = 256
        embedder.lstm_hidden = 768
        embedder.lstm_layers = 3
        embedder.window = 80
        embedder.stride = 40
        self.hp.embedder = embedder

        audio = Dotdict()
        audio.n_fft = 1200
        audio.num_freq = 601  # n_fft//2 + 1
        audio.sample_rate = 16000
        audio.hop_length = 160
        audio.win_length = 400
        audio.min_level_db = -100.0
        audio.ref_level_db = 20.0
        audio.preemphasis = 0.97
        audio.power = 0.30
        self.hp.audio = audio

    def forward(self, x, speaker):
        with torch.no_grad():
            # auto encoder encodes
            if x.size()[1] == 1:  # mono
                x = x.repeat(1, 2, 1)  # create stereo
            z = self.autoencoder.encode(x)

            # embedding d vec
            speaker = self.resampler(speaker)  # resample to 16kHz
            dvec = self.get_speaker_embedding(speaker)

        # learn the rest
        dvec = self.linear(dvec)
        dvec = dvec.repeat(1, 32, 1)

        # do addition into z
        z = z + dvec

        # auto encoder encodes z with additive embedding
        y_pred = self.autoencoder.decode(z)

        return y_pred

    def get_speaker_embedding(self, dvec_wav):
        dvec_mel = self.audio_helper.get_mel(dvec_wav)
        with torch.no_grad():
            dvec = self.embedder(dvec_mel)
            return dvec


class AutoEncoder_Speaker_PL(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(AutoEncoder_Speaker_PL, self).__init__()
        self.save_hyperparameters()

        self.autoencoder = AutoEncoder_Speaker(cfg)

        self.lr = cfg.training.learning_rate
        self.lossfn = cfg.training.lossfn
        self.cfg = cfg

        self.loss_preemphasis_hp_filter = cfg.training.loss_preemphasis_hp_filter
        self.loss_preemphasis_hp_coeff = cfg.training.loss_preemphasis_hp_coeff
        self.loss_preemphasis_aw_filter = cfg.training.loss_preemphasis_aw_filter

        """
        see types of losses
        https://github.com/csteinmetz1/auraloss
        """
        self.loss = Losses(loss_type=cfg.training.lossfn, sample_rate=cfg.dataset.sample_rate)

        if self.loss_preemphasis_hp_filter:
            self.fir_filter = PreEmphasisFilter(coeff=self.loss_preemphasis_hp_coeff)

        if self.loss_preemphasis_aw_filter:
            self.aw_filter = PreEmphasisFilter(type='aw')

    def configure_optimizers(self):
        return torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)

    def forward(self, x, speaker):
        return self.autoencoder(x, speaker)

    def _lossfn(self, y, y_pred):
        if self.loss_preemphasis_hp_filter:
            y, y_pred = self.fir_filter(y, y_pred)

        if self.loss_preemphasis_aw_filter:
            y, y_pred = self.aw_filter(y, y_pred)

        return self.loss(y, y_pred)

    def training_step(self, batch, batch_idx):
        y, y_pred = self._shared_eval_step(batch)
        loss = self._lossfn(y, y_pred)
        logs = {"loss": loss}
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return {"loss": loss, "log": logs}

    def _shared_eval_step(self, batch):
        x, y, speaker, name = batch
        y_pred = self.forward(x, speaker)
        return y, y_pred

    def validation_step(self, batch, batch_idx):
        y, y_pred = self._shared_eval_step(batch)
        loss = self._lossfn(y, y_pred)
        return {"val_loss": loss}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        self.log("val_loss_epoch", avg_loss)
        return {"avg_val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        y, y_pred = self._shared_eval_step(batch)
        loss = self._lossfn(y, y_pred)
        return {"test_loss": loss}

    def test_epoch_end(self, outs):
        avg_loss = torch.stack([x["test_loss"] for x in outs]).mean()
        logs = {"test_loss": avg_loss}
        self.log("test_loss_epoch", avg_loss)
        return {"avg_test_loss": avg_loss, "log": logs}

    def predict_step(self, batch, batch_idx):
        y, y_pred = self._shared_eval_step(batch)
        return y, y_pred
