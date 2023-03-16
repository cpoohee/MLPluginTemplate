# adapted from https://github.com/f90/Wave-U-Net-Pytorch/tree/master/model
import torch
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from omegaconf import DictConfig
from src.utils.losses import Losses, PreEmphasisFilter

class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)

class WaveUNet(nn.Module):
    def __init__(self, n_layers=12, channels_interval=24):
        super(WaveUNet, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        tmp = []
        o = input

        # Up Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        o = self.middle(o)

        # Down Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o

class WaveUNet_PL(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(WaveUNet_PL, self).__init__()
        self.save_hyperparameters()

        self.waveunet = WaveUNet(n_layers=cfg.model.n_layers,
                                 channels_interval=cfg.model.channel_intervals)
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
        self.loss = Losses(loss_type=cfg.training.lossfn, sample_rate=cfg.dataset.sample_rate, cfg=cfg)

        if self.loss_preemphasis_hp_filter:
            self.fir_filter = PreEmphasisFilter(coeff=self.loss_preemphasis_hp_coeff)

        if self.loss_preemphasis_aw_filter:
            self.aw_filter = PreEmphasisFilter(type='aw')

        self.val_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.waveunet.parameters(), lr=self.lr)

    def forward(self, x):
        return self.waveunet(x)

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
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return {"loss": loss, "log": logs}

    def _shared_eval_step(self, batch):
        x, y, dvec, name = batch
        y_pred = self.forward(x)
        return y, y_pred

    def validation_step(self, batch, batch_idx):
        y, y_pred = self._shared_eval_step(batch)
        loss = self._lossfn(y, y_pred)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_step_outputs).mean()
        logs = {"val_loss": avg_loss}
        self.log("val_loss_epoch", avg_loss)
        self.val_step_outputs.clear()
        return {"avg_val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        y, y_pred = self._shared_eval_step(batch)
        loss = self._lossfn(y, y_pred)
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_outputs).mean()
        logs = {"test_loss": avg_loss}
        self.log("test_loss_epoch", avg_loss)
        self.test_step_outputs.clear()
        return {"avg_test_loss": avg_loss, "log": logs}

    def predict_step(self, batch, batch_idx):
        ## split into block sized batches
        x, y = batch

        resized_samples = (x.size()[2] // self.cfg.dataset.block_size) * self.cfg.dataset.block_size
        x = x[:, :, 0:resized_samples]
        y = y[:, :, 0:resized_samples]

        x = torch.reshape(x, (-1, 1, self.cfg.dataset.block_size))
        y = torch.reshape(y, (-1, 1, self.cfg.dataset.block_size))

        y, y_pred = self._shared_eval_step((x, y))

        ## reshape block sized batches into one single wav
        y_pred = torch.reshape(y_pred, (1, 1, -1))
        y = torch.reshape(y, (1, 1, -1))

        return y, y_pred