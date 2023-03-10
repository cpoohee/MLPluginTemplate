# Adapted from https://github.com/archinetai/audio-encoders-pytorch

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from src.utils.losses import Losses, PreEmphasisFilter
from audio_encoders_pytorch import AutoEncoder1d
from archisound import ArchiSound



class AutoEncoder_PL(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(AutoEncoder_PL, self).__init__()
        self.save_hyperparameters()

        self.autoencoder = AutoEncoder1d(
            in_channels=1,              # Number of input channels
            channels=32,                # Number of base channels
            multipliers=[1, 1, 2, 2],   # Channel multiplier between layers (i.e. channels * multiplier[i] -> channels * multiplier[i+1])
            factors=[4, 4, 4],          # Downsampling/upsampling factor per layer
            num_blocks=[2, 2, 2]        # Number of resnet blocks per layer
        )
        # self.autoencoder = ArchiSound.from_pretrained("autoencoder1d-AT-v1")

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

    def forward(self, x):
        return self.autoencoder(x)

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
        x, y = batch
        y_pred = self.forward(x)
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