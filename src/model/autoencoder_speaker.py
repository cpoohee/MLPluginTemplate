import torch
import torch.nn as nn

import pytorch_lightning as pl
from omegaconf import DictConfig
from src.utils.losses import Losses, PreEmphasisFilter
from archisound import ArchiSound
from src.model.speaker_encoder.speaker_embedder import AudioHelper


class AutoEncoder_Speaker(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder_Speaker, self).__init__()

        # auto encoder
        self.autoencoder = ArchiSound.from_pretrained("autoencoder1d-AT-v1")

        self.bottleneck_dropout = nn.Dropout(p=cfg.model.bottleneck_dropout)

        # the autoencoder's encoder output size is [1, 32, 8192]??
        # this linear will learn from embedding sized input and will be added into the bottleneck
        fake_input = torch.randn(1, 2, cfg.dataset.block_size)
        with torch.no_grad():
            fake_z = self.autoencoder.encode(fake_input)

        out_feature_size = fake_z.size()[2] # hmm, the outfeature size is inputlen/32.

        # this linear shd be a lstm like layer, outputs /32 sized

        self.linear = nn.Linear(in_features=256,
                                out_features=out_feature_size)


    def forward(self, x, dvec):
        with torch.no_grad():
            # auto encoder encodes
            if x.size()[1] == 1:  # mono
                x = x.repeat(1, 2, 1)  # create stereo
            z = self.autoencoder.encode(x)


        # learn the rest
        dvec = self.linear(dvec)
        dvec = torch.unsqueeze(dvec,1)
        dvec = dvec.repeat(1, 32, 1)

        # do addition into z
        z = z + dvec

        # auto encoder encodes z with additive embedding
        y_pred = self.autoencoder.decode(z)

        return y_pred


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
        x, y, dvec, name = batch
        y_pred = self.forward(x, dvec)
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
