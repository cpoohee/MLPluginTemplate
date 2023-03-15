import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch import Tensor
from omegaconf import DictConfig
from src.utils.losses import Losses, PreEmphasisFilter
from audio_encoders_pytorch import TanhBottleneck
from audio_encoders_pytorch.modules import Encoder1d, Decoder1d, Bottleneck
from audio_encoders_pytorch.utils import default, prefix_dict
from typing import Any, List, Optional, Sequence, Tuple, Union
from transformers import PreTrainedModel
from transformers import PretrainedConfig

from archisound import ArchiSound

bottleneck = { 'tanh': TanhBottleneck }


class AutoEncoder1dConfig(PretrainedConfig):

    model_type = "archinetai/autoencoder1d-AT-v1"

    def __init__(
        self,
        in_channels: int = 2,
        patch_size: int = 4,
        channels: int = 32,
        multipliers: Sequence[int] = [1, 2, 4, 8, 8, 8, 1],
        factors: Sequence[int] = [2, 2, 2, 1, 1, 1],
        num_blocks: Sequence[int] = [2, 2, 8, 8, 8, 8],
        bottleneck: str = 'tanh',
        **kwargs
    ):
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.channels = channels
        self.multipliers = multipliers
        self.factors = factors
        self.num_blocks = num_blocks
        self.bottleneck = bottleneck
        super().__init__(**kwargs)

class AutoEncoder1d(PreTrainedModel):

    config_class = AutoEncoder1dConfig

    def __init__(self, config: AutoEncoder1dConfig):
        super().__init__(config)

        self.autoencoder = AE1d(
            in_channels = config.in_channels,
            patch_size = config.patch_size,
            channels = config.channels,
            multipliers = config.multipliers,
            factors = config.factors,
            num_blocks = config.num_blocks,
            bottleneck = bottleneck[config.bottleneck]()
        )

    def forward(self, *args, **kwargs):
        return self.autoencoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.autoencoder.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.autoencoder.decode(*args, **kwargs)


class AE1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        multipliers: Sequence[int],
        factors: Sequence[int],
        num_blocks: Sequence[int],
        patch_size: int = 1,
        resnet_groups: int = 8,
        out_channels: Optional[int] = None,
        bottleneck: Union[Bottleneck, List[Bottleneck]] = [],
        bottleneck_channels: Optional[int] = None,
    ):
        super().__init__()
        out_channels = default(out_channels, in_channels)

        self.encoder = Encoder1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            channels=channels,
            multipliers=multipliers,
            factors=factors,
            num_blocks=num_blocks,
            patch_size=patch_size,
            resnet_groups=resnet_groups,
            bottleneck=bottleneck,
        )

        self.decoder = Decoder1d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            channels=channels,
            multipliers=multipliers[::-1],
            factors=factors[::-1],
            num_blocks=num_blocks[::-1],
            patch_size=patch_size,
            resnet_groups=resnet_groups,
        )

    def forward(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        z, info_encoder = self.encode(x, with_info=True)
        y, info_decoder = self.decode(z, with_info=True)
        info = {
            **dict(latent=z),
            **prefix_dict("encoder_", info_encoder),
            **prefix_dict("decoder_", info_decoder),
        }
        return (y, info) if with_info else y

    def encode(
        self, x: Tensor, with_info: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Any]]:
        return self.encoder(x, with_info=with_info)

    def decode(self, x: Tensor, with_info: bool = False) -> Tensor:
        return self.decoder(x, with_info=with_info)


class AutoEncoder_Speaker(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder_Speaker, self).__init__()

        ae_config = AutoEncoder1dConfig()
        self.autoencoder = AutoEncoder1d(ae_config)
        self.autoencoder = self.autoencoder.from_pretrained(cfg.model.ae_path)

        # auto encoder
        # self.autoencoder = ArchiSound.from_pretrained("autoencoder1d-AT-v1")

        self.bottleneck_dropout = nn.Dropout(p=cfg.model.bottleneck_dropout)

        # the autoencoder's encoder output size is [1, 32, 8192]??
        # this linear will learn from embedding sized input and will be added into the bottleneck
        fake_input = torch.randn(1, 2, cfg.dataset.block_size)
        with torch.no_grad():
            fake_z = self.autoencoder.encode(fake_input)

        out_feature_size = fake_z.size()[2]  # hmm, the outfeature size is inputlen/32.

        # this linear shd be a lstm like layer, outputs /32 sized

        # self.linear = nn.Linear(in_features=256,
        #                         out_features=out_feature_size)
        #
        # self.lstm = nn.LSTM(input_size=256,
        #                     hidden_size=256,
        #                     num_layers=1,
        #                     bidirectional=True,
        #                     batch_first=True)


    def forward(self, x, dvec):
        with torch.no_grad():
            # auto encoder encodes
            if x.size()[1] == 1:  # mono
                x = x.repeat(1, 2, 1)  # create stereo
            z = self.autoencoder.encode(x)

        z = self.bottleneck_dropout(z)

        # # learn the rest
        # dvec = self.linear(dvec)
        # dvec = torch.unsqueeze(dvec,1)
        # dvec = dvec.repeat(1, 32, 1)
        #
        # # do addition into z
        # z = z + dvec

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
