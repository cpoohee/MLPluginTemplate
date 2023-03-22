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


bottleneck = {'tanh': TanhBottleneck}


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
    """
    Hugging face AE model
    """

    config_class = AutoEncoder1dConfig

    def __init__(self, config: AutoEncoder1dConfig):
        super().__init__(config)

        self.autoencoder = AE1d(
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            channels=config.channels,
            multipliers=config.multipliers,
            factors=config.factors,
            num_blocks=config.num_blocks,
            bottleneck=bottleneck[config.bottleneck]()
        )

    def forward(self, *args, **kwargs):
        return self.autoencoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.autoencoder.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.autoencoder.decode(*args, **kwargs)


class AE1d(nn.Module):
    """
    audio_encoders_pytorch's ae model
    """

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

# Style adaptive layer norm adapted from
# https://github.com/KevinMIN95/StyleSpeech/blob/main/models/Modules.py


class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channel = in_channel
        self.norm = nn.LayerNorm(in_channel, elementwise_affine=False)

        self.style = AffineLinear(style_dim, in_channel * 2)
        self.style.affine.bias.data[:in_channel] = 1
        self.style.affine.bias.data[in_channel:] = 0

    def forward(self, input, style_code):
        # style
        style = self.style(style_code)
        gamma, beta = style.chunk(2, dim=-1)
        norm = self.norm(input)
        out = gamma * norm + beta
        return out

class AutoEncoder_Speaker2(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder_Speaker2, self).__init__()

        ae_config = AutoEncoder1dConfig()
        self.autoencoder = AutoEncoder1d(ae_config)
        self.autoencoder = self.autoencoder.from_pretrained(cfg.model.ae_path)

        if cfg.model.freeze_encoder:
            for p in self.autoencoder.autoencoder.encoder.parameters():
                p.requires_grad = False

        if cfg.model.freeze_decoder:
            for p in self.autoencoder.autoencoder.decoder.parameters():
                p.requires_grad = False

        self.autoencoder = torch.compile(self.autoencoder, mode="reduce-overhead")

        self.bottleneck_dropout = nn.Dropout(p=cfg.model.bottleneck_dropout)

        # the autoencoder's encoder output size is [1, 32 channels, input//32]
        # this lstm will learn from embedding sized input and will be fused into the bottleneck z
        self.ae_channel_size = ae_config.channels
        emb_size = cfg.model.emb_size
        self.latent_slice_size = cfg.model.latent_slice_size

        # 32 ae_channel_size
        self.aslns = nn.ModuleList([StyleAdaptiveLayerNorm(in_channel=self.latent_slice_size,
                                                          style_dim=emb_size)
                                          for _ in range(0, self.ae_channel_size)])

        self.activations = nn.Tanh()  # follows the same activation output from the encoder z

    def fuse_embedding(self, z, dvec):
        # z is [b, 32 channels, xsize/32 ], dvec is [b, 256]
        z_fuses = []

        for i, asln in enumerate(self.aslns):  # for each channel
            z_channel = z[:, i, :]  # z_channel is [b, xsize/32]

            # list of slices [ [b, latent_slice_size],...]
            z_partials = torch.split(z_channel, self.latent_slice_size, dim=1)

            z_partial_outs = []
            for z_partial in z_partials:
                z_partial_out = asln(z_partial, dvec)  # [b, latent_slice_size]
                z_partial_out = self.activations(z_partial_out)
                z_partial_outs.append(z_partial_out)

            #  [b, num_z_partials ,latent_slice_size]
            z_n_partials_nslices = torch.stack(z_partial_outs, dim=1)

            # tensor[b, xsize / 32]
            z_channel_emb_out = torch.flatten(z_n_partials_nslices, start_dim=1, end_dim=- 1)
            z_fuses.append(z_channel_emb_out)

        # z_fuses is list of [tensor [b, 1 , xsize/32]]
        z_fused = torch.stack(z_fuses, dim=1)  # [b, 32 channels, xsize/32 ]
        return z_fused

    def forward(self, x, dvec):
        with torch.no_grad():
            # auto encoder encodes
            # force sum to mono, then create stereo
            x = torch.sum(x, dim=1, keepdim=True)
            x = x.repeat(1, 2, 1)  # create stereo
            z = self.autoencoder.encode(x)

        z = self.bottleneck_dropout(z)  # [b, 32 channels, xsize/32 ]

        z_fused = self.fuse_embedding(z, dvec) + z  # skip connection

        # z_fused = z

        # auto encoder encodes z fused with embedding
        y_pred = self.autoencoder.decode(z_fused)

        # sum to mono
        y_pred = torch.sum(y_pred, dim=1, keepdim=True)

        return y_pred


class AutoEncoder_Speaker_PL2(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(AutoEncoder_Speaker_PL2, self).__init__()
        self.save_hyperparameters()

        self.autoencoder = AutoEncoder_Speaker2(cfg)

        self.lr = cfg.training.learning_rate
        self.cfg = cfg

        self.loss_preemphasis_hp_filter = cfg.training.loss_preemphasis_hp_filter
        self.loss_preemphasis_hp_coeff = cfg.training.loss_preemphasis_hp_coeff
        self.loss_preemphasis_aw_filter = cfg.training.loss_preemphasis_aw_filter

        """
        see types of losses
        https://github.com/csteinmetz1/auraloss
        """
        self.loss_type = cfg.training.lossfn
        self.loss = Losses(loss_type=self.loss_type,
                           sample_rate=cfg.dataset.sample_rate,
                           cfg=self.cfg)

        if self.loss_preemphasis_hp_filter:
            self.fir_filter = PreEmphasisFilter(coeff=self.loss_preemphasis_hp_coeff)

        if self.loss_preemphasis_aw_filter:
            self.aw_filter = PreEmphasisFilter(type='aw')

        self.val_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        all_params = self.autoencoder.parameters()
        return torch.optim.Adam(all_params, lr=self.lr)

    def forward(self, x, dvec):
        return self.autoencoder(x, dvec)

    def _lossfn(self, y_pred, y, dvec):
        if self.loss_preemphasis_hp_filter:
            y_pred, y = self.fir_filter(y_pred, y)

        if self.loss_preemphasis_aw_filter:
            y_pred, y = self.aw_filter(y_pred, y)

        if self.loss_type == 'EMBLoss':
            return self.loss.forward(y_pred, dvec)

        if self.loss_type == 'EMB_MR_Loss' or self.loss_type == 'EMB_MSE_Loss':
            return self.loss.forward(y_pred, y, dvec)

        return self.loss.forward(y_pred, y)

    def training_step(self, batch, batch_idx):
        # y, y_pred, dvecs, name = self._shared_eval_step(batch)
        x, y, dvecs, name = batch
        own_dvec, target_dvec = dvecs
        if self.loss_type == 'EMBLoss' or \
                self.loss_type == 'EMB_MR_Loss' or \
                self.loss_type == 'EMB_MSE_Loss':
            y_pred = self.forward(x, target_dvec)
            loss = self._lossfn(y_pred, y, target_dvec)
        else:
            y_pred = self.forward(x, own_dvec)  # train with own dvecs
            loss = self._lossfn(y_pred, y, dvec=None)

        logs = {"loss": loss}
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        # print(self.autoencoder.projections[0].weight)
        return {"loss": loss, "log": logs}

    def _shared_eval_step(self, batch):
        x, y, dvecs, name = batch
        own_dvec, target_dvec = dvecs
        y_pred = self.forward(x, target_dvec)
        return y, y_pred, target_dvec, name

    def validation_step(self, batch, batch_idx):
        y, y_pred, dvec, name = self._shared_eval_step(batch)
        loss = self._lossfn(y_pred, y, dvec)
        self.val_step_outputs.append(loss)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_step_outputs).mean()
        logs = {"val_loss": avg_loss}
        self.log("val_loss_epoch", avg_loss)
        self.val_step_outputs.clear()
        return {"avg_val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        y, y_pred, dvec, name = self._shared_eval_step(batch)
        loss = self._lossfn(y_pred, y, dvec)
        self.test_step_outputs.append(loss)
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_outputs).mean()
        logs = {"test_loss": avg_loss}
        self.log("test_loss_epoch", avg_loss)
        self.test_step_outputs.clear()
        return {"avg_test_loss": avg_loss, "log": logs}

    # def predict_step(self, batch, batch_idx):
    #     y, y_pred, dvec, name = self._shared_eval_step(batch)
    #     return y_pred
