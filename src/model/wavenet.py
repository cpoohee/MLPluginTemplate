# Adapted from https://github.com/GuitarML/PedalNetRT/blob/master/model.py

import torch
import auraloss
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from auraloss.utils import apply_reduction


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


def _conv_stack(dilations, in_channels, out_channels, kernel_size):
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    return nn.ModuleList(
        [
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=d,
                kernel_size=kernel_size,
            )
            for i, d in enumerate(dilations)
        ]
    )


class WaveNet(nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2):
        super(WaveNet, self).__init__()
        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        internal_channels = int(num_channels * 2)
        self.hidden = _conv_stack(dilations, num_channels, internal_channels, kernel_size)
        self.residuals = _conv_stack(dilations, num_channels, num_channels, 1)
        self.input_layer = CausalConv1d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=1,
        )

        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_repeat,
            out_channels=1,
            kernel_size=1,
        )
        self.num_channels = num_channels

    def forward(self, x):
        out = x
        skips = []
        out = self.input_layer(out)

        for hidden, residual in zip(self.hidden, self.residuals):
            x = out
            out_hidden = hidden(x)

            # gated activation
            #   split (32,16,3) into two (16,16,3) for tanh and sigm calculations
            out_hidden_split = torch.split(out_hidden, self.num_channels, dim=1)
            out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])

            skips.append(out)

            out = residual(out)
            out = out + x[:, :, -out.size(2):]

        # modified "postprocess" step:
        out = torch.cat([s[:, :, -out.size(2):] for s in skips], dim=1)
        out = self.linear_mix(out)
        return out


# def error_to_signal(y, y_pred):
#     """
#     Error to signal ratio with pre-emphasis filter:
#     https://www.mdpi.com/2076-3417/10/3/766/htm
#     """
#     y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
#     return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)
#
#
# def pre_emphasis_filter(x, coeff=0.95):
#     return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


class ESRLossORG(torch.nn.Module):
    """
    Re-writing the original loss function as auroloss based module
    """

    def __init__(self, coeff=0.95, reduction='mean'):
        super(ESRLossORG, self).__init__()
        self.coeff = coeff
        self.reduction = reduction

    def forward(self, input, target):
        """
        the original loss has 1e-10 in the divisor for preventing div by zero
        """
        y = self.pre_emphasis_filter(input, self.coeff)
        y_pred = self.pre_emphasis_filter(target, self.coeff)
        losses = (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + 1e-10)
        losses = apply_reduction(losses, reduction=self.reduction)
        return losses

    def pre_emphasis_filter(self, x, coeff):
        return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


class WaveNet_PL(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(WaveNet_PL, self).__init__()
        self.wavenet = WaveNet(
            num_channels=cfg.model.num_channels,
            dilation_depth=cfg.model.dilation_depth,
            num_repeat=cfg.model.num_repeat,
            kernel_size=cfg.model.kernel_size,
        )
        self.lr = cfg.training.learning_rate
        self.lossfn = cfg.training.lossfn
        self.cfg = cfg

        self.loss_preemphasis_filter = cfg.training.loss_preemphasis_filter
        self.loss_preemphasis_coeff = cfg.training.loss_preemphasis_coeff

        """
        see types of losses
        https://github.com/csteinmetz1/auraloss
        """

        if cfg.training.lossfn == 'error_to_signal':
            self.ESRLoss = ESRLossORG()

        if cfg.training.lossfn == 'ESRLoss':
            self.ESRLoss = auraloss.time.ESRLoss()

        if cfg.training.lossfn == 'DCLoss':
            self.DCLoss = auraloss.time.DCLoss()

        if cfg.training.lossfn == 'LogCoshLoss':
            self.LogCoshLoss = auraloss.time.LogCoshLoss()

        if cfg.training.lossfn == 'SNRLoss':
            self.SNRLoss = auraloss.time.SNRLoss()

        if cfg.training.lossfn == 'SDSDRLoss':
            self.SDSDRLoss = auraloss.time.SDSDRLoss()

        if self.loss_preemphasis_filter:
            self.fir_filter = auraloss.perceptual.FIRFilter(coef=self.loss_preemphasis_coeff,
                                                            filter_type='hp')

        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.wavenet.parameters(), lr=self.lr)

    def forward(self, x):
        return self.wavenet(x)

    def _lossfn(self, y, y_pred):
        if self.loss_preemphasis_filter:
            y, y_pred = self.fir_filter(y, y_pred)

        if self.lossfn == 'error_to_signal' or self.lossfn == 'ESRLoss':
            return self.ESRLoss(y, y_pred)

        if self.lossfn == 'DCLoss':
            return self.DCLoss(y, y_pred)

        if self.lossfn == 'LogCoshLoss':
            return self.LogCoshLoss(y, y_pred)

        if self.lossfn == 'SNRLoss':
            return self.SNRLoss(y, y_pred)

        if self.lossfn == 'SDSDRLoss':
            return self.SDSDRLoss(y, y_pred)

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
