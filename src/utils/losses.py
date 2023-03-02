import torch
import auraloss
from auraloss.utils import apply_reduction


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


class PreEmphasisFilter(torch.nn.Module):
    def __init__(self, coeff=0.95):
        super(PreEmphasisFilter, self).__init__()
        self.coeff = coeff

    def forward(self, input, target):
        y = self.pre_emphasis_filter(input, self.coeff)
        y_pred = self.pre_emphasis_filter(target, self.coeff)
        return y, y_pred

    def pre_emphasis_filter(self, x, coeff):
        return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


class Losses(torch.nn.Module):
    def __init__(self, loss_type='error_to_signal'):
        super(Losses, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'error_to_signal':
            self.loss = ESRLossORG()

        elif loss_type == 'ESRLoss':
            self.loss = auraloss.time.ESRLoss()

        elif loss_type == 'DCLoss':
            self.loss = auraloss.time.DCLoss()

        elif loss_type == 'LogCoshLoss':
            self.loss = auraloss.time.LogCoshLoss()

        elif loss_type == 'SNRLoss':
            self.loss = auraloss.time.SNRLoss()

        elif loss_type == 'SDSDRLoss':
            self.loss = auraloss.time.SDSDRLoss()

        elif loss_type == 'MSELoss':
            self.loss = torch.nn.MSELoss()

        elif loss_type == 'DC_SDSDR_SNR_Loss':
            self.lossDC = auraloss.time.DCLoss()
            self.lossSDSDR = auraloss.time.SDSDRLoss()
            self.lossSNR = auraloss.time.SNRLoss()

        elif loss_type == 'ESR_DC_Loss':
            self.lossDC = auraloss.time.DCLoss()
            self.lossESR = auraloss.time.ESRLoss()

        else:
            assert False

    def forward(self, input, target):
        if self.loss_type == 'DC_SDSDR_SNR_Loss':
            lossDC = self.lossDC(input, target)
            lossSDSDR = self.lossSDSDR(input, target)
            lossSNR = self.lossSNR(input, target)
            loss = lossDC * 10000.0 + lossSDSDR + lossSNR * 10.0 # loss weighting but chosen from experiments
            return loss

        if self.loss_type == 'ESR_DC_Loss':
            lossDC = self.lossDC(input, target)
            lossESR = self.lossESR(input, target)
            loss = lossDC * 1000.0 + lossESR # loss weighting but chosen from experiments
            return loss

        return self.loss(input, target)
