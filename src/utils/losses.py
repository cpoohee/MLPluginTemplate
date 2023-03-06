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
    def __init__(self, coeff=0.95, type= 'hp', fs=44100):
        super(PreEmphasisFilter, self).__init__()
        self.coeff = coeff
        self.type = type

        if self.type == 'aw':
            self.aw = auraloss.perceptual.FIRFilter(filter_type="aw", fs=fs)

    def forward(self, input, target):
        if self.type == 'hp':
            y = self.pre_emphasis_filter(input, self.coeff)
            y_pred = self.pre_emphasis_filter(target, self.coeff)
            return y, y_pred

        elif self.type == 'aw':
            y, y_pred = self.aw(input, target)
            return y, y_pred

    def pre_emphasis_filter(self, x, coeff):
        return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


class Losses(torch.nn.Module):
    def __init__(self, loss_type='error_to_signal', sample_rate='44100'):
        super(Losses, self).__init__()
        self.loss_type = loss_type
        self.sample_rate = sample_rate

        # time domain
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

        # frequency domain
        elif loss_type == 'STFTLoss':
            self.loss = auraloss.freq.STFTLoss(fft_size=4096,
                                               win_length=4096,
                                               hop_size=1024,
                                               w_phs=0.2,
                                               sample_rate=self.sample_rate,
                                               device=torch.device("cpu"));

        elif loss_type == 'MelSTFTLoss':
            self.loss = auraloss.freq.MelSTFTLoss(fft_size=4096,
                                                  hop_size=1024,
                                                  n_mels=128,
                                                  w_phs=0.2,
                                                  sample_rate=self.sample_rate,
                                                  device=torch.device("cpu"));

        elif loss_type == 'MultiResolutionSTFTLoss':
            self.loss = auraloss.freq.MultiResolutionSTFTLoss(
                fft_sizes=[512, 1024, 2048, 4096],
                hop_sizes=[50, 120, 240, 480],
                win_lengths=[512, 1024, 2048, 4096],
                w_phs=0.2,
                device=torch.device("cpu"));

        elif loss_type == 'RandomResolutionSTFTLoss':
            self.loss = auraloss.freq.RandomResolutionSTFTLoss(device=torch.device("cpu"));

        # combination losses
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
            loss = lossDC * 10000.0 + lossSDSDR + lossSNR # loss weighting but chosen from experiments
            return loss

        if self.loss_type == 'ESR_DC_Loss':
            lossDC = self.lossDC(input, target)
            lossESR = self.lossESR(input, target)
            loss = lossDC * 1000.0 + lossESR # loss weighting but chosen from experiments
            return loss

        if self.loss_type == 'STFTLoss' or \
                self.loss_type == 'MelSTFTLoss' or \
                self.loss_type == 'MultiResolutionSTFTLoss' or \
                self.loss_type == 'RandomResolutionSTFTLoss':
            # mps is not able to process complex types in stft, fall back to cpu
            cpudevice = torch.device('cpu')
            input_cpu = input.to(cpudevice)
            target_cpu = target.to(cpudevice)
            return self.loss(input_cpu, target_cpu)

        return self.loss(input, target)
