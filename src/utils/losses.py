import torch
import auraloss
import torchaudio.transforms as T
from omegaconf import DictConfig, OmegaConf
from auraloss.utils import apply_reduction
from src.model.speaker_encoder.speaker_embedder import SpeechEmbedder, AudioHelper

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
            y_pred = self.pre_emphasis_filter(input, self.coeff)
            y = self.pre_emphasis_filter(target, self.coeff)
            return y_pred, y

        elif self.type == 'aw':
            y_pred, y  = self.aw(input, target)
            return y_pred, y

    def pre_emphasis_filter(self, x, coeff):
        return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)


class EMBLoss(torch.nn.Module):
    """
    MSE loss of the speaker embeddings. uses predicted waveform to generate new speaker embeddings,
    which is then compared against target speaker embedding.
    """
    def __init__(self, cfg):
        super(EMBLoss, self).__init__()
        dev = torch.device(cfg.training.accelerator)

        self.embedder = SpeechEmbedder()
        chkpt_embed = torch.load(cfg.model.embedder_path, map_location=cfg.training.accelerator)
        self.embedder.load_state_dict(chkpt_embed)
        for p in self.embedder.parameters():
            p.requires_grad = False

        self.embedder.to(dev)

        self.audio_helper = AudioHelper()


        # try to be as close as librosa's resampling
        self.resampler = T.Resample(orig_freq=cfg.dataset.block_size_speaker,
                                    new_freq=self.embedder.get_target_sample_rate(),
                                    lowpass_filter_width=64,
                                    rolloff=0.9475937167399596,
                                    resampling_method="sinc_interp_kaiser",
                                    beta=14.769656459379492,
                                    ).to(dev)

        self.loss = torch.nn.MSELoss().to(dev)

    def forward(self, pred, target_dvec):
        # TODO: Test if onnx embedding inference is faster than pytorch model in nb. then implement it

        pred_dvec = self.__get_embedding_vec(pred)
        return self.loss(pred_dvec, target_dvec)

    def __get_embedding_vec(self, waveform_speaker):
        # embedding d vec
        waveform_speaker = self.resampler(waveform_speaker)  # resample to 16kHz
        waveform_speaker = waveform_speaker.squeeze()  # [16000]

        org_dev = waveform_speaker.device
        cpudevice = torch.device('cpu')
        waveform_speaker = waveform_speaker.to(cpudevice)
        dvec_mel, _, _ = self.audio_helper.get_mel_torch(waveform_speaker)
        dvec_mel = dvec_mel.to(org_dev)

        dvecs = self.embedder.batched_forward(dvec_mel)

        return dvecs


class Losses:
    def __init__(self, loss_type='error_to_signal', sample_rate='44100', cfg: DictConfig=None):
        super(Losses, self).__init__()
        self.loss_type = loss_type
        self.sample_rate = sample_rate
        self.cfg = cfg

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
            if self.cfg == None:
                self.loss = auraloss.freq.STFTLoss(fft_size=4096,
                                                   win_length=4096,
                                                   hop_size=1024,
                                                   w_phs=0.2,
                                                   sample_rate=self.sample_rate,
                                                   device=torch.device("cpu"))
            else:
                self.loss = auraloss.freq.STFTLoss(fft_size=self.cfg.training.loss.fft_size,
                                                   win_length=self.cfg.training.loss.win_length,
                                                   hop_size=self.cfg.training.loss.hop_size,
                                                   w_phs=self.cfg.training.loss.w_phs,
                                                   sample_rate=self.sample_rate,
                                                   device=torch.device("cpu"))

        elif loss_type == 'MelSTFTLoss':
            if self.cfg == None:
                self.loss = auraloss.freq.MelSTFTLoss(fft_size=4096,
                                                      hop_size=1024,
                                                      n_mels=128,
                                                      w_phs=0.2,
                                                      sample_rate=self.sample_rate,
                                                      device=torch.device("cpu"))
            else:
                self.loss = auraloss.freq.MelSTFTLoss(fft_size=self.cfg.training.loss.fft_size,
                                                      hop_size=self.cfg.training.loss.hop_size,
                                                      n_mels=self.cfg.training.loss.n_mels,
                                                      w_phs=self.cfg.training.loss.w_phs,
                                                      sample_rate=self.sample_rate,
                                                      device=torch.device("cpu"))

        elif loss_type == 'MultiResolutionSTFTLoss':
            if self.cfg == None:
                self.loss = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[512, 1024, 2048, 4096],
                    hop_sizes=[50, 120, 240, 480],
                    win_lengths=[512, 1024, 2048, 4096],
                    w_phs=0.2,
                    device=torch.device("cpu"));
            else:
                self.loss = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=OmegaConf.to_object(self.cfg.training.loss.fft_sizes),
                    win_lengths=OmegaConf.to_object(self.cfg.training.loss.win_lengths),
                    hop_sizes=OmegaConf.to_object(self.cfg.training.loss.hop_sizes),
                    w_phs=self.cfg.training.loss.w_phs,
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

        elif loss_type == 'EMBLoss':
            self.loss = EMBLoss(self.cfg)

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
