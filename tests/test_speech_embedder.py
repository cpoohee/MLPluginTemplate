import torch
import torchaudio.transforms as T
from src.model.speaker_encoder.speaker_embedder import AudioHelper
from src.model.speaker_encoder.speaker_embedder import SpeechEmbedder


def test_speech_embedder_batched_inference():
    embedder = SpeechEmbedder()
    for p in embedder.parameters():
        p.requires_grad = False

    embedder.eval()
    org_sr = 44100

    audio_helper = AudioHelper()

    # try to be as close as librosa's resampling
    resampler = T.Resample(orig_freq=44100,
                                new_freq=embedder.get_target_sample_rate(),
                                lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492,
                                )

    batch_size = 8
    wavs = torch.randn(batch_size, org_sr)

    wavs = resampler(wavs)

    org_dev = wavs.device
    cpudevice = torch.device('cpu')
    wavs = wavs.to(cpudevice)
    dvec_mel, _, _ = audio_helper.get_mel_torch(wavs)
    dvec_mel = dvec_mel.to(org_dev)

    dvecs_single = []

    for i in range(0, dvec_mel.size()[0]):
        dvec = embedder(dvec_mel[i])  # embedder functions in one batch
        dvecs_single.append(dvec)

    dvecs_single = torch.stack(dvecs_single, dim=0)

    dvecs_batched = embedder.batched_forward(dvec_mel)

    mse = torch.square(dvecs_single - dvecs_batched).mean().item()

    assert mse < 1e-5


def test_speech_embedder_batched_inference_longaudio():
    embedder = SpeechEmbedder()
    for p in embedder.parameters():
        p.requires_grad = False

    embedder.eval()
    org_sr = 44100

    audio_helper = AudioHelper()

    # try to be as close as librosa's resampling
    resampler = T.Resample(orig_freq=44100,
                                new_freq=embedder.get_target_sample_rate(),
                                lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492,
                                )

    batch_size = 8
    wavs = torch.randn(batch_size, org_sr*4)

    wavs = resampler(wavs)

    org_dev = wavs.device
    cpudevice = torch.device('cpu')
    wavs = wavs.to(cpudevice)
    dvec_mel, _, _ = audio_helper.get_mel_torch(wavs)
    dvec_mel = dvec_mel.to(org_dev)

    dvecs_single = []

    for i in range(0, dvec_mel.size()[0]):
        dvec = embedder(dvec_mel[i])  # embedder functions in one batch
        dvecs_single.append(dvec)

    dvecs_single = torch.stack(dvecs_single, dim=0)

    dvecs_batched = embedder.batched_forward(dvec_mel)

    mse = torch.square(dvecs_single - dvecs_batched).mean().item()

    assert mse < 1e-5