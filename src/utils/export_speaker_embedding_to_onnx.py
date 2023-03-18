import hydra
import os
import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import torchaudio.transforms as T
from pathlib import Path
from omegaconf import DictConfig
from src.model.speaker_encoder.speaker_embedder import SpeechEmbedder, AudioHelper


class ResampleGetMel(nn.Module):
    def __init__(self, cfg, sr):
        super(ResampleGetMel, self).__init__()
        self.audio_helper = AudioHelper()
        # try to be as close as librosa's resampling
        self.resampler = T.Resample(orig_freq=cfg.dataset.block_size_speaker,
                                    new_freq=sr,
                                    lowpass_filter_width=64,
                                    rolloff=0.9475937167399596,
                                    resampling_method="sinc_interp_kaiser",
                                    beta=14.769656459379492,
                                    )

    def forward(self, waveform_speaker):
        # embedding d vec
        waveform_speaker = self.resampler(waveform_speaker)  # resample to 16kHz
        waveform_speaker = waveform_speaker.squeeze()  # [16000]
        dvec_mel, _, _ = self.audio_helper.get_mel_torch(waveform_speaker)
        return dvec_mel


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    cur_path = Path(os.path.abspath(hydra.utils.get_original_cwd()))
    cfg.model.embedder_path = cur_path / Path(cfg.export_to_onnx.embedder_path)

    onnx_filename = cur_path / Path(cfg.export_to_onnx.export_filename)

    torch_embedder = SpeechEmbedder()
    chkpt_embed = torch.load(cfg.model.embedder_path, map_location=cfg.training.accelerator)
    torch_embedder.load_state_dict(chkpt_embed)
    torch_embedder.eval()

    resample_get_mel = ResampleGetMel(cfg, sr=torch_embedder.get_target_sample_rate())

    sample_block_size = cfg.export_to_onnx.sample_block_size
    dummy_wav = torch.randn(1, sample_block_size)
    mels = resample_get_mel(dummy_wav)
    dvec_out = torch_embedder(mels)
    emb_size = cfg.export_to_onnx.emb_size
    assert dvec_out.size()[0] == emb_size

    dummy_input = torch.randn(40, 101)
    input_names = ["mel_input1"]
    output_names = ["dvec"]

    dynamic_axes = {
        "input1": {1: 'blocksize'},  # dim 1
    }

    torch.onnx.export(
        torch_embedder,
        args=dummy_input,
        f=onnx_filename.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        # opset_version=17
    )

    onnx_embedder = onnx.load(onnx_filename.as_posix())
    onnx.checker.check_model(onnx_embedder)

    test_emb_model(torch_embedder, onnx_filename.as_posix())


def test_emb_model(torch_embedder, onnx_filename):
    ort_session = ort.InferenceSession(onnx_filename)
    dummy_input = torch.randn(40, 101)

    onnx_outputs = ort_session.run(
        None,
        {"mel_input1": dummy_input.numpy()},
    )

    torch_embedder.eval()
    torch_outputs = torch_embedder(dummy_input)

    np.testing.assert_allclose(torch_outputs.detach().numpy(), onnx_outputs[0], rtol=1e-05,
                               atol=1e-05)


if __name__ == "__main__":
    main()
