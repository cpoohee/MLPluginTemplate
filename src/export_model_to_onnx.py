import hydra
import os
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from src.model.wavenet import WaveNet_PL
from src.model.waveUnet import WaveUNet_PL
from src.model.autoencoder import AutoEncoder_PL
from src.model.autoencoder_speaker import AutoEncoder_Speaker_PL


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cur_path = Path(os.path.abspath(hydra.utils.get_original_cwd()))
    ckpt_path = cfg.export_to_onnx.checkpoint_file

    if cfg.export_to_onnx.model == 'wavenet':
        model_pl = WaveNet_PL.load_from_checkpoint(cur_path/Path(ckpt_path))
        torch_model = model_pl.wavenet

    elif cfg.export_to_onnx.model == 'waveunet':
        model_pl = WaveUNet_PL.load_from_checkpoint(cur_path / Path(ckpt_path))
        torch_model = model_pl.waveunet

    elif cfg.testing.model_name == 'AutoEncoder_PL':
        model_pl = AutoEncoder_PL.load_from_checkpoint(cur_path / Path(ckpt_path))
        torch_model = model_pl.autoencoder

    elif cfg.testing.model_name == 'AutoEncoder_Speaker_PL':
        model_pl = AutoEncoder_Speaker_PL.load_from_checkpoint(cur_path / Path(ckpt_path))
        torch_model = model_pl.autoencoder
    else:
        assert False

    sample_block_size = cfg.export_to_onnx.sample_block_size
    onnx_filename = cfg.export_to_onnx.export_filename
    torch_model.eval()

    if cfg.testing.model_name == 'AutoEncoder_Speaker_PL':
        dummy_wav = torch.randn(1, 1, sample_block_size)
        dummy_dvec = torch.randn(1, cfg.model.emb_size)
        test_out = torch_model(dummy_wav, dummy_dvec)

        dummy_input = (dummy_wav, dummy_dvec)
        assert (test_out.size() == dummy_wav.size())


    else:
        dummy_input = torch.randn(1, 1, sample_block_size)
        test_out = torch_model(dummy_input)
        assert (test_out.size() == dummy_input.size())

        input_names = ["input1"]

    dynamic_axes = {
        "input1": {2: 'blocksize'},  # dim 2
        "output1": {2: 'blocksize'}  # dim 2
    }

    output_names = ["output1"]

    torch.onnx.export(
        torch_model,
        args=dummy_input,
        f=onnx_filename,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("saved: " + onnx_filename)

    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    if cfg.testing.model_name == 'AutoEncoder_Speaker_PL':
        print("test with expected blocksize")
        test_speaker_embedding_model_onnx_output(torch_model, onnx_filename,
                                      sample_block_size=sample_block_size,
                                                 emb_size=cfg.model.emb_size)

        # print("test with larger blocksize")
        # test_speaker_embedding_model_onnx_output(torch_model, onnx_filename,
        #                                          sample_block_size=(sample_block_size * 2),
        #                                          emb_size=cfg.model.emb_size)

        # print("test with smaller blocksize ")
        # test_speaker_embedding_model_onnx_output(torch_model, onnx_filename,
        #                               sample_block_size=(sample_block_size // 2),
        #                                          emb_size=cfg.model.emb_size)

    else:
        print("test with expected blocksize")
        test_single_input_onnx_output(torch_model, onnx_filename, sample_block_size=sample_block_size)

        print("test with larger blocksize")
        test_single_input_onnx_output(torch_model, onnx_filename, sample_block_size=(sample_block_size * 2))

        print("test with smaller blocksize ")
        test_single_input_onnx_output(torch_model, onnx_filename,
                                      sample_block_size=(sample_block_size // 2))

    print("Conversion done")


def test_single_input_onnx_output(torch_model, onnx_filename, sample_block_size):
    ort_session = ort.InferenceSession(onnx_filename)
    dummy_input = torch.randn(1, 1, sample_block_size)

    onnx_outputs = ort_session.run(
        None,
        {"input1": dummy_input.numpy()},
    )
    # print(onnx_outputs[0][0])

    torch_model.eval()
    torch_outputs = torch_model(dummy_input)

    # print(torch_outputs[0])

    np.testing.assert_allclose(torch_outputs[0].detach().numpy(), onnx_outputs[0][0], rtol=1e-03, atol=1e-02)
    print("Model output delta is close... Good!")


def test_speaker_embedding_model_onnx_output(torch_model, onnx_filename, sample_block_size, emb_size):
    ort_session = ort.InferenceSession(onnx_filename)
    dummy_wav = torch.randn(1, 1, sample_block_size)
    dummy_dvec = torch.randn(1, emb_size)

    onnx_outputs = ort_session.run(
        None,
        {"input1": dummy_wav.numpy(),
         "dvec": dummy_dvec.numpy()},
    )
    # print(onnx_outputs[0][0])

    torch_model.eval()
    torch_outputs = torch_model(dummy_wav, dummy_dvec)

    # print(torch_outputs[0])

    np.testing.assert_allclose(torch_outputs[0].detach().numpy(), onnx_outputs[0][0], rtol=1e-03, atol=1e-02)
    print("Model output delta is close... Good!")


if __name__ == "__main__":
    main()
