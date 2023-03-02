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

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cur_path = Path(os.path.abspath(hydra.utils.get_original_cwd()))
    ckpt_path = cfg.export_to_onnx.checkpoint_file

    if cfg.export_to_onnx.model == 'wavenet':
        model_pl = WaveNet_PL.load_from_checkpoint(cur_path/Path(ckpt_path))
        torch_model = model_pl.wavenet
    else:
        assert False

    sample_block_size = cfg.export_to_onnx.sample_block_size
    onnx_filename = cfg.export_to_onnx.export_filename
    dummy_input = torch.randn(1, 1, sample_block_size)

    torch_model.eval()
    test_out = torch_model(dummy_input)
    assert (test_out.size() == dummy_input.size())

    input_names = ["input1"]
    output_names = ["output1"]

    torch.onnx.export(
        torch_model,
        dummy_input,
        onnx_filename,
        input_names=input_names,
        output_names=output_names
    )

    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    test_onnx_output(torch_model, onnx_filename, sample_block_size=sample_block_size)

    print("Conversion done")
    print("saved: " + onnx_filename)


def test_onnx_output(torch_model, onnx_filename, sample_block_size):
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

    np.testing.assert_allclose(torch_outputs[0].detach().numpy(), onnx_outputs[0][0], rtol=1e-03, atol=1e-05)
    print("Model output delta is close... Good!")


if __name__ == "__main__":
    main()
