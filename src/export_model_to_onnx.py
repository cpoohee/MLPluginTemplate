import hydra
import os
import torch
import torch.onnx
import onnx
import onnxruntime as ort
from pathlib import Path
from omegaconf import DictConfig
from src.model.wavenet import WaveNet_PL

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cur_path = Path(os.path.abspath(hydra.utils.get_original_cwd()))
    ckpt_path = cfg.export_to_onnx.checkpoint_file

    if cfg.export_to_onnx.model == 'wavenet':
        model_pl = WaveNet_PL.load_from_checkpoint(cur_path/Path(ckpt_path))
        model = model_pl.wavenet
    else:
        assert False

    dummy_input = torch.randn(1, 1, cfg.export_to_onnx.sample_block_size)
    onnx_filename = cfg.export_to_onnx.export_filename

    test_out = model(dummy_input)

    assert (test_out.size() == dummy_input.size())

    input_names = ["input1"]
    output_names = ["output1"]

    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        input_names=input_names,
        output_names=output_names
    )

    model = onnx.load(onnx_filename)
    onnx.checker.check_model(model)

    # TODO test pytorch output - onnx output delta

    print("Conversion done")
    print("saved: " + onnx_filename)


if __name__ == "__main__":
    main()
