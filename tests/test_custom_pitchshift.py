import torch
from torch_audiomentations import Compose
from src.datamodule.augmentations.custom_pitchshift import PitchShift_Slow

def test_custom_pitchshift():
    # a quick functionality test on the custom pitch shift for torch audiomentations
    transforms = []

    transforms.append(PitchShift_Slow(min_transpose_semitones=-0.4,
                                      max_transpose_semitones=-0.1,
                                      p=1.0,
                                      sample_rate=44100,
                                      p_mode='per_example'))

    apply_augmentation = Compose(transforms)

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Make an example tensor with white noise.
    audio_sample = torch.rand(size=(2, 1, 44100), dtype=torch.float32, device=torch_device) - 0.5

    shifted_audio_samples = apply_augmentation(audio_sample, sample_rate=44100)

    assert(torch.equal(audio_sample, shifted_audio_samples) is False)