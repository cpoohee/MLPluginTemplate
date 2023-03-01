import torch
from torch_audiomentations import Compose, Gain, PolarityInversion


def test_per_batch():
    # tests per_batch in torch_audiomentation p_mode produces the same randomised effect on all
    # samples in the batch,
    # that is, if random gain is -2, all samples in the batch will have -2 gain

    apply_augmentation = Compose(
        transforms=[
            Gain(
                min_gain_in_db=-15.0,
                max_gain_in_db=5.0,
                p=0.5,
                p_mode='per_batch',
            ),
            PolarityInversion(p=0.5, p_mode='per_batch')
        ]
    )

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make an example tensor with white noise.
    audio_sample1 = torch.rand(size=(1, 1, 44100), dtype=torch.float32, device=torch_device) - 0.5

    # create batch
    audio_samples = torch.cat([audio_sample1,audio_sample1], 0)

    # Apply augmentation.
    perturbed_audio_samples = apply_augmentation(audio_samples, sample_rate=44100)

    # check before augmentation is equal
    assert(torch.equal(audio_samples[0], audio_samples[1]))

    # check after augmentation is equal
    assert(torch.equal(perturbed_audio_samples[0], perturbed_audio_samples[1]))
