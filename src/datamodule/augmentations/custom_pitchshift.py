import librosa
import random
import torch
from torch import Tensor
from typing import Optional
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict


class PitchShift_Slow(BaseWaveformTransform):
    """
    Pitch-shift sounds up or down without changing the tempo.
    adapted from the non gpu accelerated audiomentation
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_transpose_semitones: float = -4.0,
        max_transpose_semitones: float = 4.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """
        :param sample_rate:
        :param min_transpose_semitones: Minimum pitch shift transposition in semitones (default -4.0)
        :param max_transpose_semitones: Maximum pitch shift transposition in semitones (default +4.0)
        :param mode: ``per_example``, ``per_channel``, or ``per_batch``. Default ``per_example``.
        :param p:
        :param p_mode:
        :param target_rate:
        """
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        if min_transpose_semitones > max_transpose_semitones:
            raise ValueError("max_transpose_semitones must be > min_transpose_semitones")
        if not sample_rate:
            raise ValueError("sample_rate is invalid.")
        self._sample_rate = sample_rate

        assert min_transpose_semitones >= -12
        assert max_transpose_semitones <= 12
        assert min_transpose_semitones <= max_transpose_semitones
        self.min_transpose_semitones = min_transpose_semitones
        self.max_transpose_semitones = max_transpose_semitones

        self._mode = mode

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """
        :param samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        batch_size, num_channels, num_samples = samples.shape

        if self._mode == "per_example":
            list_of_trans = []
            for b in range(0, batch_size):
                list_of_trans.append(
                    random.uniform(self.min_transpose_semitones, self.max_transpose_semitones)
                )
            self.transform_parameters["transpositions"] = list_of_trans
        elif self._mode == "per_channel":
            self.transform_parameters["transpositions"] = list(
                zip(
                    *[
                        random.uniform(self.min_transpose_semitones, self.max_transpose_semitones)
                        for i in range(num_channels)
                    ]
                )
            )
        elif self._mode == "per_batch":
            self.transform_parameters["transpositions"] = \
                random.uniform(self.min_transpose_semitones, self.max_transpose_semitones)

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        """
        :param samples: (batch_size, num_channels, num_samples)
        :param sample_rate:
        """
        batch_size, num_channels, num_samples = samples.shape

        if sample_rate is not None and sample_rate != self._sample_rate:
            raise ValueError(
                "sample_rate must match the value of sample_rate "
                + "passed into the PitchShift constructor"
            )
        sample_rate = self.sample_rate

        # convert to ndarray
        samples = samples.numpy()

        if self._mode == "per_example":
            for i in range(batch_size):
                samples[i, ...] = librosa.effects.pitch_shift(
                    samples[i][None],
                    n_steps=self.transform_parameters["transpositions"][i],
                    sr=sample_rate,
                )[0]

        elif self._mode == "per_channel":
            for i in range(batch_size):
                for j in range(num_channels):
                    samples[i, j, ...] = librosa.effects.pitch_shift(
                        samples[i][j][None][None],
                        n_steps=self.transform_parameters["transpositions"][i][j],
                        sr=sample_rate,
                    )[0][0]

        elif self._mode == "per_batch":
            samples = librosa.effects.pitch_shift(
                samples,
                n_steps=self.transform_parameters["transpositions"][0],
                sr=sample_rate
            )

        # revert to tensor
        samples = torch.from_numpy(samples)

        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
