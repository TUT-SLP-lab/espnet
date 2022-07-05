""" MixAugument module."""
from typing import Optional, Sequence, Union

from espnet2.asr.mixaug.abs_mixaug import AbsMixAug
from espnet2.layers.mix_along_axis import MixAlongAxis


class MixAug(AbsMixAug):
    def __init__(
        self,
        apply_freq_mix: bool = True,
        freq_mix_width_range: Union[int, Sequence[int]] = (0, 20),
        num_freq_mix: int = 2,
        apply_time_mix: bool = True,
        time_mix_width_range: Optional[Union[int, Sequence[int]]] = None,
        num_time_mix: int = 2,
    ) -> None:
        super().__init__()

        self.apply_freq_mix = apply_freq_mix
        self.apply_time_mix = apply_time_mix

        if self.apply_time_mix:
            self.time_mix = MixAlongAxis(
                mix_width_range=time_mix_width_range, num_mix=num_time_mix, dim="time"
            )
        else:
            self.time_mix = None
        if self.apply_freq_mix:
            self.freq_mix = MixAlongAxis(
                mix_width_range=freq_mix_width_range, num_mix=num_freq_mix, dim="freq"
            )
        else:
            self.freq_mix = None

    def forward(self, x, y, x_lengths=None, y_lengths=None):
        if self.time_mix is not None:
            x, x_lengths = self.time_mix(x, y, x_lengths, y_lengths)
        if self.freq_mix is not None:
            x, x_lengths = self.freq_mix(x, y, x_lengths, y_lengths)

        return x, x_lengths
