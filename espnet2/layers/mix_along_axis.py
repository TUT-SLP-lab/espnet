from turtle import forward
import torch
from typing import Sequence, Union

from typeguard import check_argument_types


def mix_alogn_axis(
    est_spec: torch.Tensor,
    org_spec: torch.Tensor,
    est_spec_length: torch.Tensor = None,
    org_spec_length: torch.Tensor = None,
    mix_width_range: Sequence[int] = (0, 30),
    dim: int = 1,
    num_mix: int = 2,
):
    """Mix with original and estimated speech.
    Args:
            est_spec: (Batch, Length, Freq): speech spectrogram estimated
            est_spec_length: (Length): speech spectrogram length estimated
            origin_spec: (Batch, Length, Freq): speech spectrogram estimated
            origin_spec_length: (Length): speech spectrogram length estimated
            mix_width_range: Sequence[int] = (0,30),
            dim: int = 1,
            num_mix: int = 2,
    """

    if est_spec.dim() == 4:
        # spec: (Batch, Channel, Length, Freq) -> (Batch * Channel, Length, Freq)
        est_spec = est_spec.view(-1, est_spec.size(2), est_spec.size(3))

    B = est_spec.shape[0]
    # D = Length or Freq
    D = est_spec.shape[dim]

    # mix_length: (B, num_mix,1)

    mix_length = torch.randint(
        mix_width_range[0],
        mix_width_range[1],
        (B, num_mix),
        device=est_spec.device,
    ).unsqueeze(2)

    # mix_pos: (B, num_mix,1)
    mix_pos = torch.randint(
        0, max(1, D - mix_length.max()), (B, num_mix), device=est_spec.device
    ).unsqueeze(2)

    # aran: (1, 1, D)
    aran = torch.arange(D, device=est_spec.device)[None, None, :]
    # mask: (Batch, num_mask, D)
    est_mask = (mix_pos <= aran) * (aran < (mix_pos + mix_length))
    # Multiply masks: (Batch, num_mask, D) -> (Batch, D)
    est_mask = est_mask.any(dim=1).logical_not()

    if dim == 1:
        # mask: (Batch, Length, 1)
        est_mask = est_mask.unsqueeze(2)
    elif dim == 2:
        # mask: (Batch, 1, Freq)
        est_mask = est_mask.unsqueeze(1)
    org_mask = torch.logical_not(est_mask)

    # convert to 0,1
    est_mask = est_mask.int()
    org_mask = org_mask.int()

    spec_mixed = torch.mul(est_spec, est_mask) + torch.mul(org_spec, org_mask)

    return spec_mixed, est_spec_length


class MixAlongAxis(torch.nn.Module):
    """
    Mix 2 speech alogn axis
    this support just "time" and "freq"

    """

    def __init__(
        self,
        mix_width_range: Union[int, Sequence[int]],
        num_mix: int = 2,
        dim: Union[int, str] = "time",
    ) -> None:
        assert check_argument_types()

        if isinstance(mix_width_range, int):
            mix_width_range = (0, mix_width_range)
        if len(mix_width_range) != 2:
            raise TypeError(
                f"mix_width_range must be a tuple of int and int values: "
                f"{mix_width_range}",
            )
        assert mix_width_range[1] > mix_width_range[0]
        if isinstance(dim, str):
            if dim == "time":
                dim = 1
            elif dim == "freq":
                dim = 2
            else:
                raise ValueError("dim must be int, 'time' or 'freq'")
        if dim == 1:
            self.mix_axis = "time"
        elif dim == 2:
            self.mix_axis = "freq"
        else:
            self.mix_axis = "unknown"

        super().__init__()
        self.mix_width_range = mix_width_range
        self.num_mix = num_mix
        self.dim = dim

    def extra_repr(self) -> str:
        return (
            f"mix_width_range={self.mix_width_range}, "
            f"num_mix={self.num_mix}, axis={self.mix_axis}"
        )

    def forward(
        self,
        est_spec: torch.Tensor,
        org_spec: torch.Tensor,
        est_spec_lengths: torch.Tensor = None,
        org_spec_length: torch.Tensor = None,
    ):
        """Forward function.

        Args:
            est_spec: (Batch, Length, Freq): estimated spectrogram
            org_spec: (Batch, Length, Freq): original spectrogram
        """

        return mix_alogn_axis(
            est_spec=est_spec,
            est_spec_length=est_spec_lengths,
            org_spec=org_spec,
            org_spec_length=org_spec_length,
            mix_width_range=self.mix_width_range,
            dim=self.dim,
            num_mix=self.num_mix,
        )
