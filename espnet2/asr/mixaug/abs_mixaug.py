from typing import Optional, Tuple

import torch


class AbsMixAug(torch.nn.Module):
    """Abstract class for the mixing augmentation of spectrogram

    The process-flow:

    Frontend  -> MixAug -> SpecAug -> Normalization -> Encoder -> Decoder
    """

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_lengths: torch.Tensor = None,
        y_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
