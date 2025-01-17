import dataclasses
from typing import Union

import numpy as np
import torch
from PIL.Image import Image

__all__ = ["SegmentationSample"]


@dataclasses.dataclass
class SegmentationSample:
    """
    A data class describing a single semantic segmentation sample that comes from a dataset.
    It contains both input image and target segmentation mask to train an semantic segmentation model.

    :param image:              Associated image with sample, in [H,W,C] (or H,W for greyscale) format.
    :param mask:               Associated segmentation mask with sample, in [H,W]
    """

    __slots__ = ["image", "mask"]

    image: Union[np.ndarray, torch.Tensor]
    mask: Union[np.ndarray, torch.Tensor]

    def __init__(self, image: Union[np.ndarray, torch.Tensor, Image], mask: Union[np.ndarray, torch.Tensor, Image]):
        self.image = image
        self.mask = mask
