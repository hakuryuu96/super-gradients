import dataclasses
from typing import Union

import numpy as np
import torch
from PIL.Image import Image

__all__ = ["DepthEstimationSample"]


@dataclasses.dataclass
class DepthEstimationSample:
    """
    A dataclass representing a single depth estimation sample.
    Contains input image and depth map.

    :param image:              Image of [H, W, (C if colorful)] shape.
    :param depth_map:          Depth map of [H, W] shape.
    """

    __slots__ = ["image", "mask"]

    image: Union[np.ndarray, torch.Tensor]
    depth_map: Union[np.ndarray, torch.Tensor]

    def __init__(self, image: Union[np.ndarray, torch.Tensor, Image], depth_map: Union[np.ndarray, torch.Tensor, Image]):
        self.image = image
        self.depth_map = depth_map
