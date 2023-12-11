import os

from typing import List

import cv2

from super_gradients.training.datasets.depth_estimation_datasets import AbstractDepthEstimationDataset
from super_gradients.training.transforms.depth_estimation import AbstractDepthEstimationTransform
from super_gradients.training.samples import DepthEstimationSample
from super_gradients.common.object_names import Datasets
from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory


@register_dataset(Datasets.REDWEB_DEPTH_ESTIMATION_DATASET)
class ReDWebDepthEstimationDataset(AbstractDepthEstimationDataset):
    """
    Dataset class for training depth estimation models using ReDWeb V1 dataset.

    Link to download dataset: https://sites.google.com/site/redwebcvpr18/
    Paper: https://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_Monocular_Relative_Depth_CVPR_2018_paper.pdf

    The ReDWeb V1 dataset consists of 3.6K RGB-RD images, covering both indoor and outdoor scenes.
    """

    @resolve_param("transforms", factory=TransformsFactory())
    def __init__(
        self,
        data_dir: str,
        transforms: List[AbstractDepthEstimationTransform] = [],
        images_dir: str = "Imgs",
        targets_dir: str = "RDs",
        image_extension: str = "jpg",
        target_extension: str = "png",
    ):
        """ """
        super().__init__(transforms=transforms)
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.targets_dir = targets_dir
        self.image_extension = image_extension
        self.target_extension = target_extension

        self.pair_names = [name.split(".")[0] for name in os.listdir(os.path.join(self.data_dir, self.targets_dir))]

    def __len__(self):
        return len(self.pair_names)

    def load_sample(self, index: int) -> DepthEstimationSample:
        pair_name = self.pair_names[index]

        image = cv2.imread(os.path.join(self.data_dir, self.images_dir, f"{pair_name}.{self.image_extension}"), cv2.IMREAD_COLOR)

        depth_map = cv2.imread(os.path.join(self.data_dir, self.targets_dir, f"{pair_name}.{self.target_extension}"), cv2.IMREAD_GRAYSCALE)

        return DepthEstimationSample(image=image, depth_map=depth_map)
