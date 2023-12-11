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
    def __init__(self, data_dir: str, transforms: List[AbstractDepthEstimationTransform] = [], images_dir: str = "Imgs", targets_dir: str = "RDs"):
        """ """
        super().__init__(transforms=transforms)
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.targets_dir = targets_dir

        images_path = os.path.join(self.data_dir, self.images_dir)
        targets_path = os.path.join(self.data_dir, self.targets_dir)

        self._data_sanity_check(images_path, targets_path)

        sorted_images = sorted([os.path.join(images_path, file) for file in os.listdir(images_path)])
        sorted_targets = sorted([os.path.join(targets_path, file) for file in os.listdir(targets_path)])

        self.pair_paths = list(zip(sorted_images, sorted_targets))

    def _data_sanity_check(self, image_path: str, target_path: str) -> None:
        # separating name and extension
        image_names = [x.split(".")[0] for x in os.listdir(image_path)]
        target_names = [x.split(".")[0] for x in os.listdir(target_path)]

        unique_image_names = set(image_names)
        unique_target_names = set(target_names)

        diff = unique_image_names.symmetric_difference(unique_target_names)
        if len(diff) > 0:
            raise RuntimeError(f"{len(diff)} dataset elements don't have image or target pairs in data folder. Check the following names: {', '.join(diff)}.")

    def __len__(self):
        return len(self.pair_paths)

    def load_sample(self, index: int) -> DepthEstimationSample:
        image_path, depth_map_path = self.pair_paths[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

        return DepthEstimationSample(image=image, depth_map=depth_map)
