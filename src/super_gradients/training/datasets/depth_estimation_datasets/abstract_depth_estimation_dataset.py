import abc
from typing import List

import random

from torch.utils.data.dataloader import Dataset
from super_gradients.training.samples import DepthEstimationSample


class AbstractDepthEstimationDataset(Dataset):
    """
    Abstract class for datasets for depth estimation task.

    Attemting to follow principles provided in pose_etimation_dataset.
    """

    def __init__(
        self,
        transforms: List[AbstractDepthEstimationTransform]
    ):
        pass

    @abc.abstractmethod
    def load_sample(self, index: int) -> DepthEstimationSample:
        raise NotImplementedError()

    def load_random_sample(self) -> DepthEstimationSample:
        """
        Return a random sample from the dataset

        :return: Instance of DepthEstimationSample
        """
        num_samples = len(self)
        random_index = random.randrange(0, num_samples)
        return self.load_sample(random_index)

    def __getitem__(self, index: int) -> DepthEstimationSample:
        sample = self.load_sample(index)
        sample = self.transforms.apply_to_sample(sample)
        return sample