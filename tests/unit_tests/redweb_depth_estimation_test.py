import unittest
from pathlib import Path
import cv2
import numpy as np
import os
import tempfile

from super_gradients.training.datasets.depth_estimation_datasets import ReDWebDepthEstimationDataset
from super_gradients.training.samples.depth_estimation_sample import DepthEstimationSample


class ReDWebDepthEstimationDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rw_data_dir = str(Path(__file__).parent.parent / "data" / "tinyredweb")
        self.default_dataset_params = {"data_dir": self.rw_data_dir, "images_dir": "Imgs", "targets_dir": "RDs"}
        self.transform_params = {
            "transforms": [
                {
                    "Albumentations": {
                        "Compose": {"transforms": [{"VerticalFlip": {"p": 1.0}}, {"InvertImg": {"p": 1.0}}]},
                    }
                }
            ]
        }

    def test_dataset_init(self):
        ReDWebDepthEstimationDataset(**self.default_dataset_params)

    def test_len_of_dataset(self):
        d = ReDWebDepthEstimationDataset(**self.default_dataset_params)
        self.assertEqual(len(d), 2)

    def test_dataset_output(self):
        d = ReDWebDepthEstimationDataset(**self.default_dataset_params)
        sample = d[0]
        self.assertTrue(isinstance(sample, DepthEstimationSample))
        self.assertTrue(len(sample.image.shape) == 3)
        self.assertTrue(len(sample.depth_map.shape) == 2)

    def test_smoke_dataset_with_augmentations(self):
        params_with_augs = {**self.transform_params, **self.default_dataset_params}
        d = ReDWebDepthEstimationDataset(**params_with_augs)
        self.assertTrue(isinstance(d[0], DepthEstimationSample))
        self.assertTrue(isinstance(d[1], DepthEstimationSample))

    def test_data_sanity_check_raise(self):
        with tempfile.TemporaryDirectory() as tmp_dirname:
            data_dir = os.path.join(tmp_dirname, "redwebtmp")
            os.mkdir(data_dir)
            os.mkdir(os.path.join(data_dir, "Imgs"))
            os.mkdir(os.path.join(data_dir, "RDs"))

            image = np.zeros((100, 100, 3), dtype=np.uint8)

            cv2.imwrite(os.path.join(data_dir, "Imgs", "test1.png"), image)
            cv2.imwrite(os.path.join(data_dir, "Imgs", "test2.png"), image)
            cv2.imwrite(os.path.join(data_dir, "Imgs", "test3.png"), image)

            cv2.imwrite(os.path.join(data_dir, "RDs", "test1.png"), image)

            init_params = {**self.default_dataset_params}
            init_params["data_dir"] = data_dir
            with self.assertRaises(RuntimeError) as context:
                ReDWebDepthEstimationDataset(**init_params)
            self.assertEquals(
                "2 dataset elements don't have image or target pairs in data folder. Check the following names: test2, test3.", str(context.exception)
            )

    def test_dataset_augmentations_correctness(self):
        # create small sample dataset
        with tempfile.TemporaryDirectory() as tmp_dirname:
            data_dir = os.path.join(tmp_dirname, "redwebtmp")
            os.mkdir(data_dir)
            os.mkdir(os.path.join(data_dir, "Imgs"))
            os.mkdir(os.path.join(data_dir, "RDs"))

            h_in, w_in = 100, 100
            h_out, w_out = 200, 200

            image = np.zeros((h_in, w_in, 3), dtype=np.uint8)
            depth_map = np.zeros((h_in, w_in), dtype=np.uint8)

            image[h_in // 2 :, ...] += 255
            depth_map[h_in // 2 :, ...] += 1

            # saving image to png not to lose in image quality
            cv2.imwrite(os.path.join(data_dir, "Imgs", "test.png"), image)
            cv2.imwrite(os.path.join(data_dir, "RDs", "test.png"), depth_map)

            params_with_augs = {**self.transform_params, **self.default_dataset_params}

            params_with_augs["data_dir"] = data_dir
            params_with_augs["transforms"][0]["Albumentations"]["Compose"]["transforms"].insert(
                0, {"Resize": {"p": 1.0, "height": h_out, "width": w_out, "interpolation": cv2.INTER_NEAREST}}
            )
            d = ReDWebDepthEstimationDataset(**params_with_augs)

            sample = d[0]

            # expected depth map is (200, 200) and only flipped
            self.assertTrue(sample.depth_map.shape == (h_out, w_out))

            expected_dm = np.zeros((h_out, w_out))
            expected_dm[: h_out // 2, ...] += 1

            # check that depth map just flipped vertically
            self.assertTrue(np.allclose(sample.depth_map, expected_dm))

            # expected image is (200, 200)
            self.assertTrue(sample.image.shape == (h_out, w_out, 3))

            # and invert(vflip(img)) == img
            expected_image = np.zeros((h_out, w_out, 3))
            expected_image[h_out // 2 :, ...] += 255
            self.assertTrue(np.allclose(sample.image, expected_image))
