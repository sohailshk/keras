import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.utils.module_utils import tensorflow as tf


class RandomCropTest(testing.TestCase):
    def test_random_crop(self):
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 2,
                "width": 2,
                "data_format": "channels_last",
            },
            input_shape=(1, 3, 4, 3),
            supports_masking=False,
            run_training_check=False,
            expected_output_shape=(1, 2, 2, 3),
        )
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 2,
                "width": 2,
                "data_format": "channels_last",
            },
            input_shape=(3, 4, 3),
            supports_masking=False,
            run_training_check=False,
            expected_output_shape=(2, 2, 3),
        )
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 2,
                "width": 2,
                "data_format": "channels_first",
            },
            input_shape=(1, 3, 3, 4),
            supports_masking=False,
            run_training_check=False,
            expected_output_shape=(1, 3, 2, 2),
        )
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 2,
                "width": 2,
                "data_format": "channels_first",
            },
            input_shape=(3, 3, 4),
            supports_masking=False,
            run_training_check=False,
            expected_output_shape=(3, 2, 2),
        )

    def test_random_crop_full(self):
        np.random.seed(1337)
        height, width = 8, 16
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
        else:
            input_shape = (12, 3, 8, 16)
        inp = np.random.random(input_shape)
        layer = layers.RandomCrop(height, width)
        actual_output = layer(inp, training=False)
        self.assertAllClose(inp, actual_output)

    def test_random_crop_partial(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
            output_shape = (12, 8, 8, 3)
        else:
            input_shape = (12, 3, 8, 16)
            output_shape = (12, 3, 8, 8)
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 8,
                "width": 8,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            supports_masking=False,
            run_training_check=False,
        )

    def test_predicting_with_longer_height(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
            output_shape = (12, 10, 8, 3)
        else:
            input_shape = (12, 3, 8, 16)
            output_shape = (12, 3, 10, 8)
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 10,
                "width": 8,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            supports_masking=False,
            run_training_check=False,
        )

    def test_predicting_with_longer_width(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (12, 8, 16, 3)
            output_shape = (12, 8, 18, 3)
        else:
            input_shape = (12, 3, 8, 16)
            output_shape = (12, 3, 8, 18)
        self.run_layer_test(
            layers.RandomCrop,
            init_kwargs={
                "height": 8,
                "width": 18,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            supports_masking=False,
            run_training_check=False,
        )

    def test_tf_data_compatibility(self):
        layer = layers.RandomCrop(8, 9)
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 12, 3)
            output_shape = (2, 8, 9, 3)
        else:
            input_shape = (2, 3, 10, 12)
            output_shape = (2, 3, 8, 9)
        input_data = np.random.random(input_shape)
        ds = tf.data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        output = next(iter(ds)).numpy()
        self.assertEqual(tuple(output.shape), output_shape)

    def test_dict_input(self):
        layer = layers.RandomCrop(
            3, 3, data_format="channels_last", bounding_box_format="xyxy"
        )
        data = {
            "images": np.random.random((2, 4, 5, 3)),
            "labels": np.random.random((2, 7)),
            "segmentation_masks": np.random.random((2, 4, 5, 7)),
            "bounding_boxes": {
                "boxes": np.array([[1, 2, 2, 3]]),
                "labels": np.array([0]),
            },
        }
        transformed_data = layer(data)
        self.assertEqual(
            data["images"].shape[:-1],
            transformed_data["segmentation_masks"].shape[:-1],
        )
        self.assertAllClose(data["labels"], transformed_data["labels"])
        self.assertEqual(data["bounding_boxes"]["boxes"].shape, (1, 4))
        self.assertAllClose(
            data["bounding_boxes"]["labels"],
            transformed_data["bounding_boxes"]["labels"],
        )

    def test_bounding_boxes_offsets_applied_to_correct_axes(self):
        layer = layers.RandomCrop(
            2, 4, data_format="channels_last", bounding_box_format="xyxy"
        )

        # Unbatched boxes: shape (num_boxes, 4)
        bounding_boxes = {
            "boxes": np.array([[3.0, 4.0, 5.0, 5.0]]),
            "labels": np.array([0]),
        }
        output = layer.transform_bounding_boxes(
            bounding_boxes, (3, 1, 8, 8), training=True
        )
        # Crop starts at row 3 (h_start), column 1 (w_start).
        # x coordinates must shift by w_start, y coordinates by h_start.
        expected_boxes = np.array([[2.0, 1.0, 4.0, 2.0]])
        self.assertAllClose(output["boxes"], expected_boxes)
        self.assertAllClose(output["labels"], np.array([0]))

        # Batched boxes: shape (batch, num_boxes, 4)
        bounding_boxes = {
            "boxes": np.array([[[3.0, 4.0, 5.0, 5.0]]]),
            "labels": np.array([[0]]),
        }
        output = layer.transform_bounding_boxes(
            bounding_boxes, (3, 1, 8, 8), training=True
        )
        expected_boxes = np.array([[[2.0, 1.0, 4.0, 2.0]]])
        self.assertAllClose(output["boxes"], expected_boxes)
        self.assertAllClose(output["labels"], np.array([[0]]))

    def test_bounding_boxes_clipped_to_crop_size(self):
        layer = layers.RandomCrop(
            2, 4, data_format="channels_last", bounding_box_format="xyxy"
        )

        # Unbatched boxes: shape (num_boxes, 4)
        bounding_boxes = {
            "boxes": np.array([[0.0, 0.0, 8.0, 8.0]]),
            "labels": np.array([0]),
        }
        output = layer.transform_bounding_boxes(
            bounding_boxes, (3, 1, 8, 8), training=True
        )
        # Box extends beyond the crop: it must be clipped to its size.
        # x2 = clip(8 - w_start, 0, 4) = 4 ; y2 = clip(8 - h_start, 0, 2) = 2
        expected_boxes = np.array([[0.0, 0.0, 4.0, 2.0]])
        self.assertAllClose(output["boxes"], expected_boxes)
        self.assertAllClose(output["labels"], np.array([0]))

        # Batched boxes: shape (batch, num_boxes, 4)
        bounding_boxes = {
            "boxes": np.array([[[0.0, 0.0, 8.0, 8.0]]]),
            "labels": np.array([[0]]),
        }
        output = layer.transform_bounding_boxes(
            bounding_boxes, (3, 1, 8, 8), training=True
        )
        expected_boxes = np.array([[[0.0, 0.0, 4.0, 2.0]]])
        self.assertAllClose(output["boxes"], expected_boxes)
        self.assertAllClose(output["labels"], np.array([[0]]))

    def _box_extent(self, image):
        # (x1, y1, x2, y2) of the pixels above 0.5 in an (H, W, 1) image.
        ys, xs = np.where(backend.ops.convert_to_numpy(image)[..., 0] > 0.5)
        return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])

    @parameterized.named_parameters(
        ("crop_smaller", (20, 30), "xyxy"),
        ("crop_smaller_rel", (20, 30), "rel_xyxy"),
        ("crop_larger", (50, 70), "xyxy"),
        ("crop_larger_rel", (50, 70), "rel_xyxy"),
        ("crop_larger_height_only", (50, 30), "xyxy"),
    )
    def test_bounding_boxes_follow_image_content(self, size, box_format):
        # A white rectangle on a black image: the transformed box must still
        # cover the rectangle in the output image, including when the input is
        # smaller than the target size (images get resized) and for relative
        # box formats (relative to the input, not to the crop).
        height, width = 40, 60
        images = np.zeros((1, height, width, 1), "float32")
        images[:, 4:36, 6:54] = 1.0
        boxes = np.array([[[6.0, 4.0, 54.0, 36.0]]], "float32")
        if box_format == "rel_xyxy":
            boxes = boxes / np.array([width, height, width, height])
        layer = layers.RandomCrop(
            *size,
            bounding_box_format=box_format,
            data_format="channels_last",
            seed=0,
        )
        out = layer(
            {
                "images": images,
                "bounding_boxes": {
                    "boxes": boxes,
                    "labels": np.array([[1]], "int32"),
                },
            },
            training=True,
        )
        out_boxes = backend.ops.convert_to_numpy(out["bounding_boxes"]["boxes"])
        if box_format == "rel_xyxy":
            out_boxes = out_boxes * np.array(
                [size[1], size[0], size[1], size[0]]
            )
        expected = self._box_extent(out["images"][0])
        self.assertAllClose(out_boxes[0, 0], expected, atol=0.5)

    def test_segmentation_mask_preserves_dtype_and_classes(self):
        # Masks hold discrete class indices; the resize fallback must use
        # nearest-neighbor interpolation so no new classes appear, and the
        # integer dtype must be preserved. See keras-team/keras#20857.
        images = np.random.random((2, 8, 8, 3)).astype("float32")
        masks = np.zeros((2, 8, 8, 1), dtype="uint8")
        masks[:, :4] = 4  # only classes {0, 4}
        # Target larger than input forces the resize path.
        layer = layers.RandomCrop(16, 16, seed=0)
        out = layer({"images": images, "segmentation_masks": masks})
        result = backend.ops.convert_to_numpy(out["segmentation_masks"])
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(set(np.unique(result).tolist()).issubset({0, 4}))

        # With training=False the layer is a no-op: images and masks must
        # pass through unchanged (and keep matching shapes).
        out_eval = layer(
            {"images": images, "segmentation_masks": masks}, training=False
        )
        self.assertAllClose(out_eval["images"], images)
        self.assertAllClose(out_eval["segmentation_masks"], masks)
