# YOLOv4 TensorFlow SavedModel
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

Forked from the [repo](https://github.com/hunglc007/tensorflow-yolov4-tflite) by hunglc007.

This fork generates an easy to use YOLOv4 TensorFlow SavedModel that accepts any image size, works in batched and non-batched mode and returns person detections in a convenient tf.RaggedTensor.

Compile the aforementioned SavedModel as follows:

```bash
python save_model.py --weights $weight_dir/yolov4.weights --output $output_path --input_size 416 --model yolov4
```

# API Reference

Load the saved model as

```python
import tensorflow as tf

model = tf.saved_model.load('path_to_model')
```

## Methods

### model.predict_single_image

Performs person bounding box detection on an RGB image.

```python
model.predict_single_image(
    image, threshold=0.1, nms_iou_threshold=0.65, flip_aug=False, bothflip_aug=False)
```

#### Arguments:

- **image**: a ```uint8``` Tensor of shape ```[H, W, 3]``` containing an RGB image.
- **threshold**: a ```float32``` value for thresholding detection scores (detections with lower score are discarded)
- **nms_iou_threshold**: float value for use in intersection-over-union-based (IoU) non-max suppression (NMS).
  Too low values may result in false negatives when people are close to each other in the image,
  while too high values may result in duplicates (same person detected multiple times).
- **flip_aug**: boolean specifying whether to run the image through the detector with
  horizontal flipping as well and aggregate the results (before the detector NMS step).
- **bothflip_aug**: boolean specifying whether to run the image through the detector with
  horizontal and vertical flipping as well (so 3 augmentations) and aggregate the results (before the detector NMS step).

#### Return value:
**boxes**: ```[left, top, width, height, confidence]``` for each detection box. Shape
  is ```[num_detections, 5]```.


### model.predict_multi_image

The batched (multiple input images) equivalent of ```predict_single_image```. Performs person detection
on a batch of RGB images.

```python
model.predict_multi_image(
    images, threshold=0.1, nms_iou_threshold=0.65, flip_aug=False, bothflip_aug=False)
```

Only the first argument is mandatory.

- **images**: a batch of RGB images as a ```uint8``` Tensor with shape ```[N, H, W, 3]```
- **The remaining arguments have the same type and meaning as in ```predict_single_image``` (see above).**

#### Return value:

**boxes**: ```[left, top, width, height, confidence]``` for each detection box. It is
  a ```tf.RaggedTensor``` with shape ```[N, None, 5]``` where the None stands for the ragged
  dimension (the image-specific number of detections).
