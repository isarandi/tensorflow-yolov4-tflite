# YOLOv4 TensorFlow SavedModel
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

Forked from the [repo](https://github.com/hunglc007/tensorflow-yolov4-tflite) by hunglc007.

This fork generates an easy to use TensorFlow SavedModel that accepts any image size, works in batched and non-batched mode and returns person detections in a convenient tf.RaggedTensor.

Compile the aforementioned SavedModel as follows:

```bash
python ./save_model.py --weights $weight_dir/yolov4.weights --output $output_path --input_size 416 --model yolov4
```
