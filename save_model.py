import numpy as np
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from tensorflow.keras import backend as K

import core.utils as utils
from core.yolov4 import YOLO, decode

flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416', 'path to output')
flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
flags.DEFINE_integer('input_size', 416, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
flags.DEFINE_string('framework', 'tf',
                    'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')


def main(_argv):
    save_tf()


def make_keras_detector():
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    K.set_floatx('float32')
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    input_layer = tf.keras.layers.Input([None, None, 3], dtype=tf.float32)
    input_shape = tf.shape(input_layer)[1:3]
    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    bbox_tensors = []
    prob_tensors = []
    if FLAGS.tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(
                    fm, input_shape // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            else:
                output_tensors = decode(
                    fm, input_shape // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(
                    fm, input_shape // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            elif i == 1:
                output_tensors = decode(
                    fm, input_shape // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            else:
                output_tensors = decode(
                    fm, input_shape // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1, name='boxes')
    pred_prob = tf.concat(prob_tensors, axis=1, name='scores')
    preds = (pred_bbox, pred_prob)
    return tf.keras.Model(input_layer, preds)


def save_tf():
    model = make_keras_detector()
    utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny, dtype=np.float32)

    person_detector = PersonDetector(model)
    tf.saved_model.save(person_detector, FLAGS.output)


class PersonDetector(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.bool)])
    def predict_multi_image(
            self, images, threshold=0.1, nms_iou_threshold=0.65, flip_aug=False,
            bothflip_aug=False):
        target_size = tf.convert_to_tensor(FLAGS.input_size, tf.float32)
        shape = tf.shape(images)
        h = tf.cast(shape[1], tf.float32)
        w = tf.cast(shape[2], tf.float32)
        max_side = tf.maximum(h, w)
        factor = target_size / max_side
        target_w = tf.cast(factor * w, tf.int32)
        target_h = tf.cast(factor * h, tf.int32)
        images = (tf.cast(images, tf.float32) / 255) ** 2.2

        if factor > 1:
            images = tf.image.resize(
                images, (target_h, target_w), method=tf.image.ResizeMethod.BILINEAR)
        else:
            images = tf.image.resize(
                images, (target_h, target_w), method=tf.image.ResizeMethod.AREA)

        # images = tf.cast(images, tf.float32) / 255
        images = tf.cast(images, tf.float32) ** (1 / 2.2)
        pad_h = -target_h % 32
        pad_w = -target_w % 32
        half_pad_h = pad_h // 2
        half_pad_w = pad_w // 2
        half_pad_h_float = tf.cast(half_pad_h, tf.float32)
        half_pad_w_float = tf.cast(half_pad_w, tf.float32)
        images = tf.pad(
            images, [(0, 0), (half_pad_h, pad_h - half_pad_h),
                     (half_pad_w, pad_w - half_pad_w), (0, 0)],
            constant_values=0.5)

        if bothflip_aug:
            flipped_horiz = tf.image.flip_left_right(images)
            flipped_vert = tf.image.flip_up_down(images)
            net_input = tf.concat([images, flipped_horiz, flipped_vert], axis=0)
            boxes, scores = self.model(net_input)
            padded_width = tf.cast(tf.shape(images)[2], tf.float32)
            padded_height = tf.cast(tf.shape(images)[1], tf.float32)
            boxes_normal, boxes_flipped_horiz, boxes_flipped_vert = tf.split(boxes, 3, axis=0)
            boxes_backflipped_horiz = tf.concat(
                [padded_width - boxes_flipped_horiz[..., :1], boxes_flipped_horiz[..., 1:]],
                axis=-1)
            boxes_backflipped_vert = tf.concat(
                [boxes_flipped_vert[..., :1], padded_height - boxes_flipped_vert[..., 1:2],
                 boxes_flipped_vert[..., 2:]], axis=-1)
            boxes = tf.concat(
                [boxes_normal, boxes_backflipped_horiz, boxes_backflipped_vert], axis=1)
            scores = tf.concat(tf.split(scores, 3, axis=0), axis=1)
        elif flip_aug:
            flipped = tf.image.flip_left_right(images)
            net_input = tf.concat([images, flipped], axis=0)
            boxes, scores = self.model(net_input)
            padded_width = tf.cast(tf.shape(images)[2], tf.float32)
            boxes_normal, boxes_flipped = tf.split(boxes, 2, axis=0)
            boxes_backflipped = tf.concat(
                [padded_width - boxes_flipped[..., :1], boxes_flipped[..., 1:]], axis=-1)
            boxes = tf.concat([boxes_normal, boxes_backflipped], axis=1)
            scores = tf.concat(tf.split(scores, 2, axis=0), axis=1)
        else:
            boxes, scores = self.model(images)

        # Convert from CMWH to TLBR
        boxes = tf.stack([
            boxes[..., 1] - boxes[..., 3] / 2, boxes[..., 0] - boxes[..., 2] / 2,
            boxes[..., 1] + boxes[..., 3] / 2, boxes[..., 0] + boxes[..., 2] / 2], axis=-1)

        batch_size = tf.shape(images)[0]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (batch_size, -1, 1, 4)),
            scores=tf.reshape(scores[..., 0], (batch_size, -1, 1)),
            max_output_size_per_class=150, max_total_size=150, iou_threshold=nms_iou_threshold,
            score_threshold=threshold, clip_boxes=False)

        # Convert from TLBR to LTWH
        y_factor = h / tf.cast(target_h, tf.float32)
        x_factor = w / tf.cast(target_w, tf.float32)

        boxes = tf.stack([
            (boxes[..., 1] - half_pad_w_float) * x_factor,
            (boxes[..., 0] - half_pad_h_float) * y_factor,
            (boxes[..., 3] - boxes[..., 1]) * x_factor,
            (boxes[..., 2] - boxes[..., 0]) * y_factor,
            scores], axis=-1)

        return tf.RaggedTensor.from_tensor(boxes, lengths=valid_detections)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.bool)])
    def predict_single_image(
            self, image, threshold=0.1, nms_iou_threshold=0.65, flip_aug=False, bothflip_aug=False):
        boxes = self.predict_multi_image(
            image[tf.newaxis], threshold, nms_iou_threshold, flip_aug, bothflip_aug)
        return tf.squeeze(boxes, 0)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
