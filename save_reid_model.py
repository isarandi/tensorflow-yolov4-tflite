import os
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import sys

def main():
    checkpoint_filename = sys.argv[1]
    out_path = sys.argv[2]
    tf.compat.v1.disable_eager_execution()
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    with tf.compat.v1.Session() as sess:
        with tf.compat.v1.gfile.GFile(checkpoint_filename, 'rb') as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())

        tf.import_graph_def(graph_def, name='')
        graph = tf.compat.v1.get_default_graph()
        image_t = graph.get_tensor_by_name('images:0')
        features_t = graph.get_tensor_by_name('features:0')
        signature_def = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
            inputs=dict(image=image_t), outputs=dict(features=features_t))
        os.mkdir(out_path)
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(out_path)
        builder.add_meta_graph_and_variables(
            sess, ['serve'], signature_def_map=dict(serving_default=signature_def))
        builder.save()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.enable_eager_execution()
    crop_model = tf.saved_model.load(out_path)
    shutil.rmtree(out_path)
    wrapped_model = ReIDModel(crop_model)
    tf.saved_model.save(wrapped_model, out_path)


class ReIDModel(tf.Module):
    def __init__(self, crop_model):
        super().__init__()
        self.crop_model = crop_model
        self.predict_crop = self.crop_model.signatures['serving_default']
        self.crop_size = (64, 128)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(3, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)])
    def predict_single_image(self, image, intrinsic_matrix, boxes, internal_batch_size=64):
        if tf.size(boxes) == 0:
            return tf.zeros(shape=(0, 128), dtype=tf.float32)
        ragged_boxes = tf.RaggedTensor.from_tensor(boxes[np.newaxis])
        return self.predict_multi_image(
            image[np.newaxis], intrinsic_matrix[np.newaxis], ragged_boxes, internal_batch_size)[0]

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(None, 3, 3), dtype=tf.float32),
        tf.RaggedTensorSpec(shape=(None, None, 4), ragged_rank=1, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)])
    def predict_multi_image(self, image, intrinsic_matrix, boxes, internal_batch_size=64):
        """Obtain ReID features for multiple bounding boxes specified for an image."""
        n_images = tf.shape(image)[0]
        if tf.size(boxes) == 0:
            # Special case for zero boxes provided
            result_flat = tf.zeros(shape=(0, 128), dtype=tf.float32)
            return tf.RaggedTensor.from_row_lengths(result_flat, tf.zeros(n_images, tf.int64))

        boxes_flat = boxes.flat_values
        n_box_per_image = boxes.row_lengths()
        image_id_per_box = boxes.value_rowids()
        n_total_boxes = tf.shape(boxes_flat)[0]

        if tf.shape(intrinsic_matrix)[0] == 1:
            intrinsic_matrix = tf.repeat(intrinsic_matrix, n_images, axis=0)

        Ks = tf.repeat(intrinsic_matrix, n_box_per_image, axis=0)

        if internal_batch_size == 0:
            # No batching
            results_flat = self.predict_single_batch(image, Ks, boxes_flat, image_id_per_box)
            return tf.RaggedTensor.from_row_lengths(results_flat, n_box_per_image)

        n_batches = tf.cast(tf.math.ceil(n_total_boxes / internal_batch_size), tf.int32)
        result_batches = tf.TensorArray(
            tf.float32, size=n_batches, element_shape=(None, 128), infer_shape=False)

        for i in tf.range(n_batches):
            box_batch = boxes_flat[i * internal_batch_size:(i + 1) * internal_batch_size]
            image_ids = image_id_per_box[i * internal_batch_size:(i + 1) * internal_batch_size]
            K_batch = Ks[i * internal_batch_size:(i + 1) * internal_batch_size]
            features = self.predict_single_batch(image, K_batch, box_batch, image_ids)
            result_batches = result_batches.write(i, features)

        results_flat = result_batches.concat()
        return tf.RaggedTensor.from_row_lengths(results_flat, n_box_per_image)

    def predict_single_batch(self, images, K, boxes, image_ids):
        n_box = tf.shape(boxes)[0]
        center_points = boxes[:, :2] + boxes[:, 2:4] / 2
        box_center_camspace = transf(center_points - K[:, :2, 2], tf.linalg.inv(K[:, :2, :2]))
        box_center_camspace = tf.concat(
            [box_center_camspace, tf.ones_like(box_center_camspace[:, :1])], axis=1)

        new_z = box_center_camspace / tf.linalg.norm(box_center_camspace, axis=-1, keepdims=True)
        new_x = tf.stack([new_z[:, 2], tf.zeros_like(new_z[:, 2]), -new_z[:, 0]], axis=1)
        new_y = tf.linalg.cross(new_z, new_x)
        new_R = tf.stack([new_x, new_y, new_z], axis=1)
        box_scales = tf.reduce_min(np.array(self.crop_size) / boxes[:, 2:4], axis=-1)
        new_K_mid = (tf.reshape(box_scales, [1, -1, 1, 1]) *
                     tf.reshape(K[:, :2, :2], [1, -1, 2, 2]))

        center = tf.concat([tf.fill((1, n_box, 1, 1), tf.cast(side, tf.float32))
                            for side in self.crop_size], axis=2) / 2
        intrinsic_matrix = tf.concat([
            tf.concat([new_K_mid, center], axis=3),
            tf.concat([tf.zeros((1, n_box, 1, 2), tf.float32),
                       tf.ones((1, n_box, 1, 1), tf.float32)], axis=3)], axis=2)
        new_proj_matrix = intrinsic_matrix @ new_R
        homography = K @ tf.linalg.inv(new_proj_matrix)
        homography = tf.reshape(homography, [1, n_box, 3, 3])
        homography = tf.reshape(homography, [-1, 9])
        homography = homography[:, :8] / homography[:, 8:]

        crops = perspective_transform(
            images, homography, (self.crop_size[1], self.crop_size[0]), 'BILINEAR', image_ids)
        crops = tf.reshape(crops, [1, n_box * self.crop_size[1], self.crop_size[0], 3])
        crops = tf.reshape(crops, [-1, self.crop_size[1], self.crop_size[0], 3])
        return self.predict_crop(image=crops)['features']


def transf(points, matrices):
    return tf.einsum('...k,...jk->...j', points, matrices)


def perspective_transform(images, homographies, output_shape, interpolation, image_ids):
    n_crops = tf.cast(tf.shape(homographies)[0], tf.int32)
    result_crops = tf.TensorArray(
        images.dtype, size=n_crops, element_shape=(*output_shape, 3), infer_shape=False)
    for i in tf.range(n_crops):
        tf.autograph.experimental.set_loop_options(parallel_iterations=1000)
        crop = tfa.image.transform(
            images[image_ids[i]], homographies[i], interpolation=interpolation,
            output_shape=output_shape)
        result_crops = result_crops.write(i, crop)
    return result_crops.stack()


def linspace_noend(start, stop, num):
    # Like np.linspace(endpoint=False)

    start = tf.convert_to_tensor(start)
    stop = tf.convert_to_tensor(stop, dtype=start.dtype)

    if num > 1:
        step = (stop - start) / tf.cast(num, start.dtype)
        new_stop = stop - step
        return tf.linspace(start, new_stop, num)
    else:
        return tf.linspace(start, stop, num)


if __name__ == '__main__':
    main()
