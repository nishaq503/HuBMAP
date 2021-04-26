import gc
import os
import shutil
from glob import glob
from typing import List
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from rasterio.windows import Window
from tqdm import tqdm

from model import utils

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('DEBUG')


def encoding_to_mask(encoding, shape):
    mask = np.zeros(np.prod(shape), dtype=np.uint8)
    splits = encoding.split()
    for i in range(0, len(splits), 2):
        start, length = int(splits[i]) - 1, int(splits[i + 1])
        mask[start: start + length] = 1 + i
    return mask.reshape(shape).T


def mask_to_encodings(mask, n=1):
    pixels = mask.T.flatten()
    encodings = list()
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encodings.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            encodings.append(' '.join(str(x) for x in runs))
    return encodings


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """ Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, mask, x_index: int, y_index: int):
    """ Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    feature = {
        'x_index': _int64_feature(x_index),
        'y_index': _int64_feature(y_index),
        'image': _bytes_feature(image),
    }
    if mask is not None:
        feature['mask'] = _bytes_feature(mask)
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def parse_example(example_proto, tile_size: int, mode: str):
    feature_description = {
        'x_index': tf.io.FixedLenFeature([], tf.int64),
        'y_index': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    if mode == 'train':
        feature_description['mask'] = tf.io.FixedLenFeature([], tf.string)

    single_example = tf.io.parse_single_example(example_proto, feature_description)
    x_index = single_example['x_index']
    y_index = single_example['y_index']
    image = tf.reshape(
        tf.io.decode_raw(single_example['image'], out_type=np.dtype('uint8')),
        (tile_size, tile_size),
    )

    if mode == 'train':
        mask = tf.reshape(
            tf.io.decode_raw(single_example['mask'], out_type=np.dtype('uint8')),
            (tile_size, tile_size),
        )
        return image, mask, x_index, y_index
    else:
        return image, x_index, y_index


def load_dataset(filenames: List[str], tile_size: int, mode: str):
    if mode not in {'train', 'test'}:
        raise ValueError(f'mode must be \'train\' or \'test\'. Got \'{mode}\' instead.')

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda example: parse_example(example, tile_size, mode))
    return dataset


def _make_grid(shape: Tuple[int, int], tile_size: int, min_overlap: int):
    """ Return Array of size (N, 4), where:
        N - number of tiles,
        2nd axis represents slices: x1, x2, y1, y2
    """
    def num_tiles(length: int):
        num = length // (tile_size - min_overlap) + 1
        starts = np.linspace(0, length, num=num, endpoint=False, dtype=np.int64)
        starts[-1] = length - tile_size
        ends = (starts + tile_size).clip(0, length)
        return num, starts, ends

    num_x, x1, x2 = num_tiles(shape[0])
    num_y, y1, y2 = num_tiles(shape[1])
    grid = np.zeros((num_x, num_y, 4), dtype=np.int64)

    for i in range(num_x):
        for j in range(num_y):
            grid[i, j] = x1[i], x2[i], y1[j], y2[j]

    return grid.reshape(num_x * num_y, 4)


def create_tf_records(
        mode: str,
        tile_size: int = 1024,
        min_overlap: int = 32,
        s_threshold: int = 40,  # saturation blanking threshold
        p_threshold: int = None,  # threshold for minimum number of pixels
):
    if mode == 'train':
        in_dir = utils.TRAIN_DIR
        out_dir = utils.TF_TRAIN_DIR

        masks_df = pd.read_csv(utils.TRAIN_PATH)
        filenames = [os.path.join(in_dir, f'{name}.tiff') for name in masks_df['id']]
        encodings_list = list(masks_df['encoding'])
    elif mode == 'test':
        in_dir = utils.TEST_DIR
        out_dir = utils.TF_TEST_DIR

        path_pattern = f'{in_dir}/*.tiff'
        filenames = list(glob(path_pattern))
        encodings_list = None
    else:
        raise ValueError(f'mode must be \'train\' or \'test\'. Got \'{mode}\' instead.')

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    p_threshold = 1000 * (tile_size // 256) ** 2 if p_threshold is None else p_threshold
    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
    for i, tiff_path in tqdm(enumerate(filenames)):
        tiff_name = tiff_path.split('/')[-1].split('.')[0]
        print(f'{i + 1:02d} Creating tfrecords for image: {tiff_name}')

        tiff_reader = rasterio.open(tiff_path, transform=identity)
        if tiff_reader.count == 3:
            layers = None
        else:
            layers = [rasterio.open(sub_dataset) for sub_dataset in tiff_reader.subdatasets]

        if mode == 'train':
            encodings = encodings_list[i]
            shape = tiff_reader.shape
            full_mask = encoding_to_mask(encodings, (shape[1], shape[0]))
        else:
            full_mask = None

        slices = _make_grid(tiff_reader.shape, tile_size, min_overlap)
        print(slices.shape[0])

        count = 0
        tf_tiff_path = os.path.join(out_dir, f'{tiff_name}.tfrec')
        tf_writer_options = tf.io.TFRecordOptions(compression_type='GZIP')
        x_total, x_sq_total = [], []
        with tf.io.TFRecordWriter(tf_tiff_path, options=tf_writer_options) as tf_writer:
            for (x1, x2, y1, y2) in slices:
                image = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                window = Window.from_slices((x1, x2), (y1, y2))
                if tiff_reader.count == 3:
                    image[:, :, :] = np.moveaxis(tiff_reader.read([1, 2, 3], window=window), 0, -1)
                else:
                    for fl in range(3):
                        image[:, :, fl] = layers[fl].read(window=window)

                # if the image does not contain tissue, continue
                saturation = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[1]
                if ((saturation > s_threshold).sum() <= p_threshold) or (image.sum() <= p_threshold):
                    continue
                else:
                    if count % 100 == 99:
                        print(f'{count + 1}', end=' ')

                if mode == 'train':
                    mask = np.zeros(shape=(tile_size, tile_size), dtype=np.uint8)
                    mask[0:tile_size, 0:tile_size] = full_mask[x1: x2, y1: y2]
                else:
                    mask = None

                x_total.append((image / 255.0).reshape(-1, 3).mean(0))
                x_sq_total.append(((image / 255.0) ** 2).reshape(-1, 3).mean(0))

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if mask is not None:
                    mask = mask.tobytes()
                example = serialize_example(image.tobytes(), mask, x1, y1)
                tf_writer.write(example)
                count += 1
        print()
        new_tf_tiff_path = os.path.join(out_dir, f'{tiff_name}-{count}.tfrec')
        os.rename(tf_tiff_path, new_tf_tiff_path)
        gc.collect()

        # image stats
        image_mean = np.array(x_total).mean(0)
        image_std = np.sqrt(np.array(x_sq_total).mean(0) - image_mean ** 2)
        print(f'{i + 1:02d}, image: {tiff_name}, mean: {image_mean}, std: {image_std}')
    return


if __name__ == '__main__':
    # create_tf_records('train')
    create_tf_records('test')
    # _datagen = JSData(
    #     mode='train',
    #     resp=0,
    #     batch_size=32,
    #     noisy=True
    # )
    # for _i in range(5):
    #     _v1, _v2 = _datagen[_i]
    #     print(dict(Counter(_v2['classifier'])))
