import os
import shutil
from typing import List
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from rasterio.windows import Window

from model import utils


def encoding_to_mask(encoding: str, shape: Tuple[int, int]) -> np.array:
    mask = np.zeros(np.prod(shape), dtype=np.uint8)
    splits = encoding.split()
    for i in range(0, len(splits), 2):
        start, length = int(splits[i]) - 1, int(splits[i + 1])
        mask[start: start + length] = 1
    return mask.reshape(shape).T


def mask_to_encoding(mask: np.array) -> str:
    pixels = mask.T.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    starts = np.concatenate([[0], runs[1::2]])
    lengths = runs[::2] - starts
    return ' '.join((' '.join(map(str, v)) for v in zip(starts + 1, lengths)))


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(image, mask, indices):
    """ Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    feature = {
        'indices': _bytes_feature(indices),
        'image': _bytes_feature(image),
        'mask': _bytes_feature(mask),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def parse_example(example_proto, tile_size: int):
    feature_description = {
        'indices': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
    }

    single_example = tf.io.parse_single_example(example_proto, feature_description)

    x1, x2, y1, y2 = single_example['indices']

    image = tf.reshape(
        tf.io.decode_raw(single_example['image'], out_type=np.dtype('uint8')),
        (tile_size, tile_size, 3),
    )

    mask = tf.reshape(
        tf.io.decode_raw(single_example['mask'], out_type=np.dtype('uint8')),
        (tile_size, tile_size),
    )
    return image, mask, x1, x2, y1, y2


def load_dataset(filenames: List[str], tile_size: int):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(
        lambda example: parse_example(example, tile_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return dataset


def _make_grid(shape: Tuple[int, int], tile_size: int, min_overlap: int):
    """ Return Array of size (N, 4), where:
            N - number of tiles,
            2nd axis represents slices: x1, x2, y1, y2
    """
    if tile_size > min(shape):
        raise ValueError(f'Tile size of {tile_size} is too large for image with shape {shape}')

    def tiles(length: int):
        num = length // (tile_size - min_overlap) + 1
        starts = np.linspace(0, length, num=num, endpoint=False, dtype=np.int64)
        starts[-1] = length - tile_size
        ends = (starts + tile_size).clip(0, length)
        return num, starts, ends

    num_x, x1, x2 = tiles(shape[0])
    num_y, y1, y2 = tiles(shape[1])
    grid = np.zeros((num_x, num_y, 4), dtype=np.uint64)

    for i in range(num_x):
        for j in range(num_y):
            grid[i, j] = x1[i], x2[i], y1[j], y2[j]

    return grid.reshape(num_x * num_y, 4)


def _filter_tissue(
        image,
        s_threshold: int,
        p_threshold: int,
) -> bool:
    saturation = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[1]
    not_all_black = ((saturation > s_threshold).sum() <= p_threshold)
    not_all_gray = (image.sum() <= p_threshold)  # is this redundant?
    return not_all_black or not_all_gray


def tiff_tile_generator(
        tiff_path: str,
        encoding: str,
        tile_size: int,
        min_overlap: int,
        filter_tissue: bool = True,
):
    s_threshold = 40  # saturation blanking threshold
    p_threshold = 1000 * (tile_size // 256) ** 2  # threshold for minimum number of pixels

    tiff_reader = rasterio.open(tiff_path, transform=rasterio.Affine(1, 0, 0, 0, 1, 0))
    if tiff_reader.count == 3:
        layers = None
    else:
        layers = [rasterio.open(sub_dataset) for sub_dataset in tiff_reader.subdatasets]

    full_mask = None if encoding is None else encoding_to_mask(
        encoding=encoding,
        shape=(tiff_reader.shape[1], tiff_reader.shape[0]),
    )
    slices = _make_grid(tiff_reader.shape, tile_size, min_overlap)
    print(f'num_slices: {slices.shape[0]}')

    for x1, x2, y1, y2 in slices:
        window = Window.from_slices((x1, x2), (y1, y2))
        image = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        if tiff_reader.count == 3:
            image[:, :, :] = np.moveaxis(tiff_reader.read([1, 2, 3], window=window), 0, -1)
        else:
            for channel in range(3):
                image[:, :, channel] = layers[channel].read(window=window)

        if filter_tissue and _filter_tissue(image, s_threshold, p_threshold):
            continue

        # TODO: figure out whether BGR is more efficient than RGB
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if full_mask is None:
            mask = None
        else:
            mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            mask[:, :] = full_mask[x1:x2, y1:y2]

        yield image, mask, x1, x2, y1, y2


def create_tf_records(
        tile_size: int = 512,
        min_overlap: int = 128,
        filter_tissue: bool = True,
):
    # get the names of the tiff files to be read and
    # the list of encodings of the corresponding masks
    masks_df = pd.read_csv(utils.TRAIN_PATH)
    filenames = [os.path.join(utils.TRAIN_DIR, f'{name}.tiff') for name in masks_df['id']]
    encodings_list = list(masks_df['encoding'])

    # clean out the old files and prepare to write anew.
    if os.path.exists(utils.TF_TRAIN_DIR):
        shutil.rmtree(utils.TF_TRAIN_DIR)
    os.makedirs(utils.TF_TRAIN_DIR, exist_ok=True)

    for i, tiff_path in enumerate(filenames):
        encoding = encodings_list[i]
        tiff_name = tiff_path.split('/')[-1].split('.')[0]
        print(f'{i + 1:02d} Creating tfrecords for image: {tiff_name}')

        count, glom_count = 0, 0  # to rename tfrec files later

        tfrec_name = f'{tiff_name}'
        tfrec_path = os.path.join(utils.TF_TRAIN_DIR, f'{tfrec_name}.tfrec')

        tfrec_glom_name = f'{tiff_name}-glom'
        tfrec_glom_path = os.path.join(utils.TF_TRAIN_DIR, f'{tfrec_glom_name}.tfrec')

        options = tf.io.TFRecordOptions('GZIP')  # compression slows down writing but uses only ~35% space
        with tf.io.TFRecordWriter(tfrec_glom_path, options=options) as glom_tf_writer:
            with tf.io.TFRecordWriter(tfrec_path, options=options) as tf_writer:

                for image, mask, x1, x2, y1, y2 in tiff_tile_generator(
                    tiff_path=tiff_path,
                    encoding=encoding,
                    tile_size=tile_size,
                    min_overlap=min_overlap,
                    filter_tissue=filter_tissue,
                ):
                    count += 1
                    if count % 100 == 0:  # I am impatient and require this for my sanity
                        print(f'{count}', end=' ' if count % 1000 != 0 else '\n')

                    indices = np.asarray((x1, x2, y1, y2), dtype=np.uint64)
                    # serialize and write to tfrec
                    if mask.sum() > 0:
                        glom_count += 1
                        glom_tf_writer.write(serialize_example(
                            image=image.tobytes(),
                            mask=mask.tobytes(),
                            indices=indices.tobytes(),
                        ))
                    else:
                        tf_writer.write(serialize_example(
                            image=image.tobytes(),
                            mask=mask.tobytes(),
                            indices=indices.tobytes(),
                        ))
            print(count)

        new_tfrec_path = os.path.join(utils.TF_TRAIN_DIR, f'{tfrec_name}-{count - glom_count}.tfrec')
        os.rename(tfrec_path, new_tfrec_path)

        new_tfrec_glom_path = os.path.join(utils.TF_TRAIN_DIR, f'{tfrec_glom_name}-{glom_count}.tfrec')
        os.rename(tfrec_glom_path, new_tfrec_glom_path)
    return


if __name__ == '__main__':
    create_tf_records()
    # _datagen = JSData(
    #     mode='train',
    #     resp=0,
    #     batch_size=32,
    #     noisy=True
    # )
    # for _i in range(5):
    #     _v1, _v2 = _datagen[_i]
    #     print(dict(Counter(_v2['classifier'])))
