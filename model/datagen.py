import os
import shutil
from typing import List
from typing import Optional
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

    indices = tf.reshape(
        tf.io.decode_raw(single_example['indices'], out_type=np.dtype('int64')),
        (4,),
    )

    image = tf.reshape(
        tf.io.decode_raw(single_example['image'], out_type=np.dtype('uint8')),
        (tile_size, tile_size, 3),
    )

    mask = tf.reshape(
        tf.io.decode_raw(single_example['mask'], out_type=np.dtype('uint8')),
        (tile_size, tile_size),
    )
    return image, mask, indices


def load_tfrecords(filenames: List[str], tile_size: int = None) -> tf.data.TFRecordDataset:
    tile_size = utils.GLOBALS['tile_size'] if tile_size is None else tile_size

    # dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(
        lambda example: parse_example(example, tile_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return dataset


def _make_grid(shape: Tuple[int, int], tile_size: int = None, min_overlap: int = None):
    """ Return Array of size (N, 4), where:
            N - number of tiles,
            2nd axis represents slices: x1, x2, y1, y2
    """
    tile_size = utils.GLOBALS['tile_size'] if tile_size is None else tile_size
    min_overlap = utils.GLOBALS['min_overlap'] if min_overlap is None else min_overlap

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


def _filter_tissue(image) -> bool:
    saturation = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[1]

    sat_sum = np.asarray((saturation > utils.GLOBALS['s_threshold'])).sum()
    all_black = sat_sum <= utils.GLOBALS['p_threshold']
    all_gray = (image.sum() <= utils.GLOBALS['p_threshold'])
    return all_black or all_gray


def tiff_tile_generator(
        tiff_path: str,
        encoding: Optional[str],
        tile_size: int = None,
        min_overlap: int = None,
        filter_tissue: bool = True,
):
    tile_size = utils.GLOBALS['tile_size'] if tile_size is None else tile_size
    min_overlap = utils.GLOBALS['min_overlap'] if min_overlap is None else min_overlap

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

        if filter_tissue and _filter_tissue(image):
            continue

        if full_mask is None:
            mask = None
        else:
            mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            mask[:, :] = full_mask[x1:x2, y1:y2]

        yield image, mask, x1, x2, y1, y2


def create_tf_records(tile_size: int = None, min_overlap: int = None, filter_tissue: bool = True):
    tile_size = utils.GLOBALS['tile_size'] if tile_size is None else tile_size
    min_overlap = utils.GLOBALS['min_overlap'] if min_overlap is None else min_overlap

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
        print(f'\n{i + 1:2d} Creating tfrecords for image: {tiff_name}')

        count, glom_count = 0, 0  # to rename tfrec files later

        tfrec_name = f'{tiff_name}'
        tfrec_path = os.path.join(utils.TF_TRAIN_DIR, f'{tfrec_name}.tfrec')

        tfrec_glom_name = f'{tiff_name}-glom'
        tfrec_glom_path = os.path.join(utils.TF_TRAIN_DIR, f'{tfrec_glom_name}.tfrec')

        # options = tf.io.TFRecordOptions('GZIP')  # compression slows down writing but uses only ~60% space
        options = None
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
                    if count % 1000 == 0:  # I am impatient and require this for my sanity
                        print(f'{count:5d}', end=' ' if count % 10000 != 0 else '\n')

                    indices = np.asarray((x1, x2, y1, y2), dtype=np.int64)
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
            print(f'{count:5d}')

        new_tfrec_path = os.path.join(utils.TF_TRAIN_DIR, f'{tfrec_name}-{count - glom_count}.tfrec')
        os.rename(tfrec_path, new_tfrec_path)

        new_tfrec_glom_path = os.path.join(utils.TF_TRAIN_DIR, f'{tfrec_glom_name}-{glom_count}.tfrec')
        os.rename(tfrec_glom_path, new_tfrec_glom_path)
    return


def create_tf_df():
    sizes_path = os.path.join(utils.DATA_DIR, 'sizes.tsv')
    os.system(f'du -hs {utils.TF_TRAIN_DIR}/* > {sizes_path}')
    sizes_df = pd.read_csv(sizes_path, sep='\t', names=['size', 'id'], header=None)
    os.system(f'rm {sizes_path}')

    sizes_df['id'] = sizes_df['id'].apply(lambda s: s.split('.')[0])
    sizes_df['size'] = sizes_df['size'].apply(lambda s: int((1000 if s[-1] == 'G' else 1) * float(s[:-1])))
    sizes_df['glom'] = sizes_df['id'].apply(lambda s: int('glom' in s))
    sizes_df['num_records'] = sizes_df['id'].apply(lambda s: int(s.split('-')[-1]))
    sizes_df['id'] = sizes_df['id'].apply(lambda s: s.split('-')[0].split('/')[-1])
    sizes_df['num_glom_records'] = sizes_df['num_records'] * sizes_df['glom']
    sizes_df['num_records'] = sizes_df['num_records'] * (1 - sizes_df['glom'])
    sizes_df.drop(['glom'], axis='columns', inplace=True)
    sizes_df = sizes_df.groupby('id').sum()

    train_df = pd.concat([pd.read_csv(utils.TRAIN_PATH).set_index('id'), sizes_df], axis=1)
    train_df = train_df[['size', 'num_records', 'num_glom_records', 'encoding']]
    train_df.sort_values(by='size')
    train_df.to_csv(utils.TF_TRAIN_PATH)
    return


class TrainSequence:
    def __init__(self, mode: str, tile_size: int = None, batch_size: int = None):
        batch_size = utils.GLOBALS['batch_size'] if batch_size is None else batch_size
        if batch_size % 2 == 0:
            self.batch_size: int = batch_size
        else:
            raise ValueError(f'Batch size must be a positive even integer. Got {batch_size} instead.')

        self.tile_size: int = utils.GLOBALS['tile_size'] if tile_size is None else tile_size

        train_df = pd.read_csv(utils.TF_TRAIN_PATH).set_index('id')
        split = int(train_df.shape[0] * 0.8)
        if mode == 'train':
            file_ids = train_df.index[:split]
        elif mode == 'validate':
            file_ids = train_df.index[split:]
        else:
            raise ValueError(f'mode must be one of \'train\' or \'validate\'. Got {mode} instead.')

        self.file_ids: List[str] = list(file_ids)
        self.df: pd.DataFrame = train_df.loc[self.file_ids]

        file_names = self.df.index + '-' + self.df['num_records'].apply(str) + '.tfrec'
        self.file_paths = [os.path.join(utils.TF_TRAIN_DIR, name) for name in file_names]
        assert all(map(os.path.exists, self.file_paths))

        file_names = self.df.index + '-glom-' + self.df['num_glom_records'].apply(str) + '.tfrec'
        self.glom_paths = [os.path.join(utils.TF_TRAIN_DIR, name) for name in file_names]
        assert all(map(os.path.exists, self.glom_paths))

        self.num_glom_tiles = self.df['num_glom_records'].sum()
        self.num_batches = self.num_glom_tiles // (self.batch_size // 2)
        if self.num_glom_tiles % (self.batch_size // 2) != 0:
            self.num_batches += 1

        self.tiles_tfrec = load_tfrecords(self.file_paths, tile_size=self.tile_size)
        self.tiles_tfrec: tf.data.TFRecordDataset = self.tiles_tfrec.repeat().batch(self.batch_size // 2)

        self.gloms_tfrec = load_tfrecords(self.glom_paths, tile_size=self.tile_size)
        self.gloms_tfrec: tf.data.TFRecordDataset = self.gloms_tfrec.repeat().batch(self.batch_size // 2)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for val1, val2 in zip(self.tiles_tfrec, self.gloms_tfrec):
            images1, masks1, _ = val1
            images2, masks2, _ = val2

            images = tf.concat([images1, images2], axis=0)
            masks = tf.concat([masks1, masks2], axis=0)

            ys = {
                'embedding': masks,
                'autoencoder': images,
                'mask': masks,
            }

            yield images, ys


# WARNINGS:
# NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs. The identity matrix be returned.
# 2f6ecfcdf
#
# TIFFReadDirectoryCheckOrder:Invalid TIFF directory; tags are not sorted in ascending order
# e79de561c 54f2eec69
#
# Nonstandard tile length/width ###, convert file
# 095bf7a1f 4ef6695ce 26dc41664 c68fe75ea 1e2425f28


if __name__ == '__main__':
    # create_tf_records()
    # create_tf_df()
    print(len(TrainSequence(mode='train')))
    print(len(TrainSequence(mode='validate')))
