import os
from glob import glob
from typing import List
from typing import Tuple

import cv2
import numpy
import numpy as np
import pandas
import rasterio
import tensorflow as tf
from rasterio.windows import Window
from tensorflow import keras

import utils


def encoding_to_mask(encoding: str, shape: Tuple[int, int]) -> numpy.ndarray:
    mask = numpy.zeros(shape[0] * shape[1], dtype=numpy.uint8)
    s = encoding.split()
    for i in range(len(s) // 2):
        start = int(s[2 * i]) - 1
        length = int(s[2 * i + 1])
        mask[start:start + length] = 1
    return mask.reshape(shape).T


def mask_to_encoding(mask: numpy.ndarray):
    pixels = mask.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = numpy.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def _make_grid(shape: Tuple[int, int], tile_size: int = None, min_overlap: int = None) -> numpy.ndarray:
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
        starts = numpy.linspace(0, length, num=num, endpoint=False, dtype=numpy.int64)
        starts[-1] = length - tile_size
        ends = (starts + tile_size).clip(0, length)
        return num, starts, ends

    num_x, x1, x2 = tiles(shape[0])
    num_y, y1, y2 = tiles(shape[1])
    grid = numpy.zeros((num_x, num_y, 4), dtype=numpy.uint64)

    for i in range(num_x):
        for j in range(num_y):
            grid[i, j] = x1[i], x2[i], y1[j], y2[j]

    return grid.reshape(num_x * num_y, 4)


def _get_tile(tile_size, raster, layers, x1, x2, y1, y2, *, normalize: bool = False):
    dtype = numpy.float16 if normalize else numpy.uint8
    image = numpy.zeros((tile_size, tile_size, 3), dtype=dtype)

    window = Window.from_slices((x1, x2), (y1, y2))
    if raster.count == 3:
        image[:, :, :] = numpy.moveaxis(raster.read([1, 2, 3], window=window), 0, -1)
    else:
        for channel in range(3):
            image[:, :, channel] = layers[channel].read(window=window)

    return np.asarray(image / 255, dtype=np.float32) if normalize else image


def _filter_tissue(image) -> bool:
    saturation = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[1]

    sat_sum = numpy.asarray((saturation > utils.GLOBALS['s_threshold'])).sum()
    all_black = sat_sum <= utils.GLOBALS['p_threshold']
    all_gray = (image.sum() <= utils.GLOBALS['p_threshold'])
    return all_black or all_gray


def _open_tiff(path: str):
    raster = rasterio.open(path, transform=rasterio.Affine(1, 0, 0, 0, 1, 0))
    layers = None if raster.count == 3 else [rasterio.open(sub_dataset) for sub_dataset in raster.subdatasets]
    return raster, layers


def create_tiles_df(tile_size: int = None, min_overlap: int = None, filter_tissue: bool = True):
    tile_size = utils.GLOBALS['tile_size'] if tile_size is None else tile_size
    min_overlap = utils.GLOBALS['min_overlap'] if min_overlap is None else min_overlap

    train_df = pandas.read_csv(utils.TRAIN_PATH).set_index('id')
    columns = ['id', 'x1', 'x2', 'y1', 'y2', 'encoding']
    tiles_df = pandas.DataFrame(columns=columns)

    i = 0
    for name, encoding in train_df.iterrows():
        i += 1
        print(f'{i:2d} {name} ', end='')

        raster, layers = _open_tiff(os.path.join(utils.TRAIN_DIR, f'{name}.tiff'))
        full_mask = encoding_to_mask(encoding.values[0], (raster.shape[1], raster.shape[0]))
        slices = _make_grid(raster.shape, tile_size, min_overlap)
        print(f'num_slices {slices.shape[0]}')
        count = 0
        for x1, x2, y1, y2 in slices:
            image = _get_tile(tile_size, raster, layers, x1, x2, y1, y2)
            if filter_tissue and _filter_tissue(image):
                continue

            mask = full_mask[x1:x2, y1:y2]
            values = [name, x1, x2, y1, y2, mask_to_encoding(mask)]
            tiles_df = tiles_df.append(dict(zip(columns, values)), ignore_index=True)

            count += 1
            if count % 100 == 0:
                end = '\n' if count % 1000 == 0 else ' '
                print(f'{count:4d}', end=end)
        print(f'{count:4d}')

    path = f'{utils.TILES_PATH}-{tile_size}-{min_overlap}.csv.gz'
    tiles_df.to_csv(path, index=False)
    return tiles_df


def _test_encoding_conversion():
    train_df = pandas.read_csv(utils.TRAIN_PATH).set_index('id')
    i = 0
    for name, encoding in train_df.iterrows():
        i += 1
        print(f'{i:2d} {name}')
        raster = rasterio.open(
            os.path.join(utils.TRAIN_DIR, f'{name}.tiff'),
            transform=rasterio.Affine(1, 0, 0, 0, 1, 0),
        )
        rle = encoding.values[0]
        mask = encoding_to_mask(rle, (raster.shape[1], raster.shape[0]))
        converted = mask_to_encoding(mask)
        if rle != converted:
            raise ValueError(f'mismatch in encoding with file {name}')
    return


class TrainSequence(keras.utils.Sequence):
    def __init__(
            self,
            file_ids: List[str],
            batch_size: int = None,
            tile_size: int = None,
            min_overlap: int = None,
    ):
        if len(file_ids) == 0:
            raise ValueError('no files given')

        batch_size = utils.GLOBALS['batch_size'] if batch_size is None else batch_size
        if batch_size % 2 != 0:
            raise ValueError(f'Batch Size must be even. Got {batch_size} instead.')
        self.half_batch_size = batch_size // 2

        self.tile_size = utils.GLOBALS['tile_size'] if tile_size is None else tile_size
        self.min_overlap = utils.GLOBALS['min_overlap'] if min_overlap is None else min_overlap

        file_path = f'{utils.TILES_PATH}-{self.tile_size}-{self.min_overlap}.csv.gz'
        self.tiles_df = pandas.read_csv(file_path).fillna('')

        if not set(file_ids) <= set(self.tiles_df['id']):
            raise ValueError(f'some files not found in tiles_df:\n{file_ids}')

        self.rasters = {
            file_id: _open_tiff(os.path.join(utils.TRAIN_DIR, f'{file_id}.tiff'))
            for file_id in file_ids
        }

        self.tiles_df = self.tiles_df[self.tiles_df['id'].isin(file_ids)].reset_index(drop=True)
        self.glom_indices = list(self.tiles_df[self.tiles_df['encoding'] != ''].index)
        self.blank_indices = list(self.tiles_df[self.tiles_df['encoding'] == ''].index)

        self.num_batches = min(len(self.glom_indices), len(self.blank_indices)) // self.half_batch_size
        self.batch_indices = numpy.zeros(shape=(self.num_batches, self.half_batch_size * 2), dtype=numpy.uint64)
        self.on_epoch_end()

    def on_epoch_end(self):
        numpy.random.shuffle(self.glom_indices)
        numpy.random.shuffle(self.blank_indices)

        for i in range(self.num_batches):
            start, end = i * self.half_batch_size, (i + 1) * self.half_batch_size
            self.batch_indices[i, :self.half_batch_size] = self.glom_indices[start:end]
            self.batch_indices[i, self.half_batch_size:] = self.blank_indices[start:end]
        return

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index: int):
        return self.get_batch(self.batch_indices[index])

    def get_batch(self, indices):
        images = numpy.zeros(shape=(self.half_batch_size * 2, self.tile_size, self.tile_size, 3), dtype=numpy.float32)
        masks = numpy.zeros(shape=(self.half_batch_size * 2, self.tile_size, self.tile_size), dtype=numpy.uint8)

        for i, b in enumerate(indices):
            file_id, x1, x2, y1, y2, encoding = list(self.tiles_df.iloc[b])
            raster, layers = self.rasters[file_id]
            image = _get_tile(self.tile_size, raster, layers, x1, x2, y1, y2, normalize=True)
            mask = encoding_to_mask(encoding, (self.tile_size, self.tile_size))

            flip = numpy.random.uniform()
            if flip < 0.25:
                image = numpy.flipud(image)
                mask = numpy.flipud(mask)
            elif 0.25 <= flip < 0.5:
                image = numpy.fliplr(image)
                mask = numpy.fliplr(mask)
            elif 0.5 <= flip < 0.75:
                image = numpy.rot90(image)
                mask = numpy.rot90(mask)
            else:
                image = numpy.rot90(image, k=3)
                mask = numpy.rot90(mask, k=3)

            if numpy.random.uniform() < 0.25:
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv_image)
                s = np.clip(s * 1.5, 0, 1)
                hsv_image = cv2.merge([h, s, v])
                image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            images[i] = image
            masks[i] = mask

        images = tf.convert_to_tensor(images, dtype=tf.float32)
        masks = tf.convert_to_tensor(masks, dtype=tf.uint8)
        return images, masks


if __name__ == '__main__':
    # _test_encoding_conversion()
    # _tiles_df = create_tiles_df()
    # print(_tiles_df.shape)
    # print(_tiles_df.head())
    _file_ids = [_name.split('/')[-1].split('.')[0] for _name in glob(f'{utils.TRAIN_DIR}/*.tiff')]
    _train_gen = TrainSequence(_file_ids[:1])
    print(len(_train_gen.glom_indices), len(_train_gen.blank_indices))
    print(len(_train_gen))
    _images, _mask = _train_gen[len(_train_gen) - 1]
    print(tf.shape(_images), tf.shape(_mask))
