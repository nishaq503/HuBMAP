import os
from typing import Tuple

import cv2
import numpy
import pandas
import rasterio
from rasterio.windows import Window

from model import utils


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


def _get_tile(tile_size, raster, layers, x1, x2, y1, y2):
    window = Window.from_slices((x1, x2), (y1, y2))
    image = numpy.zeros((tile_size, tile_size, 3), dtype=numpy.uint8)
    if raster.count == 3:
        image[:, :, :] = numpy.moveaxis(raster.read([1, 2, 3], window=window), 0, -1)
    else:
        for channel in range(3):
            image[:, :, channel] = layers[channel].read(window=window)
    return image


def _filter_tissue(image) -> bool:
    saturation = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[1]

    sat_sum = numpy.asarray((saturation > utils.GLOBALS['s_threshold'])).sum()
    all_black = sat_sum <= utils.GLOBALS['p_threshold']
    all_gray = (image.sum() <= utils.GLOBALS['p_threshold'])
    return all_black or all_gray


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

        raster = rasterio.open(
            os.path.join(utils.TRAIN_DIR, f'{name}.tiff'),
            transform=rasterio.Affine(1, 0, 0, 0, 1, 0),
        )
        layers = None if raster.count == 3 else [rasterio.open(sub_dataset) for sub_dataset in raster.subdatasets]
        full_mask = encoding_to_mask(
            encoding=encoding.values[0],
            shape=(raster.shape[1], raster.shape[0]),
        )

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


if __name__ == '__main__':
    # _test_encoding_conversion()
    _tiles_df = create_tiles_df()
    print(_tiles_df.shape)
    print(_tiles_df.head())
