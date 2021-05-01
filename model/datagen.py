import os
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from model import utils


def encoding_to_mask(encoding: str, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(np.prod(shape), dtype=np.uint8)
    splits = encoding.split()
    for i in range(0, len(splits), 2):
        start, length = int(splits[i]) - 1, int(splits[i + 1])
        mask[start: start + length] = 1
    return mask.reshape(shape).T


def mask_to_encoding(mask: np.ndarray) -> str:
    pixels = mask.T.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    starts = np.concatenate([[0], runs[1::2]])
    lengths = runs[::2] - starts
    return ' '.join((' '.join(map(str, v)) for v in zip(starts + 1, lengths)))


def _make_grid(shape: Tuple[int, int], tile_size: int = None, min_overlap: int = None) -> np.ndarray:
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

    raster = rasterio.open(tiff_path, transform=rasterio.Affine(1, 0, 0, 0, 1, 0))
    layers = None if raster.count == 3 else [rasterio.open(sub_dataset) for sub_dataset in raster.subdatasets]

    full_mask: Optional[np.ndarray] = None if encoding is None else encoding_to_mask(
        encoding=encoding,
        shape=(raster.shape[1], raster.shape[0]),
    )
    slices = _make_grid(raster.shape, tile_size, min_overlap)

    for x1, x2, y1, y2 in slices:
        window = Window.from_slices((x1, x2), (y1, y2))
        image = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        if raster.count == 3:
            image[:, :, :] = np.moveaxis(raster.read([1, 2, 3], window=window), 0, -1)
        else:
            for channel in range(3):
                image[:, :, channel] = layers[channel].read(window=window)

        if filter_tissue and _filter_tissue(image):
            continue

        image = np.asarray(image, dtype=np.float32) / 255.0

        if full_mask is None:
            mask = None
        else:
            mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            mask[:, :] = full_mask[x1:x2, y1:y2]

        yield image, mask, x1, x2, y1, y2


def create_tiles_df():
    pass


if __name__ == '__main__':
    _train_df = pd.read_csv(utils.TRAIN_PATH).set_index('id')
    _tiff_paths = [os.path.join(utils.TRAIN_DIR, f'{_id}.tiff') for _id in _train_df.index]
    _encodings = list(_train_df['encoding'])
    list(map(lambda _pe: tiff_tile_generator(_pe[0], _pe[1]), zip(_tiff_paths, _encodings)))
