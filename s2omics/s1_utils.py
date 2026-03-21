import itertools
from PIL import Image
import tifffile
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import torch
import random


Image.MAX_IMAGE_PIXELS = None
MAX_SAVEFIG_DIMENSION = 65000
MAX_SAVEFIG_PIXELS = 100000000
MAX_JPEG_DIMENSION = 65000


def _is_tiff(filename):
    lower = filename.lower()
    return lower.endswith('.tif') or lower.endswith('.tiff') or lower.endswith('.ome.tif')

def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename, verbose=True):
    if _is_tiff(filename):
        img = tifffile.imread(filename)
        return img
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img

def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask


def get_image_filename(prefix):
    for suffix in ['.tiff', '.tif', '.ome.tif', '.png', '.svs']:
        filename = prefix + suffix
        if os.path.exists(filename):
            return filename
    raise FileNotFoundError(f'Image not found for prefix: {prefix}')


def save_image(img, filename):
    mkdir(filename)
    if _is_tiff(filename.lower()):
        tifffile.imwrite(filename, img)
    else:
        Image.fromarray(img).save(filename)
    print(filename)


def save_figure_safely(fig, filename, dpi=300, format=None,
                       max_dimension=MAX_SAVEFIG_DIMENSION,
                       max_pixels=MAX_SAVEFIG_PIXELS,
                       dimension_margin=0.95,
                       prefer_tiff_when_rescaled=True,
                       **savefig_kwargs):
    mkdir(filename)
    figure_width, figure_height = fig.get_size_inches()
    figure_width = max(float(figure_width), 1.0)
    figure_height = max(float(figure_height), 1.0)
    max_inches = max(figure_width, figure_height)
    area_inches = figure_width * figure_height
    safe_dimension = max(int(max_dimension * dimension_margin), 1)
    safe_pixels = max(int(max_pixels * dimension_margin), 1)
    dimension_limited_dpi = max(int(safe_dimension // max_inches), 1)
    pixel_limited_dpi = max(int(np.sqrt(safe_pixels / area_inches)), 1)
    safe_dpi = min(int(dpi), dimension_limited_dpi, pixel_limited_dpi)

    output_filename = filename
    output_format = format or os.path.splitext(filename)[1].lstrip('.')
    if output_format == '':
        output_format = None

    if output_format is not None:
        output_format = output_format.lower()
    projected_width = int(np.ceil(figure_width * safe_dpi))
    projected_height = int(np.ceil(figure_height * safe_dpi))
    if (
        prefer_tiff_when_rescaled
        and safe_dpi < int(dpi)
        and output_format in {'jpg', 'jpeg'}
        and max(projected_width, projected_height) > MAX_JPEG_DIMENSION
    ):
        output_filename = os.path.splitext(filename)[0] + '.tiff'
        output_format = 'tiff'

    fig.savefig(output_filename, dpi=safe_dpi, format=output_format, **savefig_kwargs)
    return output_filename, safe_dpi


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_string(filename):
    return read_lines(filename)[0]


def write_lines(strings, filename):
    mkdir(filename)
    with open(filename, 'w') as file:
        for s in strings:
            file.write(f'{s}\n')
    print(filename)


def write_string(string, filename):
    return write_lines([string], filename)


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def load_tsv(filename, index=True):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, sep='\t', header=0, index_col=index_col)
    print(f'Dataframe loaded from {filename}')
    return df


def save_tsv(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = '\t'
    x.to_csv(filename, **kwargs)
    print(filename)


def load_yaml(filename, verbose=False):
    with open(filename, 'r') as file:
        content = yaml.safe_load(file)
    if verbose:
        print(f'YAML loaded from {filename}')
    return content


def save_yaml(filename, content):
    with open(filename, 'w') as file:
        yaml.dump(content, file)
    print(file)


def join(x):
    return list(itertools.chain.from_iterable(x))


def get_most_frequent(x):
    # return the most frequent element in array
    uniqs, counts = np.unique(x, return_counts=True)
    return uniqs[counts.argmax()]


def sort_labels(labels, descending=True):
    labels = labels.copy()
    isin = labels >= 0
    labels_uniq, labels[isin], counts = np.unique(
            labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)
    order = c.argsort()
    rank = order.argsort()
    labels[isin] = rank[labels[isin]]
    return labels, labels_uniq[order]