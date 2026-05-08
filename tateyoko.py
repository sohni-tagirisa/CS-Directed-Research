#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image
from imageio.v3 import imread
from skimage.draw import line, rectangle_perimeter
from skimage.color import rgb2gray
from skimage.measure import label as skimage_label, regionprops
from skimage.morphology import flood
from skimage.util import invert

RNG = Random(8675309)

Line = tuple[tuple[float, float], tuple[float, float]]

STATE = {
    'filepath': Path(),
    'step': 0,
    'time': datetime.now(),
}


def reset_state():
    STATE['filepath'] = Path()
    STATE['step'] = 0
    STATE['time'] = datetime.now()


def check_time(message=''):
    """Print out the current and elapsed time."""
    prev_time = STATE['time']
    curr_time = datetime.now()
    if message:
        print(f'{prev_time.isoformat()} (+{(curr_time - prev_time).seconds}); {message}')
    else:
        print(f'{prev_time.isoformat()} (+{(curr_time - prev_time).seconds})')
    STATE['time'] = curr_time


def save_image(array, filename=None):
    """Save the array as an image, with an auto-incremented filename."""
    image = Image.fromarray(array)
    if filename is None:
        filename = f'{STATE["filepath"].stem}-step{STATE["step"]:02d}.png'
    image.save(filename)
    STATE['step'] += 1


def crop(array):
    """Crop to the contents of the page."""
    # assume the topleft pixel is the background and flood it
    flood_mask = flood(array, (0, 0))
    # label all regions that were not flooded (skimage_label() considers 0 as background by default)
    flooded = (invert(flood_mask) * np.ones(array.shape) * 255).astype(np.uint8)
    labels = skimage_label(flooded)
    # find the largest region
    largest_region = max(
        regionprops(labels),
        key=(lambda region: region.area),
    )
    # crop
    min_row, min_col, max_row, max_col = largest_region.bbox
    return array[min_row:max_row, min_col:max_col]


def identify_characters_borders(array):
    """Identify character and border (and artifact) regions."""
    character_regions = {}
    border_regions = {}
    min_dimension = min(array.shape[0], array.shape[1]) // 100
    max_dimension = min(array.shape[0], array.shape[1]) // 20
    labels = skimage_label(array)
    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        width = max_col - min_col # the width of the region
        height = max_row - min_row # the height of the region
        if width < min_dimension and height < min_dimension:
            # discard small image artifacts
            continue
        is_character = (
            width < max_dimension # width less than 1/20 of the image
            and height < max_dimension # height less than 1/20 of the image
            and (width / height) < 10 # width:height ratio less than 10
            and (height / width) < 10 # height:width ratio less than 10
            # FIXME add density (within bounding box) requirement?
        )
        if is_character:
            character_regions[region.label] = region
        else:
            border_regions[region.label] = region
    return labels, character_regions, border_regions


def visualize(*mask_colors, background=None):
    """Visualize masks on a background.

    Colors are always integer RGB tuples.

    Parameters:
        mask_colors (list[array, color]): A list of masks and their colors.
        background (array | color | None): The background image to draw on.
    """
    # create the background
    if isinstance(background, np.ndarray):
        # if it's an image, use it
        result = background.astype(np.uint8)
    else:
        if background is None:
            background = (0, 0, 0)
        # if not, create it by first figuring out the size of the masks
        height = max(mask.shape[0] for mask, _ in mask_colors)
        width = max(mask.shape[1] for mask, _ in mask_colors)
        # figure out the color as a tuple of integers
        result = (
            np.repeat(
                background,
                height * width,
            ).reshape(
                (height, width, 3),
                order='F',
            ).astype(np.uint8)
        )
    # apply each mask
    for mask, color in mask_colors:
        result = np.ma.masked_array(
            result,
            np.repeat(mask, result.shape[2]).reshape(result.shape),
            fill_value=color,
        ).filled()
    # save the image
    save_image(result)
    return result


def sum_dimension(array, dimension):
    """Sum an array along the rows or the columns."""
    if dimension == 'row':
        return np.sum(array, axis=1)
    elif dimension == 'col':
        return np.sum(array, axis=0)
    else:
        raise ValueError(' '.join([
            'dimension should be either "row" or "col"',
            f'but got "{dimension}"',
        ]))


def find_basins(array, dimension, threshold):
    # count the pixels along the dimension
    pixel_counts = sum_dimension(array, dimension)
    character_ratios = pixel_counts / len(pixel_counts)
    valleys = np.nonzero(character_ratios < threshold)[0]
    basins = [] # type: list[tuple[int, int]]
    first_value = None
    prev_value = None
    # determine the start and end of basins
    for value in valleys:
        if first_value is None:
            first_value = value
        elif value > prev_value + 1:
            basins.append((first_value, prev_value))
            first_value = None
        prev_value = value
    basins.append((first_value, value))
    return basins


def create_basins_mask(character_mask, row_basins, col_basins):
    mask = np.zeros(character_mask.shape).astype(np.uint8)
    for min_row, max_row in row_basins:
        for min_col, max_col in col_basins:
            mask[min_row:max_row, min_col:max_col] = 1
            mask[min_row, :] = 1
            mask[max_row, :] = 1
            mask[:, min_col] = 1
            mask[:, max_col] = 1
    return mask


def hash_grid_radius_offsets(max_radius):
    """Generate the offsets for each radius away."""
    yield 0, [(0, 0)]
    for radius in range(1, max_radius):
        radius_keys = [
            (-radius, -radius),
            (-radius, radius),
            (radius, -radius),
            (radius, radius),
        ]
        for dim1_diff in range(-radius + 1, radius):
            for dim2_diff in (-radius, radius):
                radius_keys.append((dim1_diff, dim2_diff))
                radius_keys.append((dim2_diff, dim1_diff))
        yield radius, radius_keys


def centroid_crosses_border(centroid1, centroid2, no_mans_coords):
    return any(
        (coord in no_mans_coords) for coord
        in zip(*line(*(round(x) for x in (*centroid1, *centroid2))))
    )


def k_nearest_neighbors_hash(regions, k, grid_size, no_mans_coords):
    """Find the k nearest neighbors for each region.

    This implementation of kNN uses a hash grid to avoid unnecessary distance
    calculations. A hash grid assigns every point to a larger grid cell - for
    example, if the grid size is 10, the points (5, 13) and (6, 16) would both
    be assigned to grid cell (0, 10), while (57, 34) would be assigned to (50,
    30). To find the k nearest neighbors for a point, start by only checking
    distances to other points in the same grid cell, then to points one cell
    away, then two away, etc., until k neighbors are found. This avoids the need
    to check against points far away, in turn meaning that it is more efficient
    than the naive O(n^2) approach of checking every point against every other
    point. Empirically, for ~1300 points, this implementation is ~30% faster
    than the naive algorithm.

    Because grid cells are square, and because a point can be anywhere within a
    cell, care must be taken when determining whether neighbors are actually the
    nearest ones. (1, 1) and (19, 19) are one cell away (squared distance: 648),
    but are further apart than (1, 1) and (1, 21) despite the latter being two
    cells away (squared distance: 400). More generally, cells that are a radius
    `r` away could contain points `(r-1)*grid_size` to `sqrt(2)*r*grid_size`
    apart. The algorithm therefore only considers points nearer than the lower
    bound as confirmed, while keeping track of the further away points to be
    added later.
    """
    # initialize the hash grid by putting each region in the appropriate grid cell
    keys = []
    hash_grid = defaultdict(list)
    all_nearest_neighbors = {} # type: dict[int, list[int]]
    for region in regions:
        key = (region.centroid[0] // grid_size, region.centroid[1] // grid_size)
        keys.append(key)
        hash_grid[key].append(region)
        all_nearest_neighbors[region.label] = []
    # initialize result variables
    distance_cache = {}
    max_radius = 3
    max_distance = max_radius * max_radius * grid_size * grid_size
    # loop over each region to look for its nearest neighbors
    for this_key, this_region in zip(keys, regions):
        this_centroid = this_region.centroid
        away_neighbors = []
        near_neighbors = []
        for radius, offsets in hash_grid_radius_offsets(max_radius):
            # pre-calculate the maximum distance we will consider, accounting for grid squareness
            radius_distance = radius * radius * grid_size * grid_size
            # loop over the grid cells in the larger radius
            for offset in offsets:
                that_key = (this_key[0] + offset[0], this_key[1] + offset[1])
                # loop over the regions in that grid cell
                for that_region in hash_grid[that_key]:
                    # skip over the original region
                    if that_region.label == this_region.label:
                        continue
                    that_centroid = that_region.centroid
                    # retrieve or calculate the distance between regions
                    distance_key = tuple(sorted([this_centroid, that_centroid]))
                    if distance_key not in distance_cache:
                        dx = this_centroid[0] - that_centroid[0]
                        dy = this_centroid[1] - that_centroid[1]
                        distance_cache[distance_key] = dx * dx + dy * dy
                    distance = distance_cache[distance_key]
                    if distance < max_distance:
                        away_neighbors.append((distance, that_region))
            # add regions that were too far away but are now eligible
            new_away_neighbors = []
            for distance, that_region in away_neighbors:
                if distance > radius_distance:
                    new_away_neighbors.append((distance, that_region))
                    continue
                # check if connecting the centroids will cross no man's land
                if not centroid_crosses_border(this_region.centroid, that_region.centroid, no_mans_coords):
                    near_neighbors.append((distance, that_region))
            away_neighbors = new_away_neighbors
            # if there are enough near neighbors, this region is done
            if len(near_neighbors) >= k:
                break
        all_nearest_neighbors[this_region.label] = [
            that_region.label for _, that_region in sorted(near_neighbors)[:k]
        ]
        result = all_nearest_neighbors[this_region.label]
        assert len(result) == len(set(result))
    # return the list of nearest neighbors
    return all_nearest_neighbors


def k_nearest_neighbors_naive(regions, k, _, no_mans_land):
    no_mans_coords = set(zip(*np.nonzero(no_mans_land)))
    regions_dict = {region.label: region for region in regions}
    region_labels = sorted(regions_dict.keys())
    all_nearest_neighbors = {}
    for i, region_label1 in enumerate(region_labels):
        region1 = regions_dict[region_label1]
        centroid1 = region1.centroid
        neighbors = []
        for region_label2 in region_labels[i+1:]:
            region2 = regions_dict[region_label2]
            centroid2 = region2.centroid
            dx = centroid1[0] - centroid2[0]
            dy = centroid1[1] - centroid2[1]
            distance = dx * dx + dy * dy
            if not centroid_crosses_border(centroid1, centroid2, no_mans_coords):
                neighbors.append((distance, region2))
        all_nearest_neighbors[region_label1] = [
            region2.label for _, region2 in sorted(neighbors)[:k]
        ]
    return all_nearest_neighbors


def find(union_find, i):
    path = set()
    rep = i
    while union_find[rep] != rep:
        path.add(rep)
        rep = union_find[rep]
    for node in path:
        union_find[node] = rep
    return rep


def union(union_find, i, j):
    path = set()
    rep = i
    while union_find[rep] != rep:
        path.add(rep)
        rep = union_find[rep]
    path.add(rep)
    rep = j
    while union_find[rep] != rep:
        path.add(rep)
        rep = union_find[rep]
    path.add(rep)
    for node in path:
        union_find[node] = rep
    return rep


def find_connected_components(neighbors):
    # use union-find to identify connected components
    union_find = {label: label for label in neighbors}
    for label, nearest_neighbors in neighbors.items():
        for neighbor in nearest_neighbors:
            union(union_find, label, neighbor)
    # extract out the connected components
    components = defaultdict(set)
    for label in neighbors:
        rep = find(union_find, label)
        components[rep].add(label)
    return list(components.values())


def visualize_components(regions, labels, components):
    # initialize the image to all black
    array = np.zeros((*labels.shape, 3)).astype(np.uint8)
    # for each component
    for component in components:
        # randomly pick a color for this component
        rgb = (
            RNG.randrange(128, 255),
            RNG.randrange(128, 255),
            RNG.randrange(128, 255),
        )
        # color the regions
        region_mask = np.isin(
            labels,
            [regions[region_index].label for region_index in component],
        )
        array = np.ma.masked_array(
            array,
            np.repeat(region_mask, array.shape[2]).reshape(array.shape),
            fill_value=rgb,
        ).filled()
        # color the bounding box
        min_rows, min_cols, max_rows, max_cols = zip(*(
            regions[region_index].bbox for region_index in component
        ))
        perimeter_mask = rectangle_perimeter(
            start=(min(min_rows), min(min_cols)),
            end=(max(max_rows), max(max_cols)),
            shape=array.shape,
            clip=True,
        )
        array[perimeter_mask] = rgb
    save_image(array)
    return array


def pipeline(path, args):
    reset_state()
    STATE['filepath'] = path
    # read the image
    array = imread(path)
    check_time('read in the image')
    save_image(array)
    # convert to black-and-white
    array = (rgb2gray(array) * 255 > 127) * np.ones(array.shape[:2])
    array = (array * 255).astype(np.uint8)
    # crop to just the page
    array = crop(array)
    check_time('cropped the image')
    save_image(array)
    # separate characters from borders
    array = invert(array)
    labels, character_regions, border_regions = identify_characters_borders(array)
    character_mask = np.isin(labels, list(character_regions.keys()))
    visualize(
        (
            np.isin(labels, list(border_regions.keys())),
            (255, 255, 255),
        ),
        (character_mask, (0, 255, 0)),
    )
    check_time('visualized characters and borders')
    # find rows and columns where there are no characters
    # the character mask has a 1 where there are characters and 0 where there aren't
    row_basins = find_basins(character_mask, 'row', args.border_ratio_threshold)
    col_basins = find_basins(character_mask, 'col', args.border_ratio_threshold)
    visualize(
        (np.isin(labels, list(border_regions.keys())), (255, 255, 255)),
        (character_mask, (0, 255, 0)),
        (
            create_basins_mask(character_mask, row_basins, col_basins),
            (255, 0, 0),
        ),
    )
    check_time('visualized non-character gaps')
    return # FIXME
    # find nearest neighbors and visualize
    border_mask = np.zeros(labels.shape).astype(bool)
    border_mask[np.isin(labels, list(border_regions.keys()))] = True
    border_coords = set(zip(*np.nonzero(border_mask)))
    nearest_neighbors = k_nearest_neighbors_hash(
        character_regions.values(),
        args.k,
        min(array.shape[0], array.shape[1]) // 20,
        border_coords,
    )
    check_time('found the k nearest neighbors')
    components = find_connected_components(nearest_neighbors)
    check_time('found the connected components')
    visualize_components(character_regions, labels, components)
    check_time('visualized connected components')


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('images', metavar='image', type=Path, nargs='+')
    arg_parser.add_argument('-k', default=3, type=int)
    arg_parser.add_argument('--border-ratio-threshold', default=0.05, type=float)
    args = arg_parser.parse_args()
    args.images = sorted(set(path.expanduser().resolve() for path in args.images))
    for image_path in args.images:
        pipeline(image_path, args)


if __name__ == '__main__':
    main()