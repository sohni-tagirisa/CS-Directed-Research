#!/usr/bin/env python3
"""
visualization.py

What gets drawn:
  - Blue lines: the raw (pre-shrink) gap boundaries detected by find_basins
  - Pink lines: the shrunk gap boundaries from the ShrunkGraph
  - Green shading: original basin bands + filled intersection square per node
  - Dark-pink squares: the shrunk intersection rectangle for each node,
    showing exactly how the shrink algorithm moved the boundary
"""

import sys
import numpy as np
import cv2
from skimage.util import invert
from skimage.measure import label as skimage_label, regionprops
from skimage.morphology import flood

# BorderGraph and its helpers live in subvalley.py
from subvalley import find_basins, basin_center, BorderGraph



# image loading & preprocessing

def load_and_preprocess(image_path):
    """
    Load an image from disk and convert it to a binary ink mask.

    Returns:
        img_bgr  : the original colour image (H x W x 3, uint8, BGR order)
                   kept so we can draw coloured overlays on it later
        inverted : binary array where 1 = ink pixel, 0 = background
                   this is the convention used throughout — ink is "foreground"
    """
 
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"could not load: {image_path}")

    # convert to grayscale: each pixel becomes a single 0-255 brightness value
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # threshold at 127: pixels brighter than 127 become 1 (white/background),
    # darker pixels become 0 (ink). astype(np.uint8) ensures values are 0 or 1.
    binary = (gray > 127).astype(np.uint8)  # white=1, black/ink=0

    # flip convention: now ink=1, background=0
    # this makes summing pixels equivalent to counting ink
    inverted = (1 - binary).astype(np.uint8)
    return img_bgr, inverted

# cropping

def crop_to_content(array):
    """
    Remove the blank border around the page content.

    Strategy: flood-fill from the top-left corner (which is assumed to be
    background), then find the largest connected region that was NOT flooded
    — that region is the page content.

    Returns:
        cropped: the sub-array containing just the page
        r0, c0: the row and column offsets of the crop within the original
                     image — needed later to convert cropped coordinates back
                     to original-image coordinates when drawing
    """
    # flood fill starting at pixel (0,0), marks all background pixels reachable
    # from the top-left corner. this removes the white margin around the page.
    flood_mask = flood(array, (0, 0))

    # invert the flood mask so the non-background pixels are 255, then label
    # each connected region with a unique integer id
    flooded = (invert(flood_mask) * 255).astype(np.uint8)
    labels  = skimage_label(flooded)
    regions = regionprops(labels)

    if not regions:
        # nothing found, return the array unchanged with zero offsets
        return array, 0, 0

    # the largest region by pixel area is the page content
    largest     = max(regions, key=lambda r: r.area)
    r0, c0, r1, c1 = largest.bbox # bounding box of the page

    # slice out just the page, and remember the offset so we can un-crop later
    return array[r0:r1, c0:c1], r0, c0


# character vs. border classification

def identify_characters_borders(array):
    """
    Classify every connected ink region as either a character or a border line.

    Characters are small, roughly square blobs (individual kanji / kana).
    Borders are long thin regions (the printed grid lines framing the columns).

    The size thresholds are expressed as fractions of the shorter image dimension
    so the logic scales gracefully across images of different resolutions.

    Returns:
        labels: integer label array- every pixel has the id of its region
        character_regions: dict { label_id -> regionprops object } for characters
        border_regions: dict { label_id -> regionprops object } for borders
    """
    character_regions, border_regions = {}, {}

    # 1/100 of the shorter dimension: anything smaller than this is noise / dust
    min_dimension = min(array.shape[0], array.shape[1]) // 100
    # 1/20 of the shorter dimension: anything larger is probably a border line
    max_dimension = min(array.shape[0], array.shape[1]) // 20

    # label every connected component; skimage_label assigns a unique integer
    # to each group of touching non-zero pixels
    labels = skimage_label(array)

    for region in regionprops(labels):
        r0, c0, r1, c1 = region.bbox
        w = c1 - c0  # width of the bounding box
        h = r1 - r0  # height of the bounding box

        # skip specks that are too small to be meaningful
        if w < min_dimension and h < min_dimension:
            continue

        # a character must be:
        # - smaller than max_dimension in both directions (not a long border line)
        # - aspect ratio between 1:5 and 5:1 
        is_character = (
            w < max_dimension and h < max_dimension
            and (w / h) < 5 and (h / w) < 5
        )

        if is_character:
            character_regions[region.label] = region
        else:
            border_regions[region.label] = region

    return labels, character_regions, border_regions



# Character mask

def make_character_mask(labels, character_regions):
    """
    Build a binary mask that is 1 wherever there is a character pixel, 0 elsewhere.

    This is what find_basins operates on, we want to find gaps between
    characters, so we exclude border lines and only look at character ink.
    """
    mask = np.zeros(labels.shape, dtype=np.uint8)
    # set every pixel whose label belongs to a character region to 1
    mask[np.isin(labels, list(character_regions.keys()))] = 1
    return mask

# Main visualization function

def visualize_graph(img_bgr, graph: BorderGraph, cropped,
                    row_offset, col_offset,
                    output_path="output_visualization.png"):
    """
 Parameters:
        img_bgr: original colour image to draw on
        graph: the BorderGraph built from the character mask
        cropped: the cropped binary array (needed for its shape)
        row_offset: how many rows were removed from the top during cropping
        col_offset: how many columns were removed from the left during cropping
        output_path: where to save the result
    """

    collapsed = 0 # edges where x1_shrunk >= x2_shrunk (gap completely closed)
    no_change = 0 # edges where neither boundary moved at all
    shrunk_ok = 0 # edges where at least one boundary moved inward

    for edge in graph.edges:
        delta1 = abs(edge.x1_shrunk - edge.x1)
        delta2 = abs(edge.x2_shrunk - edge.x2)
        if edge.x1_shrunk >= edge.x2_shrunk:
            collapsed += 1
        elif delta1 == 0 and delta2 == 0:
            no_change += 1
        else:
            shrunk_ok += 1

    print(f"\nShrink Diagnostic")
    print(f"  total edges: {len(graph.edges)}")
    print(f"  shrunk (moved): {shrunk_ok}")
    print(f"  no change: {no_change}")
    print(f"  collapsed (x1>=x2): {collapsed}")


    print(f"\n  Sample edges (first 10 with actual shrink):")
    count = 0
    for edge in graph.edges:
        d1 = edge.x1_shrunk - edge.x1
        d2 = edge.x2 - edge.x2_shrunk
        if d1 != 0 or d2 != 0:
            print(f"    {edge.orientation} edge: x1={edge.x1}->{edge.x1_shrunk} (+{d1}), "
                  f"x2={edge.x2}->{edge.x2_shrunk} (-{d2}), "
                  f"span {edge.x2-edge.x1}->{edge.x2_shrunk-edge.x1_shrunk}")
            count += 1
            if count >= 10:
                break

    """Coordinate helpers 
    All positions in graph / cropped coordinates start from (0,0) at the #top-left of the cropped image. To draw on the original full image we
    must add back the row_offset and col_offset from the crop step. We also clamp to the image bounds to avoid cv2 crashing on out-of-range pixels. """

    out = img_bgr.copy()   # work on a copy so the original is untouched
    h, w = out.shape[:2]   # full image height and width

    def ry(r):
        """Convert a cropped row coordinate to a full-image row, clamped to [0, h-1]."""
        return max(0, min(int(r) + row_offset, h - 1))

    def cx(c):
        """Convert a cropped col coordinate to a full-image col, clamped to [0, w-1]."""
        return max(0, min(int(c) + col_offset, w - 1))

    """ Layer 1: Blue lines — raw (pre-shrink) gap boundaries
    Each GapEdge stores x1 and x2: the two boundaries of the gap it represents.
    For a horizontal edge (gap runs left-right between two col basins):
        x1 = rightmost col of the left basin  (left wall of the gap)
        x2 = leftmost  col of the right basin (right wall of the gap)
    For a vertical edge (gap runs top-bottom between two row basins):
        x1 = bottom row of the upper basin (top wall of the gap)
        x2 = top row of the lower basin (bottom wall of the gap)
    
    We draw each boundary as a short perpendicular tick line so the whole
    grid looks connected, the tick spans from the neighbouring node above
    to the neighbouring node below (for H edges), or left to right (for V edges). """

    for edge in graph.edges:
        if edge.x1 >= edge.x2:
            # degenerate edge, the two basins are touching or overlapping, skip it
            continue

        if edge.orientation == 'H':
            """Horizontal edge
                The gap runs horizontally. x1 and x2 are column positions.
                We draw vertical blue tick lines at those columns.
                The tick height spans from the row-basin center above to the one below, so the ticks from adjacent edges
                visually connect into a continuous line."""

            # find the index of this edge's row basin in the sorted basin list
            ri = next((i for i, rb in enumerate(graph.row_basins)
                        if basin_center(rb) == edge.node_a.row), None)
            if ri is not None:
                # center of the row basin above this one (or top of image if first)
                r_top = ry(basin_center(graph.row_basins[ri - 1])) if ri > 0 \
                        else ry(0)
                # center of the row basin below this one (or bottom of image if last)
                r_bot = ry(basin_center(graph.row_basins[ri + 1])) \
                        if ri < len(graph.row_basins) - 1 \
                        else ry(cropped.shape[0] - 1)
            else:
                # fallback: draw a zero-height tick at the node center
                r_top = ry(edge.node_a.row)
                r_bot = r_top

            # vertical blue tick at the left wall of the gap (x1)
            cv2.line(out, (cx(edge.x1), r_top), (cx(edge.x1), r_bot), (255, 0, 0), 1)
            # vertical blue tick at the right wall of the gap (x2)
            cv2.line(out, (cx(edge.x2), r_top), (cx(edge.x2), r_bot), (255, 0, 0), 1)

        else:
            """Vertical edge 
            The gap runs vertically. x1 and x2 are row positions.
            We draw horizontal blue tick lines at those rows.
            The tick width spans from the col-basin center to the left to the one to the right."""

            ci = next((i for i, cb in enumerate(graph.col_basins)
                        if basin_center(cb) == edge.node_a.col), None)
            if ci is not None:
                # center of the col basin to the left (or left edge of image if first)
                c_left  = cx(basin_center(graph.col_basins[ci - 1])) if ci > 0 \
                          else cx(0)
                # center of the col basin to the right (or right edge if last)
                c_right = cx(basin_center(graph.col_basins[ci + 1])) \
                          if ci < len(graph.col_basins) - 1 \
                          else cx(cropped.shape[1] - 1)
            else:
                # fallback: use the node's own col basin extent
                c_left  = cx(edge.node_a.col_start)
                c_right = cx(edge.node_a.col_end)

            # horizontal blue tick at the top wall of the gap (x1)
            cv2.line(out, (c_left, ry(edge.x1)), (c_right, ry(edge.x1)), (255, 0, 0), 1)
            # horizontal blue tick at the bottom wall of the gap (x2)
            cv2.line(out, (c_left, ry(edge.x2)), (c_right, ry(edge.x2)), (255, 0, 0), 1)

    """ Layer 2: Pink lines, shrunk gap boundaries 
    After the shrink algorithm runs, each edge's boundaries may have moved
    outward (x1 decreased, x2 increased) to account for characters bleeding
    into the gap region. build_shrunk_graph() constructs a new lightweight
    graph whose nodes sit at the actual shrunk boundary positions.
    We draw every edge in that graph as a simple pink line. """

    PINK = (180, 105, 255)   # BGR colour for the shrunk boundary lines
    DARK_PINK = (120, 50, 180)   # darker shade used for the shrunk node squares

    shrunk_graph = graph.build_shrunk_graph()
    shrunk_graph.report()

    for edge in shrunk_graph.edges:
        # each ShrunkEdge connects two ShrunkNodes; draw a line between them
        p1 = (cx(edge.node_a.col), ry(edge.node_a.row))
        p2 = (cx(edge.node_b.col), ry(edge.node_b.row))
        cv2.line(out, p1, p2, PINK, 1)

    """ Layer 3: Green shading- original basin bands + node squares 
    We visualize the pre-shrink nodes the same way tateyoko's create_basins_mask
    does: draw a full-width horizontal band for every row basin and a full-height
    vertical band for every col basin. Their overlap at each grid intersection
    produces a brighter cross that highlights the node location. """

    overlay = out.copy()

    # shade every row basin as a horizontal green band spanning the full image width
    for rb in graph.row_basins:
        # rb[0] = first row of the basin, rb[1] = last row of the basin
        cv2.rectangle(overlay,
                      (0,              ry(rb[0])), # top-left corner: col=0
                      (out.shape[1]-1, ry(rb[1])), # bottom-right corner: col=full width
                      (0, 255, 0), -1) # -1 thickness = filled rectangle

    # shade every col basin as a vertical green band spanning the full image height
    for cb in graph.col_basins:
        cv2.rectangle(overlay,
                      (cx(cb[0]), 0), # top-left corner: row=0
                      (cx(cb[1]), out.shape[0]-1),  # bottom-right: row=full height
                      (0, 255, 0), -1)

    # blend: 30% of the green overlay + 70% of the current image
    cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)

    # draw 1-px green outlines along every basin edge
    for rb in graph.row_basins:
        cv2.line(out, (0, ry(rb[0])), (out.shape[1]-1, ry(rb[0])), (0, 200, 0), 1)  # top edge
        cv2.line(out, (0, ry(rb[1])), (out.shape[1]-1, ry(rb[1])), (0, 200, 0), 1)  # bottom edge
    for cb in graph.col_basins:
        cv2.line(out, (cx(cb[0]), 0), (cx(cb[0]), out.shape[0]-1), (0, 200, 0), 1)  # left edge
        cv2.line(out, (cx(cb[1]), 0), (cx(cb[1]), out.shape[0]-1), (0, 200, 0), 1)  # right edge

    for node in graph.nodes.values():
        r0 = ry(node.row_start); r1 = ry(node.row_end)
        c0 = cx(node.col_start); c1 = cx(node.col_end)
        if node.is_isolated:
            color = (255, 0, 255) #magenta, false positive
        elif node.is_dead_end:
            color = (0, 0, 255) #red
        else:
            color = (0, 255, 0) #green
        cv2.rectangle(out, (c0, r0), (c1, r1), color, -1)

    """Layer 4: Dark-pink squares: shrunk node rectangles
    Each node's "shrunk rectangle" is computed from the x1_shrunk / x2_shrunk
    values of its four adjacent edges. It shows where the shrink algorithm
    believes the actual gap boundaries are after removing character intrusions.
    
    For a node N:
       H edge where N = node_a (gap runs to the right of N):
           x1_shrunk is the new left wall of that gap -> N's right boundary shrinks to x1_shrunk
       H edge where N = node_b (gap runs to the left of N):
           x2_shrunk is the new right wall of that gap -> N's left boundary shrinks to x2_shrunk
       V edge where N = node_a (gap runs below N):
           x1_shrunk is the new top wall of that gap -> N's bottom boundary shrinks to x1_shrunk
       V edge where N = node_b (gap runs above N):
           x2_shrunk is the new bottom wall of that gap → N's top boundary shrinks to x2_shrunk """

    for node in graph.nodes.values():
        # start from the original basin extents as the fallback bounds
        top   = node.row_start
        bot   = node.row_end
        left  = node.col_start
        right = node.col_end

        for edge in node.edges:
            # skip edges that collapsed entirely (no valid gap remains)
            if edge.x1_shrunk >= edge.x2_shrunk:
                continue

            if edge.orientation == 'H':
                if edge.node_a is node:
                    # this edge departs to the right: x1_shrunk is where the gap
                    # actually starts, that becomes the new right boundary of N
                    right = min(right, edge.x1_shrunk)
                else:
                    # this edge arrives from the left: x2_shrunk is where the gap
                    # actually ends, that becomes the new left boundary of N
                    left = max(left, edge.x2_shrunk)
            else:
                if edge.node_a is node:
                    # this edge departs downward: x1_shrunk is the new bottom boundary of N
                    bot = min(bot, edge.x1_shrunk)
                else:
                    # this edge arrives from above: x2_shrunk is the new top boundary of N
                    top = max(top, edge.x2_shrunk)

        # only draw if the shrunk rectangle is non-degenerate
        # (if all edges collapsed the node may have zero area)
        if right > left and bot > top:
            cv2.rectangle(out,
                          (cx(left), ry(top)), # top-left corner of shrunk rect
                          (cx(right), ry(bot)), # bottom-right corner of shrunk rect
                          DARK_PINK, -1) # filled dark-pink rectangle

    cv2.imwrite(output_path, out)
    print(f"saved: {output_path}")


# Entry point

def main():
    if len(sys.argv) < 2:
        print("usage: python visualization.py <image_path> [output_path]")
        print("       [--threshold FLOAT]   basin threshold, default 0.05")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = "output_visualization.png"
    threshold = 0.05 # ink-density ratio below which a row/col is a basin
    merge_threshold = 3 # merge adjacent basins separated by <= this many pixels

    # parse optional command-line flags
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--threshold' and i+1 < len(sys.argv):
            threshold = float(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--merge' and i+1 < len(sys.argv):
            merge_threshold = int(sys.argv[i+1]); i += 2
        elif not sys.argv[i].startswith('--'):
            output_path = sys.argv[i]; i += 1
        else:
            i += 1

    """Step 1: load 
    Read the image from disk and convert it to a binary ink mask.
    img_bgr is kept for drawing on later; inverted is used for analysis. """
    print(f"loading: {image_path}")
    img_bgr, inverted = load_and_preprocess(image_path)
    print(f"image shape: {inverted.shape}")

    """Step 2: crop 
    Remove the blank margin around the page so basin detection isn't thrown
    off by large empty regions outside the document border.
    row_offset and col_offset record how much was cropped so we can convert
    coordinates back to full-image space when drawing."""
    print("cropping...")
    cropped, row_offset, col_offset = crop_to_content(inverted)
    print(f"cropped: {cropped.shape}, offset: ({row_offset}, {col_offset})")

    """Step 3: classify regions 
    Label every connected ink component and decide whether it is a character
    (small, squarish) or a border line (large or very elongated). """
    print("separating characters from borders...")
    labels, char_regions, border_regions = identify_characters_borders(cropped)
    print(f"  {len(char_regions)} characters, {len(border_regions)} borders")

    """Step 4: character mask 
    Build a binary array that is 1 only where character pixels are.
    Border lines are excluded so they don't interfere with basin detection. """
    char_mask = make_character_mask(labels, char_regions)

    """Step 5: find basins
    A basin is a contiguous run of rows (or columns) where the ink density
    falls below `threshold`. These are the gaps between rows/columns of text.
    merge_threshold merges basins that are very close together (≤ 3 px apart)
    to avoid splitting a single gap into two basins due to a stray ink pixel. """
    print(f"finding basins (threshold={threshold}, merge_threshold={merge_threshold})...")
    row_basins = find_basins(char_mask, 'row', threshold, merge_threshold=merge_threshold)
    col_basins = find_basins(char_mask, 'col', threshold, merge_threshold=merge_threshold)
    print(f"  row basins: {len(row_basins)}")
    print(f"  col basins: {len(col_basins)}")

    if not row_basins or not col_basins:
        print("WARNING: no basins found, try raising --threshold")
        sys.exit(1)

    """Step 6: build graph 
    BorderGraph creates one GapNode per (row_basin x col_basin) intersection,
    connects adjacent nodes with GapEdges, runs the shrink algorithm on every
    edge, and flags isolated/dead-end nodes and weak edges.
    """
    print("building BorderGraph...")
    graph = BorderGraph(char_mask, row_basins, col_basins)
    graph.report()

    """Step 7: render
    Draw all four layers on top of the original colour image and save. """
    print("rendering...")
    visualize_graph(img_bgr, graph, cropped, row_offset, col_offset, output_path)
    print("done.")


if __name__ == "__main__":
    main()
