#!/usr/bin/env python3
"""
GapNode:
    - row_start, row_end, col_start, col_end  (basin extents)
    - connected edges list

GapEdge:
    - strength (% of segment that is black, 0.0-1.0)
    - x1, x2  (raw endpoints along the varying axis)
    - x1_shrunk, x2_shrunk  (after shrink algorithm)
    - orientation 'H' or 'V'

BorderGraph:
    - nodes at every (row_basin x col_basin) intersection
    - horizontal edges: same row_basin, adjacent col_basins
    - vertical edges: same col_basin, adjacent row_basins
    - shrink algorithm on every edge:
        while pixel_sum_at(x1) >= SHRINK_MIN: x1++
        while pixel_sum_at(x2) >= SHRINK_MIN: x2--
    - isolated nodes flagged  (degree == 0, likely false positive)
    - dead-end nodes flagged  (degree == 1, border chain broken)
    - weak edges flagged (strength < threshold, unreliable)
"""

""" This import makes all type annotations in this file lazy (treated as strings)
rather than evaluated immediately. this lets us reference GapEdge inside GapNode
and GapNode inside GapEdge without python complaining that one isn't defined yet."""
from __future__ import annotations

# dataclass lets us define simple data-holding classes without writing __init__ manually.
# field() is used to set default values for mutable fields like lists.
from dataclasses import dataclass, field

# these are just type hint helpers (List, Dict, etc.) used for documentation purposes
from typing import List, Dict, Tuple, Optional

import numpy as np

# utility: summing pixels along rows or columns

def sum_dimension(array, dimension):
    # axis=1 sums across columns, giving one total per row (how much ink is in each row)
    if dimension == 'row':
        return np.sum(array, axis=1)
    # axis=0 sums across rows, giving one total per column (how much ink is in each column)
    elif dimension == 'col':
        return np.sum(array, axis=0)
    else:
        raise ValueError(f'dimension must be "row" or "col", got "{dimension}"')

# finding basins: contiguous bands of mostly-empty rows or columns

def find_basins(array, dimension, threshold, merge_threshold=0):
    """ Find contiguous runs of rows/cols where character pixel ratio < threshold.
    Returns list of (start, end) index pairs.
    merge_threshold: merge adjacent basins whose gap is <= this many pixels.
    """
    # count how many ink pixels are in each row (or column)
    pixel_counts = sum_dimension(array, dimension)

    # for rows: divide by the number of columns to get a density ratio per row
    # for cols: divide by the number of rows to get a density ratio per column
    # ex. a ratio of 0.02 means only 2% of that row/col is ink- likely a gap
    length = array.shape[1] if dimension == 'row' else array.shape[0]
    character_ratios = pixel_counts / length

    # find every row/col index where the ratio falls below the threshold
    # np.nonzero returns the indices where the condition is true
    valleys = np.nonzero(character_ratios < threshold)[0]

    # if no low-density rows/cols were found at all, return empty
    if len(valleys) == 0:
        return []

    # group the low-density indices into contiguous runs.
    # for example, if valleys = [10, 11, 12, 20, 21] we get two basins: (10,12) and (20,21)
    basins = []
    first_value = None
    prev_value = None
    for value in valleys:
        if first_value is None:
            # start of a new basin
            first_value = value
        elif value > prev_value + 1:
            # there's a gap between this index and the previous one,
            # so the previous basin has ended- save it and start a new one
            basins.append((first_value, prev_value))
            first_value = value
        prev_value = value

    # saves the last basin after the loop ends
    basins.append((first_value, prev_value))

    # merge adjacent basins whose gap is within merge_threshold pixels
    if merge_threshold > 0 and len(basins) > 1:
        merged = [basins[0]]
        for basin in basins[1:]:
            if basin[0] - merged[-1][1] <= merge_threshold:
                merged[-1] = (merged[-1][0], basin[1])
            else:
                merged.append(basin)
        basins = merged

    return basins


def basin_center(basin: Tuple[int, int]) -> int:
    # returns the middle index of a (start, end) basin
    # used wherever we need a single representative coordinate for a basin
    return (basin[0] + basin[1]) // 2


# GapNode: one intersection point in the grid

@dataclass
class GapNode:
    """Represents one intersection point where a horizontal gap (row basin) crosses a vertical gap 
    (column basin), storing the extents of both basins and a list of connected edges

    row_start / row_end : extent of the row basin at this node
    col_start / col_end : extent of the col basin at this node
    row / col : center point of the basin (used for lookup and drawing)
    edges : all edges attached to this node
    """
    # the center row coordinate of this node (midpoint of its row basin)
    row: int
    # the center column coordinate of this node (midpoint of its col basin)
    col: int
    # the full vertical extent of the row basin that passes through this node
    row_start: int
    row_end:   int
    # the full horizontal extent of the col basin that passes through this node
    col_start: int
    col_end:   int
    # all GapEdge objects attached to this node (populated later by BorderGraph)
    # field(default_factory=list) means each node gets its own fresh empty list
    edges: List[GapEdge] = field(default_factory=list)

    # traversal helpers

    def neighbors(self) -> List[GapNode]:
        # returns all nodes directly connected to this one via an edge
        # e.other(self) gives the node on the other end of edge e
        return [e.other(self) for e in self.edges]

    def get_edge_to(self, other: GapNode) -> Optional[GapEdge]:
        # finds the direct edge connecting this node to a specific other node
        # returns None if no such edge exists
        for e in self.edges:
            if e.other(self) is other:
                return e
        return None

    # topology

    @property
    def degree(self) -> int:
        # how many edges are attached to this node (i.e. how many neighbors it has)
        return len(self.edges)

    @property
    def is_isolated(self) -> bool:
        """No neighbors, likely a false positive intersection."""
        # degree 0 means nothing connects here, this intersection shouldn't exist
        return self.degree == 0

    @property
    def is_dead_end(self) -> bool:
        """Only one neighbor, border chain is probably broken here."""
        # degree 1 means a line arrives here but goes nowhere, the grid is broken at this point
        return self.degree == 1

    # these are needed so GapNode objects can be stored in sets and used as dict keys
    def __hash__(self):   return hash((self.row, self.col))
    def __eq__(self, o):  return isinstance(o, GapNode) and self.row == o.row and self.col == o.col
    def __repr__(self):   return f"GapNode(row={self.row}, col={self.col}, degree={self.degree})"


# GapEdge: the border segment between two adjacent nodes

@dataclass
class GapEdge:
    """One edge between two adjacent GapNodes.

    orientation : 'H' = horizontal (same row_basin, varying col)
                  'V' = vertical   (same col_basin, varying row)

    x1, x2: raw endpoints along the varying axis (col for H, row for V)
    x1_shrunk, x2_shrunk: endpoints after shrink algorithm removes character intrusions
                    shrink walks x1 forward while pixel sum >= SHRINK_MIN
                    shrink walks x2 backward while pixel sum >= SHRINK_MIN

    strength: fraction of pixels in [x1_shrunk, x2_shrunk] that are ink (1)
        0.0 = all white (no gap), 1.0 = all black (solid border)
    is_valid: False when strength < WEAK_EDGE_THRESHOLD
    """
    # the two nodes this edge connects
    node_a:      GapNode
    node_b:      GapNode
    # 'H' for horizontal (runs left-right), 'V' for vertical (runs top-bottom)
    orientation: str
    # raw start and end positions along the varying axis before shrinking
    x1: int = 0
    x2: int = 0
    # positions after shrinking inward past character intrusions
    x1_shrunk: int = 0
    x2_shrunk: int = 0
    # ink density of the shrunk segment: 0.0 = pure gap, 1.0 = solid ink
    strength:  float = 0.0
    # set to False if strength is below the weak edge threshold
    is_valid:  bool  = True

    def other(self, node: GapNode) -> GapNode:
        # given one endpoint, return the other endpoint
        return self.node_b if node is self.node_a else self.node_a

    @property
    def is_weak(self) -> bool:
        # an edge is weak if its ink density is too low to be a reliable gap
        return self.strength < BorderGraph.WEAK_EDGE_THRESHOLD

    def __repr__(self):
        return (f"GapEdge({self.orientation} "
                f"[{self.node_a.row},{self.node_a.col}]"
                f"<->[{self.node_b.row},{self.node_b.col}] "
                f"strength={self.strength:.2f} valid={self.is_valid})")


# shrunk graph: the refined graph built from shrunk boundary positions

@dataclass
class ShrunkNode:
    row: int
    col: int
    edges: List = field(default_factory=list)

    @property
    def degree(self) -> int:
        return len(self.edges)

    def neighbors(self):
        return [e.other(self) for e in self.edges]

    def __hash__(self):   return hash((self.row, self.col))
    def __eq__(self, o):  return isinstance(o, ShrunkNode) and self.row == o.row and self.col == o.col
    def __repr__(self):   return f"ShrunkNode(row={self.row}, col={self.col}, degree={self.degree})"


@dataclass
class ShrunkEdge:
    node_a: ShrunkNode
    node_b: ShrunkNode
    orientation: str #'H' or 'V'

    def other(self, node):
        return self.node_b if node is self.node_a else self.node_a

    def __repr__(self):
        return (f"ShrunkEdge({self.orientation} "
                f"[{self.node_a.row},{self.node_a.col}]"
                f"<->[{self.node_b.row},{self.node_b.col}])")


@dataclass
class ShrunkGraph:
    nodes: Dict[Tuple[int, int], ShrunkNode]
    edges: List[ShrunkEdge]

    def report(self):
        print(f"\nShrunkGraph")
        print(f"  nodes: {len(self.nodes)}")
        print(f"  edges: {len(self.edges)}")
        h_edges = sum(1 for e in self.edges if e.orientation == 'H')
        v_edges = sum(1 for e in self.edges if e.orientation == 'V')
        print(f"  horizontal edges: {h_edges}")
        print(f"  vertical edges:   {v_edges}")


# BorderGraph: the full grid graph
class BorderGraph:

    # edges with ink density below this are flagged as weak/unreliable
    WEAK_EDGE_THRESHOLD = 0.4

    def __init__(self,
                 binary_array: np.ndarray,
                 row_basins:   List[Tuple[int, int]],
                 col_basins:   List[Tuple[int, int]]):
        """
        binary_array: 2D uint8, 1 = ink, 0 = background/gap
        row_basins: list of (start, end) from find_basins(..., 'row', ...)
        col_basins: list of (start, end) from find_basins(..., 'col', ...)
        """
        # store the binary image so edge methods can look up pixel values
        self.binary = binary_array
        self.row_basins = row_basins
        self.col_basins = col_basins

        # nodes keyed by (row_center, col_center) for fast lookup
        self.nodes: Dict[Tuple[int,int], GapNode] = {}
        self.edges: List[GapEdge] = []

        # problem lists populated during _flag_problems()
        self.isolated_nodes: List[GapNode] = []
        self.dead_end_nodes: List[GapNode] = []
        self.weak_edges: List[GapEdge] = []

        # run all four build steps immediately on construction
        self._build()

    def _build(self):
        # step 1: create one node per basin intersection
        self._make_nodes()
        # step 2: connect adjacent nodes with edges
        self._connect_nodes()
        # step 3: trim edge endpoints past character intrusions
        self._shrink_edges()
        # step 4: flag isolated nodes, dead ends, and weak edges
        self._flag_problems()

    # step 1: nodes

    def _make_nodes(self):
        """One GapNode per (row_basin x col_basin) crossing."""
        # loop over every combination of row basin and column basin
        # each combination is one intersection point in the grid
        for rb in self.row_basins:
            for cb in self.col_basins:
                # find the center coordinate of each basin
                row = basin_center(rb)
                col = basin_center(cb)
                # create the node and store it, keyed by its center coordinates
                self.nodes[(row, col)] = GapNode(
                    row=row,       col=col,
                    row_start=rb[0], row_end=rb[1],
                    col_start=cb[0], col_end=cb[1],
                )

    # step 2: edges 

    def _connect_nodes(self):
        """Horizontal: for each row_basin, connect adjacent col_basin pairs.
        Vertical: for each col_basin, connect adjacent row_basin pairs.
        Edge is attached to BOTH endpoints so the graph is traversable by
        calling node.neighbors() without ever touching self.edges.
        """
        # horizontal edges: walk along each row basin and connect neighboring column basins
        # ex. if there are 5 col basins in a row, this creates 4 horizontal edges per row
        for rb in self.row_basins:
            row = basin_center(rb)
            for i in range(len(self.col_basins) - 1):
                # take two adjacent column basins
                cb_a = self.col_basins[i]
                cb_b = self.col_basins[i + 1]
                col_a = basin_center(cb_a)
                col_b = basin_center(cb_b)
                # look up the nodes that sit at these two intersections
                node_a = self.nodes[(row, col_a)]
                node_b = self.nodes[(row, col_b)]
                # the actual gap between these two col basins runs from the
                # end of cb_a to the start of cb_b, use those as raw endpoints
                # so the shrink walks inward from the real gap boundaries,
                # not from the node centers
                gap_x1 = cb_a[1] # rightmost col of the left basin
                gap_x2 = cb_b[0] # leftmost  col of the right basin
                edge = GapEdge(
                    node_a=node_a, node_b=node_b,
                    orientation='H',
                    x1=gap_x1, x2=gap_x2,
                    x1_shrunk=gap_x1, x2_shrunk=gap_x2,
                )
                self.edges.append(edge)
                # attach the edge to both nodes so either node can find it
                node_a.edges.append(edge)
                node_b.edges.append(edge)

        # vertical edges: walk along each column basin and connect neighboring row basins
        # same logic as above but rotated 90 degrees
        for cb in self.col_basins:
            col = basin_center(cb)
            for i in range(len(self.row_basins) - 1):
                rb_a = self.row_basins[i]
                rb_b = self.row_basins[i + 1]
                row_a = basin_center(rb_a)
                row_b = basin_center(rb_b)
                node_a = self.nodes[(row_a, col)]
                node_b = self.nodes[(row_b, col)]
                # the actual gap between these two row basins runs from the
                # end of rb_a to the start of rb_b, use those as raw endpoints
                gap_x1 = rb_a[1] # bottom row of the upper basin
                gap_x2 = rb_b[0] # top row of the lower basin
                edge = GapEdge(
                    node_a=node_a, node_b=node_b,
                    orientation='V',
                    x1=gap_x1, x2=gap_x2,
                    x1_shrunk=gap_x1, x2_shrunk=gap_x2,
                )
                self.edges.append(edge)
                node_a.edges.append(edge)
                node_b.edges.append(edge)

    # step 3: shrink basins where characters bleed into the gap

    def _shrink_edges(self):
        """Precompute how tall/wide the perpendicular scan slice should be for
        each basin. this uses the midpoint between adjacent basin edges.
        example: if row basin i ends at row 50 and row basin i+1 starts at
        row 60, the midpoint is 55. this is used as the boundary between
        the two cells when counting ink in a vertical slice. """
        self._shrink_debug_count = 0

        self._row_scan = {}
        for i, rb in enumerate(self.row_basins):
            rc = basin_center(rb)
            r_top = 0 if i == 0 else (self.row_basins[i-1][1] + rb[0]) // 2
            r_bot = (self.binary.shape[0]) if i == len(self.row_basins)-1 else (rb[1] + self.row_basins[i+1][0]) // 2
            self._row_scan[rc] = (r_top, r_bot)

        self._col_scan = {}
        for i, cb in enumerate(self.col_basins):
            cc = basin_center(cb)
            c_left  = 0 if i == 0 else (self.col_basins[i-1][1] + cb[0]) // 2
            c_right = (self.binary.shape[1]) if i == len(self.col_basins)-1 else (cb[1] + self.col_basins[i+1][0]) // 2
            self._col_scan[cc] = (c_left, c_right)

        # run the shrink on every edge
        for edge in self.edges:
            if edge.orientation == 'H':
                self._shrink_H(edge)
            else:
                self._shrink_V(edge)

        # collapse per-edge shrunk bounds to one pair per gap so the
        # visualization shows two pink lines per gap instead of one per row
        self._aggregate_shrunk_per_gap()

        # measure strength against the aggregated (final) bounds
        for edge in self.edges:
            edge.strength = self._measure(edge)

    def _aggregate_shrunk_per_gap(self):
        """_shrink_H / _shrink_V compute x1_shrunk and x2_shrunk independently
        per row (for H-edges) or per column (for V-edges). Different rows
        scan different ink and land on slightly different boundaries, so
        the shrunk graph ends up with many staggered ticks per gap.
        
        Group edges that share the same gap (same column-pair for H,
        same row-pair for V) and replace each member's bounds with the
        group's outer envelope: min(x1_shrunk), max(x2_shrunk). this
        picks the most aggressive shrink found in any row/column so the
        aggregated gap covers the worst character bleed anywhere in it."""
        h_groups = {}
        v_groups = {}
        for edge in self.edges:
            if edge.orientation == 'H':
                h_groups.setdefault((edge.node_a.col, edge.node_b.col), []).append(edge)
            else:
                v_groups.setdefault((edge.node_a.row, edge.node_b.row), []).append(edge)

        for group in (*h_groups.values(), *v_groups.values()):
            x1_agg = min(e.x1_shrunk for e in group)
            x2_agg = max(e.x2_shrunk for e in group)
            for e in group:
                e.x1_shrunk = x1_agg
                e.x2_shrunk = x2_agg

    def _find_minimum_pos(self, slices_iter, start_falling=False):
        """Walks through (position, ink_count) pairs looking for the minimum.
        
        start_falling=False (default): starts in RISING phase, waits for ink
        to increase before tracking the minimum. use this when the walk starts
        inside a clean basin (node center) and needs to skip past the initial
        low-ink region before finding the valley between characters and the gap.
        
        start_falling=True: starts in FALLING phase, tracks the minimum from
        the very first step. use this when the walk starts at a boundary position
        that already has some ink and moves toward cleaner whitespace.
        
        in both cases: once tracking, returns the minimum position as soon as
        ink rises again (we passed the valley). if the walk ends while still
        falling, returns the lowest position seen. """

        phase = 'FALLING' if start_falling else 'RISING'
        prev_count = None
        prev_pos   = None
        min_count  = None
        min_pos    = None

        for pos, count in slices_iter:
            if phase == 'RISING':
                # waiting for ink to go up
                if prev_count is not None and count > prev_count:
                    # ink just rose, the boundary is the last position before the rise
                    phase = 'FALLING'
                    min_count = prev_count
                    min_pos   = prev_pos
                prev_count = count
                prev_pos   = pos
            else:
                # tracking the minimum
                if min_count is None or count < min_count:
                    min_count = count
                    min_pos   = pos
                else:
                    # ink rose again, we passed the valley
                    return min_pos

        # reached the end without rising again, return whatever minimum we found
        if min_pos is not None:
            return min_pos
        return None

    def _shrink_H(self, edge: GapEdge):
        # horizontal edge: gap runs left-right between two column basins.
        # we walk column by column from node_a center to node_b center.
        # at each column we count ink pixels in a vertical slice from r_top to r_bot.
        # the minimum tells us where the actual gap boundary is.
        r_top, r_bot = self._row_scan.get(edge.node_a.row,
                                           (edge.node_a.row_start, edge.node_a.row_end + 1))
        band_h = r_bot - r_top
        if band_h <= 0:
            return

        na_col = edge.node_a.col
        nb_col = edge.node_b.col
        if na_col >= nb_col:
            return

        walk_lines = []

        # walk leftward from x1 toward node_a to find x1_shrunk
        def fwd():
            for c in range(edge.x1, na_col - 1, -1):
                ink = int(np.sum(self.binary[r_top:r_bot, c] == 1))
                walk_lines.append(f"    fwd c={c}  ink={ink}  ratio={ink/band_h:.3f}")
                yield c, ink
        x1 = self._find_minimum_pos(fwd(), start_falling=True)
        if x1 is None:
            x1 = edge.x1

        # walk rightward from x2 toward node_b to find x2_shrunk
        def rev():
            for c in range(edge.x2, nb_col + 1):
                ink = int(np.sum(self.binary[r_top:r_bot, c] == 1))
                walk_lines.append(f"    rev c={c}  ink={ink}  ratio={ink/band_h:.3f}")
                yield c, ink
        x2 = self._find_minimum_pos(rev(), start_falling=True)
        if x2 is None:
            x2 = edge.x2

        # clamp: the shrunk boundary can only move outward from the original.
        # x1_shrunk <= x1 (move left, basin gets smaller on the left side)
        # x2_shrunk >= x2 (move right, basin gets smaller on the right side)
        # this makes the gap wider, not narrower.
        edge.x1_shrunk = min(x1, edge.x1)
        edge.x2_shrunk = max(x2, edge.x2)

        # debug: print ratio before/after for the first 10 H edges
        if self._shrink_debug_count < 10:
            self._shrink_debug_count += 1
            def col_ratio(c):
                ink = int(np.sum(self.binary[r_top:r_bot, c] == 1))
                return ink / band_h if band_h > 0 else 0.0
            x1_delta    = edge.x1_shrunk - edge.x1
            x2_delta    = edge.x2_shrunk - edge.x2
            span_before = edge.x2 - edge.x1
            span_after  = edge.x2_shrunk - edge.x1_shrunk
            print(f"[SHRINK H #{self._shrink_debug_count}] row={edge.node_a.row}  node_a.col={edge.node_a.col}  node_b.col={edge.node_b.col}  band_h={band_h}")
            print(f"  stored:  x1={edge.x1} (ratio={col_ratio(edge.x1):.3f})  x2={edge.x2} (ratio={col_ratio(edge.x2):.3f})  span={span_before}")
            print(f"  found:   x1={x1} (ratio={col_ratio(x1):.3f})  x2={x2} (ratio={col_ratio(x2):.3f})")
            print(f"  clamped: x1={edge.x1_shrunk} (ratio={col_ratio(edge.x1_shrunk):.3f}, {x1_delta:+d})  x2={edge.x2_shrunk} (ratio={col_ratio(edge.x2_shrunk):.3f}, {x2_delta:+d})  span={span_after}")
            print('\n'.join(walk_lines))

    def _shrink_V(self, edge: GapEdge):
        # vertical edge: gap runs top-bottom between two row basins.
        # same logic as _shrink_H but rotated — walk row by row from
        # node_a center to node_b center, counting ink in a horizontal slice.
        c_left, c_right = self._col_scan.get(edge.node_a.col,
                                              (edge.node_a.col_start, edge.node_a.col_end + 1))
        band_w = c_right - c_left
        if band_w <= 0:
            return

        na_row = edge.node_a.row
        nb_row = edge.node_b.row
        if na_row >= nb_row:
            return

        # walk downward from node_a toward node_b to find x1_shrunk
        def fwd():
            for r in range(na_row, nb_row + 1):
                yield r, int(np.sum(self.binary[r, c_left:c_right] == 1))
        x1 = self._find_minimum_pos(fwd())
        if x1 is None:
            x1 = edge.x1

        # walk upward from node_b toward node_a to find x2_shrunk
        def rev():
            for r in range(nb_row, na_row - 1, -1):
                yield r, int(np.sum(self.binary[r, c_left:c_right] == 1))
        x2 = self._find_minimum_pos(rev())
        if x2 is None:
            x2 = edge.x2

        # clamp: same as _shrink_H but vertical
        # x1_shrunk <= x1 (move up, basin gets smaller on top)
        # x2_shrunk >= x2 (move down, basin gets smaller on bottom)
        edge.x1_shrunk = min(x1, edge.x1)
        edge.x2_shrunk = max(x2, edge.x2)

    def _measure(self, edge: GapEdge) -> float:
        """Fraction of pixels in the shrunk segment that are ink/1 (0.0-1.0).
        Uses the full basin band width, not just a single row/col.
        """
        # if the shrunk segment has collapsed to zero length, return 0
        if edge.x2_shrunk <= edge.x1_shrunk:
            return 0.0

        if edge.orientation == 'H':
            # for a horizontal edge, cut out the rectangle:
            # rows r0:r1 (the full row basin height) x cols x1_shrunk:x2_shrunk (the trimmed span)
            r0 = edge.node_a.row_start
            r1 = edge.node_a.row_end + 1
            band = self.binary[r0:r1, edge.x1_shrunk:edge.x2_shrunk]
        else:
            # for a vertical edge, cut out the rectangle:
            # rows x1_shrunk:x2_shrunk (the trimmed span) x cols c0:c1 (the full col basin width)
            c0 = edge.node_a.col_start
            c1 = edge.node_a.col_end + 1
            band = self.binary[edge.x1_shrunk:edge.x2_shrunk, c0:c1]

        # guard against an empty slice
        if band.size == 0:
            return 0.0

        # count how many pixels are ink (value == 1) and divide by total pixels
        # result: 0.0 = completely empty gap, 1.0 = completely filled with ink
        return float(np.sum(band == 1)) / band.size

    # step 4: flag problem areas

    def _flag_problems(self):
        # check every node: is it isolated (degree 0) or a dead end (degree 1)?
        for node in self.nodes.values():
            if node.is_isolated:
                self.isolated_nodes.append(node)
            elif node.is_dead_end:
                self.dead_end_nodes.append(node)
        # check every edge: is it too weak to be a reliable gap separator?
        for edge in self.edges:
            if edge.is_weak:
                edge.is_valid = False  # mark it invalid
                self.weak_edges.append(edge)

    # traversal

    def bfs(self, start_row: int, start_col: int) -> List[GapNode]:
        # look up the starting node by its coordinates — return empty if not found
        start = self.nodes.get((start_row, start_col))
        if not start:
            return []

        visited = set()   # nodes we've already processed
        queue = [start] # nodes waiting to be processed, starting with the root
        order = []      # the final list of nodes in visit order

        while queue:
            # take the next node off the front of the queue
            node = queue.pop(0)
            # skip it if we've already visited it (can arrive via multiple paths)
            if node in visited:
                continue
            # mark it visited and add it to our result
            visited.add(node)
            order.append(node)
            # add all unvisited neighbors to the back of the queue
            queue.extend(nb for nb in node.neighbors() if nb not in visited)

        return order

    def connected_components(self) -> List[List[GapNode]]:
        visited    = set()
        components = []

        for node in self.nodes.values():
            if node not in visited:
                # run bfs from this unvisited node to find everything reachable from it
                comp = self.bfs(node.row, node.col)
                # mark all of them visited so we don't start a new bfs from them
                visited.update(comp)
                # this group of nodes is one connected component
                components.append(comp)

        # if the grid is intact, there should be exactly one component
        # multiple components means some gap lines are broken or missing
        return components

    def build_shrunk_graph(self):
        # builds a separate graph from the shrunk boundary positions.

        """Each original edge that didn't collapse (x1_shrunk < x2_shrunk)
        produces two pink boundary lines, one at x1_shrunk and one at x2_shrunk.
        these lines are perpendicular to the original edge direction and span
        from one neighboring node center to the next (node-to-node)."""

        """For a horizontal original edge, the pink ticks are vertical lines.
        For a vertical original edge, the pink ticks are horizontal lines.
        Each tick becomes one ShrunkEdge with a node at each end.
        Nodes at the same (row, col) get merged so the graph connects where tick endpoints meet."""

        shrunk_nodes = {}  # (row, col) -> ShrunkNode
        shrunk_edges = []
        seen_edges = set()  # tracks which node pairs already have an edge
        seen_ticks = set()  # tracks which tick positions have been drawn:
                              # H-edge ticks keyed by (row_basin_center, c)
                              # V-edge ticks keyed by (r, col_basin_center)

        def get_or_create(row, col):
            # look up or create a node at this position
            key = (int(row), int(col))
            if key not in shrunk_nodes:
                shrunk_nodes[key] = ShrunkNode(row=key[0], col=key[1])
            return shrunk_nodes[key], key

        for edge in self.edges:
            # skip collapsed edges, these have no valid gap left
            if edge.x1_shrunk >= edge.x2_shrunk:
                continue

            if edge.orientation == 'H':
                """Horizontal original edge -> pink ticks are vertical lines.
                figure out where the tick starts and ends vertically
                by finding the neighboring row basin centers (node above and below). """
                ri = next((i for i, rb in enumerate(self.row_basins) if basin_center(rb) == edge.node_a.row), None)
                if ri is not None:
                    r_top = basin_center(self.row_basins[ri - 1]) if ri > 0 else 0
                    r_bot = basin_center(self.row_basins[ri + 1]) if ri < len(self.row_basins) - 1 else self.binary.shape[0] - 1
                else:
                    r_top = edge.node_a.row_start
                    r_bot = edge.node_a.row_end
                
                row_key = edge.node_a.row
                for c in (edge.x1_shrunk, edge.x2_shrunk):
                    tick_key = (row_key, int(c))
                    if tick_key in seen_ticks:
                        continue
                    seen_ticks.add(tick_key)
                    na, ka = get_or_create(r_top, c)
                    nb, kb = get_or_create(r_bot, c)
                    pair = frozenset([ka, kb])
                    if pair not in seen_edges:
                        seen_edges.add(pair)
                        se = ShrunkEdge(node_a=na, node_b=nb, orientation='V')
                        shrunk_edges.append(se)
                        na.edges.append(se)
                        nb.edges.append(se)
            else:
                """ Vertical original edge -> pink ticks are horizontal lines.
                figure out where the tick starts and ends horizontally
                by finding the neighboring col basin centers (node left and right). """
                ci = next((i for i, cb in enumerate(self.col_basins) if basin_center(cb) == edge.node_a.col), None)
                if ci is not None:
                    c_left = basin_center(self.col_basins[ci - 1]) if ci > 0 else 0
                    c_right = basin_center(self.col_basins[ci + 1]) if ci < len(self.col_basins) - 1 else self.binary.shape[1] - 1
                else:
                    c_left = edge.node_a.col_start
                    c_right = edge.node_a.col_end
                """ Make one horizontal edge at x1_shrunk and one at x2_shrunk.
                    deduplicate by (r, col_basin_center): if two V-edges in the
                    same col basin produce the same shrunk row we only draw one """
                # tick, preventing the triple-line bug.
                col_key = edge.node_a.col
                for r in (edge.x1_shrunk, edge.x2_shrunk):
                    tick_key = (int(r), col_key)
                    if tick_key in seen_ticks:
                        continue
                    seen_ticks.add(tick_key)
                    na, ka = get_or_create(r, c_left)
                    nb, kb = get_or_create(r, c_right)
                    pair = frozenset([ka, kb])
                    if pair not in seen_edges:
                        seen_edges.add(pair)
                        se = ShrunkEdge(node_a=na, node_b=nb, orientation='H')
                        shrunk_edges.append(se)
                        na.edges.append(se)
                        nb.edges.append(se)

        return ShrunkGraph(nodes=shrunk_nodes, edges=shrunk_edges)

    def report(self):
        comps = self.connected_components()
        print(f"\n BorderGraph")
        print(f"  row basins: {len(self.row_basins)}")
        print(f"  col basins: {len(self.col_basins)}")
        print(f"  nodes: {len(self.nodes)}")
        print(f"  edges: {len(self.edges)}")
        print(f"  connected components: {len(comps)}")
        print(f"  isolated nodes: {len(self.isolated_nodes)}  likely false positives")
        print(f"  dead-end nodes: {len(self.dead_end_nodes)}  border chain broken")
        print(f"  weak edges: {len(self.weak_edges)}  strength < {self.WEAK_EDGE_THRESHOLD}")
        print(f"  valid edges: {sum(1 for e in self.edges if e.is_valid)}")
        if len(comps) > 1:
            print(f"  WARNING: {len(comps)} disconnected components,  gaps in border detected")


def extract_subvalleys_at_intersections(binary_array, gap_rows, gap_cols, window_size=20):
    subvalleys     = []
    all_coordinates = []

    # for each gap row/col crossing, sample a small window of pixels around it
    for row in gap_rows:
        for col in gap_cols:
            # clamp the window to the image boundaries so we don't go out of bounds
            r0 = max(0, row - window_size)
            r1 = min(binary_array.shape[0], row + window_size)
            c0 = max(0, col - window_size)
            c1 = min(binary_array.shape[1], col + window_size)

            # cut out the small window of pixels around this intersection
            window = binary_array[r0:r1, c0:c1]

            # count ink pixels (value == 1) in the window
            ink_px   = int(np.sum(window == 1))
            total_px = window.size

            if total_px > 0:
                # record summary stats for this intersection
                subvalleys.append({
                    'row': row, 'col': col,
                    'window_bounds': (r0, r1, c0, c1),
                    'black_pixels': ink_px,
                    'total_pixels': total_px,
                    'black_ratio':  ink_px / total_px,
                })
                # also record the absolute coordinates of every ink pixel in the window
                rr, cc = np.where(window == 1)
                for r, c in zip(rr, cc):
                    all_coordinates.append((r0+r, c0+c))

    return {'subvalleys': subvalleys, 'all_coordinates': all_coordinates}


def report_subvalleys(subvalleys):
    if not subvalleys:
        print("no sub-valleys detected.")
        return
    print(f"\ndetected {len(subvalleys)} sub-valleys:")
    for i, sv in enumerate(subvalleys):
        print(f"  [{i+1}] row={sv['row']} col={sv['col']} black_ratio={sv['black_ratio']:.3f}")
