# Detecting Table Structure in Document Images

Computer-vision research by Sohni Tagirisa under Professor Justin Li at
Occidental College (Spring 2026).

This project explores how whitespace and border geometry can recover table
structure from document images. The pipeline separates text from border
components, finds low-density row and column “basins,” builds a graph over
their intersections, and refines boundaries where character strokes intrude
into whitespace. The result can be visualized and used to crop individual
table cells for downstream analysis.

## What is implemented

- Connected-component classification for text and border regions
- Row and column basin detection with configurable merging
- A graph representation of candidate table borders
- Boundary refinement and weak/dead-end edge diagnostics
- Cell extraction and layered OpenCV visualizations

## Tech stack

Python, NumPy, SciPy, scikit-image, OpenCV, Pillow, and ImageIO.

## Setup

```bash
git clone https://github.com/sohni-tagirisa/CS-Directed-Research.git
cd CS-Directed-Research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Analyze one or more document images and crop detected table cells:

```bash
python new_next_door.py path/to/document.png
```

Render the detected basins, graph edges, and refined boundaries:

```bash
python visualization.py path/to/document.png output_visualization.png
```

Use `--threshold` to tune whitespace detection and `--merge_threshold`
(or `--merge` for the visualization script) to combine nearby basins.

## Repository guide

- `new_next_door.py`: end-to-end analysis and cell extraction
- `subvalley.py`: basin detection and graph data structures
- `visualization.py`: diagnostic rendering for the graph pipeline

## Research status

This repository contains an active research prototype. The current focus is
making boundary recovery robust to broken rules, noisy scans, and characters
that overlap the whitespace between cells.
