# ClayArgus

Full-local macOS photogrammetry pipeline. Photos in, Fusion 360-ready mesh out. No cloud services, no data leaves your machine.

## What it does

```
Photos → QA culling → Scale detection → Subject framing → Preview → 3D solve → Mesh optimization → .obj
```

ClayArgus analyzes your photos for blur, exposure, and duplicates; detects scale references (coins, bananas) for real-world sizing; crops to the subject using Apple Vision; then runs Object Capture for the 3D solve and PyMeshLab for decimation and repair — producing a mesh ready for Fusion 360.

## Requirements

- macOS 14+ (Sonoma)
- Apple Silicon (M1 or later)
- Xcode 15+ command line tools
- Python 3.12+

## Setup

**1. Build the Swift CLI:**

```bash
cd src/swift
swift build -c release
```

**2. Install the Python package:**

```bash
pip install -e .
```

## Usage

```bash
clayarg run ./photos --name my-project
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | (required) | Project name, used for output filenames |
| `--detail` | `medium` | Object Capture detail: `preview`, `reduced`, `medium`, `full`, `raw` |
| `--poly-count` | `10000` | Target triangle count after decimation |
| `--format` | `obj` | Output format: `obj`, `stl`, `ply` |
| `--sensitivity` | `normal` | Feature sensitivity: `normal`, `high` |
| `--output-dir` | `<input_dir>/clayarg_<name>` | Output directory |
| `--skip-qa` | off | Skip image quality analysis |
| `--blur-threshold` | `100.0` | Laplacian variance threshold (higher = stricter) |
| `--exposure-tolerance` | `0.15` | Max fraction of clipped pixels allowed |
| `--duplicate-threshold` | `3` | Perceptual hash hamming distance (higher = stricter, 0 = off) |
| `--scale-obj` | none | Which detected object is the scale reference: `smaller` or `larger` |
| `--scale-ref` | none | Reference type: `quarter`, `dime`, `nickel`, `penny`, `euro`, `pound`, `loonie`, `toonie`, `coin`, `banana` |
| `--scale-mm` | none | Reference object diameter in mm (required with `--scale-ref coin`) |
| `--skip-framing` | off | Skip Apple Vision subject framing |
| `--crop-padding` | `0.1` | Padding around detected subject (fraction of bbox) |

### Examples

```bash
# Basic run
clayarg run ./photos --name phoebe

# Auto-scale from a quarter in the photos
clayarg run ./photos --name phoebe --scale-obj smaller --scale-ref quarter

# Custom coin with known diameter
clayarg run ./photos --name phoebe --scale-obj smaller --scale-ref coin --scale-mm 30

# Banana for scale
clayarg run ./photos --name phoebe --scale-obj smaller --scale-ref banana
```

### Pipeline Phases

```
I.   Image Quality Analysis     — blur, exposure, duplicate detection
II.  Scale Detection            — Hough circles (coins) or YOLO (banana)
III. Subject Framing            — Apple Vision objectness saliency auto-crop
IV.  Visual Preview             — MJPEG flythrough + confirm before solve
V.   3D Solve                   — Object Capture via Swift CLI
VI.  Mesh Optimization          — PyMeshLab decimation, repair, scaling
```

### Output

```
photos/clayarg_phoebe/
├── phoebe_raw.obj             # High-fidelity model from Object Capture
├── phoebe_fusion.obj          # Decimated, repaired mesh for Fusion 360
├── cropped/                   # Auto-cropped images (from Phase III)
├── preview.mjpeg              # Quick flythrough of the camera path
└── processing_report.json     # Pipeline log (quality scores, scale, culling decisions)
```

## Architecture

The pipeline has two independently built components:

**Swift CLI** (`clayarg-capture`) — Standalone executable wrapping `PhotogrammetrySession`. Emits JSON progress lines to stdout. Can be used independently of the Python orchestrator.

**Python orchestrator** (`clayarg`) — Manages the full pipeline: image QA, scale detection (Hough circles for coins, YOLOv8n via cv2.dnn for other objects), Apple Vision subject framing, Swift CLI subprocess, and PyMeshLab mesh optimization.

See [docs/Spec.md](docs/Spec.md) for the full technical specification.

## Roadmap

| Version | Status | Scope |
|---------|--------|-------|
| v0.1 | Done | Swift CLI + PyMeshLab pipeline |
| v0.2 | Done | Image quality culling (blur, exposure, duplicates), MJPEG preview |
| v0.3 | Current | Apple Vision subject framing, scale detection (coins + banana), HEIC dedup |
| v0.4 | Planned | In-app review UI, config files, resume-from-phase |

## License

AGPL-3.0 — due to bundled YOLOv8n model (Ultralytics AGPL). See [LICENSE](LICENSE) for details.
