# ClayArgus (`clayarg`) — Technical Specification v3

A full-local macOS photogrammetry pipeline that turns a folder of photos into a Fusion 360-ready mesh. Privacy-first: no cloud services, no data leaves the machine.

---

## 1. System Architecture

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestrator** | Python 3.12+ | Manages the pipeline, EXIF data, pre-flight checks, and subprocess calls. |
| **Image QA** | Pillow / OpenCV | Traditional IQA (Laplacian variance, exposure analysis) for blur and quality culling. |
| **Subject Framing** | Apple Vision (PyObjC) | `VNGenerateForegroundInstanceMaskRequest` for subject isolation and bounding box extraction. Runs on Neural Engine. |
| **3D Solve** | **Swift CLI** (Object Capture API) | Standalone Swift executable wrapping `PhotogrammetrySession`. Called via subprocess. |
| **Post-Process** | **PyMeshLab** | Decimation, scaling, and manifold repairs. |

### Key Constraint: Sequential Execution

Phases run strictly in order. No concurrent GPU contention.

```
Pre-flight → Phase I (QA + Framing) → Phase II (Preview) → Phase III (3D Solve) → Phase IV (Optimize)
```

---

## 2. Pre-flight Checks

Before any processing, the orchestrator validates the environment:

- [ ] Swift CLI binary present and executable
- [ ] Input folder exists and contains supported image files (JPG, HEIC, DNG)
- [ ] Sufficient disk space for intermediate outputs
- [ ] macOS 14+ (required for Object Capture API and `VNGenerateForegroundInstanceMaskRequest`)

Failures produce clear, actionable error messages.

---

## 3. Pipeline Phases

### Setup: Project Initialization

Before processing begins, the user provides:

- **Project name:** Used for output filenames (`<name>_raw.usdz`, `<name>_fusion.obj`).
- **Source folder:** Path to the image set.

### Phase I: Image Quality & Framing

**Step 1 — Automated Quality Culling (Pillow/OpenCV)**

Traditional image quality analysis on full-resolution images:
- **Blur detection:** Laplacian variance with configurable threshold.
- **Exposure analysis:** Histogram-based over/underexposure flagging.
- **Duplicate detection:** Perceptual hashing to flag near-identical shots.

Images are scored and ranked. Those below the quality threshold are flagged for removal (user can override in-app).

**Step 2 — Subject Framing (Apple Vision)**

Images that pass quality culling are processed via `VNGenerateForegroundInstanceMaskRequest` (macOS 14+):

- The Vision framework isolates the foreground subject and produces a pixel-accurate segmentation mask.
- A tight bounding box is derived from the mask and used to crop/center the subject.
- ~50-200ms per image on Apple Silicon Neural Engine. No model downloads required.
- Zero-shot: works on any object without a description.

Images where the subject is partially out of frame (mask touches image edges) are flagged for user review.

### Phase II: Visual Confirmation (MJPEG)

- The orchestrator generates a `preview.mjpeg` from the cleaned/cropped image set — a quick flythrough of the camera path.
- Individual images remain available for in-app review.
- **User action:** Confirm the set looks good, or manually exclude/include images before proceeding.

### Phase III: 3D Solve (Swift CLI)

- The orchestrator calls the Swift CLI binary via subprocess, passing the image folder and detail level.
- The Swift CLI wraps `PhotogrammetrySession` and reports progress via stdout (structured JSON lines).
- **Detail level** is configurable (default: `medium`). Options: `preview`, `reduced`, `medium`, `full`, `raw`.
- **Error propagation:** Non-zero exit codes and stderr are captured. Common failures (insufficient overlap, too few images) produce user-friendly messages.
- **Output:** `.obj` file (high-poly, pre-optimization).

### Phase IV: Mesh Optimization (PyMeshLab)

- **Decimation:** Reduce triangle count to target (default: **10,000**, configurable).
- **Manifold repair:** Close holes, remove non-manifold edges.
- **Tri-to-quad conversion:** Optional step for Fusion 360 T-Spline workflows.
- **Scaling:** User provides a known dimension (e.g., "wingspan = 142mm"). No automated scale detection.
- **Output:** `.obj` file (additional formats configurable: STL, PLY).

---

## 4. Swift CLI Specification

The Swift CLI is a **separate, independently built executable**. It has no Python dependencies.

```
Usage: clayarg-capture <input-dir> <output-file> [options]
  --detail <level>     preview|reduced|medium|full|raw (default: medium)
  --progress           Emit JSON progress lines to stdout
  --sensitivity <val>  Feature sensitivity: normal|high (default: normal)
```

**Build requirements:** Xcode 15+, macOS 14+ SDK, Swift 5.9+.

Progress output format (one JSON object per line):
```json
{"phase": "processing", "progress": 0.45, "message": "Generating point cloud"}
{"phase": "completed", "output": "/path/to/output.usdz"}
{"phase": "error", "message": "Insufficient image overlap", "code": 1}
```

---

## 5. Inputs and Outputs

### Inputs
- **Project name:** Identifier used for output filenames.
- **Source folder:** JPG, HEIC, or DNG (ProRAW) images. DNG files are passed directly to Object Capture (which handles RAW development internally).
- **Configuration:** Target poly count, scale dimension, detail level, output format.

### Outputs
- **`<name>_raw.obj`:** High-fidelity model from Object Capture.
- **`<name>_fusion.obj`:** Optimized, scaled mesh ready for Fusion 360.
- **`processing_report.json`:** Structured log of all pipeline decisions — quality scores, crop actions, culled images with reasons, solve parameters, decimation stats.

---

## 6. Configuration

```toml
# clayarg.toml (project-level, or ~/.config/clayarg/config.toml for defaults)

[output]
poly_count = 10000          # Target triangle count for decimation
format = "obj"              # obj, stl, ply
scale_dimension_mm = 0      # 0 = no scaling applied

[solve]
detail_level = "medium"     # preview, reduced, medium, full, raw

[quality]
blur_threshold = 100.0      # Laplacian variance threshold (higher = stricter)
exposure_tolerance = 0.15   # Fraction of clipped pixels allowed
```

---

## 7. Hardware Requirements

| Component | Minimum (M1/M2, 16GB) | Recommended (M3 Pro/Max, 36GB+) |
| :--- | :--- | :--- |
| **Detail Level** | `reduced` or `medium` | `full` or `raw` |
| **Practical Limits** | ~40 images | 100+ images |

**Note:** Processing times depend heavily on image count, resolution, and detail level. No specific time guarantees are made.

---

## 8. Error Handling

| Failure | Response |
| :--- | :--- |
| Swift CLI not found | Pre-flight fails with build instructions |
| Vision framework framing fails | Pass image through uncropped (fail-open) |
| Object Capture insufficient overlap | Report which images lack coverage, suggest re-shooting angles |
| PyMeshLab degenerate geometry | Retry with less aggressive decimation, warn user |
| Pipeline interrupted mid-phase | Intermediate outputs preserved; user can resume from last completed phase |

---

## 9. Development Phases

| Version | Scope |
| :--- | :--- |
| **v0.1** | Swift CLI + PyMeshLab pipeline. Folder → Object Capture → decimate → OBJ. Manual scale. No AI. |
| **v0.2** | Pre-flight checks. Traditional IQA culling (blur, exposure, duplicates). MJPEG preview. |
| **v0.3** | Apple Vision subject framing. Auto-crop via foreground instance mask. |
| **v0.4** | In-app image review UI. Configuration file support. Resume-from-phase. |
