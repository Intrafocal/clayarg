"""Scale reference detection — derive real-world scale from a known object."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Known reference objects and their longest dimension in mm
# Detection method is inferred: coins use Hough circles, others use YOLO
KNOWN_REFERENCES: dict[str, float | None] = {
    # Coins (detected via Hough circles)
    "quarter": 24.26,     # US quarter
    "dime": 17.91,        # US dime
    "nickel": 21.21,      # US nickel
    "penny": 19.05,       # US penny
    "euro": 23.25,        # 1 euro coin
    "pound": 23.43,       # UK 1 pound coin
    "loonie": 26.50,      # Canadian dollar
    "toonie": 28.00,      # Canadian 2 dollar
    "coin": None,         # Custom coin — requires --scale-mm
    # Objects (detected via YOLO)
    "banana": 178.0,
}

# COCO class IDs for YOLO-detectable references
_YOLO_CLASSES: dict[str, int] = {
    "banana": 46,
}

_COIN_REFS = {"quarter", "dime", "nickel", "penny", "euro", "pound", "loonie", "toonie", "coin"}

# Lazy-loaded YOLO model
_yolo_net = None


def _get_yolo_net():
    """Load YOLOv8n ONNX model via cv2.dnn (lazy singleton)."""
    global _yolo_net
    if _yolo_net is None:
        model_path = Path(__file__).parent / "models" / "yolov8n.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        _yolo_net = cv2.dnn.readNetFromONNX(str(model_path))
    return _yolo_net


def _detect_yolo_object(
    image_path: Path,
    target_class_id: int,
    conf_threshold: float = 0.25,
) -> tuple[int, int, int, int] | None:
    """Detect a specific COCO object in an image via YOLOv8n.

    Returns (x1, y1, x2, y2) in original image coordinates for the
    highest-confidence detection, or None.
    """
    net = _get_yolo_net()
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()[0].T  # (8400, 84)

    class_scores = outputs[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    # Filter for target class above threshold
    mask = (class_ids == target_class_id) & (confidences > conf_threshold)
    if not mask.any():
        return None

    # Pick highest confidence
    indices = np.where(mask)[0]
    best = indices[np.argmax(confidences[indices])]

    # Convert box from 640x640 space to original
    cx, cy, bw, bh = outputs[best, :4]
    x_scale = w / 640.0
    y_scale = h / 640.0
    x1 = int((cx - bw / 2) * x_scale)
    y1 = int((cy - bh / 2) * y_scale)
    x2 = int((cx + bw / 2) * x_scale)
    y2 = int((cy + bh / 2) * y_scale)

    return (x1, y1, x2, y2)


@dataclass
class ScaleEstimate:
    reference_name: str
    reference_mm: float
    subject_mm: float
    pixels_per_mm: float
    samples: int
    confidence: str


def _get_subject_bbox(image_path: Path) -> tuple[float, float, float, float] | None:
    """Get the largest objectness saliency bbox (the main subject).

    Uses objectness rather than attention because objectness gives a tighter
    bbox around the actual object, while attention includes surrounding context.
    """
    import Cocoa
    import Vision

    url = Cocoa.NSURL.fileURLWithPath_(str(image_path))
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    request = Vision.VNGenerateObjectnessBasedSaliencyImageRequest.alloc().init()
    success, error = handler.performRequests_error_([request], None)

    if not success or error:
        return None

    results = request.results()
    if not results or not results[0].salientObjects():
        return None

    objects = results[0].salientObjects()
    largest = max(objects, key=lambda o: o.boundingBox().size.width * o.boundingBox().size.height)
    bb = largest.boundingBox()
    return (bb.origin.x, bb.origin.y, bb.size.width, bb.size.height)


def _find_coin_radius(
    images: list[Path],
    work_size: int = 1024,
) -> list[tuple[Path, float, float]]:
    """Find the coin across all images using Hough circle detection + consensus.

    Strategy: detect all circles in all images, find the radius that appears
    most consistently (coins produce a tight cluster because they're the same
    size at similar camera distances). Return per-image coin measurements.

    Returns list of (image_path, coin_diameter_full_res_px, scale_factor).
    """
    # Pass 1: collect all circles from all images
    all_circles: list[tuple[int, int, Path, float]] = []  # (radius, img_idx, path, sf)

    for idx, img_path in enumerate(images):
        img = Image.open(img_path)
        sf = work_size / max(img.size)
        work = img.resize(
            (int(img.width * sf), int(img.height * sf)),
            Image.Resampling.LANCZOS,
        )
        gray = np.array(work.convert("L"))
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=int(min(gray.shape) * 0.15),
        )
        if circles is not None:
            for c in np.round(circles[0]).astype(int):
                all_circles.append((int(c[2]), idx, img_path, sf))

    if not all_circles:
        return []

    # Pass 2: find the tightest radius cluster (the coin)
    # Bucket radii and find the peak
    radii = [c[0] for c in all_circles]
    bucket_size = 5
    from collections import Counter
    buckets = Counter(r // bucket_size for r in radii)

    # The coin bucket: most frequent, or if tied, the smaller radius
    peak_bucket = min(buckets, key=lambda b: (-buckets[b], b))
    peak_center = peak_bucket * bucket_size + bucket_size // 2

    # Select circles within ±bucket_size of the peak
    coin_circles = [
        c for c in all_circles
        if abs(c[0] - peak_center) <= bucket_size
    ]

    # Deduplicate per image (one coin per image, closest to peak)
    seen_images: dict[int, tuple] = {}
    for c in coin_circles:
        radius, idx, path, sf = c
        if idx not in seen_images or abs(radius - peak_center) < abs(seen_images[idx][0] - peak_center):
            seen_images[idx] = c

    # Return per-image coin diameter in full-res pixels
    results = []
    for radius, idx, path, sf in seen_images.values():
        coin_diameter_px = (radius * 2) / sf
        results.append((path, coin_diameter_px, sf))

    return results


def _measure_ref_yolo(
    images: list[Path],
    reference_name: str,
) -> list[tuple[Path, float]]:
    """Detect a YOLO-class reference object and return its pixel size per image.

    Returns list of (image_path, ref_longest_dimension_px).
    """
    class_id = _YOLO_CLASSES.get(reference_name)
    if class_id is None:
        return []

    results = []
    for img_path in images:
        bbox = _detect_yolo_object(img_path, class_id)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        longest = max(x2 - x1, y2 - y1)
        results.append((img_path, float(longest)))

    return results


def estimate_scale(
    images: list[Path],
    scale_obj: str,
    ref_mm: float,
    reference_name: str = "custom",
) -> ScaleEstimate | None:
    """Estimate real-world scale from a known reference object.

    Coins are detected via Hough circle consensus. Other objects (banana, etc.)
    are detected via YOLOv8n. The subject is measured via objectness saliency.

    Args:
        images: List of image paths to analyze.
        scale_obj: Which object is the reference — "smaller" or "larger".
        ref_mm: Known longest dimension of the reference object in mm.
        reference_name: Display name of the reference (e.g., "quarter", "banana").

    Returns:
        ScaleEstimate with the derived scale, or None if insufficient data.
    """
    # Detect reference object using appropriate method
    is_coin = reference_name in _COIN_REFS
    if is_coin:
        ref_detections = _find_coin_radius(images)
        # ref_detections: list of (path, diameter_px, scale_factor)
        ref_measurements = [(p, d) for p, d, _ in ref_detections]
    else:
        ref_measurements = _measure_ref_yolo(images, reference_name)
        # ref_measurements: list of (path, longest_dim_px)

    if not ref_measurements:
        return None

    measurements: list[float] = []
    px_per_mm_values: list[float] = []

    for img_path, ref_px in ref_measurements:
        px_per_mm = ref_px / ref_mm

        # Get subject extent from objectness saliency
        bbox = _get_subject_bbox(img_path)
        if bbox is None:
            continue

        img = Image.open(img_path)
        subj_w = bbox[2] * img.width
        subj_h = bbox[3] * img.height
        subj_longest = max(subj_w, subj_h)

        subj_mm = subj_longest / px_per_mm
        measurements.append(subj_mm)
        px_per_mm_values.append(px_per_mm)

    if not measurements:
        return None

    # Remove outliers (>2 stdev from median) if we have enough samples
    if len(measurements) >= 5:
        med = statistics.median(measurements)
        std = statistics.stdev(measurements)
        filtered = [(m, p) for m, p in zip(measurements, px_per_mm_values)
                     if abs(m - med) < 2 * std]
        if filtered:
            measurements = [f[0] for f in filtered]
            px_per_mm_values = [f[1] for f in filtered]

    avg_mm = statistics.mean(measurements)
    avg_px_per_mm = statistics.mean(px_per_mm_values)

    # Confidence based on measurement consistency
    if len(measurements) >= 3:
        cv = statistics.stdev(measurements) / avg_mm
        confidence = "high" if cv < 0.15 else "low"
    elif len(measurements) >= 2:
        cv = statistics.stdev(measurements) / avg_mm
        confidence = "high" if cv < 0.20 else "low"
    else:
        confidence = "low"

    return ScaleEstimate(
        reference_name=reference_name,
        reference_mm=ref_mm,
        subject_mm=round(avg_mm, 1),
        pixels_per_mm=round(avg_px_per_mm, 2),
        samples=len(measurements),
        confidence=confidence,
    )
