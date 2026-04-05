"""Phase I, Step 2: Subject framing via Apple Vision saliency detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class FramingResult:
    path: Path
    original_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]  # (left, upper, right, lower)
    edge_flags: list[str]  # Which edges the subject touches


def _get_saliency_bbox(image_path: Path) -> tuple[float, float, float, float] | None:
    """Get the largest object's bounding box via Apple Vision objectness saliency.

    Uses objectness-based saliency which can detect multiple objects. When
    multiple are found, returns only the largest (by area) — this excludes
    scale references and other small objects in the scene.

    Falls back to attention-based saliency if objectness returns nothing.

    Returns normalized (x, y, width, height) with origin at bottom-left,
    or None if detection fails.
    """
    import Cocoa
    import Vision

    url = Cocoa.NSURL.fileURLWithPath_(str(image_path))

    # Try objectness-based first — detects distinct objects
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    request = Vision.VNGenerateObjectnessBasedSaliencyImageRequest.alloc().init()
    success, error = handler.performRequests_error_([request], None)

    if success and not error:
        results = request.results()
        if results and results[0].salientObjects():
            objects = results[0].salientObjects()
            # Pick the largest object by area
            best = max(objects, key=lambda o: o.boundingBox().size.width * o.boundingBox().size.height)
            bb = best.boundingBox()
            return (bb.origin.x, bb.origin.y, bb.size.width, bb.size.height)

    # Fallback to attention-based saliency
    handler2 = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    request2 = Vision.VNGenerateAttentionBasedSaliencyImageRequest.alloc().init()
    success2, error2 = handler2.performRequests_error_([request2], None)

    if not success2 or error2:
        return None

    results2 = request2.results()
    if not results2 or not results2[0].salientObjects():
        return None

    bb = results2[0].salientObjects()[0].boundingBox()
    return (bb.origin.x, bb.origin.y, bb.size.width, bb.size.height)


def _normalized_to_pixel(
    bbox: tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    padding: float = 0.1,
) -> tuple[int, int, int, int]:
    """Convert Vision normalized bbox to pixel crop box with padding.

    Vision uses bottom-left origin; PIL uses top-left origin.
    Returns (left, upper, right, lower) for PIL.Image.crop().
    """
    x, y, w, h = bbox

    # Add padding (fraction of bbox dimensions)
    pad_w = w * padding
    pad_h = h * padding
    x = max(0.0, x - pad_w)
    y = max(0.0, y - pad_h)
    w = min(1.0 - x, w + 2 * pad_w)
    h = min(1.0 - y, h + 2 * pad_h)

    # Convert to pixel coordinates (flip y-axis for PIL)
    left = int(x * img_width)
    right = int((x + w) * img_width)
    upper = int((1.0 - y - h) * img_height)  # Flip y
    lower = int((1.0 - y) * img_height)

    return (left, upper, right, lower)


def _check_edge_flags(
    bbox: tuple[float, float, float, float],
    threshold: float = 0.02,
) -> list[str]:
    """Check if the saliency bbox touches image edges (subject may be cut off)."""
    x, y, w, h = bbox
    flags = []
    if x < threshold:
        flags.append("left")
    if y < threshold:
        flags.append("bottom")
    if (x + w) > (1.0 - threshold):
        flags.append("right")
    if (y + h) > (1.0 - threshold):
        flags.append("top")
    return flags


def analyze_framing(image_path: Path, padding: float = 0.1) -> FramingResult | None:
    """Analyze a single image for subject framing.

    Returns FramingResult with crop coordinates, or None if detection fails.
    """
    bbox = _get_saliency_bbox(image_path)
    if bbox is None:
        return None

    img = Image.open(image_path)
    crop_box = _normalized_to_pixel(bbox, img.width, img.height, padding=padding)
    edge_flags = _check_edge_flags(bbox)

    return FramingResult(
        path=image_path,
        original_size=(img.width, img.height),
        crop_box=crop_box,
        edge_flags=edge_flags,
    )


def run_framing(
    images: list[Path],
    output_dir: Path,
    padding: float = 0.1,
) -> tuple[list[Path], list[FramingResult]]:
    """Crop all images to their detected subject.

    Args:
        images: List of image paths to process.
        output_dir: Directory to write cropped images.
        padding: Fraction of bbox to add as padding around the subject.

    Returns:
        Tuple of (cropped image paths, framing results for reporting).
    """
    cropped_dir = output_dir / "cropped"
    cropped_dir.mkdir(parents=True, exist_ok=True)

    cropped_paths: list[Path] = []
    results: list[FramingResult] = []

    for img_path in images:
        result = analyze_framing(img_path, padding=padding)

        if result is None:
            # Fail-open: copy original if detection fails
            cropped_paths.append(img_path)
            continue

        results.append(result)

        # Crop and save
        img = Image.open(img_path)
        cropped = img.crop(result.crop_box)
        out_path = cropped_dir / img_path.name
        cropped.save(out_path, quality=95)
        cropped_paths.append(out_path)

    return cropped_paths, results
