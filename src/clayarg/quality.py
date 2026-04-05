"""Phase I, Step 1: Image quality analysis — blur, exposure, and duplicate detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class ImageScore:
    path: Path
    blur_score: float = 0.0  # Laplacian variance — higher is sharper
    overexposed_frac: float = 0.0  # Fraction of clipped-high pixels
    underexposed_frac: float = 0.0  # Fraction of clipped-low pixels
    phash: str = ""  # Perceptual hash hex string
    flagged: bool = False
    flag_reasons: list[str] = field(default_factory=list)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Compute Laplacian variance as a blur metric. Higher = sharper."""
    # 3x3 Laplacian kernel convolved manually via numpy to avoid OpenCV dependency
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    laplacian = np.zeros_like(gray, dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            laplacian += kernel[di, dj] * padded[di : di + h, dj : dj + w]
    return float(np.var(laplacian))


def _exposure_fractions(
    gray: np.ndarray, low_thresh: int = 10, high_thresh: int = 245
) -> tuple[float, float]:
    """Return fraction of pixels that are clipped low / clipped high."""
    total = gray.size
    under = float(np.sum(gray < low_thresh)) / total
    over = float(np.sum(gray > high_thresh)) / total
    return under, over


def _perceptual_hash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute a simple perceptual hash (average hash)."""
    resized = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS).convert("L")
    pixels = np.array(resized, dtype=np.float64)
    mean = pixels.mean()
    bits = (pixels > mean).flatten()
    # Pack bits into hex string
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:0{hash_size * hash_size // 4}x}"


def _hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hex hash strings."""
    n1 = int(h1, 16)
    n2 = int(h2, 16)
    return bin(n1 ^ n2).count("1")


def analyze_image(path: Path) -> ImageScore | None:
    """Compute quality metrics for a single image. Returns None if unreadable."""
    try:
        img = Image.open(path)
        img.load()  # Force decode to catch format issues early
    except Exception:
        return None
    # Work on a reasonable size for analysis
    img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    gray = np.array(img.convert("L"), dtype=np.float64)

    blur = _laplacian_variance(gray)
    under, over = _exposure_fractions(gray.astype(np.uint8))
    phash = _perceptual_hash(img)

    return ImageScore(
        path=path,
        blur_score=blur,
        overexposed_frac=over,
        underexposed_frac=under,
        phash=phash,
    )


def run_quality_analysis(
    images: list[Path],
    blur_threshold: float = 100.0,
    exposure_tolerance: float = 0.15,
    duplicate_threshold: int = 5,
) -> list[ImageScore]:
    """Analyze all images and flag those below quality thresholds.

    Args:
        images: List of image file paths.
        blur_threshold: Laplacian variance below this = flagged as blurry.
        exposure_tolerance: Fraction of clipped pixels above this = flagged.
        duplicate_threshold: Hamming distance below this = flagged as duplicate.

    Returns:
        List of ImageScore objects, sorted by blur_score descending (sharpest first).
    """
    scores = []
    skipped = []
    for p in images:
        result = analyze_image(p)
        if result is None:
            skipped.append(p)
        else:
            scores.append(result)

    # Flag blur
    for s in scores:
        if s.blur_score < blur_threshold:
            s.flagged = True
            s.flag_reasons.append(f"blurry (score {s.blur_score:.1f} < {blur_threshold})")

    # Flag exposure
    for s in scores:
        if s.overexposed_frac > exposure_tolerance:
            s.flagged = True
            s.flag_reasons.append(
                f"overexposed ({s.overexposed_frac:.1%} clipped > {exposure_tolerance:.0%})"
            )
        if s.underexposed_frac > exposure_tolerance:
            s.flagged = True
            s.flag_reasons.append(
                f"underexposed ({s.underexposed_frac:.1%} clipped > {exposure_tolerance:.0%})"
            )

    # Flag duplicates — mark the later one in each duplicate pair
    for i in range(len(scores)):
        if scores[i].flagged and "duplicate" in str(scores[i].flag_reasons):
            continue
        for j in range(i + 1, len(scores)):
            dist = _hamming_distance(scores[i].phash, scores[j].phash)
            if dist <= duplicate_threshold:
                scores[j].flagged = True
                scores[j].flag_reasons.append(
                    f"duplicate of {scores[i].path.name} (distance {dist})"
                )

    # Sort by sharpness (best first)
    scores.sort(key=lambda s: s.blur_score, reverse=True)
    return scores
