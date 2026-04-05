"""Phase II: MJPEG preview generation for quick visual confirmation."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


def generate_mjpeg(
    images: list[Path],
    output_file: Path,
    frame_width: int = 960,
    jpeg_quality: int = 75,
) -> Path:
    """Generate an MJPEG file from a list of images.

    Each frame is a JPEG-encoded image written sequentially. Standard MJPEG
    viewers will play this as a slideshow/flythrough of the camera path.

    Args:
        images: Ordered list of image paths.
        output_file: Where to write the .mjpeg file.
        frame_width: Resize frames to this width (maintains aspect ratio).
        jpeg_quality: JPEG compression quality (1-100).

    Returns:
        Path to the generated MJPEG file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        for img_path in images:
            img = Image.open(img_path)
            img = img.convert("RGB")

            # Resize maintaining aspect ratio
            ratio = frame_width / img.width
            new_size = (frame_width, int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Encode as JPEG and write
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            f.write(buf.getvalue())

    return output_file
