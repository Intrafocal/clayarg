"""Phase III: 3D Solve — calls the Swift CLI and streams progress."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SolveResult:
    output_path: Path
    warnings: list[str]


class SolveError(Exception):
    def __init__(self, message: str, code: int = 1):
        self.code = code
        super().__init__(message)


def find_swift_cli() -> Path:
    """Locate the clayarg-capture binary."""
    # Check next to the Python package first (installed layout)
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "bin" / "clayarg-capture",
        pkg_dir.parent.parent / "src" / "swift" / ".build" / "release" / "clayarg-capture",
        pkg_dir.parent.parent / "src" / "swift" / ".build" / "debug" / "clayarg-capture",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Fall back to PATH
    result = subprocess.run(["which", "clayarg-capture"], capture_output=True, text=True)
    if result.returncode == 0:
        return Path(result.stdout.strip())

    raise SolveError(
        "clayarg-capture not found. Build it with: "
        "cd src/swift && swift build -c release",
        code=2,
    )


def run_solve(
    input_dir: Path,
    output_file: Path,
    detail: str = "medium",
    sensitivity: str = "normal",
    on_progress: callable | None = None,
) -> SolveResult:
    """Run Object Capture via the Swift CLI.

    Args:
        input_dir: Folder containing source images.
        output_file: Where to write the .usdz output.
        detail: Detail level (preview/reduced/medium/full/raw).
        sensitivity: Feature sensitivity (normal/high).
        on_progress: Optional callback(fraction, message) for progress updates.

    Returns:
        SolveResult with the output path and any warnings.

    Raises:
        SolveError on failure.
    """
    cli = find_swift_cli()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(cli),
        str(input_dir),
        str(output_file),
        "--detail", detail,
        "--sensitivity", sensitivity,
        "--progress",
    ]

    warnings: list[str] = []

    # Send stderr to a temp file to avoid pipe deadlock
    import tempfile
    stderr_file = tempfile.TemporaryFile(mode="w+t")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_file, text=True)

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        phase = msg.get("phase")

        if phase == "processing" and on_progress:
            on_progress(msg.get("progress", 0), msg.get("message", ""))
        elif phase == "warning":
            warnings.append(msg.get("message", ""))
        elif phase == "error":
            proc.wait()
            stderr_file.close()
            raise SolveError(msg.get("message", "Unknown error"), code=msg.get("code", 1))
        elif phase == "completed":
            pass  # Wait for process to finish

    proc.wait()

    if proc.returncode != 0:
        stderr_file.seek(0)
        stderr = stderr_file.read()
        stderr_file.close()
        raise SolveError(f"Swift CLI exited with code {proc.returncode}: {stderr}", code=proc.returncode)

    stderr_file.close()

    if not output_file.exists():
        raise SolveError("Processing completed but output file was not created")

    return SolveResult(output_path=output_file, warnings=warnings)
