"""ClayArgus CLI — entry point for the photogrammetry pipeline."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

from clayarg import __version__
from clayarg.quality import run_quality_analysis
from clayarg.framing import run_framing
from clayarg.preview import generate_mjpeg
from clayarg.scale import KNOWN_REFERENCES, estimate_scale
from clayarg.solve import SolveError, find_swift_cli, run_solve
from clayarg.optimize import optimize_mesh


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".heic", ".dng", ".png"}

# Pillow can read these directly; HEIC/DNG need special handling
PILLOW_READABLE = {".jpg", ".jpeg", ".png"}


def _collect_images(input_dir: Path) -> list[Path]:
    """Collect image files, preferring JPG over HEIC when both exist."""
    all_images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
    )
    if not all_images:
        return []

    # Deduplicate: if a JPG conversion exists alongside HEIC, prefer JPG
    stems_with_jpg = {p.stem for p in all_images if p.suffix.lower() in {".jpg", ".jpeg"}}
    deduped = []
    for p in all_images:
        if p.suffix.lower() == ".heic":
            # Skip HEIC if a JPG with matching stem (or stem derived from HEIC name) exists
            # Handles both "IMG_001.jpg" and "IMG_001.HEIC.JPG" patterns
            if p.stem in stems_with_jpg or f"{p.name}.JPG" in {x.name for x in all_images}:
                continue
        deduped.append(p)
    return deduped


def preflight(input_dir: Path) -> list[Path]:
    """Validate environment and inputs. Returns list of image files."""
    errors = []

    # macOS version check
    mac_ver = platform.mac_ver()[0]
    if mac_ver:
        major = int(mac_ver.split(".")[0])
        if major < 14:
            errors.append(f"macOS 14+ required (found {mac_ver})")
    else:
        errors.append("Could not determine macOS version")

    # Swift CLI check
    try:
        find_swift_cli()
    except Exception:
        errors.append(
            "clayarg-capture not found. Build it with: cd src/swift && swift build -c release"
        )

    # Input directory
    if not input_dir.is_dir():
        errors.append(f"'{input_dir}' is not a directory")
    else:
        images = _collect_images(input_dir)
        if not images:
            errors.append(
                f"No supported images in '{input_dir}' "
                f"(supported: {', '.join(SUPPORTED_EXTENSIONS)})"
            )

    # Disk space — warn if < 2GB free
    usage = shutil.disk_usage(input_dir if input_dir.is_dir() else Path.cwd())
    free_gb = usage.free / (1024**3)
    if free_gb < 2.0:
        errors.append(f"Low disk space: {free_gb:.1f} GB free (recommend 2+ GB)")

    if errors:
        for e in errors:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    return images


def progress_callback(fraction: float, message: str) -> None:
    bar_width = 30
    filled = int(bar_width * fraction)
    bar = "█" * filled + "░" * (bar_width - filled)
    # Clear entire line, then write progress
    print(f"\r\033[K  [{bar}] {fraction:.0%}  {message}", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="clayarg",
        description="ClayArgus — full-local macOS photogrammetry pipeline",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    # `clayarg run` — the main pipeline
    run_parser = sub.add_parser("run", help="Run the photogrammetry pipeline")
    run_parser.add_argument("input_dir", type=Path, help="Folder containing source images")
    run_parser.add_argument("--name", required=True, help="Project name (used for output filenames)")
    run_parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <input_dir>/clayarg_<name>)")
    run_parser.add_argument("--detail", default="medium", choices=["preview", "reduced", "medium", "full", "raw"])
    run_parser.add_argument("--sensitivity", default="normal", choices=["normal", "high"])
    run_parser.add_argument("--poly-count", type=int, default=10_000, help="Target face count (default: 10000)")
    run_parser.add_argument("--scale-obj", default=None, choices=["smaller", "larger"], help="Which detected object is the scale reference")
    run_parser.add_argument("--scale-ref", default=None, choices=list(KNOWN_REFERENCES.keys()), help="Known reference object (e.g., quarter, euro, loonie)")
    run_parser.add_argument("--scale-mm", type=float, default=None, help="Reference object size in mm (alternative to --scale-ref)")
    run_parser.add_argument("--format", default="obj", choices=["obj", "stl", "ply"], dest="output_format")
    run_parser.add_argument("--skip-qa", action="store_true", help="Skip image quality analysis")
    run_parser.add_argument("--blur-threshold", type=float, default=100.0, help="Blur threshold (default: 100.0)")
    run_parser.add_argument("--exposure-tolerance", type=float, default=0.15, help="Exposure clipping tolerance (default: 0.15)")
    run_parser.add_argument("--duplicate-threshold", type=int, default=3, help="Perceptual hash hamming distance for duplicates (default: 3, higher=stricter, 0=off)")
    run_parser.add_argument("--skip-framing", action="store_true", help="Skip Apple Vision subject framing/crop")
    run_parser.add_argument("--crop-padding", type=float, default=0.1, help="Padding around detected subject (default: 0.1)")

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(args)


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or args.input_dir / f"clayarg_{args.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"{args.name}_raw.obj"
    mesh_path = output_dir / f"{args.name}_fusion.{args.output_format}"
    preview_path = output_dir / "preview.mjpeg"
    report_path = output_dir / "processing_report.json"

    report: dict = {"project": args.name, "phases": {}}

    # Pre-flight
    print(f"ClayArgus v{__version__}")
    print(f"Project: {args.name}")
    images = preflight(args.input_dir)
    print(f"Found {len(images)} images in '{args.input_dir}'")
    print()

    # Phase I: Image Quality Analysis
    if not args.skip_qa:
        print("Phase I: Image Quality Analysis")
        t0 = time.monotonic()
        scores = run_quality_analysis(
            images,
            blur_threshold=args.blur_threshold,
            exposure_tolerance=args.exposure_tolerance,
            duplicate_threshold=args.duplicate_threshold,
        )
        elapsed = time.monotonic() - t0

        flagged = [s for s in scores if s.flagged]
        passed = [s for s in scores if not s.flagged]

        if flagged:
            print(f"  Flagged {len(flagged)} of {len(scores)} images:")
            for s in flagged:
                reasons = "; ".join(s.flag_reasons)
                print(f"    {s.path.name}: {reasons}")
        print(f"  {len(passed)} images passed quality checks ({elapsed:.1f}s)")

        report["phases"]["quality"] = {
            "elapsed_s": round(elapsed, 1),
            "total": len(scores),
            "passed": len(passed),
            "flagged": len(flagged),
            "details": [
                {
                    "file": s.path.name,
                    "blur_score": round(s.blur_score, 1),
                    "overexposed": round(s.overexposed_frac, 4),
                    "underexposed": round(s.underexposed_frac, 4),
                    "flagged": s.flagged,
                    "reasons": s.flag_reasons,
                }
                for s in scores
            ],
        }

        # Use only passing images for subsequent phases
        images = [s.path for s in passed]
        if not images:
            print("\nError: All images were flagged. Use --skip-qa to bypass.", file=sys.stderr)
            sys.exit(1)
        print()

    # Phase II: Scale Detection (must run before framing crops out the reference)
    scale_mm = None
    has_scale_args = args.scale_obj and args.scale_ref
    if has_scale_args:
        # Resolve reference size
        ref_mm = KNOWN_REFERENCES[args.scale_ref]
        ref_name = args.scale_ref
        if ref_mm is None:
            # Generic "coin" — needs --scale-mm
            if not args.scale_mm:
                print("Error: --scale-ref coin requires --scale-mm <diameter>", file=sys.stderr)
                sys.exit(1)
            ref_mm = args.scale_mm
            ref_name = f"coin ({args.scale_mm}mm)"
        elif args.scale_mm:
            print("Error: --scale-mm not needed with a named reference", file=sys.stderr)
            sys.exit(1)

        print(f"Phase II: Scale Detection ({ref_name})")
        t0 = time.monotonic()
        scale_est = estimate_scale(images, args.scale_obj, ref_mm, reference_name=ref_name)
        elapsed = time.monotonic() - t0

        if scale_est is None:
            print("  Could not detect two objects in enough images for scale estimation.")
            print("  Continuing without scale.")
            print()
        else:
            print(f"  Reference: {scale_est.reference_name} ({scale_est.reference_mm}mm)")
            print(f"  Sampled from {scale_est.samples} images (confidence: {scale_est.confidence})")
            print(f"  Estimated subject longest dimension: {scale_est.subject_mm:.1f}mm")
            print()

            report["phases"]["scale"] = {
                "elapsed_s": round(elapsed, 1),
                "reference": scale_est.reference_name,
                "reference_mm": scale_est.reference_mm,
                "subject_mm": round(scale_est.subject_mm, 1),
                "pixels_per_mm": scale_est.pixels_per_mm,
                "samples": scale_est.samples,
                "confidence": scale_est.confidence,
            }

            # Confirm with user — accept, reject, or correct
            try:
                response = input(
                    f"Apply scale ({scale_est.subject_mm:.1f}mm)? [Y/n/correct value in mm] "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(0)
            if not response or response.lower() in ("y", "yes"):
                scale_mm = scale_est.subject_mm
                print(f"  Scale will be applied: {scale_mm:.1f}mm")
            elif response.lower() in ("n", "no"):
                print("  Scale not applied.")
            else:
                try:
                    scale_mm = float(response)
                    print(f"  Scale will be applied: {scale_mm:.1f}mm (user corrected)")
                except ValueError:
                    print("  Invalid value. Scale not applied.")
            print()
    elif args.scale_obj and not args.scale_ref:
        print("Error: --scale-obj requires --scale-ref", file=sys.stderr)
        sys.exit(1)
    elif args.scale_ref and not args.scale_obj:
        print("Error: --scale-ref requires --scale-obj (smaller or larger)", file=sys.stderr)
        sys.exit(1)

    # Phase III: Subject Framing
    if args.skip_framing:
        print("Skipping Phase III: Subject Framing")
        print()
    else:
        print("Phase III: Subject Framing (Apple Vision)")
        t0 = time.monotonic()
        images, framing_results = run_framing(
            images, output_dir, padding=args.crop_padding
        )
        elapsed = time.monotonic() - t0

        edge_warnings = [r for r in framing_results if r.edge_flags]
        if edge_warnings:
            print(f"  Warning: {len(edge_warnings)} images have subject near edge:")
            for r in edge_warnings:
                edges = ", ".join(r.edge_flags)
                print(f"    {r.path.name}: touches {edges}")

        cropped_count = sum(1 for r in framing_results)
        failed_count = len(images) - cropped_count
        print(f"  Cropped {cropped_count} images ({elapsed:.1f}s)", end="")
        if failed_count:
            print(f" ({failed_count} passed through uncropped)")
        else:
            print()

        report["phases"]["framing"] = {
            "elapsed_s": round(elapsed, 1),
            "cropped": cropped_count,
            "uncropped": failed_count,
            "edge_warnings": len(edge_warnings),
            "details": [
                {
                    "file": r.path.name,
                    "crop_box": list(r.crop_box),
                    "edge_flags": r.edge_flags,
                }
                for r in framing_results
            ],
        }
        print()

    # Phase IV: Preview
    print("Phase IV: Visual Preview")
    t0 = time.monotonic()
    generate_mjpeg(images, preview_path)
    elapsed = time.monotonic() - t0
    print(f"  Generated {preview_path} ({len(images)} frames, {elapsed:.1f}s)")
    print(f"  Review the preview and individual images before proceeding.")
    report["phases"]["preview"] = {
        "elapsed_s": round(elapsed, 1),
        "frames": len(images),
        "output": str(preview_path),
    }
    print()

    # Wait for user confirmation before the expensive solve
    try:
        response = input("Proceed with 3D solve? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)
    if response and response not in ("y", "yes"):
        print("Aborted.")
        sys.exit(0)
    print()

    # Phase V: 3D Solve
    # Clean stale outputs that would cause Object Capture to fail
    for stale in (raw_path, raw_path.with_suffix(".usdz")):
        if stale.exists():
            stale.unlink()
    print("Phase V: 3D Solve")
    print(f"  Detail: {args.detail} | Sensitivity: {args.sensitivity}")
    t0 = time.monotonic()
    try:
        solve_result = run_solve(
            input_dir=args.input_dir,
            output_file=raw_path,
            detail=args.detail,
            sensitivity=args.sensitivity,
            on_progress=progress_callback,
        )
    except SolveError as e:
        print(f"\nError during 3D solve: {e}", file=sys.stderr)
        sys.exit(e.code)

    elapsed = time.monotonic() - t0
    print(f"\n  Done in {elapsed:.1f}s → {solve_result.output_path}")
    report["phases"]["solve"] = {
        "elapsed_s": round(elapsed, 1),
        "output": str(solve_result.output_path),
        "warnings": solve_result.warnings,
    }
    print()

    # Phase VI: Mesh Optimization
    print("Phase VI: Mesh Optimization")
    print(f"  Target: {args.poly_count} faces | Format: {args.output_format}")
    t0 = time.monotonic()
    try:
        opt_result = optimize_mesh(
            input_file=raw_path,
            output_file=mesh_path,
            target_faces=args.poly_count,
            scale_dimension_mm=scale_mm,
            output_format=args.output_format,
        )
    except Exception as e:
        print(f"\nError during optimization: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.monotonic() - t0
    print(f"  {opt_result.original_faces} → {opt_result.final_faces} faces")
    if opt_result.scale_applied:
        print(f"  Scale factor: {opt_result.scale_applied:.4f}")
    print(f"  Done in {elapsed:.1f}s → {opt_result.output_path}")

    report["phases"]["optimize"] = {
        "elapsed_s": round(elapsed, 1),
        "original_faces": opt_result.original_faces,
        "final_faces": opt_result.final_faces,
        "scale_applied": opt_result.scale_applied,
        "output": str(opt_result.output_path),
    }

    # Write report
    report_path.write_text(json.dumps(report, indent=2))
    print()
    print(f"Report: {report_path}")
    print("Done.")
