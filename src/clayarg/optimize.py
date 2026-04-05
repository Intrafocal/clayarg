"""Phase IV: Mesh optimization via PyMeshLab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymeshlab


@dataclass
class OptimizeResult:
    output_path: Path
    original_faces: int
    final_faces: int
    scale_applied: float | None


def optimize_mesh(
    input_file: Path,
    output_file: Path,
    target_faces: int = 10_000,
    scale_dimension_mm: float | None = None,
    output_format: str = "obj",
) -> OptimizeResult:
    """Decimate, repair, and optionally scale a mesh for Fusion 360.

    Args:
        input_file: Path to the input mesh (.usdz, .obj, .ply, etc).
        output_file: Where to write the optimized mesh.
        target_faces: Target triangle count after decimation.
        scale_dimension_mm: If provided, scale the mesh so its bounding box
            longest axis equals this value in mm.
        output_format: Output format (obj, stl, ply).

    Returns:
        OptimizeResult with stats.
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_file))

    original_faces = ms.current_mesh().face_number()

    # Decimation first — repair can change topology in ways that block edge collapse
    if original_faces > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            qualitythr=0.3,
        )

    # Manifold repair after decimation
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=100)
    ms.meshing_remove_unreferenced_vertices()

    # Scaling
    scale_applied = None
    if scale_dimension_mm and scale_dimension_mm > 0:
        bbox = ms.current_mesh().bounding_box()
        longest_axis = max(bbox.dim_x(), bbox.dim_y(), bbox.dim_z())
        if longest_axis > 0:
            scale_factor = scale_dimension_mm / longest_axis
            ms.compute_matrix_from_scaling_or_normalization(
                scalecenter="barycenter",
                axisx=scale_factor,
                axisy=scale_factor,
                axisz=scale_factor,
                uniformflag=False,
            )
            scale_applied = scale_factor

    final_faces = ms.current_mesh().face_number()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(output_file))

    return OptimizeResult(
        output_path=output_file,
        original_faces=original_faces,
        final_faces=final_faces,
        scale_applied=scale_applied,
    )
