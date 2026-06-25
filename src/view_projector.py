"""
View projector: generates 2D orthographic views from a 3D STEP part.

Uses CadQuery's built-in SVG exporter for projection + hidden line removal,
then converts the SVG paths to DXF-ready line segments.
"""

import cadquery as cq
import numpy as np
from svgpathtools import svg2paths, Path as SvgPath
from typing import List, Tuple
import io
import os
import tempfile


# Standard orthographic view directions (projectionDir in CadQuery)
VIEWS = {
    "front":  (0, -1, 0),
    "top":    (0, 0, 1),
    "right":  (1, 0, 0),
    "iso":    (1, -1, 1),  # isometric, optional
}


def export_view_svg(shape: cq.Workplane, view_name: str, output_path: str) -> dict:
    """
    Export a single orthographic view as SVG using CadQuery's projector.

    Returns metadata about the view (width, height in SVG units).
    """
    proj_dir = VIEWS.get(view_name, VIEWS["front"])

    cq.exporters.export(
        shape,
        output_path,
        exportType="SVG",
        opt={
            "projectionDir": proj_dir,
            "showAxes": False,
            "showHidden": True,  # include hidden lines as dashed
            "marginLeft": 10,
            "marginTop": 10,
            "strokeWidth": 0.5,
            "strokeColor": (0, 0, 0),
            "hiddenColor": (100, 100, 100),
            "showOutline": True,
        },
    )

    # Read back the SVG to get dimensions
    with open(output_path, "r") as f:
        content = f.read()

    # Parse width/height from SVG (may be on separate lines)
    import re
    width, height = 0, 0
    w_match = re.search(r'width="([\d.]+)', content)
    h_match = re.search(r'height="([\d.]+)', content)
    if w_match:
        width = float(w_match.group(1))
    if h_match:
        height = float(h_match.group(1))

    return {"view": view_name, "width": width, "height": height, "path": output_path}


def svg_to_line_segments(svg_path: str) -> List[dict]:
    """
    Parse an SVG file and extract line segments for DXF output.

    Returns list of segments: {"type": "line"|"hidden", "points": [[x,y], ...]}
    """
    paths, attributes = svg2paths(svg_path)

    segments = []
    for i, (path, attr) in enumerate(zip(paths, attributes)):
        # Determine if this is a hidden line (dashed stroke)
        is_hidden = "dash" in str(attr.get("stroke-dasharray", "")) or \
                    "dash" in str(attr.get("style", ""))

        # Convert path segments to points
        points = []
        for segment in path:
            if hasattr(segment, "start"):
                start = segment.start
                end = segment.end
                points.append([start.real, -start.imag])  # flip Y for CAD coords
                points.append([end.real, -end.imag])

                # For curves, sample intermediate points
                if hasattr(segment, "control1"):
                    # Cubic bezier: sample N points
                    n_samples = 10
                    for t in np.linspace(0.1, 0.9, n_samples - 1):
                        pt = segment.point(t)
                        points.append([pt.real, -pt.imag])

        if len(points) >= 2:
            segments.append({
                "type": "hidden" if is_hidden else "visible",
                "points": points,
            })

    return segments


def project_all_views(shape: cq.Workplane, output_dir: str = None) -> dict:
    """
    Generate all standard orthographic views from a 3D shape.

    Returns:
        {
            "front": {"segments": [...], "width": w, "height": h},
            "top": {...},
            "right": {...},
        }
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="cad_views_")

    results = {}
    for view_name in ["front", "top", "right"]:
        svg_path = os.path.join(output_dir, f"{view_name}.svg")

        try:
            meta = export_view_svg(shape, view_name, svg_path)
            segments = svg_to_line_segments(svg_path)
            results[view_name] = {
                "segments": segments,
                "width": meta["width"],
                "height": meta["height"],
            }
            print(f"  [{view_name}] {len(segments)} segments, {meta['width']:.0f} x {meta['height']:.0f}")
        except Exception as e:
            print(f"  [{view_name}] ERROR: {e}")
            results[view_name] = {"segments": [], "width": 0, "height": 0, "error": str(e)}

    return results
