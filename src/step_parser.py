"""
STEP file parser using CadQuery / OpenCascade.

Loads a STEP file, extracts structured geometry metadata that the LLM
needs to make dimensioning decisions, and provides the raw shape for
projection.
"""

import cadquery as cq
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class FaceInfo:
    """Metadata about a single face on the part."""
    face_type: str  # 'planar', 'cylindrical', 'conical', 'other'
    area: float
    normal: Optional[list] = None  # direction vector for planar faces
    radius: Optional[float] = None  # for cylindrical/conical faces


@dataclass
class HoleInfo:
    """Metadata about a hole feature."""
    radius: float
    depth: float  # mm, -1 if through
    position: list  # [x, y, z] center
    is_through: bool


@dataclass
class PartMetadata:
    """Structured geometry description sent to the LLM."""
    filename: str
    bounding_box: dict  # {min: [x,y,z], max: [x,y,z], size: [dx,dy,dz]}
    volume: float  # mm³
    surface_area: float  # mm²
    face_count: int
    edge_count: int
    vertex_count: int
    faces: list = field(default_factory=list)  # list of FaceInfo dicts
    holes: list = field(default_factory=list)  # list of HoleInfo dicts
    is_symmetric_x: bool = False
    is_symmetric_y: bool = False
    is_symmetric_z: bool = False
    estimated_material: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary(self) -> str:
        """Human-readable summary for the LLM prompt."""
        bb = self.bounding_box
        size = bb["size"]
        lines = [
            f"Part: {self.filename}",
            f"Bounding box: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm",
            f"Volume: {self.volume:.0f} mm³",
            f"Surface area: {self.surface_area:.0f} mm²",
            f"Faces: {self.face_count} ({len(self.holes)} holes detected)",
        ]
        if self.holes:
            hole_descs = []
            for h in self.holes[:10]:  # cap for prompt size
                depth = "through" if h["is_through"] else f"{h['depth']:.1f}mm deep"
                hole_descs.append(f"  Ø{h['radius']*2:.1f}mm, {depth}, at {h['position']}")
            lines.append("Holes:")
            lines.extend(hole_descs)
        return "\n".join(lines)


def load_step(filepath: str) -> cq.Workplane:
    """Load a STEP file and return a CadQuery Workplane object."""
    return cq.importers.importStep(filepath)


def extract_metadata(filepath: str, shape: Optional[cq.Workplane] = None) -> PartMetadata:
    """
    Extract structured metadata from a STEP part.

    Args:
        filepath: path to the .step / .stp file
        shape: pre-loaded CadQuery workplane (optional, saves a reload)
    """
    if shape is None:
        shape = load_step(filepath)

    solid = shape.val()
    filename = filepath.split("/")[-1].split("\\")[-1]

    # Bounding box
    bb = solid.BoundingBox()
    bounding_box = {
        "min": [round(bb.xmin, 3), round(bb.ymin, 3), round(bb.zmin, 3)],
        "max": [round(bb.xmax, 3), round(bb.ymax, 3), round(bb.zmax, 3)],
        "size": [round(bb.xlen, 3), round(bb.ylen, 3), round(bb.zlen, 3)],
    }

    # Volume and surface area
    volume = solid.Volume()
    surface_area = solid.Area()

    # Face analysis
    faces_list = []
    holes_list = []

    try:
        for face in shape.faces().vals():
            face_type = "other"
            radius = None
            normal = None
            area = face.Area()

            try:
                # Get surface type from the geometry object class name
                # Newer CadQuery/OCP returns Geom_Plane, Geom_CylindricalSurface, etc.
                surf = face._geomAdaptor()
                typename = type(surf).__name__

                if "Plane" in typename:
                    face_type = "planar"
                    try:
                        normal = [round(n, 4) for n in face.normalAt().normalized().toTuple()]
                    except Exception:
                        pass
                elif "Cylindrical" in typename:
                    face_type = "cylindrical"
                    try:
                        cyl = surf.Cylinder()
                        radius = round(cyl.Radius(), 3)
                    except Exception:
                        pass
                elif "Conical" in typename:
                    face_type = "conical"
                    try:
                        cone = surf.Cone()
                        radius = round(cone.RefRadius(), 3)
                    except Exception:
                        pass
            except Exception:
                pass

            faces_list.append(asdict(FaceInfo(
                face_type=face_type,
                area=round(area, 3),
                normal=normal,
                radius=radius,
            )))

            # Detect holes: cylindrical faces with small radius relative to bbox
            if face_type == "cylindrical" and radius is not None:
                max_dim = max(bb.xlen, bb.ylen, bb.zlen)
                if radius < max_dim * 0.3:  # heuristic: hole if small vs part
                    center = face.Center()
                    is_through = False
                    depth = -1.0
                    try:
                        # Check if the cylinder spans the full dimension in its axis direction
                        # This is a rough heuristic
                        verts = face.Vertices()
                        if verts:
                            z_coords = [v.Z for v in verts]
                            if max(z_coords) - min(z_coords) >= bb.zlen * 0.9:
                                is_through = True
                            else:
                                depth = round(max(z_coords) - min(z_coords), 3)
                    except Exception:
                        pass

                    holes_list.append(asdict(HoleInfo(
                        radius=radius,
                        depth=depth,
                        position=[round(center.x, 3), round(center.y, 3), round(center.z, 3)],
                        is_through=is_through,
                    )))
    except Exception as e:
        # Face iteration can fail on complex geometries; continue with what we have
        print(f"  [warn] face analysis incomplete: {e}")

    # Deduplicate holes by position (tolerance 0.1mm)
    unique_holes = []
    for h in holes_list:
        is_dup = False
        for uh in unique_holes:
            if (abs(h["position"][0] - uh["position"][0]) < 0.1 and
                abs(h["position"][1] - uh["position"][1]) < 0.1 and
                abs(h["position"][2] - uh["position"][2]) < 0.1 and
                abs(h["radius"] - uh["radius"]) < 0.1):
                is_dup = True
                break
        if not is_dup:
            unique_holes.append(h)

    # Symmetry check (rough: compare bounding box halves)
    is_sym_x = abs(bb.xlen - 2 * (bb.center.x - bb.xmin)) < 0.01
    is_sym_y = abs(bb.ylen - 2 * (bb.center.y - bb.ymin)) < 0.01
    is_sym_z = abs(bb.zlen - 2 * (bb.center.z - bb.zmin)) < 0.01

    # Safe edge/vertex counts (avoid selector syntax issues across CadQuery versions)
    try:
        edge_count = len(shape.edges().vals())
    except Exception:
        edge_count = 0
    try:
        vertex_count = len(shape.vertices().vals())
    except Exception:
        vertex_count = 0

    return PartMetadata(
        filename=filename,
        bounding_box=bounding_box,
        volume=round(volume, 3),
        surface_area=round(surface_area, 3),
        face_count=len(faces_list),
        edge_count=edge_count,
        vertex_count=vertex_count,
        faces=faces_list[:50],  # cap for prompt size
        holes=unique_holes[:20],
        is_symmetric_x=is_sym_x,
        is_symmetric_y=is_sym_y,
        is_symmetric_z=is_sym_z,
    )
