"""
AI dimensioning engine: sends geometry metadata to an LLM and gets back
dimensioning decisions for the 2D drawing.

The LLM does NOT do geometry math. It makes engineering decisions:
- Which views to dimension
- What features need dimensions (overall size, holes, critical features)
- What tolerances to apply
- What annotations to add (surface finish, thread specs, etc.)

The CAD library executes the actual dimension placement.
"""

import json
import os
from typing import Optional


SYSTEM_PROMPT = """You are a mechanical engineering drafting expert. You receive structured geometry data extracted from a 3D STEP file and must decide how to dimension the 2D manufacturing drawing.

Your job:
1. Decide which dimensions to place on which views
2. Specify dimension types (linear, diameter, radius)
3. Suggest tolerances for critical features
4. Add manufacturing annotations (surface finish, thread specs, etc.)

Rules:
- Dimension overall width, depth, and height on the appropriate views
- Dimension every hole with a diameter callout
- Use standard drafting practices (ANSI/ISO)
- Don't over-dimension. Each feature gets dimensioned exactly once
- Provide coordinates relative to the view's local origin (0,0 = bottom-left of that view)
- Coordinates are in mm

You MUST respond with valid JSON only, no markdown, no explanation. The JSON schema:

{
  "dimensions": [
    {
      "view": "front" | "top" | "right",
      "type": "linear" | "diameter" | "radius",
      "from": [x, y],      // for linear: start point
      "to": [x, y],        // for linear: end point
      "offset": 15,         // dimension line offset from part edge (mm)
      "center": [x, y],    // for diameter/radius: center point
      "radius": 5.0,        // for diameter/radius: radius value
      "label": "Ø10",       // display label
      "tolerance": "+/-0.1" // optional
    }
  ],
  "annotations": [
    {
      "view": "front",
      "position": [x, y],
      "text": "Surface finish Ra 3.2"
    }
  ],
  "notes": [
    "All dimensions in mm",
    "Remove all sharp edges"
  ]
}
"""


def generate_dimensions(
    metadata,
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-8",
) -> dict:
    """
    Call the LLM with part metadata and get back dimensioning decisions.

    Args:
        metadata: PartMetadata object from step_parser
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        model: LLM model to use

    Returns: dict with "dimensions", "annotations", "notes"
    """
    try:
        import anthropic
    except ImportError:
        print("  [error] anthropic package not installed. Skipping AI dimensioning.")
        return _fallback_dimensions(metadata)

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("  [warn] No ANTHROPIC_API_KEY set. Using fallback dimensions.")
        return _fallback_dimensions(metadata)

    client = anthropic.Anthropic(api_key=key)

    # Build the prompt from metadata
    part_summary = metadata.to_summary()
    bb = metadata.bounding_box
    size = bb["size"]

    user_msg = f"""Here is the geometry data for a 3D part that needs a 2D manufacturing drawing:

{part_summary}

The bounding box size is {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm.
- Front view shows X (width) x Z (height)
- Top view shows X (width) x Y (depth)
- Right view shows Y (depth) x Z (height)

For each view, the local coordinate origin (0,0) is at the bottom-left corner.
The part fills the view, so use the bounding box dimensions to estimate positions.

Provide the dimensioning JSON now."""

    print(f"  Sending geometry to {model}...")
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    # Parse JSON from response
    text = response.content[0].text.strip()

    # Strip any markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    try:
        result = json.loads(text)
        print(f"  AI returned {len(result.get('dimensions', []))} dimensions")
        return result
    except json.JSONDecodeError as e:
        print(f"  [warn] LLM returned invalid JSON: {e}")
        print(f"  Raw response: {text[:200]}...")
        return _fallback_dimensions(metadata)


def _fallback_dimensions(metadata) -> dict:
    """
    Generate basic dimensions without AI, using just the bounding box.
    This is used when no API key is available or the LLM fails.
    """
    bb = metadata.bounding_box
    size = bb["size"]
    w, d, h = size

    dims = [
        # Front view: width (horizontal) and height (vertical)
        {
            "view": "front",
            "type": "linear",
            "from": [0, 0],
            "to": [w, 0],
            "offset": 15,
            "label": f"{w:.1f}",
        },
        {
            "view": "front",
            "type": "linear",
            "from": [0, 0],
            "to": [0, h],
            "offset": 15,
            "label": f"{h:.1f}",
        },
        # Top view: width and depth
        {
            "view": "top",
            "type": "linear",
            "from": [0, 0],
            "to": [w, 0],
            "offset": 15,
            "label": f"{w:.1f}",
        },
        {
            "view": "top",
            "type": "linear",
            "from": [0, 0],
            "to": [0, d],
            "offset": 15,
            "label": f"{d:.1f}",
        },
        # Right view: depth and height
        {
            "view": "right",
            "type": "linear",
            "from": [0, 0],
            "to": [d, 0],
            "offset": 15,
            "label": f"{d:.1f}",
        },
        {
            "view": "right",
            "type": "linear",
            "from": [0, 0],
            "to": [0, h],
            "offset": 15,
            "label": f"{h:.1f}",
        },
    ]

    # Add hole diameters on the view that shows them (front by default)
    for hole in metadata.holes:
        pos = hole["position"]
        dims.append({
            "view": "front",
            "type": "diameter",
            "center": [pos[0], pos[2]],  # X, Z in front view
            "radius": hole["radius"],
            "label": f"Ø{hole['radius'] * 2:.1f}",
        })

    return {
        "dimensions": dims,
        "annotations": [],
        "notes": ["All dimensions in mm", "Generated without AI (fallback mode)"],
    }
