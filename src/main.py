"""
Main pipeline: STEP → parse → project views → AI dimension → DXF

Usage:
    python -m src.main samples/bracket.step --output output/bracket.dxf
    python -m src.main samples/bracket.step --no-ai  # skip LLM, use fallback dims
"""

import os
import sys
import time
import click

from .step_parser import load_step, extract_metadata
from .view_projector import project_all_views
from .ai_dim import generate_dimensions
from .dxf_writer import write_dxf


@click.command()
@click.argument("step_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output DXF path (default: same name as input)")
@click.option("--no-ai", is_flag=True, help="Skip LLM dimensioning, use fallback bounding-box dims")
@click.option("--model", default="accounts/fireworks/models/glm-5p2", help="LLM model ID")
@click.option("--provider", default="fireworks", help="LLM provider: fireworks or anthropic")
@click.option("--verbose", "-v", is_flag=True, help="Print detailed progress")
def main(step_file, output, no_ai, model, provider, verbose):
    """Convert a 3D STEP file to a 2D manufacturing DXF drawing."""
    start = time.time()

    # Resolve output path
    if output is None:
        base = os.path.splitext(step_file)[0]
        output = base + ".dxf"

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CAD Drawing AI")
    print(f"  Input:  {step_file}")
    print(f"  Output: {output}")
    print(f"  AI:     {'off' if no_ai else 'on (' + provider + ': ' + model + ')'}")
    print(f"{'='*60}\n")

    # Step 1: Load and parse
    print("[1/4] Loading STEP file and extracting geometry...")
    shape = load_step(step_file)
    metadata = extract_metadata(step_file, shape)

    if verbose:
        print(f"\n{metadata.to_summary()}\n")
    else:
        bb = metadata.bounding_box["size"]
        print(f"  Part: {metadata.filename}")
        print(f"  Size: {bb[0]:.1f} x {bb[1]:.1f} x {bb[2]:.1f} mm")
        print(f"  Faces: {metadata.face_count}, Holes: {len(metadata.holes)}")

    # Step 2: Project views
    print("\n[2/4] Projecting orthographic views...")
    views = project_all_views(shape)

    # Step 3: AI dimensioning
    print("\n[3/4] Generating dimensions...")
    if no_ai:
        from .ai_dim import _fallback_dimensions
        dim_result = _fallback_dimensions(metadata)
        print(f"  Fallback: {len(dim_result['dimensions'])} dimensions")
    else:
        dim_result = generate_dimensions(metadata, model=model, provider=provider)

    # Step 4: Write DXF
    print("\n[4/4] Writing DXF drawing...")
    write_dxf(views, metadata, dim_result["dimensions"], output)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Drawing: {output}")
    print(f"  Dimensions: {len(dim_result['dimensions'])}")
    if dim_result.get("notes"):
        print(f"  Notes:")
        for note in dim_result["notes"]:
            print(f"    - {note}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
