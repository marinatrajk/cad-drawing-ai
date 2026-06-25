"""
Generate sample STEP files for testing the pipeline.
Run: python -m src.generate_samples
"""

import os
import cadquery as cq


SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")


def make_bracket():
    """L-bracket with mounting holes."""
    part = (
        cq.Workplane("XY")
        .box(80, 60, 8, centered=False)  # horizontal plate
        .faces(">Z")
        .workplane()
        .pushPoints([(15, 15), (65, 15), (15, 45), (65, 45)])
        .hole(5)  # 4x M5 mounting holes, through
        # vertical plate
        .faces(">Y")
        .workplane(offset=60)
        .box(80, 40, 8, centered=(True, False, True))
        .faces(">X")
        .workplane(offset=-40)
        .center(0, 15)
        .hole(6.5)  # clearance hole
    )
    path = os.path.join(SAMPLES_DIR, "bracket.step")
    part.val().exportStep(path)
    print(f"  Created: {path}")
    return path


def make_plate():
    """Simple flat plate with holes and a cutout."""
    part = (
        cq.Workplane("XY")
        .box(120, 80, 10, centered=False)
        .faces(">Z")
        .workplane()
        .pushPoints([(20, 20), (100, 20), (20, 60), (100, 60)])
        .hole(4.2)  # 4x M4 clearance
        .center(60, 40)
        .slot2D(40, 20, 0)
        .cutThruAll()  # center slot
    )
    path = os.path.join(SAMPLES_DIR, "plate.step")
    part.val().exportStep(path)
    print(f"  Created: {path}")
    return path


def make_shaft():
    """Stepped shaft / axle."""
    part = (
        cq.Workplane("XY")
        .circle(10)
        .extrude(50)
        .faces("<Z")
        .workplane()
        .circle(15)
        .extrude(10)  # larger base
        .faces(">Z")
        .workplane()
        .circle(7)
        .extrude(20)  # reduced tip
    )
    path = os.path.join(SAMPLES_DIR, "shaft.step")
    part.val().exportStep(path)
    print(f"  Created: {path}")
    return path


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    print("Generating sample STEP files...")
    make_bracket()
    make_plate()
    make_shaft()
    print("\nDone. Test with:")
    print("  python -m src.main samples/bracket.step --no-ai")
    print("  python -m src.main samples/plate.step")


if __name__ == "__main__":
    main()
