# CAD Drawing AI

Convert 3D STEP files into 2D manufacturing drawings (DXF) using AI for dimensioning decisions.

## What it does

```
STEP file → parse geometry → project orthographic views → AI decides dimensions → output DXF
```

The LLM doesn't do geometry math. It makes engineering decisions (which views to dimension, what features need callouts, what tolerances to apply). The Python CAD libraries handle the actual projection and drawing.

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/cad-drawing-ai.git
cd cad-drawing-ai
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample parts for testing
python -m src.generate_samples

# 4. Run without AI first (bounding-box dimensions only)
python -m src.main samples/bracket.step --no-ai

# 5. Run with AI (needs Anthropic API key)
export ANTHROPIC_API_KEY="sk-ant-..."
python -m src.main samples/bracket.step
```

Output: `samples/bracket.dxf` — open in any CAD viewer (SolidWorks, AutoCAD, LibreCAD, etc.)

## Web interface (optional)

```bash
python -m src.server
# Open http://localhost:8000 in your browser
```

Upload a STEP file, get a DXF back. Useful for demos.

## How it works

### 1. STEP parsing (`src/step_parser.py`)
Uses [CadQuery](https://github.com/CadQuery/cadquery) (OpenCascade bindings) to load the STEP file and extract:
- Bounding box dimensions
- Volume and surface area
- Face types (planar, cylindrical, conical)
- Hole detection (radius, depth, through/not-through)
- Symmetry checks

### 2. View projection (`src/view_projector.py`)
Generates standard orthographic views (front, top, right) as SVG using CadQuery's projector with hidden line removal, then converts SVG paths to line segments for DXF output.

### 3. AI dimensioning (`src/ai_dim.py`)
Sends the structured geometry metadata to an LLM (Anthropic Claude) with a system prompt that encodes drafting best practices. The LLM returns JSON specifying:
- Which dimensions go on which views
- Dimension types (linear, diameter, radius)
- Tolerances for critical features
- Manufacturing annotations and notes

If no API key is set, it falls back to simple bounding-box dimensions.

### 4. DXF assembly (`src/dxf_writer.py`)
Uses [ezdxf](https://github.com/mozman/ezdxf) to assemble the projected views and AI dimensions into a DXF file with:
- Visible and hidden line layers
- Dimension layer
- Title block with part info

## Project structure

```
cad-drawing-ai/
├── src/
│   ├── __init__.py
│   ├── step_parser.py       # STEP → geometry metadata
│   ├── view_projector.py    # 3D → 2D orthographic views
│   ├── ai_dim.py            # LLM dimensioning decisions
│   ├── dxf_writer.py        # Assemble views + dims → DXF
│   ├── main.py              # CLI entry point
│   ├── server.py            # Optional web interface
│   └── generate_samples.py  # Generate test STEP files
├── samples/                 # Generated test parts
├── requirements.txt
└── README.md
```

## Current limitations

- **Simple parts only.** Complex geometry (lofts, sweeps, fillets everywhere) may not project cleanly.
- **No GD&T.** Geometric dimensioning and tolerancing is not yet supported.
- **Coordinate mapping is approximate.** The AI gets bounding-box-level position info, not pixel-perfect coordinates. Dimension placement will need refinement.
- **No section views.** Only standard orthographic projections.
- **No assembly drawings.** Single parts only.

## Roadmap

- [ ] Section views
- [ ] Isometric view in corner
- [ ] GD&T symbols
- [ ] Thread callouts
- [ ] Bill of materials
- [ ] Custom title blocks per company
- [ ] Batch processing
- [ ] SolidWorks add-in (direct integration)

## Tech stack

| Component | Library |
|-----------|---------|
| CAD parsing | CadQuery / OpenCascade |
| DXF output | ezdxf |
| AI | Anthropic Claude API |
| CLI | Click |
| Web server | FastAPI + Uvicorn |

## License

MIT
