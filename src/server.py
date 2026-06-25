"""
Optional web server: upload a STEP file, get back a DXF drawing.

Usage:
    python -m src.server
    # then open http://localhost:8000

Requires: pip install fastapi uvicorn python-multipart
"""

import os
import tempfile
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .step_parser import load_step, extract_metadata
from .view_projector import project_all_views
from .ai_dim import generate_dimensions, _fallback_dimensions
from .dxf_writer import write_dxf

app = FastAPI(title="CAD Drawing AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = tempfile.mkdtemp(prefix="cad_uploads_")


@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html>
<head><title>CAD Drawing AI</title>
<style>
  body { font-family: system-ui; max-width: 600px; margin: 80px auto; padding: 20px; }
  h1 { color: #333; }
  .upload { border: 2px dashed #aaa; padding: 40px; text-align: center; border-radius: 8px; }
  input[type=file] { margin: 10px 0; }
  button { padding: 10px 24px; font-size: 16px; cursor: pointer; background: #2563eb; color: white; border: none; border-radius: 4px; }
  button:hover { background: #1d4ed8; }
  .status { margin-top: 20px; color: #666; }
</style>
</head>
<body>
  <h1>CAD Drawing AI</h1>
  <p>Upload a STEP file, get a 2D manufacturing drawing (DXF).</p>
  <div class="upload">
    <form action="/convert" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".step,.stp" required>
      <br>
      <label><input type="checkbox" name="no_ai"> Skip AI (use fallback dimensions)</label>
      <br><br>
      <button type="submit">Convert</button>
    </form>
  </div>
</body>
</html>"""


@app.post("/convert")
async def convert(file: UploadFile = File(...), no_ai: bool = False):
    if not file.filename.endswith((".step", ".stp")):
        raise HTTPException(400, "Only STEP (.step/.stp) files are supported")

    # Save uploaded file
    step_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(step_path, "wb") as f:
        f.write(await file.read())

    start = time.time()

    # Pipeline
    shape = load_step(step_path)
    metadata = extract_metadata(step_path, shape)
    views = project_all_views(shape)

    if no_ai or not os.environ.get("ANTHROPIC_API_KEY"):
        dim_result = _fallback_dimensions(metadata)
    else:
        dim_result = generate_dimensions(metadata)

    dxf_path = step_path.replace(".step", ".dxf").replace(".stp", ".dxf")
    write_dxf(views, metadata, dim_result["dimensions"], dxf_path)

    elapsed = time.time() - start
    print(f"Converted {file.filename} in {elapsed:.1f}s")

    return FileResponse(dxf_path, filename=os.path.basename(dxf_path),
                        media_type="application/dxf")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
