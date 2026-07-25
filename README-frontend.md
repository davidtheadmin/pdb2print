# pdb2print — custom front end

A polished, single-screen desktop-style UI that replaces the Gradio `app.py`,
plus a thin FastAPI layer wrapping the existing geometry pipeline. **The
`pdb2print/` geometry package is untouched.**

## Files

- `frontend/index.html` — self-contained single-page app (dark tool UI,
  interactive `<model-viewer>` preview, segmented toggles, preset chips,
  advanced-settings drawer). `<model-viewer>` loads from CDN.
- `server.py` — FastAPI app. Serves the front end and exposes
  `POST /api/generate`, which calls `pdb2print.pipeline.build_all(...)` and
  writes GLB / 3MF / per-chain STL-zip outputs.
- `requirements-server.txt` — FastAPI/uvicorn deps on top of the core
  `requirements.txt`.

## Run

```bash
pip install -r requirements.txt -r requirements-server.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

Then open http://localhost:7860 . Try example IDs **1UBQ** (protein), **1BNA**
(DNA), or **1ZAA** (zinc-finger–DNA complex, exercises both paths).

> **Legacy:** the old Gradio UI still runs with `python app.py`.

## Deploy to Hugging Face Spaces (CPU)

Use a **Docker** or **FastAPI**-style Space that runs
`uvicorn server:app --host 0.0.0.0 --port 7860`. No GPU or paid services are
required. The Space needs outbound network access to fetch PDB IDs from RCSB;
uploaded files work without network access.

## API contract

`POST /api/generate` (multipart form): parameter fields + optional `file`.

The response is **`text/event-stream`**: zero or more `event: progress`
(`{frac, msg}`) lines, then a single `event: result` carrying the JSON below.
Validation problems (bad file type, missing source, unparseable params)
short-circuit to plain JSON with a 4xx status, so the client tells the two apart
by content-type.

```json
{
  "ok": true,
  "warning": null,
  "report": "Built N chain(s): …",
  "glb_url": "/files/<token>/1zaa_pdb2print_1p5mm.glb",
  "threemf_url": "/files/<token>/1zaa_pdb2print_1p5mm.3mf",
  "stl_url": "/files/<token>/1zaa_pdb2print_1p5mm_stl.zip",
  "chains": [{ "id": "A", "name": "Zif268", "color": "#D93333" }],
  "connections": [{ "a": "A", "b": "E", "method": "magnet", "applied": true,
                    "note": "" }],
  "size_mm": [61.2, 44.0, 38.5],
  "scale_used": 1.5
}
```

Output files are named after the structure and scale, not `out.3mf`, so a
downloads folder with several builds stays readable. The uuid directory (not the
filename) is what keeps concurrent builds apart.

Connection fields beyond the original set: `socket`, `socket_wall`,
`magnet_fit_clearance`.

If `build_all` fails the watertight gate it returns `ok:false` with the message
in both `warning` and `report`. If only 3MF export fails, GLB + STL URLs are
still returned and the 3MF error is placed in `warning` (shown in dark red).
