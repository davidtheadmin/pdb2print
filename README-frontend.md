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
Returns:

```json
{
  "ok": true,
  "warning": null,
  "report": "Built N chain(s): …",
  "glb_url": "/files/<token>/out.glb",
  "threemf_url": "/files/<token>/out.3mf",
  "stl_url": "/files/<token>/out_stl.zip",
  "chains": [{ "id": "A", "color": "#D93333" }]
}
```

If `build_all` fails the watertight gate it returns `ok:false` with the message
in both `warning` and `report`. If only 3MF export fails, GLB + STL URLs are
still returned and the 3MF error is placed in `warning` (shown in dark red).
