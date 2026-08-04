"""Thin FastAPI layer for pdb2print.

Serves the static front end in ``frontend/`` and wraps the existing geometry
pipeline. All geometry lives in the ``pdb2print`` package; this module only
maps HTTP form fields to :class:`PrintParams`, calls ``build_all``, writes the
export files, and returns their URLs. Run with::

    uvicorn server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from typing import Optional

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pdb2print.config import (
    PrintParams, Representation, MinWallMode, BaseStyle, BackboneStyle,
    ConnectionParams, NoMagnetMethod, MagnetShape, StandParams, ColumnShape,
    PlaqueRelief, PlaqueFont,
    MoleculeType, LigandStyle, color_for_index, _palette_index,
)
from pdb2print.pipeline import build_all, BuildCancelled
from pdb2print import export
from pdb2print import cache as cache_mod
from pdb2print.cache import Cache, DEFAULT_CACHE_DIR


HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(HERE, "frontend")

# Neither extension has an entry in Python's built-in table, and python:3.12-slim
# ships no /etc/mime.types, so StaticFiles fell back to text/plain in the
# container -- which is in Caddy's default `encode` match list, so the proxy
# spent CPU gzipping a 3MF (already a zip) on every download.
mimetypes.add_type("model/3mf", ".3mf")
mimetypes.add_type("model/gltf-binary", ".glb")

# Every generation writes its outputs into a fresh sub-directory here, which is
# served read-only at /files/<token>/... for both <model-viewer> and downloads.
OUTPUT_ROOT = tempfile.mkdtemp(prefix="pdb2print_out_")

# The build cache. Shipped entries live in the repo and survive a restart, which
# the temp directory above deliberately does not.
#
# PDB2PRINT_CACHE_RO=1 turns off writing on misses. Set it wherever the disk is
# ephemeral (a free Space resets on every cold start): the pre-generated entries
# are still served, and a miss is simply rebuilt each time rather than written to
# a filesystem that will throw the result away anyway.
CACHE_DIR = os.environ.get("PDB2PRINT_CACHE_DIR", DEFAULT_CACHE_DIR)
CACHE_READ_ONLY = os.environ.get("PDB2PRINT_CACHE_RO", "").strip().lower() in {
    "1", "true", "on", "yes"}


def _cache_max_bytes():
    """Cache ceiling from PDB2PRINT_CACHE_MAX_GB, or the module default."""
    raw = os.environ.get("PDB2PRINT_CACHE_MAX_GB", "").strip()
    if not raw:
        return None
    try:
        return int(float(raw) * 1024 ** 3)
    except ValueError:
        return None


cache = Cache(CACHE_DIR, read_only=CACHE_READ_ONLY,
              max_bytes=_cache_max_bytes())
os.makedirs(CACHE_DIR, exist_ok=True)

UPLOAD_EXTS = {".pdb", ".ent", ".cif", ".mmcif", ".bcif"}

#: Largest structure file this will write to disk. The biggest things people
#: legitimately bring here are whole-virus mmCIFs, comfortably under this.
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024

#: Server-side bounds for the parameters that decide how much memory a build
#: allocates. The sliders advertise exactly these, and nothing enforced them
#: here: _map_params took a bare float(), resolve_surface_grid only ever refines
#: *finer*, and SURFACE_VOXEL_BUDGET caps refinement rather than a supplied
#: value -- so a crafted POST asking for grid_spacing 0.15 at scale 3.0 on a
#: small protein allocated about 2.5 GB, on a public endpoint with no rate
#: limiting.
#:
#: Clamped rather than rejected, because these arrive from a slider: a value out
#: of range is a bug or an attack, not a user choice worth an error page.
_PARAM_BOUNDS = {
    "scale": (0.2, 6.0),
    "grid_spacing": (0.2, 1.5),
    "min_wall": (0.0, 5.0),
    "probe_radius": (0.6, 5.0),
    "connector_diameter": (1.5, 12.0),
    "surface_padding": (0.0, 2.0),
}


def _bounded(fields: dict, name: str) -> float:
    """``fields[name]`` as a float, clamped to the range the slider offers.

    Raises like the bare ``float()`` it replaces when the field is missing or
    unparseable, so a malformed request still gets its 400.
    """
    lo, hi = _PARAM_BOUNDS[name]
    value = float(fields[name])
    if value != value:                        # NaN survives every comparison
        raise ValueError(f"{name} is not a number")
    return max(lo, min(hi, value))

#: Bumped whenever the front end and the server change together.
#:
#: The front end carries the same string, and the preview compares them. Three
#: separate rounds were lost to the same misreading: index.html is refetched on
#: every page load and server.py is not, so a change to both shows up as the new
#: control appearing and doing the old thing — which is indistinguishable from a
#: bug in the new control, and sends everybody looking in the wrong place.
CODE_STAMP = '2026-08-02.1'

#: When this process started, for /api/health.
_STARTED = time.time()


# --------------------------------------------------------------------------
# Admission control
# --------------------------------------------------------------------------
#: How many builds may hold the geometry core at once.
#:
#: One, because two do not each run at half speed — they run at a quarter or
#: worse. Measured on the 4-vCPU box, 2026-07-30: idle 0.3% CPU, one build
#: 105%, *two builds 420%* against a 400% ceiling with a run queue of 8, no
#: swapping and no steal. The extra demand is not work, it is contention —
#: both builds live in this one process, so every GIL-holding stretch
#: serialises, and when manifold3d's threaded booleans do overlap they ask for
#: more cores than exist.
#:
#: So queueing is not a throttle, it is the faster option: one build at full
#: speed followed by the next beats two crawling, and the only thing the second
#: visitor gives up is the illusion that something is happening — which is why
#: the wait reports a position rather than showing a frozen bar.
#:
#: Raise it with PDB2PRINT_MAX_BUILDS on a box with cores to spare.
def _max_builds() -> int:
    raw = os.environ.get("PDB2PRINT_MAX_BUILDS", "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


class _BuildGate:
    """Admits a bounded number of builds; tells the rest where they are.

    A plain semaphore would do the admitting, but it cannot answer "how many
    ahead of me", and that number is the whole point: a queued visitor who is
    told their position is waiting, and one who is not is watching a hang.
    """

    def __init__(self, limit: int):
        self._sem = asyncio.Semaphore(limit)
        #: Every live request in arrival order — the ones building *and* the
        #: ones queued.  Running tickets stay in the list on purpose: drop them
        #: on acquire and the first queued visitor finds itself at index 0 and
        #: is told nobody is ahead of them, while they wait.
        self._tickets: list = []
        self._issued = 0

    def take_ticket(self) -> int:
        self._issued += 1
        self._tickets.append(self._issued)
        return self._issued

    def abandon(self, ticket: int) -> None:
        if ticket in self._tickets:
            self._tickets.remove(ticket)

    def position(self, ticket: int) -> int:
        """How many requests are ahead of ``ticket``; 0 once it is at the front."""
        try:
            return self._tickets.index(ticket)
        except ValueError:
            return 0

    async def acquire(self, ticket: int) -> None:
        await self._sem.acquire()

    def release(self, ticket: int) -> None:
        self.abandon(ticket)
        self._sem.release()


_GATE: Optional[_BuildGate] = None


def _gate() -> _BuildGate:
    """The process-wide build gate, created on the running loop's first use."""
    global _GATE
    if _GATE is None:
        _GATE = _BuildGate(_max_builds())
    return _GATE

# --------------------------------------------------------------------------
# Worker pools
# --------------------------------------------------------------------------
#: Builds and stand previews get their own thread pools, deliberately.
#:
#: They used to share asyncio's default executor, which is
#: ``min(32, cpu_count + 4)`` threads -- 8 on the 4-vCPU box.  ``/api/stand/preview``
#: is ungated on purpose (it is meant to be a sub-second sketch), so a few
#: seconds of dragging a stand slider can put every one of those threads to
#: work.  A build admitted by the gate at that moment is submitted to a pool
#: with no free thread and simply never starts: ``work()`` does not run, so
#: ``__done__`` is never queued, so ``_stream_events`` sends keepalives forever
#: -- and because a keepalive is bytes on the wire, the front end's silence
#: detector never fires either.  Nothing anywhere reports a problem.  That is
#: the freeze only a refresh clears.
#:
#: Separate pools make it structurally impossible: a preview cannot consume a
#: build thread, and a build cannot consume a preview one.
_BUILD_POOL: Optional[ThreadPoolExecutor] = None
_PREVIEW_POOL: Optional[ThreadPoolExecutor] = None


def _build_pool() -> ThreadPoolExecutor:
    """Threads for the geometry core. Sized to the gate, plus one.

    The spare covers the moment a build has finished its work but its thread has
    not yet been handed back to the pool; without it a gate of one can stall for
    as long as that takes.
    """
    global _BUILD_POOL
    if _BUILD_POOL is None:
        _BUILD_POOL = ThreadPoolExecutor(
            max_workers=_max_builds() + 1, thread_name_prefix="pdb2print-build")
    return _BUILD_POOL


def _preview_pool() -> ThreadPoolExecutor:
    """Threads for the stand sketch. Small on purpose.

    Two: enough that one preview in flight does not delay the next, few enough
    that a burst of them can never become the machine's workload.
    """
    global _PREVIEW_POOL
    if _PREVIEW_POOL is None:
        _PREVIEW_POOL = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="pdb2print-preview")
    return _PREVIEW_POOL


def _sse(kind: str, data) -> str:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n"


#: How long a stream may go quiet before it sends a keepalive.
#:
#: Progress is reported once per chain, and nothing reports from inside one, so
#: a large structure at fine settings can legitimately say nothing for minutes.
#: That is indistinguishable from a dead connection — to the browser, to the
#: proxy, and to the user — so the difference is made explicit here rather than
#: guessed at by whoever is watching.
_HEARTBEAT_S = 20.0


async def _stream_events(queue: asyncio.Queue):
    """Yield SSE frames from ``queue`` until ``__done__``, keeping the line warm.

    The heartbeat is an SSE *comment*: no event name, no data, ignored by the
    front end's parser — but bytes on the wire, which is the whole job. It gives
    the browser's read something to return, it gives Caddy something to flush,
    and it means a silent connection really is a broken one.

    The pending ``get`` is held across timeouts rather than cancelled and
    remade. Cancelling a queue getter is the kind of thing that loses an item
    once in a thousand builds and is never reproducible afterwards.
    """
    getter = None
    try:
        while True:
            if getter is None:
                getter = asyncio.ensure_future(queue.get())
            done, _pending = await asyncio.wait({getter}, timeout=_HEARTBEAT_S)
            if not done:
                yield ": keepalive\n\n"
                continue
            kind, data = getter.result()
            getter = None
            if kind == "__done__":
                return
            yield _sse(kind, data)
    finally:
        if getter is not None:
            getter.cancel()


def _queue_message(ahead: int) -> str:
    if ahead == 1:
        return "Waiting for the server — 1 build ahead of you…"
    return f"Waiting for the server — {ahead} builds ahead of you…"


#: How long the gate will wait for an abandoned worker before taking the slot.
#:
#: A Python thread cannot be killed from outside, so a build wedged inside a C
#: extension would otherwise hold the gate shut for the life of the process.
#: Past this the slot is handed on regardless: the box then briefly runs two
#: builds, which is worse than one and far better than none.
_SLOT_MAX_HOLD_S = 900.0

#: How long an admitted build may sit unstarted before the stream gives up.
_BUILD_START_TIMEOUT_S = 90.0


def _release_on_completion(gate: _BuildGate, ticket: int, worker) -> None:
    """Release ``ticket`` when ``worker`` finishes, or after the hold ceiling.

    Both paths run on the event loop -- ``add_done_callback`` on the future
    returned by ``run_in_executor`` is scheduled there, and so is ``call_later``
    -- so touching the gate's semaphore from here is safe.  The guard matters:
    releasing twice would let two builds through a gate of one.
    """
    loop = asyncio.get_running_loop()
    state = {"done": False, "timer": None}

    def _release(*_args) -> None:
        if state["done"]:
            return
        state["done"] = True
        if state["timer"] is not None:
            state["timer"].cancel()
        gate.release(ticket)

    state["timer"] = loop.call_later(_SLOT_MAX_HOLD_S, _release)
    worker.add_done_callback(_release)


async def _watch_start(queue: asyncio.Queue, started: threading.Event) -> None:
    """Say something if an admitted build never reaches a worker thread.

    Belt to the separate pools' braces.  The failure this catches was silent by
    construction -- no worker means no ``__done__``, and the keepalive kept the
    connection looking healthy -- so it is worth one message even though the
    pool split should have made it unreachable.
    """
    deadline = time.monotonic() + _BUILD_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if started.is_set():
            return
        await asyncio.sleep(0.5)
    if started.is_set():
        return
    queue.put_nowait(("result", _error_payload(
        "The server accepted this build but could not start it. "
        "Please try again in a moment.")))
    queue.put_nowait(("__done__", None))


def _hand_back_slot(gate: _BuildGate, ticket: int, waiter, acquired: bool,
                    worker=None) -> None:
    """Give the slot back, whether the build ran or the client walked away.

    The awkward case is a client that disconnects while queued: the acquire may
    have completed in the moment between the last position check and this
    cleanup, in which case the slot is ours and nobody is going to use it.  Not
    checking for that leaks a slot per abandoned request, and a gate of one
    leaks itself shut on the first cancelled queue.

    When the slot *was* acquired the release now waits for ``worker`` -- the
    future for the thread actually doing the geometry.  Handing it back on
    disconnect instead, which is what this did, gives the slot to the next
    visitor while the abandoned build is still running: cancellation is only
    polled between chains and an in-flight boolean cannot be interrupted, so
    four Cancel-then-Generate cycles produced four concurrent builds under a
    gate of one.  That is exactly the contention the gate exists to prevent, and
    it feeds itself -- a slower box means more cancelling, which means more
    builds.
    """
    if acquired:
        if worker is None or worker.done():
            gate.release(ticket)
        else:
            _release_on_completion(gate, ticket, worker)
        return
    waiter.cancel()
    if waiter.done() and not waiter.cancelled() and waiter.exception() is None:
        gate.release(ticket)     # won the slot in the last instant; hand it on
    else:
        gate.abandon(ticket)


async def _wait_for_slot(gate: _BuildGate, ticket: int, waiter):
    """Yield queue-position events until ``waiter`` has the slot.

    The events are ordinary ``progress`` frames at fraction 0, so the existing
    front end shows them with no change: the bar sits at the start and the
    caption says where you are.  Re-sent every couple of seconds, both to track
    the queue moving and to keep the connection warm through the proxy.
    """
    while True:
        ahead = gate.position(ticket)
        if ahead:
            # ``queued`` is what the front end keys its banner off.  It rides on
            # the progress event rather than travelling as its own event type so
            # that a browser holding a cached copy of the old page still shows
            # the message in the report line instead of silently ignoring an
            # event it has never heard of.
            yield _sse("progress", {"frac": 0.0, "msg": _queue_message(ahead),
                                    "queued": ahead})
        done, _pending = await asyncio.wait({waiter}, timeout=2.0)
        if done:
            return

class _ImmutableStatic(StaticFiles):
    """StaticFiles that tells the browser the bytes will never change.

    Only safe where the URL changes when the content does, which is true of both
    places it is used: a cache entry's directory is its content hash, and the
    vendored viewer carries its version in the filename.  ``get_response`` is
    the documented extension point, so this does not depend on Starlette's
    internals.
    """

    #: Files under these mounts whose name does *not* change with their
    #: content. Only one so far: the cache writes a human-readable listing at
    #: its root, and that is rewritten in place on every store.
    MUTABLE = {"index.json"}

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if os.path.basename(path) not in self.MUTABLE:
            response.headers["Cache-Control"] = ("public, max-age=31536000, "
                                                 "immutable")
        return response


app = FastAPI(title="pdb2print")
app.mount("/files", StaticFiles(directory=OUTPUT_ROOT), name="files")
app.mount("/cache", _ImmutableStatic(directory=CACHE_DIR), name="cache")
_VENDOR_DIR = os.path.join(FRONTEND_DIR, "vendor")
if os.path.isdir(_VENDOR_DIR):
    # Mounted ahead of the catch-all below so the immutable header wins.
    app.mount("/vendor", _ImmutableStatic(directory=_VENDOR_DIR), name="vendor")


def _rgb_to_hex(rgb) -> str:
    r, g, b = (max(0, min(255, round(c * 255))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


@app.get("/")
def index() -> FileResponse:
    # The whole UI is one HTML file, so a browser holding a cached copy is a
    # browser running last week's front end against this week's API — which
    # presents as a fixed bug that is still there, and sends everyone hunting in
    # the wrong place.
    #
    # no-cache, not no-store: both revalidate on every load, so neither can ever
    # serve stale code, but no-cache permits a conditional request and a 304
    # while no-store forbids even that. The page is 68 KB gzipped and it was
    # being re-sent in full on every visit for no benefit.
    return FileResponse(
        os.path.join(FRONTEND_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


#: Every admitted build, while it is running: what it is, when it started, and
#: what it last said.
#:
#: The freeze has been chased three times by reasoning about the code, because
#: from the outside a build that is wedged and a build that is legitimately
#: spending four minutes inside one boolean look identical -- both are silent,
#: both keep the connection warm. This turns that into a question anyone can
#: answer in a browser: open /api/debug and read how long it has been since the
#: build last said anything, and what it was doing at the time.
_LIVE: dict = {}
_LIVE_LOCK = threading.Lock()


def _live_start(ticket: int, what: str, source: str) -> None:
    with _LIVE_LOCK:
        _LIVE[ticket] = {"what": what, "source": source, "started": time.time(),
                         "frac": 0.0, "msg": "admitted, not started yet",
                         "said_at": time.time(),
                         "thread": threading.current_thread().name}


def _live_say(ticket: int, frac, msg) -> None:
    with _LIVE_LOCK:
        entry = _LIVE.get(ticket)
        if entry is not None:
            entry["frac"] = float(frac)
            entry["msg"] = str(msg)
            entry["said_at"] = time.time()
            entry["thread"] = threading.current_thread().name


def _live_end(ticket: int) -> None:
    with _LIVE_LOCK:
        _LIVE.pop(ticket, None)


@app.get("/api/debug")
def debug_state() -> JSONResponse:
    """What the server is doing *right now*. Open it in a browser when it hangs.

    Read it in this order:

    * ``builds`` empty and your page still spinning -> nothing is running. The
      request never got in, or it finished and the answer was lost on the way
      back. A browser problem or a stream problem, not a geometry one.
    * a build with a large ``silent_for`` -> it is alive and stuck inside one
      operation. ``msg`` names which one. That is a geometry problem and the
      structure and settings in ``source`` reproduce it.
    * a build with a large ``waiting_to_start`` -> it was admitted and never
      reached a worker thread. That is the pool starvation the separate build
      and preview pools exist to prevent, and it would mean they are not.
    * ``queued`` non-zero with nothing in ``builds`` -> a slot was taken and
      never handed back. That wedges the site shut and is its own bug.

    Times are seconds. No geometry, no side effects; safe to hit at any time.
    """
    now = time.time()
    gate = _GATE
    with _LIVE_LOCK:
        builds = []
        for ticket, e in sorted(_LIVE.items()):
            started = e["started"]
            builds.append({
                "ticket": ticket, "what": e["what"], "source": e["source"],
                "running_for": round(now - started, 1),
                "silent_for": round(now - e["said_at"], 1),
                "waiting_to_start": (round(now - started, 1)
                                     if e["frac"] <= 0.0 else 0.0),
                "frac": round(e["frac"], 3), "msg": e["msg"],
                "thread": e["thread"],
            })

    def pool(p):
        if p is None:
            return None
        return {"threads": len(getattr(p, "_threads", []) or []),
                "max": getattr(p, "_max_workers", None),
                "queued": getattr(getattr(p, "_work_queue", None), "qsize",
                                  lambda: None)()}

    return JSONResponse({
        "now": round(now - _STARTED, 1),
        "max_builds": _max_builds(),
        "tickets_live": list(getattr(gate, "_tickets", []) or []) if gate else [],
        "tickets_issued": getattr(gate, "_issued", 0) if gate else 0,
        "slots_free": getattr(getattr(gate, "_sem", None), "_value", None)
                      if gate else None,
        "builds": builds,
        "build_pool": pool(_BUILD_POOL),
        "preview_pool": pool(_PREVIEW_POOL),
        "recent_tokens": len(_RECENT_BUILDS),
    })


@app.get("/api/health")
def health() -> JSONResponse:
    """What the *running process* actually has. Open it in a browser.

    Written after losing two rounds to the same misdiagnosis twice over: a
    setting appears in the panel, does nothing, and there is no way to tell from
    the outside whether the cause is a missing dependency, a stale process still
    running last week's code, or a real bug. All three look identical, and
    guessing between them costs a round trip each time.

    So this reports the loaded code rather than the code on disk — the fields
    actually present on the dataclasses, the fonts that actually resolve, the
    dependencies that actually import — and compares the newest source file
    against the process start time, which is the one question a person cannot
    answer by looking at their editor.
    """
    import dataclasses

    from pdb2print import config as cfg, typeset
    from pdb2print.pipeline import BuildReport

    def version(name: str) -> str:
        try:
            module = __import__(name)
            return str(getattr(module, "__version__", "present"))
        except Exception as exc:
            return f"MISSING — {type(exc).__name__}: {exc}"

    newest, newest_name = 0.0, ""
    package = os.path.dirname(os.path.abspath(cfg.__file__))
    sources = [os.path.abspath(__file__)]
    for dirpath, _dirs, names in os.walk(package):
        sources.extend(os.path.join(dirpath, n) for n in names
                       if n.endswith(".py"))
    for path in sources:
        try:
            stamp = os.path.getmtime(path)
        except OSError:
            continue
        if stamp > newest:
            newest, newest_name = stamp, path
    stale = newest > _STARTED

    stand_fields = {f.name for f in dataclasses.fields(cfg.StandParams)}
    report_fields = {f.name for f in dataclasses.fields(BuildReport)}
    code = {
        "plaque_split": "plaque_split" in stand_fields,
        "plaque_relief": "plaque_relief" in stand_fields,
        "plaque_font": "plaque_font" in stand_fields,
        "column_pins": "column_pins" in stand_fields,
        "apron_rake": "apron_rake_deg" in stand_fields,
        "build_report_carries_title": "title" in report_fields,
        "stand_preview_endpoint": any(
            getattr(route, "path", "") == "/api/stand/preview"
            for route in app.routes),
    }

    fonts = {}
    for key in ("line", "sans", "serif"):
        face = typeset.face(key)
        fonts[key] = {"using": face.key, "ok": face.key == key,
                      "why": typeset.unavailable(face)}

    problems = []
    if stale:
        problems.append(
            f"This process started {time.strftime('%H:%M:%S', time.localtime(_STARTED))} "
            f"but {os.path.basename(newest_name)} was changed at "
            f"{time.strftime('%H:%M:%S', time.localtime(newest))} — it is running "
            f"older code than you have on disk. Restart the server.")
    problems.extend(f["why"] for f in fonts.values() if f["why"])
    problems.extend(f"{name} is missing" for name in
                    ("fontTools", "mapbox_earcut")
                    if version(name).startswith("MISSING"))

    return JSONResponse({
        "ok": not problems,
        "code_stamp": CODE_STAMP,
        "problems": problems or ["nothing — the running server is up to date."],
        "process_started": time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(_STARTED)),
        "newest_source": {
            "file": os.path.relpath(newest_name, os.path.dirname(
                os.path.abspath(__file__))) if newest_name else None,
            "changed": time.strftime("%Y-%m-%d %H:%M:%S",
                                     time.localtime(newest)) if newest else None,
            "newer_than_this_process": stale,
        },
        "code_loaded": code,
        "fonts": fonts,
        "dependencies": {name: version(name) for name in
                         ("fontTools", "mapbox_earcut", "manifold3d",
                          "trimesh", "biotite")},
    })


def _error_payload(message: str) -> dict:
    """The standard failure JSON shape (matches the streamed result event)."""
    return {"ok": False, "warning": message, "report": message,
            "glb_url": None, "threemf_url": None, "stl_url": None, "chains": []}


def _bool(x) -> bool:
    """Parse a permissive form boolean ("true"/"1"/"on"/"yes")."""
    return str(x).strip().lower() in {"1", "true", "on", "yes"}


#: Characters allowed in a generated download filename; everything else in a
#: user-supplied name collapses to "_" so the name is safe on every OS and
#: needs no quoting in a URL path.
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


def _download_stem(source: str, params: PrintParams) -> str:
    """A descriptive base filename for this build, e.g. ``1zaa_pdb2print_1p5mm``.

    Downloads used to land in the user's folder as ``out.3mf`` / ``out_stl.zip``
    for every structure, which is useless the moment you have built more than
    one. The stem carries what distinguishes two builds in a downloads folder:
    which structure, and at what scale.

    ``source`` is either a bare PDB ID or the path of an uploaded file; either
    way only the basename (without extension) contributes.
    """
    stem = os.path.splitext(os.path.basename(source))[0]
    stem = _SAFE_STEM.sub("_", stem).strip("._-").lower()[:40] or "model"
    scale = f"{params.scale_mm_per_angstrom:g}".replace(".", "p")
    return f"{stem}_pdb2print_{scale}mm"


def _map_connections(fields: dict) -> ConnectionParams:
    """Build a :class:`ConnectionParams` from the (small) connection form fields."""
    return ConnectionParams(
        connect=_bool(fields.get("connect", False)),
        use_magnets=_bool(fields.get("use_magnets", False)),
        no_magnet_method=NoMagnetMethod(fields.get("no_magnet_method", "inflate")),
        connector_diameter_mm=(_bounded(fields, "connector_diameter")
                               if "connector_diameter" in fields else 4.0),
        magnet_thickness_mm=float(fields.get("magnet_thickness", 2.0)),
        magnet_shape=MagnetShape(fields.get("magnet_shape", "round")),
        # Floored at zero rather than at one: zero now means "no joint on these
        # interfaces" and has to survive, but a negative count is nonsense that
        # would read as a veto by accident.
        magnet_count=max(0, int(float(fields.get("magnet_count", 1)))),
        dna_magnet_count=max(0, int(float(fields.get("dna_magnet_count", 1)))),
        # Capped like every other free-text field that reaches the builder: one
        # line per joint, and a structure has a handful, so 2000 characters is
        # far more than any real model needs and still bounds the field.
        joint_overrides=str(fields.get("joint_overrides", "") or "")[:2000],
        socket=_bool(fields.get("socket", True)),
        socket_wall_mm=float(fields.get("socket_wall", 1.5)),
        magnet_fit_clearance_mm=float(fields.get("magnet_fit_clearance", 0.2)),
        basepair_connect=_bool(fields.get("basepair_connect", False)),
    )


def _map_params(fields: dict) -> PrintParams:
    """Map raw HTTP form fields to a :class:`PrintParams` (may raise ValueError)."""
    return PrintParams(
        scale_mm_per_angstrom=_bounded(fields, "scale"),
        grid_spacing_mm=_bounded(fields, "grid_spacing"),
        min_wall_mm=_bounded(fields, "min_wall"),
        min_wall_mode=MinWallMode(fields["min_wall_mode"]),
        protein_representation=Representation(fields["protein_rep"]),
        nucleic_representation=Representation(fields["nucleic_rep"]),
        backbone_style=BackboneStyle(fields["backbone_style"]),
        base_style=BaseStyle(fields["base_style"]),
        nucleic_radius_mm=float(fields["nucleic_radius"]),
        protein_tube_radius_mm=float(fields.get("protein_tube_radius", 1.2)),
        cartoon_helix_width_mm=float(fields.get("cartoon_helix_width", 4.5)),
        cartoon_strand_width_mm=float(fields.get("cartoon_strand_width", 4.0)),
        cartoon_coil_radius_mm=float(fields.get("cartoon_coil_radius", 0.9)),
        slab_thickness_mm=float(fields["slab_thickness"]),
        slab_scale=float(fields["base_width"]),
        connector_radius_mm=float(fields["connector_radius"]),
        atom_radius_mm=float(fields["atom_radius"]),
        bond_radius_mm=float(fields["bond_radius"]),
        backbone_atom_radius_mm=float(fields.get("backbone_atom_radius", 1.0)),
        backbone_bond_radius_mm=float(fields.get("backbone_bond_radius", 0.5)),
        probe_radius_ang=_bounded(fields, "probe_radius"),
        surface_atom_padding_ang=_bounded(fields, "surface_padding"),
        # Defaults to False when the field is absent, matching the checkbox: a
        # caller that says nothing about ligands gets the plain structure, which is
        # both the old behaviour and the conservative one.
        include_ligands=_bool(fields.get("include_ligands", False)),
        # Capped like every other free-text field that reaches the builder. A
        # few characters per chain, and a structure has tens at most.
        exclude_chains=str(fields.get("exclude_chains", "") or "")[:2000],
        ligand_style=LigandStyle(fields.get("ligand_style", "ball_stick")),
        ligand_atom_mm=float(fields.get("ligand_atom", 2.2)),
        ligand_bond_mm=float(fields.get("ligand_bond", 1.2)),
        ligand_vdw_scale=float(fields.get("ligand_vdw_scale", 1.0)),
        connections=_map_connections(fields),
    )


#: How long an uncached build's files stay downloadable under /files/.
#:
#: Most builds are cached and their temp copy is deleted immediately. This only
#: covers the ones that are not — uploads, and builds whose 3MF the watertight
#: gate refused — which would otherwise sit in the container until it restarted.
#: Two hours is far longer than the gap between a build finishing and someone
#: clicking download, while still bounding the leak.
OUTPUT_TTL_SECONDS = 2 * 60 * 60


def _sweep_output_root(now: Optional[float] = None) -> int:
    """Delete per-build output directories older than the TTL.

    Cheap enough to run on every build: one ``stat`` per directory, against work
    that takes tens of seconds. Doing it here rather than on a timer avoids a
    background task that has to be shut down cleanly.

    A build in progress is safe — its directory was created moments ago, so it
    cannot be older than the TTL.
    """
    now = time.time() if now is None else now
    removed = 0
    try:
        names = os.listdir(OUTPUT_ROOT)
    except OSError:
        return 0
    for name in names:
        path = os.path.join(OUTPUT_ROOT, name)
        try:
            if not os.path.isdir(path):
                continue
            if now - os.path.getmtime(path) < OUTPUT_TTL_SECONDS:
                continue
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


# --------------------------------------------------------------------------
# Recently built meshes, kept in memory for the display stand
# --------------------------------------------------------------------------
#: token -> {"built", "params", "source"} for the last few completed builds.
#:
#: The stand is generated from the *finished* meshes and changes nothing about
#: how they were built, so re-running the pipeline to add one would spend a
#: minute of marching cubes reproducing geometry the server had in its hands a
#: moment ago — and would spend it again for every change of column count. This
#: is the short-lived hand-back that makes the button feel like a button.
#:
#: It is a cache of convenience and never one of record: a miss (server
#: restarted, entry evicted, or the build came from the disk cache and so never
#: existed in memory) falls back to a full rebuild, which is correct, just slow.
_RECENT_BUILDS: "OrderedDict[str, dict]" = OrderedDict()
_RECENT_LOCK = threading.Lock()

#: How many *mesh-holding* builds to keep. Deliberately small: a large complex
#: is hundreds of megabytes of triangles, and this process also has to mesh the
#: next one.
_RECENT_MAX = 2

#: How many tokens to keep in total. A cache hit starts out remembering no
#: geometry — a key and a params object, bytes not megabytes — so those are
#: worth holding far longer than the meshes are.
#:
#: They no longer *stay* that way: standing up a cached model reopens its
#: meshes and _remember_reopened keeps them, which promotes that token to
#: mesh-holding and lets it evict an earlier real build under _RECENT_MAX. That
#: is the intended trade — the alternative was reopening the zip on every
#: preview — but it does mean "add a stand to a cached model" can cost the
#: previous build its place, and a stand on *that* one would then rebuild.
_RECENT_TOKENS = 64


def _remember_build(token: str, source: str, params: PrintParams, built,
                    meta: Optional[dict] = None, hit: Optional[dict] = None) -> None:
    """Record what a token was built from, so a stand does not re-derive it.

    ``built`` are the finished meshes when this process made them, and ``None``
    when the result came off the disk cache — in which case ``hit`` is that
    entry, and the meshes can be reopened from its per-chain STL zip.

    Recording ``params`` matters as much as either.  The stand routes rebuild
    their own ``PrintParams`` from the form the browser posts, and that form is
    read at the moment the user clicks "Create display stand" — so a setting
    nudged between generating and standing produced a different cache key, a
    miss, and a full rebuild of a model that was sitting right there.  The
    token now says what the model on screen was actually built from.
    """
    with _RECENT_LOCK:
        _RECENT_BUILDS[token] = {
            "built": built, "params": params, "source": source, "hit": hit,
            # What the plaque would print. Kept with the build because that is
            # the last moment it is cheap: an uploaded file is deleted when the
            # build finishes, and a fetched one is a download away.
            "meta": dict(meta or {}),
        }
        _RECENT_BUILDS.move_to_end(token)
        _trim_recent_locked()


def _trim_recent_locked() -> None:
    """Apply both ceilings. The caller holds ``_RECENT_LOCK``.

    Two of them, because the two kinds of entry cost wildly different amounts.
    Oldest first, and only mesh-holding entries count against the small one.
    """
    meshy = [k for k, e in _RECENT_BUILDS.items()
             if e.get("built") is not None or e.get("built_loose") is not None]
    for stale in meshy[:max(0, len(meshy) - _RECENT_MAX)]:
        _RECENT_BUILDS.pop(stale, None)
    while len(_RECENT_BUILDS) > _RECENT_TOKENS:
        _RECENT_BUILDS.popitem(last=False)


def _remember_reopened(token: str, built, params=None,
                       slot: str = "built") -> None:
    """Hold on to meshes reopened from the disk cache.

    Reopening is not cheap -- unzip the per-chain STL zip, load every chain, run
    the repair pass -- and the stand preview asks for it once per slider nudge.
    Measured at 6.7s on a 1.3M-face model, so a three-second drag was a minute
    of duplicated disk work, every byte of it thrown away, in the same threads
    the builds want.  Nothing here was memoised: there was no cache of any kind
    in front of this.

    The entry becomes mesh-holding, so it now counts against ``_RECENT_MAX``
    like any other -- which is the right accounting, because it costs the same.
    """
    if not token:
        return
    with _RECENT_LOCK:
        entry = _RECENT_BUILDS.get(token)
        if entry is None:
            return
        entry[slot] = built
        if params is not None:
            # The meshes and the params have to describe the same build. This
            # branch can be reached with an entry whose own hit failed to reopen
            # and a *different* cache entry that succeeded, and storing the
            # meshes without their params would leave the token answering with a
            # pair that never existed.
            entry["params"] = params
        # A watertight reopen satisfies both callers, so it clears both flags.
        # A relaxed one only clears its own — the strict attempt genuinely
        # failed and re-trying it costs the full reopen to learn that again.
        entry.pop("reopen_failed_loose", None)
        if slot == "built":
            entry.pop("reopen_failed", None)
        _RECENT_BUILDS.move_to_end(token)
        _trim_recent_locked()


def _remember_reopen_failed(token: str, slot: str = "reopen_failed") -> None:
    """Remember that this entry's meshes could not be reopened.

    The failure is the common case, not the rare one: an STL round trip cannot
    reproduce a watertight mesh, so most hits decline at the gate in
    ``_built_from_cache``.  Without this the next preview pays the full reopen
    cost to reach the same answer, and the one after that pays it again.
    """
    if not token:
        return
    with _RECENT_LOCK:
        entry = _RECENT_BUILDS.get(token)
        if entry is not None:
            entry[slot] = True


def _recall_build(token: str):
    if not token:
        return None
    with _RECENT_LOCK:
        entry = _RECENT_BUILDS.get(token)
        if entry is not None:
            _RECENT_BUILDS.move_to_end(token)
        return entry


def _cached_result(meta: dict, source: str = "") -> dict:
    """Rebuild the result payload for a cache hit.

    Everything except the three URLs was stored verbatim at build time, so a hit
    is indistinguishable from a fresh build to the front end — same report text,
    same chain list and colours, same printed size. Only the URLs are rewritten,
    to point into the cache directory instead of a per-build temp directory.
    """
    files = meta.get("files") or {}
    key = meta["key"]

    def url(kind):
        name = files.get(kind)
        return f"/cache/{key}/{name}" if name else None

    result = dict(meta.get("result") or {})

    # Entries written before the plaque carried its own text have no
    # ``plaque_meta``, and a cache hit skips the build that would have read it —
    # so a structure served from cache arrived with no name on its plaque and no
    # hint as to why. Backfilled here rather than by throwing the cache away:
    # the pre-generated entries are the whole reason a popular structure is
    # instant, and re-reading one header is cheaper than re-meshing a complex by
    # several orders of magnitude. ``resolve_source`` memoises the fetch, so an
    # ID costs at most one download per process however many hits it serves.
    if source and not (result.get("plaque_meta") or {}).get("pdb_id"):
        result["plaque_meta"] = _plaque_meta(source)

    result.update({
        "ok": True,
        "glb_url": url("glb"),
        "threemf_url": url("threemf"),
        "stl_url": url("stl_zip"),
        "cached": True,
    })
    return result


def _run_and_export(source: str, params: PrintParams, progress,
                    should_cancel=None, on_model=None) -> dict:
    """Blocking build + export; returns the result JSON dict (never raises).

    ``progress(frac, msg)`` is forwarded straight into ``build_all`` and reused
    for the export phase, so the SSE stream keeps ticking after meshing too.
    ``should_cancel`` is polled by the pipeline so a disconnected client stops
    the build instead of leaving it to run to completion unwatched.

    ``on_model(payload)``, when given, is called the moment the GLB is on disk
    and before the slower exports run, so the viewer can show the model while
    the downloads are still being written.  Optional because a cache hit never
    reaches it and the CLI has nothing to send it to.
    """
    # Cache first: the overwhelming majority of requests are a handful of famous
    # structures at preset settings, and serving those as static files is the
    # difference between a download and a minute of marching cubes. Checked
    # before the cancellation machinery matters, because a hit finishes far
    # faster than a client can disconnect.
    try:
        hit = cache.lookup(source, params)
    except Exception:
        hit = None          # a broken cache must never take the app down
    if hit:
        progress(1.0, "Loaded from cache.")
        result = _cached_result(hit, source)
        # A hit used to return here carrying no token, which quietly cost the
        # stand everything: with nothing to recall, it re-derived the params
        # from the live settings form and looked the build up again, and any
        # drift in that form meant a full rebuild of the model on screen. The
        # token holds no geometry — just this entry and the params that found
        # it — so the stand can reopen the STL zip directly.
        token = uuid.uuid4().hex
        _remember_build(token, source, params, None,
                        result.get("plaque_meta"), hit=hit)
        result["build_token"] = token
        return result

    try:
        report = build_all(source, params, progress=progress,
                           should_cancel=should_cancel)
    except BuildCancelled:
        # Nobody is listening any more; unwind quietly rather than reporting.
        return _error_payload("Build cancelled.")
    except Exception as exc:  # watertight-gate RuntimeError, ValueError, etc.
        return _error_payload(str(exc))

    progress(0.96, "Writing export files…")
    _sweep_output_root()
    token = uuid.uuid4().hex
    out_dir = os.path.join(OUTPUT_ROOT, token)
    os.makedirs(out_dir, exist_ok=True)

    # Hold the meshes so "Create display stand" can work from them instead of
    # meshing the whole structure again. Done before the export rather than
    # after, so an export that fails on disk space still leaves the build
    # reachable.
    plaque_meta = {
        "pdb_id": _plaque_id(source),
        "title": report.title,
        "why": "" if report.title else
               "this structure's header has no title record",
    }
    _remember_build(token, source, params, report.built, plaque_meta)

    # Files are named after the structure so a download is self-describing; the
    # uuid directory (not the filename) is what keeps concurrent builds apart.
    stem = _download_stem(source, params)

    chains = [
        {"id": chain.chain_id, "name": chain.name,
         "color": _rgb_to_hex(color_for_index(_palette_index(chain, i)))}
        for i, (chain, _mesh) in enumerate(report.built)
    ]

    # Every chain the structure offered, whether or not it was built. The list
    # the user picks from has to include the ones they switched off, or there is
    # no way to switch them back on — and the ones that failed to mesh, because
    # a chain that is missing from the model for a reason the report explains
    # should not also be missing from the list that explains it.
    _made = {id(chain) for chain, _m in report.built}
    parts = [
        {"index": int(chain.index if chain.index is not None else i),
         "id": chain.chain_id,
         "name": chain.display_name(),
         "kind": getattr(chain.mtype, "value", str(chain.mtype)),
         "built": id(chain) in _made,
         "color": _rgb_to_hex(color_for_index(_palette_index(chain, i)))}
        for i, chain in enumerate(getattr(report, "candidates", None) or
                                  [c for c, _m in report.built])
    ]

    # Overall printed bounding box (mm) across every built chain, plus the scale
    # that produced it, so the UI can show the size live as the scale changes.
    size_mm = None
    if report.built:
        mins = [min(m.bounds[0][k] for _, m in report.built) for k in range(3)]
        maxs = [max(m.bounds[1][k] for _, m in report.built) for k in range(3)]
        size_mm = [float(maxs[k] - mins[k]) for k in range(3)]

    # A full disk surfaces here first, because the exporters write before the
    # cache does. Caught explicitly so it reports as a server problem the
    # operator can act on, rather than as an unhandled traceback that reads like
    # the structure was at fault.
    _DISK_FULL = ("The server ran out of disk space while writing the export "
                  "files. This is not a problem with your structure. ")
    try:
        export.write_glb(report.built, os.path.join(out_dir, f"{stem}.glb"),
                         markers=report.connection_markers)
    except OSError as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        return _error_payload(_DISK_FULL + f"({exc.strerror or exc})")

    # Show the model now, rather than when the *downloads* are ready.
    #
    # The viewer needs the GLB and nothing else, but the result event was held
    # back until the STL zip and the 3MF were both written -- measured at 3.1s
    # and 14.1s on a 1.3M-face model, so on a large structure the user sat
    # watching an already-finished build for another seventeen seconds. The 3MF
    # is slow for a structural reason (lib3mf builds a Python object per vertex
    # and per triangle: 9.8s of that 14s), so sending the model early is the fix
    # available without touching the exporter.
    #
    # Deliberately not the whole result. The download links and the stand button
    # need files that are not on disk yet, and offering them here would be
    # offering a 404.
    sent_model = False
    if on_model is not None:
        try:
            on_model({"glb_url": f"/files/{token}/{stem}.glb",
                      "chains": chains, "size_mm": size_mm,
                      "scale_used": params.scale_mm_per_angstrom})
            sent_model = True
        except Exception:
            pass                      # a preview must never be able to fail a build
        progress(0.97, "Model ready — writing the download files…")

    try:
        export.write_stl_zip(report.built, os.path.join(out_dir, f"{stem}_stl.zip"))
    except OSError as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        return _error_payload(_DISK_FULL + f"({exc.strerror or exc})")

    threemf_url = None
    warning = None
    try:
        export.write_3mf(report.built, os.path.join(out_dir, f"{stem}.3mf"))
        threemf_url = f"/files/{token}/{stem}.3mf"
    except RuntimeError as exc:
        # 3MF unavailable / non-manifold — still ship GLB + STL, surface message.
        warning = str(exc)
    except OSError as exc:
        warning = f"Could not write the 3MF: {exc.strerror or exc}"

    progress(1.0, "Done.")
    result = {
        "ok": True,
        "warning": warning,
        "report": report.summary(),
        "glb_url": f"/files/{token}/{stem}.glb",
        "threemf_url": threemf_url,
        "stl_url": f"/files/{token}/{stem}_stl.zip",
        "chains": chains,
        "parts": parts,
        "connections": report.connections,
        "size_mm": size_mm,
        "scale_used": params.scale_mm_per_angstrom,
        # What a plaque would print for this structure. Sent with every build so
        # the stand sketch can show the real ID and the real title from the
        # moment it opens, rather than the words "Structure name" standing in
        # for them until the first solve comes back — a preview whose whole
        # point is showing what you get should not be showing a placeholder.
        "plaque_meta": plaque_meta,
        # Names the in-memory meshes for a follow-up stand request. Deliberately
        # excluded from what gets cached below: a token identifies one process's
        # memory, and serving a stale one from disk would send the client
        # chasing a build that no longer exists.
        "build_token": token,
    }

    # Store on miss. Only a build that produced a 3MF is worth keeping — that is
    # the artefact people come for, and an entry without one would be served in
    # place of a retry that might succeed. Caching is strictly an optimisation,
    # so any failure here is swallowed: the user already has their files.
    if threemf_url:
        stored = None
        try:
            stored = cache.store(
                source, params,
                files={
                    "threemf": os.path.join(out_dir, f"{stem}.3mf"),
                    "glb": os.path.join(out_dir, f"{stem}.glb"),
                    "stl_zip": os.path.join(out_dir, f"{stem}_stl.zip"),
                },
                meta={
                    "stem": stem,
                    # What each object *was*, so a later display-stand request
                    # can reopen this entry as geometry instead of re-meshing
                    # the whole structure. Cheap to write and the only thing
                    # standing between a cache hit and a second full build.
                    "objects": cache_mod.describe_objects(report.built),
                    # The URLs are per-build and meaningless once this temp
                    # directory is gone; a hit rewrites them from the entry.
                    "result": {k: v for k, v in result.items()
                               if k not in ("glb_url", "threemf_url", "stl_url",
                                            "build_token")},
                },
            )
        except Exception:
            pass

        # Once the files are in the cache, the copy under OUTPUT_ROOT is dead
        # weight — the cache serves the same bytes from a stable URL. Keeping
        # both doubled disk use per build, and nothing ever deleted the temp
        # copy, so it accumulated until the container restarted.
        #
        # Except the GLB, when the viewer has already been sent to it. Rewriting
        # that URL would make model-viewer reload the same geometry from a
        # different address — throwing away however the user had orbited it in
        # the seconds since — and deleting the file first would blank the viewer
        # outright if the fetch were still in flight. So the early URL stays
        # valid and only the two downloads move; _sweep_output_root clears the
        # directory on a later build. The 3MF and the STL zip are the big ones
        # anyway, and nobody is holding a link to those yet.
        if stored:
            result["threemf_url"] = f"/cache/{stored}/{stem}.3mf"
            result["stl_url"] = f"/cache/{stored}/{stem}_stl.zip"
            if sent_model:
                for name in (f"{stem}.3mf", f"{stem}_stl.zip"):
                    try:
                        os.remove(os.path.join(out_dir, name))
                    except OSError:
                        pass
            else:
                result["glb_url"] = f"/cache/{stored}/{stem}.glb"
                shutil.rmtree(out_dir, ignore_errors=True)

    return result


@app.post("/api/generate")
async def generate(
    pdb_id: str = Form(""),
    scale: float = Form(1.5),
    grid_spacing: float = Form(0.5),
    min_wall: float = Form(1.0),
    min_wall_mode: str = Form("uniform"),
    protein_rep: str = Form("surface"),
    nucleic_rep: str = Form("tube_slab"),
    backbone_style: str = Form("tube"),
    base_style: str = Form("slab"),
    nucleic_radius: float = Form(1.2),
    protein_tube_radius: float = Form(1.2),
    cartoon_helix_width: float = Form(4.5),
    cartoon_strand_width: float = Form(4.0),
    cartoon_coil_radius: float = Form(0.9),
    slab_thickness: float = Form(1.2),
    base_width: float = Form(1.0),
    connector_radius: float = Form(0.6),
    atom_radius: float = Form(1.0),
    bond_radius: float = Form(0.5),
    backbone_atom_radius: float = Form(1.0),
    backbone_bond_radius: float = Form(0.5),
    probe_radius: float = Form(1.4),
    surface_padding: float = Form(0.0),
    include_ligands: str = Form("false"),
    exclude_chains: str = Form(""),
    ligand_style: str = Form("ball_stick"),
    ligand_atom: float = Form(2.2),
    ligand_bond: float = Form(1.2),
    ligand_vdw_scale: float = Form(1.0),
    # --- connector / joinery system ---
    connect: str = Form("false"),
    use_magnets: str = Form("false"),
    no_magnet_method: str = Form("inflate"),
    connector_diameter: float = Form(4.0),
    magnet_thickness: float = Form(2.0),
    magnet_shape: str = Form("round"),
    magnet_count: int = Form(1),
    dna_magnet_count: int = Form(1),
    socket: str = Form("true"),
    socket_wall: float = Form(1.5),
    magnet_fit_clearance: float = Form(0.2),
    joint_overrides: str = Form(""),
    basepair_connect: str = Form("false"),
    file: UploadFile | None = None,
):
    """Build a model and stream progress as Server-Sent Events.

    The response is ``text/event-stream``: zero or more ``event: progress`` lines
    (``{frac, msg}``) followed by a single ``event: result`` line carrying the
    same JSON shape the endpoint returned before streaming. Validation problems
    (bad file type, missing source, unparseable params) short-circuit to a plain
    JSON error with a 4xx status, so the client can tell the two apart by
    content-type.
    """
    # 1. resolve source: uploaded file OR a PDB-ID string
    source = None
    tmp_upload_dir = None
    if file is not None and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in UPLOAD_EXTS:
            return JSONResponse(
                _error_payload(f"Unsupported file type '{ext}'. "
                               f"Accepted: {', '.join(sorted(UPLOAD_EXTS))}."),
                status_code=400,
            )
        tmp_upload_dir = tempfile.mkdtemp(prefix="pdb2print_up_")
        source = os.path.join(tmp_upload_dir, os.path.basename(file.filename))
        # Bounded copy. There was no size limit anywhere on this path, and this
        # is a public endpoint. A backstop rather than the real fix: Starlette
        # has already spooled the entire body to disk by the time this function
        # runs, so the limit that actually matters is request_body { max_size }
        # in the Caddyfile. This one stops the disk filling up behind it.
        oversize, written = False, 0
        with open(source, "wb") as fh:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    oversize = True
                    break
                fh.write(chunk)
        if oversize:
            shutil.rmtree(tmp_upload_dir, ignore_errors=True)
            return JSONResponse(
                _error_payload(f"That file is over the "
                               f"{_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."),
                status_code=413,
            )
    elif pdb_id.strip():
        source = pdb_id.strip()

    if not source:
        return JSONResponse(
            _error_payload("Enter a PDB ID or upload a .pdb/.cif file."),
            status_code=400,
        )

    # 2. map fields -> PrintParams
    try:
        params = _map_params({
            "scale": scale, "grid_spacing": grid_spacing, "min_wall": min_wall,
            "min_wall_mode": min_wall_mode, "protein_rep": protein_rep,
            "nucleic_rep": nucleic_rep, "backbone_style": backbone_style,
            "base_style": base_style, "nucleic_radius": nucleic_radius,
            "protein_tube_radius": protein_tube_radius,
            "cartoon_helix_width": cartoon_helix_width,
            "cartoon_strand_width": cartoon_strand_width,
            "cartoon_coil_radius": cartoon_coil_radius,
            "slab_thickness": slab_thickness, "base_width": base_width,
            "connector_radius": connector_radius, "atom_radius": atom_radius,
            "bond_radius": bond_radius,
            "backbone_atom_radius": backbone_atom_radius,
            "backbone_bond_radius": backbone_bond_radius,
            "probe_radius": probe_radius,
            "surface_padding": surface_padding,
            "include_ligands": include_ligands,
            "exclude_chains": exclude_chains,
            "ligand_style": ligand_style, "ligand_atom": ligand_atom,
            "ligand_bond": ligand_bond,
            "ligand_vdw_scale": ligand_vdw_scale,
            "connect": connect, "use_magnets": use_magnets,
            "no_magnet_method": no_magnet_method,
            "connector_diameter": connector_diameter,
            "magnet_thickness": magnet_thickness, "magnet_shape": magnet_shape,
            "magnet_count": magnet_count, "dna_magnet_count": dna_magnet_count,
            "socket": socket, "socket_wall": socket_wall,
            "magnet_fit_clearance": magnet_fit_clearance,
            "joint_overrides": joint_overrides,
            "basepair_connect": basepair_connect,
        })
    except (ValueError, TypeError) as exc:
        if tmp_upload_dir:
            shutil.rmtree(tmp_upload_dir, ignore_errors=True)
        return JSONResponse(
            _error_payload(f"Invalid parameter: {exc}"), status_code=400,
        )

    # 3. stream progress while the (blocking) build runs in a worker thread.
    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        # Set when the client goes away (Cancel button, closed tab, dropped
        # connection).  The worker polls it through ``should_cancel`` so an
        # abandoned build stops at the next chain instead of tying up a core
        # meshing a complex nobody is waiting for any more.
        cancelled = threading.Event()
        # Set by the worker as its first act, so the watchdog below can tell
        # "still meshing" from "never started".
        started = threading.Event()

        def progress(frac, msg):
            _live_say(ticket, frac, msg)
            loop.call_soon_threadsafe(
                queue.put_nowait, ("progress", {"frac": frac, "msg": msg}))

        def model_ready(payload):
            """The GLB exists; put it on screen without waiting for the rest."""
            loop.call_soon_threadsafe(queue.put_nowait, ("model", payload))

        def work():
            started.set()
            # ``__done__`` is queued from an outer finally, and that placement is
            # the whole point of it: the reader loop below ends on that event and
            # on nothing else, so any path that skips it leaves the browser
            # waiting on a connection that will never speak again.  There is no
            # timeout on a fetch stream — that is the freeze only a refresh
            # clears.  Whatever happens above, the stream gets its terminator.
            try:
                try:
                    result = _run_and_export(source, params, progress,
                                             should_cancel=cancelled.is_set,
                                             on_model=model_ready)
                except Exception as exc:  # _run_and_export shouldn't raise
                    result = _error_payload(str(exc))
                finally:
                    if tmp_upload_dir:
                        shutil.rmtree(tmp_upload_dir, ignore_errors=True)
                loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("__done__", None))

        # Wait for a slot before starting any geometry.  The upload has already
        # been written to disk by this point, so a queued request costs a temp
        # file and an open connection and nothing else.
        gate = _gate()
        ticket = gate.take_ticket()
        waiter = asyncio.ensure_future(gate.acquire(ticket))
        acquired = False
        worker = None
        watchdog = None
        try:
            async for frame in _wait_for_slot(gate, ticket, waiter):
                yield frame
            acquired = True

            _live_start(ticket, "generate", str(source))
            worker = loop.run_in_executor(_build_pool(), work)
            watchdog = asyncio.ensure_future(_watch_start(queue, started))
            async for frame in _stream_events(queue):
                yield frame
        finally:
            _live_end(ticket)
            # Reached on normal completion *and* when the client disconnects
            # (the generator is closed / cancelled).  Setting it after a normal
            # finish is harmless — the worker has already returned.
            cancelled.set()
            if watchdog is not None:
                watchdog.cancel()
            if tmp_upload_dir and not acquired:
                shutil.rmtree(tmp_upload_dir, ignore_errors=True)
            _hand_back_slot(gate, ticket, waiter, acquired, worker)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# Display stand
# --------------------------------------------------------------------------
def _column_shape(raw) -> ColumnShape:
    """The column style asked for, or ``square`` if it is one we no longer offer.

    ``taper`` — the obelisk — was withdrawn as a style. Old links, cached form
    state and a browser tab left open over the change can all still send it, and
    none of those deserve a 500; they get the default instead.
    """
    try:
        shape = ColumnShape(str(raw or "square"))
    except ValueError:
        return ColumnShape.SQUARE
    return ColumnShape.SQUARE if shape == ColumnShape.TAPER else shape


def _map_stand(fields: dict) -> StandParams:
    """Build a :class:`StandParams` from the stand form fields."""
    return StandParams(
        enabled=True,
        columns=int(float(fields.get("stand_columns", 0) or 0)),
        orbit_theta_deg=float(fields.get("orbit_theta", 0.0) or 0.0),
        orbit_phi_deg=float(fields.get("orbit_phi", 75.0) or 75.0),
        roll_deg=float(fields.get("orbit_roll", 0.0) or 0.0),
        plate_margin_mm=float(fields.get("plate_margin", 7.0) or 7.0),
        plate_thickness_mm=float(fields.get("plate_thickness", 4.0) or 4.0),
        plate_corner_mm=float(fields.get("plate_corner", 4.0) or 4.0),
        column_diameter_mm=float(fields.get("column_diameter", 8.0) or 8.0),
        column_flared=_bool(fields.get("column_flared", True)),
        column_edge_frac=float(fields.get("column_edge_frac", 0.45) or 0.45),
        cradle_clearance_mm=float(fields.get("cradle_clearance", 0.35) or 0.35),
        cradle_depth_mm=float(fields.get("cradle_depth", 4.0) or 4.0),
        stand_off_mm=float(fields.get("stand_off", 6.0) or 6.0),
        plaque=_bool(fields.get("plaque", True)),
        plaque_pdb_id=_bool(fields.get("plaque_pdb_id", True)),
        plaque_title_text=str(fields.get("plaque_title_text", "") or "")[:120],
        plaque_note=str(fields.get("plaque_note", "") or "")[:80],
        plaque_scalebar=_bool(fields.get("plaque_scalebar", True)),
        plaque_legend=_bool(fields.get("plaque_legend", True)),
        plaque_legend_labels=str(fields.get("plaque_legend_labels", "") or "")[:2000],
        plaque_tile=_bool(fields.get("plaque_tile", True)),
        plaque_relief=PlaqueRelief(fields.get("plaque_relief", "raised")),
        plaque_text_mm=float(fields.get("plaque_text", 5.0) or 5.0),
        plaque_font=PlaqueFont(fields.get("plaque_font", "sans")),
        plaque_info_mm=max(0.0, min(200.0,
                                    float(fields.get("plaque_info_mm", 0) or 0))),
        # Not a control any more. One nozzle plus a little covers every printer
        # this is aimed at, and a number nobody can check by looking at the
        # sketch is a number that gets set wrong.
        plaque_min_stroke_mm=0.45,
        apron_rake_deg=float(fields.get("apron_rake", 0.0) or 0.0),
        column_shape=_column_shape(fields.get("column_shape", "square")),
        column_pins=_bool(fields.get("column_pins", False)),
        pin_diameter_mm=float(fields.get("pin_diameter", 4.0) or 4.0),
        pin_depth_mm=float(fields.get("pin_depth", 3.0) or 3.0),
    )


def _plaque_id(source: str) -> str:
    """What the plaque calls this structure: its ID, or an uploaded file's name."""
    stem = os.path.splitext(os.path.basename(source))[0]
    return stem.upper() if len(stem) <= 8 else stem


def _plaque_meta(source: str) -> dict:
    """The PDB ID and structure title the plaque prints, from scratch.

    The slow path, and the fallback. Prefer :func:`_stand_meta`, which uses what
    the build already read: this one re-resolves the source, which for a PDB ID
    means going back to RCSB and for a deleted upload means failing.

    ``why`` records what went wrong rather than dropping it. A title that is
    absent because the header has none and a title that is absent because the
    file could not be opened look identical on a plaque, and only one of them is
    the user's problem.
    """
    from pdb2print import io as p2p_io
    from pdb2print.names import structure_title

    meta = {"pdb_id": _plaque_id(source), "title": None, "why": ""}
    try:
        meta["title"] = structure_title(p2p_io.resolve_source(source))
        if not meta["title"]:
            meta["why"] = "this structure's header has no title record"
    except Exception as exc:
        meta["why"] = f"the structure could not be re-read ({exc})"
    return meta


def _stand_meta(source: str, token: str) -> dict:
    """The plaque's text, from the build that produced these meshes if possible."""
    entry = _recall_build(token)
    if entry and entry.get("meta", {}).get("pdb_id"):
        return dict(entry["meta"])
    return _plaque_meta(source)


def _backfill_names(objects, source: str) -> None:
    """Fill in subunit names on objects recovered from filenames alone.

    Mutates in place, and never raises: a plaque that says "Chain A" is a
    disappointment, and a stand that failed because RCSB was slow would be a
    bug. Any failure here leaves the fallback names exactly as they were.

    Uploaded structures are skipped -- their file is long deleted and there is
    no id to fetch -- so those keep falling back, as they always have.
    """
    source = (source or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9]{4,8}", source):
        return                                  # an upload, not a PDB id
    tmp = None
    try:
        from pdb2print.io import fetch_pdb_id
        from pdb2print.names import chain_names
        tmp = tempfile.mkdtemp(prefix="pdb2print_names_")
        names = chain_names(fetch_pdb_id(source, tmp))
        for obj in objects:
            if getattr(obj, "name", None):
                continue
            name = names.get(str(getattr(obj, "chain_id", "")))
            if name:
                obj.name = name
    except Exception:
        return                                  # the fallback names still work
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _meshes_from_3mf(hit: dict, names):
    """The entry's 3MF reopened as one mesh per name in ``names``, or ``None``.

    Worth trying before the STL zip, and the reason is not float precision.
    An STL is a *triangle soup*: it carries no vertex indices at all, so a
    loader has to re-weld coincident corners by tolerance, and on an organic
    surface with near-coincident sheets that weld produces duplicate faces and
    4-valent edges.  That is why a reopened cache entry is almost never
    watertight, and why looser tolerances and a full pymeshlab repair pass both
    failed to recover one.

    A 3MF stores explicit triangle indices — ``export.write_3mf`` writes
    ``tri.Indices`` per face — so there is nothing to guess.  The coordinates
    are float32 either way; the topology is what was being lost.

    Matched by **object name**, never by position.  ``graph.nodes_geometry``
    is not build order: trimesh walks the scene graph with a LIFO queue, so
    three build items come back as 1, 3, 2 and eight as 1, 8, 7, 6, 5, 4, 3, 2.
    A count check cannot see that — the count is right and only the pairing is
    wrong — and the result would be watertight, so it would sail through the
    gate and attach every chain's name and colour to a different chain's
    geometry.  ``write_3mf`` sets each object's name from
    ``chain.object_name()`` and ``CachedObject.object_name()`` reproduces it, so
    the join is exact.

    Deliberately strict: anything unexpected returns ``None`` and the caller
    falls back to the STL path, so the worst case is exactly what happened
    before.
    """
    name = (hit.get("files") or {}).get("threemf")
    if not name or not name.lower().endswith(".3mf"):
        return None                     # older entries fell back to a GLB
    path = os.path.join(cache.entry_dir(hit["key"]), name)
    if not os.path.isfile(path):
        return None
    try:
        import trimesh
        loaded = trimesh.load(path, file_type="3mf", process=False)
    except Exception:
        return None

    if isinstance(loaded, trimesh.Trimesh):
        # A single-object 3MF carries no ambiguity to resolve.
        return [loaded] if len(names) == 1 else None

    graph = getattr(loaded, "graph", None)
    geometry = getattr(loaded, "geometry", None)
    if graph is None or not geometry:
        return None
    by_name = {}
    try:
        # Through the scene graph rather than geometry.values(), so a build item
        # placed with a transform is honoured instead of silently loading at the
        # origin.
        for node in graph.nodes_geometry:
            transform, gname = graph[node]
            mesh = geometry.get(gname)
            if mesh is None or gname in by_name:
                return None              # missing, or a name used twice
            mesh = mesh.copy()
            mesh.apply_transform(transform)
            by_name[gname] = mesh
    except Exception:
        return None

    if set(by_name) != set(names) or len(by_name) != len(names):
        # A stand entry, a components object, a name trimesh made unique behind
        # our back, or anything else unexpected. Not worth guessing — take the
        # old road.
        return None
    return [by_name[n] for n in names]


def _built_from_cache(hit: dict, source: str = "", require_watertight: bool = True):
    """Reopen a cache entry as ``[(object, mesh), ...]``, or ``None``.

    The per-chain STL zip an entry already carries *is* the finished geometry —
    the same meshes the 3MF was written from, after the connections pass. Paired
    with the object metadata stored beside it, that is everything a display stand
    needs, so a structure served from cache can be stood up without meshing it a
    second time.

    This matters more than it looks. The in-memory registry only holds builds
    this process actually performed, and the whole point of the cache is that
    the popular structures are *never* built — so before this, the exact
    structures most likely to be asked for were the ones guaranteed to pay for a
    full rebuild before they could get a stand.
    """
    import io as _io
    import zipfile

    import trimesh

    from pdb2print import meshops

    # The 3MF first, because it keeps its indices — see _meshes_from_3mf. Needs
    # the object list up front, which only the stored metadata can give; entries
    # old enough to lack it go straight to the zip, which can name them.
    meta_objects = cache_mod.objects_from_meta(hit)
    if meta_objects:
        indexed = _meshes_from_3mf(
            hit, [obj.object_name() for obj in meta_objects])
        if indexed is not None:
            out = []
            for obj, mesh in zip(meta_objects, indexed):
                try:
                    mesh = meshops.repair(mesh)
                except Exception:
                    out = None
                    break
                if require_watertight and not mesh.is_watertight:
                    out = None
                    break
                out.append((obj, mesh))
            if out:
                return out

    name = (hit.get("files") or {}).get("stl_zip")
    if not name:
        return None
    path = os.path.join(cache.entry_dir(hit["key"]), name)
    if not os.path.isfile(path):
        return None

    by_label = {}
    order = []
    try:
        with zipfile.ZipFile(path) as zf:
            # Zip order is write order is build order, which is the order the
            # colour palette was handed out in. Keep it.
            for entry in zf.namelist():
                if not entry.lower().endswith(".stl"):
                    continue
                with zf.open(entry) as fh:
                    mesh = trimesh.load(_io.BytesIO(fh.read()), file_type="stl")
                label = os.path.splitext(os.path.basename(entry))[0]
                by_label[label] = mesh
                order.append(label)
    except Exception:
        return None

    # Entries written before the metadata existed still carry everything needed
    # in their filenames, so fall back to those rather than rebuilding.
    objects = cache_mod.objects_from_meta(hit)
    if not objects:
        objects = cache_mod.objects_from_labels(order)
        # ...everything except the subunit names, which a filename cannot hold.
        # Left alone, a model reopened from one of those entries -- which is
        # every pre-generated one shipped in the repo -- puts "Chain A" on its
        # plaque where a freshly built one puts "Zif268 (A)": same structure,
        # same settings, a worse plaque, for a reason no user can see.
        #
        # The names live in the file header, and reading a header is a parse,
        # not a build -- no meshing, no geometry core -- so re-read it rather
        # than let a cache hit quietly cost you the legend.
        _backfill_names(objects, source)
    if not objects:
        return None

    built = []
    for obj in objects:
        mesh = by_label.get(obj.label())
        if mesh is None:
            return None            # a partial model is worse than a rebuild
        # An STL is a triangle soup written in float32, so what comes back is
        # not the mesh that went in: vertices that were one point are now
        # several, and putting them back together is a tolerance, not an
        # identity.  Usually it closes.  When it does not, the crack is
        # invisible right up until the 3MF exporter's watertight gate refuses
        # the finished stand — and by then the user has watched a stand being
        # built and cannot download it, which is the worst place to find out.
        #
        # So a reopened mesh has to clear the same bar a freshly built one
        # does.  If it cannot, this route declines and the caller rebuilds:
        # slower, and correct, which is the right way round.
        try:
            mesh = meshops.repair(mesh)
        except Exception:
            return None
        # ``require_watertight`` is off only for the live stand preview.  The
        # column solver does no boolean work at all — no reference to _manifold
        # anywhere in it — so it does not need a closed surface; the gate is
        # here for the 3MF exporter, which the preview never reaches.  With it
        # on for previews, a cache-served model showed the generic sketch
        # forever and paid ~6.7s per nudge to decide that.
        if require_watertight and not mesh.is_watertight:
            return None
        built.append((obj, mesh))
    return built or None


def _stand_meshes(source: str, params: PrintParams, token: str, progress=None,
                  require_watertight: bool = True):
    """``(built, base_params, where)`` for a stand, without ever rebuilding.

    Two routes, in cost order: the meshes this process still has in memory for
    ``token``, then the disk cache entry for this exact build — whose per-chain
    STL zip *is* the finished geometry. Returns ``(None, params, None)`` when
    neither has it, and leaves the decision about whether a rebuild is worth it
    to the caller: it is, for a stand the user asked for and is watching a
    progress bar for; it is not, for a preview that has to answer in a moment or
    not at all.
    """
    # Two memo slots, because the two callers want different things and one
    # must not be served the other's answer. A watertight set satisfies both; a
    # relaxed one satisfies only the preview, and handing it to a real stand
    # build would fail at the 3MF gate after the user watched it being made.
    built_key = "built" if require_watertight else "built_loose"
    failed_key = "reopen_failed" if require_watertight else "reopen_failed_loose"

    entry = _recall_build(token)
    if entry is not None and entry.get("built") is not None:
        if progress:
            progress(0.30, "Using the model already built…")
        return entry["built"], entry["params"], "meshes in memory"
    if (entry is not None and not require_watertight
            and entry.get("built_loose") is not None):
        return entry["built_loose"], entry["params"], "the cached build"
    # Key of an entry already tried and declined this session, so the disk
    # lookup below does not reopen the very same zip a second time.  It did:
    # the lookup reconstructs the key /api/generate stored, which for a token
    # carrying a hit is that hit -- so a preview on a cache-served model paid
    # the reopen cost twice before answering "not ready".
    declined_key = None
    if entry is not None and entry.get("hit"):
        declined_key = (entry.get("hit") or {}).get("key")
        if not entry.get(failed_key):
            # The token came from a cache hit: no meshes were ever made in this
            # process, but the entry that served it is known exactly, so there is
            # nothing to look up and nothing to get wrong.
            if progress:
                progress(0.20, "Reopening the cached model…")
            built = _built_from_cache(entry["hit"], source, require_watertight)
            if built is not None:
                _remember_reopened(token, built, slot=built_key)
                return built, entry["params"], "the cached build"
            _remember_reopen_failed(token, slot=failed_key)
    try:
        # Look the build up the way the build itself was stored: with the stand
        # switched OFF. canonical_params drops the whole stand block when it is
        # disabled, and that is the key an ordinary /api/generate wrote. Asking
        # with a stand enabled builds a key carrying every stand field, which
        # nothing on disk can ever match -- so every model served from cache
        # reported "not in memory", the preview fell back to its generic sketch,
        # and a real stand quietly paid for a full rebuild it did not need.
        import dataclasses as _dc
        base = params
        if getattr(params, "stand", None) is not None:
            base = _dc.replace(params,
                               stand=_dc.replace(params.stand, enabled=False))
        hit = cache.lookup(source, base)
    except Exception:
        hit = None
    if hit and declined_key is not None and hit.get("key") == declined_key:
        hit = None                    # same entry, already tried and declined
    if hit:
        if progress:
            progress(0.20, "Reopening the cached model…")
        built = _built_from_cache(hit, source, require_watertight)
        if built is not None:
            _remember_reopened(token, built, params, slot=built_key)
            return built, params, "the cached build"
    return None, params, None


def _run_stand(source: str, params: PrintParams, token: str, progress,
               should_cancel=None, on_model=None) -> dict:
    """Generate a display stand and export the model standing on it.

    Uses the in-memory meshes for ``token`` when they are still there, and falls
    back to a full rebuild when they are not — which happens after a restart, an
    eviction, or when the original build was served from the disk cache and so
    never existed as meshes in this process at all.

    **Nothing here is written to the disk cache, deliberately.**  A stand is a
    one-off: it is keyed on an orientation the user arranged by hand and will
    almost never arrange identically twice, so an entry would be written once
    and read never — while costing the same hundreds of megabytes per structure
    that the pre-generated entries are budgeted in.  The cache exists to make
    the *famous structure at preset settings* instant, and a stand is the
    opposite of that.  (The rebuild fallback below calls ``build_all`` directly
    rather than ``_run_and_export`` for the same reason: it must not store
    either.)
    """
    from pdb2print import stand as stand_mod

    built, base_params, where = _stand_meshes(source, params, token, progress)

    if built is None:
        progress(0.02, "Rebuilding the model to stand it…")
        try:
            report = build_all(source, params, progress=lambda f, m: progress(
                0.02 + 0.55 * f, m), should_cancel=should_cancel)
        except BuildCancelled:
            return _error_payload("Build cancelled.")
        except Exception as exc:
            return _error_payload(str(exc))
        built = report.built
        base_params = params
        where = "a fresh build"

    # The stand block is the only thing that may differ from the build that
    # produced these meshes: everything else has to match, or the stand would be
    # sized for geometry that is not there.
    import dataclasses
    effective = dataclasses.replace(base_params, stand=params.stand)

    progress(0.62, "Placing the stand…")
    try:
        oriented, stand_parts, notes = stand_mod.build_stand(
            built, effective, meta=_stand_meta(source, token))
    except Exception as exc:
        return _error_payload(f"Could not build the display stand: {exc}")

    combined = list(oriented) + list(stand_parts)

    progress(0.86, "Writing export files…")
    _sweep_output_root()
    out_token = uuid.uuid4().hex
    out_dir = os.path.join(OUTPUT_ROOT, out_token)
    os.makedirs(out_dir, exist_ok=True)
    stem = _download_stem(source, effective) + "_stand"

    # The viewer legend names the *molecule*. A base plate and a lettering tile
    # are not chains and listing them there turns a legend into an inventory —
    # the point of the pills is to say which colour is which subunit, and a
    # "stand_plaque_tile" pill answers a question nobody asked.
    #
    # Worked out before the exports rather than after, because the early preview
    # below needs it: the viewer wants the GLB and the pills beside it, and
    # neither has anything to do with a zip.
    chains = [
        {"id": getattr(chain, "chain_id", "-"),
         "name": chain.display_name(),
         "color": _rgb_to_hex(color)}
        for (chain, _mesh), color in zip(combined, export.object_colors(combined))
        if getattr(chain, "mtype", None) != MoleculeType.STAND
    ]

    mins = [min(m.bounds[0][k] for _, m in combined) for k in range(3)]
    maxs = [max(m.bounds[1][k] for _, m in combined) for k in range(3)]
    size_mm = [float(maxs[k] - mins[k]) for k in range(3)]

    try:
        # The preview is written Y-up so it stands the right way in the viewer
        # and the orbit's poles agree with the model's own up; the print files
        # stay Z-up, which is what a slicer means by up.
        export.write_glb(stand_mod.to_view_frame(combined),
                         os.path.join(out_dir, f"{stem}.glb"))
    except OSError as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        return _error_payload(
            "The server ran out of disk space while writing the export files. "
            f"({exc.strerror or exc})")

    # Show the stand now, the same way the model itself is shown now: the viewer
    # needs the GLB and nothing else, and the stand's zip and 3MF are written
    # from the model *plus* a plate, columns and a few hundred letter solids, so
    # they are slower here than they are for the model alone. Holding the
    # finished stand off screen until a 3MF exists is the same wait that was
    # taken off /api/generate, for the same reason.
    #
    # Not the whole result: the download links point at files that are still
    # being written, and offering them here would be offering a 404.
    if on_model is not None:
        try:
            on_model({"glb_url": f"/files/{out_token}/{stem}.glb",
                      "chains": chains, "size_mm": size_mm,
                      "scale_used": effective.scale_mm_per_angstrom})
        except Exception:
            pass                  # a preview must never be able to fail a build
        progress(0.94, "Stand ready — writing the download files…")

    try:
        export.write_stl_zip(combined, os.path.join(out_dir, f"{stem}_stl.zip"))
    except OSError as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        return _error_payload(
            "The server ran out of disk space while writing the export files. "
            f"({exc.strerror or exc})")

    threemf_url = None
    warning = None
    try:
        export.write_3mf(combined, os.path.join(out_dir, f"{stem}.3mf"))
        threemf_url = f"/files/{out_token}/{stem}.3mf"
    except (RuntimeError, OSError) as exc:
        warning = str(exc)

    progress(1.0, "Done.")
    # Say what was actually produced, and from which switches. A plaque element
    # that should not be there is otherwise only visible as a shape in a
    # preview, which makes "it is still doing X" impossible to check against
    # what the server believes it did.
    sp = effective.stand
    lines = [
        f"Display stand: {len(stand_parts)} object(s) — "
        + ", ".join(part.object_name() for part, _m in stand_parts),
        f"  from {where or 'a fresh build'}"
        f" · columns {sp.columns or 'auto'} {sp.column_shape.value}"
        f"{' flared' if sp.column_flared else ' straight'}"
        f" · orbit {sp.orbit_theta_deg:.0f}/{sp.orbit_phi_deg:.0f}"
        f" roll {sp.roll_deg:.0f}",
        f"  plaque={sp.plaque} id={sp.plaque_pdb_id} "
        f"name={sp.plaque_title_text!r} "
        f"scale={sp.plaque_scalebar} legend={sp.plaque_legend} "
        f"tile={sp.plaque_tile} {sp.plaque_relief.value} "
        f"font={sp.plaque_font.value} "
        f"rake={sp.apron_rake_deg:.0f}°"
        f"{' · pinned columns' if sp.column_pins else ''}"
        + (f" note={sp.plaque_note!r}" if sp.plaque_note else ""),
    ]
    lines.extend(f"  ! {n}" for n in notes)
    return {
        "ok": True,
        "warning": warning,
        "report": "\n".join(lines),
        "glb_url": f"/files/{out_token}/{stem}.glb",
        "threemf_url": threemf_url,
        "stl_url": f"/files/{out_token}/{stem}_stl.zip",
        "chains": chains,
        "connections": [],
        "size_mm": size_mm,
        "scale_used": effective.scale_mm_per_angstrom,
        "stand_notes": notes,
        # The stand build is not remembered: standing it again would start from
        # the already-rotated meshes and tilt it twice.
        "build_token": token,
    }


def _stand_form(form: dict) -> dict:
    """The stand routes take the form wholesale, so they owe it the defaults.

    ``_map_params`` indexes the fields the generate endpoint declares with a
    ``Form(...)`` default, so those defaults have to be supplied here too —
    rather than have a second thirty-parameter signature drift out of step with
    the first.
    """
    fields = {
        "scale": 1.5, "grid_spacing": 0.5, "min_wall": 1.0,
        "min_wall_mode": "uniform", "protein_rep": "surface",
        "nucleic_rep": "tube_slab", "backbone_style": "tube",
        "base_style": "slab", "nucleic_radius": 1.2, "slab_thickness": 1.2,
        "base_width": 1.0, "connector_radius": 0.6, "atom_radius": 1.0,
        "bond_radius": 0.5, "probe_radius": 1.4, "surface_padding": 0.0,
    }
    fields.update({k: v for k, v in form.items() if v != ""})
    return fields


@app.post("/api/stand/preview")
async def stand_preview(request: Request):
    """Solve a stand's layout — and nothing else — for the live sketch.

    Everything expensive about a stand happens *after* the layout: carving a
    cradle out of a surface mesh, sweeping a few hundred stroke solids, meshing
    and writing the result. The arithmetic that decides where the plate goes,
    how deep the apron has to be and which points on the underside the columns
    will rise to is a fraction of a second, and it is the whole of what a
    preview needs.

    The point is that the front end does not *re-derive* any of it. A sketch
    that solves for its own column positions is a second implementation of the
    hardest judgement in this module, and the day the two disagree the picture
    will be confidently wrong — which is worse than a picture that admits it is
    generic.

    Never rebuilds. If the meshes are neither in memory nor in the cache this
    answers ``ready: false`` at once and the sketch falls back to its generic
    drawing, because a preview that takes a minute is not a preview.
    """
    form = dict(await request.form())
    token = str(form.get("build_token", "")).strip()
    source = str(form.get("pdb_id", "")).strip()
    if not source:
        remembered = _recall_build(token)
        if remembered is not None:
            source = remembered["source"]
    if not source:
        return JSONResponse({"ok": True, "ready": False, "reason": "no-model"})

    try:
        params = _map_params(_stand_form(form))
        params.stand = _map_stand(_stand_form(form))
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse(
            _error_payload(f"Invalid parameter: {exc}"), status_code=400)

    def work():
        from pdb2print import stand as stand_mod
        import dataclasses

        built, base_params, _where = _stand_meshes(source, params, token,
                                                   require_watertight=False)
        if built is None:
            return {"ok": True, "ready": False, "reason": "not-in-memory",
                "stamp": CODE_STAMP}
        effective = dataclasses.replace(base_params, stand=params.stand)
        layout = stand_mod.solve_layout(built, effective,
                                        meta=_stand_meta(source, token))
        if layout is None:
            return {"ok": True, "ready": False, "reason": "empty"}
        summary = stand_mod.layout_summary(layout, effective)
        summary.update({"ok": True, "ready": True, "stamp": CODE_STAMP})
        return summary

    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(_preview_pool(), work)
    except Exception as exc:                 # a preview must never take the page down
        return JSONResponse({"ok": True, "ready": False, "reason": str(exc)})
    return JSONResponse(payload)


@app.post("/api/stand")
async def stand(request: Request):
    """Add a display stand to a model that has already been built.

    Takes the same form fields as ``/api/generate`` plus a ``build_token`` and
    the stand block, and streams progress the same way, so the front end reuses
    its whole result-handling path.
    """
    form = dict(await request.form())
    token = str(form.get("build_token", "")).strip()
    source = str(form.get("pdb_id", "")).strip()

    # An *uploaded* structure has no PDB ID to resend, and the temp file it was
    # read from is deleted the moment its build finishes. The remembered build
    # is therefore the only route to standing one — so take the source from
    # there when the form cannot supply it, and only refuse when neither can.
    if not source:
        remembered = _recall_build(token)
        if remembered is not None:
            source = remembered["source"]
    if not source:
        return JSONResponse(
            _error_payload("A display stand needs a model that is still in "
                           "memory on the server. Generate the model again, "
                           "then add the stand."),
            status_code=400)

    fields = _stand_form(form)

    try:
        params = _map_params(fields)
        params.stand = _map_stand(fields)
    except (ValueError, TypeError, KeyError) as exc:
        return JSONResponse(
            _error_payload(f"Invalid parameter: {exc}"), status_code=400)

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()
        started = threading.Event()

        def progress(frac, msg):
            _live_say(ticket, frac, msg)
            loop.call_soon_threadsafe(
                queue.put_nowait, ("progress", {"frac": frac, "msg": msg}))

        def model_ready(payload):
            """The GLB exists; put the stand on screen without waiting for the rest."""
            loop.call_soon_threadsafe(queue.put_nowait, ("model", payload))

        def work():
            started.set()
            # Same guarantee as /api/generate: the terminator is queued from an
            # outer finally so no failure above can leave the stream silent.
            try:
                try:
                    result = _run_stand(source, params, token, progress,
                                        should_cancel=cancelled.is_set,
                                        on_model=model_ready)
                except Exception as exc:
                    result = _error_payload(str(exc))
                loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("__done__", None))

        # Stands go through the same gate as builds.  A stand that hits the
        # cache is nearly free, but one that misses runs the whole pipeline —
        # and a gate that only counted /api/generate would let a stand and a
        # build saturate the box between them, which is the case it exists for.
        gate = _gate()
        ticket = gate.take_ticket()
        waiter = asyncio.ensure_future(gate.acquire(ticket))
        acquired = False
        worker = None
        watchdog = None
        try:
            async for frame in _wait_for_slot(gate, ticket, waiter):
                yield frame
            acquired = True

            _live_start(ticket, "stand", str(source))
            worker = loop.run_in_executor(_build_pool(), work)
            watchdog = asyncio.ensure_future(_watch_start(queue, started))
            async for frame in _stream_events(queue):
                yield frame
        finally:
            _live_end(ticket)
            cancelled.set()
            if watchdog is not None:
                watchdog.cancel()
            _hand_back_slot(gate, ticket, waiter, acquired, worker)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the rest of the front-end assets (kept last so /api and / win first).
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
