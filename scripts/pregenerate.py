#!/usr/bin/env python
"""Build the shipped cache from ``scripts/cache_spec.json``.

Run from the repo root::

    python scripts/pregenerate.py                 # everything missing
    python scripts/pregenerate.py --only 1ZAA     # one structure
    python scripts/pregenerate.py --preset "Molecular"   # one preset only
    python scripts/pregenerate.py --force         # rebuild even if cached
    python scripts/pregenerate.py --list          # show the plan, build nothing

Why it goes through ``server._run_and_export``
----------------------------------------------
The cache is only useful if a shipped entry is found by a real request, and a
request is matched by hashing its parameters.  Anything this script did
differently from the server — a different export call, a different result
payload, a parameter set up by hand — would produce entries that look right on
disk and are never once served.  So the script drives the server's own build path
and lets the caching happen where it always does.  Importing ``server`` costs a
FastAPI import in a build script, which is a cheap price for that guarantee.

The exit code is the number of failures, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pdb2print import presets                      # noqa: E402
from pdb2print.cache import Cache, DEFAULT_CACHE_DIR, key_for   # noqa: E402
from pdb2print.config import ConnectionParams      # noqa: E402

DEFAULT_SPEC = os.path.join(HERE, "cache_spec.json")


def _coerce_overrides(raw: dict) -> dict:
    """Turn JSON overrides into things ``dataclasses.replace`` will accept.

    Only ``connections`` needs help: it is a nested dataclass, so a plain JSON
    object would be handed straight through and every later access would fail on
    a dict.  Building it here means the spec can switch the joinery on for one
    structure without a preset existing for it — which is what the reference
    print needs, since joinery is deliberately not part of any preset.
    """
    out = dict(raw or {})
    conn = out.get("connections")
    if isinstance(conn, dict):
        out["connections"] = ConnectionParams(**conn)
    return out


def load_spec(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data.get("entries", [])
    seen = {}
    for i, e in enumerate(entries):
        if not e.get("pdb_id"):
            raise ValueError(f"Spec entry {i} has no pdb_id.")
        preset = e.get("preset", presets.DEFAULT_PRESET)
        if preset not in presets.PRESET_NAMES:
            raise ValueError(
                f"Spec entry {e['pdb_id']} asks for unknown preset {preset!r}. "
                f"Known: {', '.join(presets.PRESET_NAMES)}."
            )
        e["preset"] = preset
        e["overrides"] = _coerce_overrides(e.get("overrides"))
        # A repeat lands on an existing key, so it is a wasted build rather than
        # a second entry. Caught here rather than after the first hour of work.
        sig = (e["pdb_id"].upper(), preset,
               json.dumps(e["overrides"], sort_keys=True, default=str))
        if sig in seen:
            raise ValueError(
                f"Spec entries {seen[sig]} and {i} are identical "
                f"({e['pdb_id']}, {preset}); they would share one cache key."
            )
        seen[sig] = i
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--only", action="append", default=[],
                    help="PDB ID to build (repeatable)")
    ap.add_argument("--preset", action="append", default=[],
                    help="only entries using this preset (repeatable)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild entries that are already cached")
    ap.add_argument("--list", action="store_true",
                    help="print the plan and exit")
    args = ap.parse_args()

    entries = load_spec(args.spec)
    if args.only:
        want = {s.strip().upper() for s in args.only}
        entries = [e for e in entries if e["pdb_id"].upper() in want]
    if args.preset:
        entries = [e for e in entries if e["preset"] in set(args.preset)]
    if not entries:
        print("Nothing to do — no spec entry matched.")
        return 0

    # The directory the *server* writes to, not the module default.  Those are
    # the same thing only when PDB2PRINT_CACHE_DIR is unset -- which is not how
    # the VPS runs, so the plan was read from one directory while the builds
    # went to another: every entry reported "not cached", --force deleted
    # nothing, and the index was written where the site never looks.
    cache_dir = os.environ.get("PDB2PRINT_CACHE_DIR", DEFAULT_CACHE_DIR)
    cache = Cache(cache_dir)
    print(f"cache: {cache_dir}")
    if os.environ.get("PDB2PRINT_CACHE_RO", "").strip().lower() in {
            "1", "true", "on", "yes"}:
        print("PDB2PRINT_CACHE_RO is set — every build would be discarded. "
              "Unset it and run again.")
        return 1

    # Resolve the plan before building anything, so --list is informative and a
    # bad preset name fails immediately rather than forty minutes in.
    plan = []
    for e in entries:
        params = presets.params_for(e["preset"], **(e.get("overrides") or {}))
        key = key_for(e["pdb_id"], params)
        cached = cache.lookup_key(key) is not None
        plan.append((e, params, key, cached))

    todo = [p for p in plan if args.force or not p[3]]
    print(f"{len(plan)} entr{'y' if len(plan) == 1 else 'ies'} in the spec, "
          f"{len(plan) - len(todo)} already cached, {len(todo)} to build.\n")

    if args.list:
        for e, _params, key, cached in plan:
            mark = "cached" if cached else "     -"
            print(f"  [{mark}] {e['pdb_id']:<6} {e['preset']:<12} {key}  "
                  f"{e.get('name', '')}")
        return 0

    # Imported here, not at module scope: it spins up the FastAPI app and a temp
    # output directory, which --list has no business doing.
    import server                                   # noqa: E402

    failures, built = [], 0
    for n, (e, params, key, _cached) in enumerate(todo, 1):
        pdb_id = e["pdb_id"]
        label = f"[{n}/{len(todo)}] {pdb_id} ({e['preset']})"
        print(f"{label} …", flush=True)

        if args.force:
            import shutil
            shutil.rmtree(cache.entry_dir(key), ignore_errors=True)

        started = time.time()
        last = [""]

        def progress(frac, msg):
            # One line per distinct stage; a percentage that ticks 400 times is
            # noise in a build log.
            if msg != last[0]:
                last[0] = msg
                print(f"      {int(frac * 100):3d}%  {msg}", flush=True)

        try:
            result = server._run_and_export(pdb_id, params, progress)
        except Exception as exc:
            result = {"ok": False, "warning": f"{type(exc).__name__}: {exc}"}

        elapsed = time.time() - started
        if not result.get("ok"):
            reason = result.get("warning") or "unknown error"
            print(f"      FAILED after {elapsed:.1f}s — {reason}\n")
            failures.append((pdb_id, e["preset"], reason))
            continue

        stored = cache.lookup_key(key)
        if not stored:
            # The build worked but nothing landed in the cache — almost always a
            # missing 3MF (the watertight gate refused it), which is exactly the
            # case that must not be silently shipped.
            reason = result.get("warning") or "no 3MF was produced"
            print(f"      BUILT but NOT CACHED after {elapsed:.1f}s — {reason}\n")
            failures.append((pdb_id, e["preset"], f"not cached: {reason}"))
            continue

        chains = result.get("chains") or []
        size = result.get("size_mm") or []
        dims = " × ".join(f"{d:.0f}" for d in size) + " mm" if size else "?"
        print(f"      ok in {elapsed:.1f}s — {len(chains)} object(s), {dims}, "
              f"key {key}\n")
        built += 1

    cache.write_index()
    print(f"Built {built}, failed {len(failures)}. Index written to "
          f"{os.path.join(cache.root, 'index.json')}.")
    if failures:
        print("\nFailures:")
        for pdb_id, preset, reason in failures:
            print(f"  {pdb_id} ({preset}): {reason}")
    return len(failures)


if __name__ == "__main__":
    raise SystemExit(main())
