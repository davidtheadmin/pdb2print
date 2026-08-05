#!/usr/bin/env python3
"""Would this checkout still find the entries already in the cache?

A cache key is a hash of ``CACHE_VERSION``, the canonicalised source and the
canonicalised params.  Anything that changes any of the three renames every key
at once, and the entries do not move with them: they stay on disk, unreachable,
counting against the size cap until something evicts them.  That has already
happened twice on this project and both times it was noticed weeks later.

This reads what is actually on disk and re-derives each key under the code in
*this* checkout.  It builds nothing and writes nothing, so it is safe to run
against a live cache directory, and it takes about a second for a few hundred
entries.

    python scripts/verify_cache_keys.py                 # the configured cache
    python scripts/verify_cache_keys.py /path/to/cache  # somewhere else

Exit status is 0 when every entry is still reachable and 1 when any is not, so
it can gate a deploy.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdb2print.cache import CACHE_VERSION, DEFAULT_CACHE_DIR  # noqa: E402
from pdb2print.io import canonical_pdb_id  # noqa: E402


def _key(version, source, params) -> str:
    """``cache.key_for``'s hash, over values already canonicalised."""
    blob = json.dumps({"v": version, "source": source, "params": params},
                      sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:20]


def main(argv) -> int:
    root = argv[1] if len(argv) > 1 else os.environ.get(
        "PDB2PRINT_CACHE_DIR", DEFAULT_CACHE_DIR)
    if not os.path.isdir(root):
        print(f"no cache directory at {root}", file=sys.stderr)
        return 1

    entries = sorted(n for n in os.listdir(root)
                     if os.path.isdir(os.path.join(root, n))
                     and not n.endswith(".tmp"))
    if not entries:
        print(f"{root}: empty")
        return 0

    reachable, stale, broken, moved = [], [], [], []
    sources: dict = {}
    for name in entries:
        meta_path = os.path.join(root, name, "meta.json")
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as exc:
            broken.append((name, str(exc)))
            continue

        version = meta.get("cache_version", CACHE_VERSION)
        source = meta.get("source", "")
        params = meta.get("params")
        if params is None:
            broken.append((name, "no params in meta.json"))
            continue

        sources.setdefault(source, []).append(name)

        # Already dead: written by a different CACHE_VERSION, so the key it
        # would hash to today is not the key it is filed under.
        if version != CACHE_VERSION:
            stale.append((name, version, source))
            continue

        # What this checkout would compute for the same build.  ``params`` is
        # stored post-canonicalisation, and canonical_params only ever *drops*
        # fields, so a new field cannot change it retroactively -- the source is
        # the only part this checkout could see differently.
        now = canonical_pdb_id(source) or source.strip().upper()
        if _key(CACHE_VERSION, now, params) == name:
            reachable.append(name)
        else:
            moved.append((name, source, now))

    print(f"{root}")
    print(f"  {len(entries)} entries, {len(sources)} distinct structures")
    print(f"  reachable by this checkout : {len(reachable)}")
    print(f"  stale (old CACHE_VERSION)  : {len(stale)}")
    print(f"  key would move             : {len(moved)}")
    print(f"  unreadable                 : {len(broken)}")

    for name, version, source in stale[:10]:
        print(f"    stale  {name}  v{version}  {source}")
    if len(stale) > 10:
        print(f"    ... and {len(stale) - 10} more")
    for name, source, now in moved[:10]:
        print(f"    MOVED  {name}  source {source!r} -> {now!r}")
    for name, why in broken[:10]:
        print(f"    broken {name}  {why}")

    if stale:
        total = 0
        for name, _v, _s in stale:
            for dirpath, _d, files in os.walk(os.path.join(root, name)):
                total += sum(os.path.getsize(os.path.join(dirpath, f))
                             for f in files)
        print(f"\n  stale entries occupy {total / 1e9:.2f} GB. They can never be "
              f"served.\n  enforce_limit evicts them before live ones, but only "
              f"once the cap is hit.")

    if moved:
        print("\nSome keys would move: deploying this checkout orphans those "
              "entries.", file=sys.stderr)
        return 1
    print("\nEvery live entry is still reachable. This checkout can be deployed "
          "without losing the cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
