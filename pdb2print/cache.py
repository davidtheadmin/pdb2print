"""Persistent cache of finished builds, keyed by structure + parameters.

Why this exists
---------------
A cold build of a large complex is the slowest thing the app does, and the
overwhelming majority of requests are for a handful of famous structures at
default settings.  Serving those as static files turns the common case into a
download and leaves the compute budget for genuinely novel requests.

The store is a plain directory of finished artefacts.  Nothing here knows how to
build anything: :func:`lookup` either finds a complete entry or it does not, and
the caller falls back to the pipeline.

Two design points worth knowing
-------------------------------
*Only PDB IDs are cached.*  An uploaded file would have to be keyed by content
hash to be safe — two people's ``model.pdb`` are not the same structure — and an
upload is a one-off by nature, so it is never worth the disk.

*The key is a hash of the normalised parameters, not a hand-picked subset.*
Listing "the fields that matter" by hand is a bug waiting to happen: someone adds
a parameter, forgets the list, and the cache starts serving output built with the
wrong settings.  Hashing everything is safe by default.  The normalisation in
:func:`canonical_params` then removes only fields that provably cannot reach the
geometry given the chosen representations — which raises the hit rate without
ever letting a meaningful difference collide.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Dict, List, Optional

from .config import PrintParams, Representation, BaseStyle, BackboneStyle
from .io import looks_like_pdb_id

#: Where the shipped cache lives.  A directory inside the repo (rather than a
#: temp dir) is the whole point: a free Space's filesystem resets on restart, so
#: anything written to ``tempfile.mkdtemp`` is gone by the next cold start.
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

#: Bumped when a change to the geometry code invalidates previously built
#: artefacts.  The parameters can be identical across such a change, so the
#: params hash alone cannot notice it — this is the manual escape hatch.
#: Bump it and every existing entry is ignored.
#:
#: 2 — base-pair links reworked: sized from the rung rather than the backbone
#:     tube, flat-faced and overlapping instead of domed and clearance-gapped,
#:     and the pairing register now gated on base-plane coplanarity so a strand
#:     with a one-base overhang no longer pairs every base to its neighbour's
#:     partner.  Same parameters, different geometry.
CACHE_VERSION = 2

#: Artefact kinds an entry must hold to count as complete.  A half-written entry
#: (a crash mid-export, a killed container) must never be served, so lookup
#: checks for all of these rather than trusting the directory's existence.
#:
#: Files are stored under their *descriptive* names — ``1zaa_pdb2print_1p5mm.3mf``
#: rather than ``model.3mf`` — because the front end takes the download filename
#: from the last segment of the URL.  Fixed names on disk would mean every
#: cached download landing in someone's folder as ``model.3mf``, which is exactly
#: the problem the naming scheme was introduced to fix.  ``meta.json`` maps kind
#: to filename so lookup stays a dictionary read rather than a directory scan.
REQUIRED_KINDS = ("threemf", "glb", "stl_zip")


# --------------------------------------------------------------------------
# key
# --------------------------------------------------------------------------
def _plain(value):
    """Convert dataclasses/enums/tuples into something ``json.dumps`` accepts."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float):
        # Slider values arrive as strings and become floats, so 0.1+0.2 style
        # drift is real.  Rounding well below any dimension anyone can print
        # keeps 2.4000000000000004 and 2.4 on the same key.
        return round(value, 6)
    return value


def canonical_params(params: PrintParams) -> dict:
    """The parameter dict reduced to what can actually affect the output.

    Every removal below is a field the geometry cannot read given the chosen
    representations, so dropping it can only merge keys that would have produced
    byte-identical results.  The effect in practice: someone who nudged a cartoon
    slider, then built a surface model, still hits the cached surface build.

    Anything not explicitly dropped stays in the key.  New parameters are
    therefore included automatically and conservatively.
    """
    data = _plain(params)

    protein_rep = data.get("protein_representation")
    nucleic_rep = data.get("nucleic_representation")
    reps = {protein_rep, nucleic_rep}

    def drop(*names):
        for n in names:
            data.pop(n, None)

    # min_wall_mode selects a branch inside meshops.enforce_min_wall, and that
    # pass returns early for every representation there is (MIN_WALL_EXEMPT
    # holds SURFACE, TUBE_SLAB and CARTOON, and geometry.py always stamps the
    # representation onto the mesh).  It therefore cannot reach the geometry at
    # all.  Its UI control has been removed for the same reason; dropping it
    # here means the entries cached before that removal are still found.
    drop("min_wall_mode")

    # Cartoon dimensions are read only by the cartoon builder.
    if Representation.CARTOON.value not in reps:
        drop("cartoon_helix_width_mm", "cartoon_strand_width_mm",
             "cartoon_coil_radius_mm", "cartoon_arrow_width_factor",
             "cartoon_arrow_residues", "cartoon_samples_per_residue")

    # Surface tuning is read only when something is meshed as a surface.
    if Representation.SURFACE.value not in reps:
        drop("probe_radius_ang", "surface_atom_padding_ang")

    # The protein tube radius belongs to the protein "tubes" style alone.  It is
    # keyed off the protein representation only, never the nucleic one, because
    # the nucleic tube has its own radius.
    if protein_rep != "tubes":
        drop("protein_tube_radius_mm")

    if nucleic_rep != Representation.TUBE_SLAB.value:
        drop("nucleic_radius_mm", "slab_thickness_mm", "slab_scale",
             "connector_radius_mm", "spline_samples_per_residue",
             "base_style", "backbone_style", "atom_radius_mm", "bond_radius_mm",
             "backbone_atom_radius_mm", "backbone_bond_radius_mm")
    else:
        base = data.get("base_style")
        backbone = data.get("backbone_style")
        molecule = BaseStyle.MOLECULE.value
        # The bases and the backbone have *separate* ball-and-stick sizes, so
        # each pair is keyed off its own style alone.
        if base != molecule:
            drop("atom_radius_mm", "bond_radius_mm")
        if backbone != BackboneStyle.MOLECULE.value:
            drop("backbone_atom_radius_mm", "backbone_bond_radius_mm")
        # Slab/rod geometry is replaced wholesale by the molecule base style.
        if base == molecule:
            drop("slab_thickness_mm", "slab_scale", "connector_radius_mm")
        # A rod rung is round and reaches the backbone itself, so it reads only
        # its thickness — the plate's footprint scale and strut do not apply.
        elif base == BaseStyle.ROD.value:
            drop("slab_scale", "connector_radius_mm")
        # No tube is swept in the molecule backbone style.
        if backbone == BackboneStyle.MOLECULE.value:
            drop("nucleic_radius_mm")

    # The connections block is large and almost entirely irrelevant when the
    # pass is off.  Keep the two switches so "off" cannot collide with "on".
    conn = data.get("connections") or {}
    if not (conn.get("connect") or conn.get("basepair_connect")):
        data["connections"] = {"connect": False, "basepair_connect": False}
    elif not conn.get("connect"):
        # Base-pair connection only: nothing about the chain-join joinery is read.
        data["connections"] = {
            "connect": False,
            "basepair_connect": True,
            "basepair_max_dist_ang": conn.get("basepair_max_dist_ang"),
        }
    elif not conn.get("use_magnets"):
        for k in ("magnet_thickness_mm", "magnet_shape", "magnet_count",
                  "dna_magnet_count", "magnet_fit_clearance_mm",
                  "magnet_depth_clearance_mm", "magnet_chamfer_mm"):
            conn.pop(k, None)
        # "inflate" grows the surfaces together and never reads a diameter or
        # builds a collar; "bridge" reads both.
        if conn.get("no_magnet_method") == "inflate":
            for k in ("connector_diameter_mm", "socket", "socket_wall_mm",
                      "path_clearance_mm"):
                conn.pop(k, None)
    else:
        conn.pop("no_magnet_method", None)   # unused once magnets are on

    return data


def key_for(source: str, params: PrintParams) -> str:
    """Stable cache key for ``source`` built with ``params``.

    Deterministic across processes and machines: ``sort_keys`` removes dict
    ordering, and the normalisation above removes irrelevant fields, so the same
    request always lands on the same key.
    """
    payload = {
        "v": CACHE_VERSION,
        "source": source.strip().upper(),
        "params": canonical_params(params),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:20]


def is_cacheable(source: str) -> bool:
    """True if ``source`` is something worth (and safe) to cache.

    Uploads are excluded: keyed by name they would collide across users, and
    keying them by content hash costs a full read to serve a file that by its
    nature is wanted once.
    """
    return looks_like_pdb_id(source)


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------
class Cache:
    """A directory of finished builds.

    ``read_only`` is the setting to use on a deployment with an ephemeral disk:
    the shipped entries are still served, and misses are simply built each time
    rather than written to a filesystem that will not survive a restart.
    """

    def __init__(self, root: str = DEFAULT_CACHE_DIR, read_only: bool = False):
        self.root = os.path.abspath(root)
        self.read_only = read_only

    # -- paths ----------------------------------------------------------
    def entry_dir(self, key: str) -> str:
        return os.path.join(self.root, key)

    def _index_path(self) -> str:
        return os.path.join(self.root, "index.json")

    # -- read -----------------------------------------------------------
    def lookup(self, source: str, params: PrintParams) -> Optional[dict]:
        """Return the stored metadata for a complete entry, else ``None``."""
        if not is_cacheable(source):
            return None
        return self.lookup_key(key_for(source, params))

    def lookup_key(self, key: str) -> Optional[dict]:
        d = self.entry_dir(key)
        if not os.path.isdir(d):
            return None
        try:
            with open(os.path.join(d, "meta.json"), "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            return None
        # Partial entries are treated as absent: a build killed mid-export must
        # never be served as a finished one.
        files = meta.get("files") or {}
        if not all(k in files for k in REQUIRED_KINDS):
            return None
        if not all(os.path.exists(os.path.join(d, f)) for f in files.values()):
            return None
        meta["key"] = key
        meta["dir"] = d
        return meta

    def index(self) -> List[dict]:
        """Every complete entry's metadata, newest first where dates exist."""
        if not os.path.isdir(self.root):
            return []
        out = []
        for name in sorted(os.listdir(self.root)):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue
            meta = self.lookup_key(name)
            if meta:
                out.append(meta)
        return out

    # -- write ----------------------------------------------------------
    def store(self, source: str, params: PrintParams, files: Dict[str, str],
              meta: Optional[dict] = None) -> Optional[str]:
        """Copy ``files`` into a new entry and write its metadata.

        ``files`` maps a kind from :data:`REQUIRED_KINDS` to an existing path;
        each file keeps its own basename inside the entry.  Returns the key, or
        ``None`` if the entry was not written.

        The entry is assembled in a sibling ``.tmp`` directory and moved into
        place only once every file is there, so a crash halfway through leaves no
        directory that a later lookup could mistake for a finished build.
        """
        if self.read_only or not is_cacheable(source):
            return None
        missing = [k for k in REQUIRED_KINDS if k not in files]
        if missing:
            raise ValueError("Refusing to store an incomplete entry; missing: "
                             + ", ".join(missing))

        key = key_for(source, params)
        final = self.entry_dir(key)
        if os.path.isdir(final):
            return key

        staging = final + ".tmp"
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging, exist_ok=True)
        try:
            names = {}
            for kind, path in files.items():
                name = os.path.basename(path)
                shutil.copyfile(path, os.path.join(staging, name))
                names[kind] = name
            payload = dict(meta or {})
            payload.update({
                "key": key,
                "source": source.strip().upper(),
                "cache_version": CACHE_VERSION,
                "files": names,
                "params": canonical_params(params),
            })
            with open(os.path.join(staging, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            os.replace(staging, final)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            return None
        return key

    def write_index(self) -> str:
        """Write ``index.json`` — a human-readable listing of what is cached."""
        entries = [{k: v for k, v in m.items() if k != "dir"} for m in self.index()]
        path = self._index_path()
        os.makedirs(self.root, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"cache_version": CACHE_VERSION,
                       "count": len(entries),
                       "entries": entries}, fh, indent=2, sort_keys=True)
        return path
