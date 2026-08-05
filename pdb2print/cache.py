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
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Dict, List, Optional

from .config import (
    PrintParams, Representation, BaseStyle, BackboneStyle, MoleculeType,
    LigandStyle, HBondMode,
)
from .io import looks_like_pdb_id, canonical_pdb_id

#: Where the shipped cache lives.  A directory inside the repo (rather than a
#: temp dir) is the whole point: a free Space's filesystem resets on restart, so
#: anything written to ``tempfile.mkdtemp`` is gone by the next cold start.
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

#: Default ceiling on the whole store, overridable with ``PDB2PRINT_CACHE_MAX_GB``.
#:
#: 20 GB is chosen against a 100 GB disk that also carries the OS, a multi-GB
#: Docker image and Caddy's certificate storage.  It holds roughly a thousand
#: entries at the measured average of ~20 MB — far more than the handful of
#: structures that actually repeat — while leaving the machine plenty of room
#: to keep working.
DEFAULT_MAX_BYTES = 20 * 1024 ** 3

#: Free space below which the cache stops writing, regardless of its own cap.
#:
#: The cap alone is not enough: the disk is shared with the OS, the Docker image
#: and Caddy's certificate storage, so it can fill for reasons that have nothing
#: to do with the cache. Caching is only ever an optimisation, so it is the first
#: thing that should give way — better a slow site than one that cannot renew its
#: certificate or write an export.
MIN_FREE_BYTES = 2 * 1024 ** 3

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
#: 3 — bound ligands are exported instead of discarded, and their host is carved
#:     into a pocket around them.  Every pre-existing entry for a ligand-bearing
#:     structure therefore holds *both* the wrong object list and the wrong
#:     protein geometry.  Adding ``include_ligands`` to the parameters already
#:     changes the hash on its own, so this bump is belt-and-braces rather than
#:     the mechanism — but the mechanism is a hash of a dataclass and the cost of
#:     being wrong here is serving someone a model with no drug in it, so the
#:     explicit invalidation is worth the one-line diff.
#: 4: the joint search changed in every dimension this round -- seed axis,
#: support smoothing, a depth term, three score terms made scale-relative, and
#: the back-taper gate -- so a v3 entry is last round's magnets. Bumped in the
#: same commit as the geometry, because /cache/* is now served immutable and a
#: browser holding a stale entry has no way to find out otherwise.
#: 5: the cartoon's arrowheads and the stand's columns both changed shape. A
#: strand's last segment used to hold the arrow's section and then jump to the
#: coil tube; it tapers now. A column used to be a plain difference against the
#: model, which left material standing above anything it cut; it is a
#: downward-only cut now, and the column is nudged off any splinter the cut
#: would leave on its top. Same parameters, different mesh, in both cases.
#: 6: not a geometry change — with nothing switched off and no override set a
#: build meshes exactly as it did at 5. It is a *payload* change, which is the
#: same problem wearing different clothes. An entry stores the finished
#: result dict verbatim and ``_cached_result`` serves it back, so a hit on an
#: older entry hands the front end a payload with no ``parts`` list and
#: connections with no ``ai``/``bi``. The chains-and-joints panel reads exactly
#: those, so on the popular structures the pre-generated entries exist for — the
#: ones most likely to be tried first — the panel would simply never appear, and
#: the stale chain list from the previous structure would be left on screen
#: pointing at the wrong model. The entry also has no per-object ``index``, so a
#: display stand raised from one would recolour the whole model.
CACHE_VERSION = 6

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

    # ``include_ligands`` is never dropped, under any combination of the
    # representations.  It is the one parameter that changes *which objects exist*
    # rather than how one of them is shaped: with it on, a ligand-bearing
    # structure gains an object and its host gains a pocket, and with it off both
    # are absent.  There is no representation and no style under which those two
    # outputs are the same file, so there is never a case for merging the keys.
    # Listed here explicitly because everything else in this function is a
    # removal, and a reader looking for "where is the new field handled" should
    # find an answer rather than an absence.
    #
    # Its *thickness* is a different matter and does drop out: with ligands off
    # nothing reads it, so someone who nudged the ligand slider and then switched
    # ligands off still hits the build they would have got anyway.
    # ``exclude_chains`` is never dropped either, for exactly the reason
    # ``include_ligands`` is not: it changes *which objects exist*, and it also
    # changes the ones that remain — a chain with a neighbour switched off is no
    # longer carved to fit it and no longer offered a joint to it. There is no
    # setting under which "with chain B" and "without chain B" are the same file.
    #
    # Empty is removed rather than kept, the same as ``joint_overrides``: an
    # empty list means today's behaviour, so it has to mean today's key, or
    # every entry already in ``cache/`` would be orphaned by a field none of
    # them ever set.
    if not data.get("exclude_chains"):
        data.pop("exclude_chains", None)

    if not data.get("include_ligands"):
        drop("ligand_style", "ligand_atom_mm", "ligand_bond_mm",
             "ligand_vdw_scale")
    else:
        # Each ligand style reads a different subset, so the rest can merge.
        style = data.get("ligand_style")
        if style == LigandStyle.SURFACE.value:
            # Sized entirely by the surface controls, which are keyed already.
            drop("ligand_atom_mm", "ligand_bond_mm", "ligand_vdw_scale")
        elif style == LigandStyle.SPACEFILL.value:
            drop("ligand_bond_mm")
        elif style == LigandStyle.STICKS.value:
            drop("ligand_atom_mm", "ligand_vdw_scale")
        else:
            drop("ligand_vdw_scale")

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
             "cartoon_arrow_residues", "cartoon_samples_per_residue",
             "cartoon_hbonds")

    # Hydrogen-bond struts, off, are dropped even when the cartoon *is* built —
    # the same rule as an empty ``joint_overrides`` two blocks down, and for the
    # same reason.  Off means the ribbon comes out with the identical vertices
    # and faces it had before the field existed, so off has to mean the
    # identical *key* as well.  Leaving it in would change the canonical dict
    # for every ordinary cartoon build and orphan every cartoon entry in
    # ``cache/`` — including the pre-generated ones shipped in the repo — to
    # record a setting that changed nothing.  That is what a ``CACHE_VERSION``
    # bump would have cost, and this is why the feature did not need one.
    if data.get("cartoon_hbonds") in (None, HBondMode.NONE.value):
        data.pop("cartoon_hbonds", None)

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
    # Per-pair joint overrides have to survive *every* branch below, including
    # the two that replace the whole dict.  A ``join`` changes the interference
    # carve, and that pass runs whether or not the connect switch is on — so
    # dropping the field where connect is off would let two different override
    # sets collide on one cache entry and serve the wrong geometry.
    #
    # An *empty* override is dropped instead, in every branch.  Empty means
    # today's behaviour, so it must also mean today's key: leaving the field in
    # would change the canonical dict for every ordinary build and orphan the
    # whole existing cache, including the entries shipped in the repo.  That is
    # why ``CACHE_VERSION`` does not need a bump for this feature.
    _overrides = conn.get("joint_overrides") or ""
    if not _overrides:
        conn.pop("joint_overrides", None)

    def _keep_overrides(reduced: dict) -> dict:
        if _overrides:
            reduced["joint_overrides"] = _overrides
        return reduced

    if not (conn.get("connect") or conn.get("basepair_connect")):
        data["connections"] = _keep_overrides(
            {"connect": False, "basepair_connect": False})
    elif not conn.get("connect"):
        # Base-pair connection only: nothing about the chain-join joinery is
        # read — except the overrides, which reach the fit pass, not the joinery.
        data["connections"] = _keep_overrides({
            "connect": False,
            "basepair_connect": True,
            "basepair_max_dist_ang": conn.get("basepair_max_dist_ang"),
        })
    elif not conn.get("use_magnets"):
        for k in ("magnet_thickness_mm", "magnet_shape",
                  "magnet_fit_clearance_mm",
                  "magnet_depth_clearance_mm", "magnet_chamfer_mm"):
            conn.pop(k, None)
        # "inflate" grows the surfaces together and never reads a diameter or
        # builds a collar, and "overlap" builds nothing at all; "bridge" reads
        # both.
        if conn.get("no_magnet_method") in ("inflate", "overlap"):
            for k in ("connector_diameter_mm", "socket", "socket_wall_mm",
                      "path_clearance_mm",
                      # Inflate never runs the seat search, so the two counts
                      # are genuinely unread here — but *only* here.  They used
                      # to be dropped for the whole no-magnets branch, which was
                      # wrong: the bridge reads exactly these two fields to
                      # decide how many rods to drop, so two bridge builds
                      # asking for different numbers of rods hashed the same.
                      # Now that zero is a veto, that collision would also serve
                      # a bridged model to someone who asked for no joints.
                      "magnet_count", "dna_magnet_count"):
                conn.pop(k, None)
    else:
        conn.pop("no_magnet_method", None)   # unused once magnets are on

    # The display stand is generated after the build, and is off for every
    # ordinary one, so the whole block drops out when it is off.  That is not
    # merely a merge of equivalent keys: it leaves the dict for a normal build
    # *byte-identical to the one produced before the stand existed*, so every
    # entry already sitting in ``cache/`` — including the pre-generated ones
    # shipped in the repo — keeps hitting, instead of being orphaned by a field
    # that build never read.  Bumping CACHE_VERSION would throw them all away to
    # no purpose.
    if not (data.get("stand") or {}).get("enabled"):
        drop("stand")

    return data


@dataclass
class CachedObject:
    """A built object rebuilt from an entry's metadata, without its atoms.

    A cache hit serves finished files and never touches the geometry core, so
    the ``Chain`` objects that produced them are long gone.  The display stand
    does not need them: it works on the meshes, and asks an object only for its
    name, its chain id and its molecule type.  This carries exactly that, and
    satisfies the same small interface the exporters use — the same duck-typing
    that lets a ligand, and a stand part, ride through as a chain.

    Deliberately *not* a ``Chain`` with an empty atom array: something holding
    itself out as a chain with no atoms would be a trap for every caller that
    reasonably expects to be able to read them.
    """

    chain_id: str
    mtype: MoleculeType
    name: Optional[str] = None
    res_name: Optional[str] = None
    res_id: Optional[int] = None
    #: The chain's position in the structure, carried through the cache because
    #: it is what the palette reads. Without it a stand raised from a cache hit
    #: fell back to the built position and recoloured the whole model — visibly,
    #: the moment the stand appeared, and in the exported 3MF.
    index: Optional[int] = None
    _label: str = ""

    n_atoms: int = 0
    n_residues: int = 0

    @property
    def is_ligand(self) -> bool:
        return self.mtype == MoleculeType.LIGAND

    def label(self) -> str:
        return self._label or f"chain_{self.chain_id}_{self.mtype.value}"

    def display_name(self) -> str:
        return self.name if self.name else f"Chain {self.chain_id}"

    def object_name(self) -> str:
        if self.is_ligand:
            code = self.res_name or "LIG"
            return f"ligand_{code}{'' if self.res_id is None else self.res_id}"
        return (f"{self.name} ({self.chain_id})" if self.name
                else f"Chain {self.chain_id}")


def describe_objects(built) -> list:
    """The per-object metadata an entry needs to be reopened as geometry.

    Stored alongside the files so a cache hit can be turned back into a list of
    ``(object, mesh)`` pairs — which is what lets a display stand be added to a
    cached build without re-meshing the structure from scratch.
    """
    out = []
    for chain, _mesh in built:
        mtype = getattr(chain, "mtype", None)
        out.append({
            "label": chain.label(),
            "chain_id": str(getattr(chain, "chain_id", "?")),
            "mtype": mtype.value if mtype is not None else "protein",
            "name": getattr(chain, "name", None),
            "res_name": getattr(chain, "res_name", None),
            "res_id": getattr(chain, "res_id", None),
            "index": getattr(chain, "index", None),
        })
    return out


def objects_from_meta(meta: dict) -> list:
    """Rebuild :class:`CachedObject` instances from stored metadata."""
    out = []
    for item in (meta.get("objects") or []):
        try:
            mtype = MoleculeType(item.get("mtype", "protein"))
        except ValueError:
            mtype = MoleculeType.PROTEIN
        out.append(CachedObject(
            chain_id=str(item.get("chain_id", "?")),
            mtype=mtype,
            name=item.get("name"),
            res_name=item.get("res_name"),
            res_id=item.get("res_id"),
            index=item.get("index"),
            _label=str(item.get("label", "")),
        ))
    return out


def objects_from_labels(labels: List[str]) -> list:
    """Recover objects from STL filenames alone, for entries with no metadata.

    Every entry written before ``objects`` existed still has a per-chain STL zip,
    and ``Chain.label`` encodes what is needed: ``chain_A_nucleic``,
    ``ligand_HEM141_A``.  Parsing it back means a cache built last week can still
    be stood up without re-meshing, rather than every pre-generated structure —
    exactly the popular ones — paying for a full rebuild until the day it is
    regenerated.

    ``labels`` must be in the zip's own order, which is the order the objects
    were built in and therefore the order the colour palette was assigned in.
    Sorting here would recolour the model.

    Names are not recoverable this way, so the legend falls back to chain ids.
    A ligand's code and number cannot be split apart reliably either (CCD codes
    contain digits), so the whole tag is kept as the code.
    """
    out = []
    for label in labels:
        if label.startswith("chain_"):
            rest = label[len("chain_"):]
            chain_id, _, raw = rest.rpartition("_")
            try:
                mtype = MoleculeType(raw)
            except ValueError:
                continue
            out.append(CachedObject(chain_id=chain_id or "?", mtype=mtype,
                                    _label=label))
        elif label.startswith("ligand_"):
            rest = label[len("ligand_"):]
            tag, _, chain_id = rest.rpartition("_")
            out.append(CachedObject(chain_id=chain_id or "?",
                                    mtype=MoleculeType.LIGAND,
                                    res_name=tag or None, _label=label))
    return out


def key_for(source: str, params: PrintParams) -> str:
    """Stable cache key for ``source`` built with ``params``.

    Deterministic across processes and machines: ``sort_keys`` removes dict
    ordering, and the normalisation above removes irrelevant fields, so the same
    request always lands on the same key.

    The source is **canonicalised, not just upper-cased**.  A PDB entry has two
    names now — ``1UBQ`` and ``pdb_00001ubq`` are the same structure — and
    hashing them separately would build the same model twice, store it twice and
    serve neither to the other.  ``canonical_pdb_id`` collapses to the
    4-character form wherever an entry has one, so every key written before
    extended IDs existed is still exactly the key this produces; the fallback is
    the old expression, for the uploads and oddities that are not IDs at all.
    """
    payload = {
        "v": CACHE_VERSION,
        "source": canonical_pdb_id(source) or source.strip().upper(),
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
    """A directory of finished builds, bounded by a total size cap.

    ``read_only`` is the setting to use on a deployment with an ephemeral disk:
    the shipped entries are still served, and misses are simply built each time
    rather than written to a filesystem that will not survive a restart.

    ``max_bytes`` bounds the store.  Without it the cache grows forever, and the
    failure is nastier than it sounds: the exporters write before the cache does,
    so a disk filled by cached builds takes *new builds* down with it, and Caddy
    loses the room it needs to renew its certificate.  When the cap is exceeded,
    least-recently-used entries are deleted until the store fits again.
    """

    def __init__(self, root: str = DEFAULT_CACHE_DIR, read_only: bool = False,
                 max_bytes: Optional[int] = None):
        self.root = os.path.abspath(root)
        self.read_only = read_only
        self.max_bytes = DEFAULT_MAX_BYTES if max_bytes is None else max_bytes

    # -- paths ----------------------------------------------------------
    def entry_dir(self, key: str) -> str:
        return os.path.join(self.root, key)

    def _index_path(self) -> str:
        return os.path.join(self.root, "index.json")

    def _stamp_path(self, key: str) -> str:
        """The file whose mtime records when this entry was last served.

        A separate marker rather than the directory's own mtime, because reading
        a file does not update the directory, and rewriting ``meta.json`` on
        every hit would mean a write on the hot path just to record a read.
        """
        return os.path.join(self.entry_dir(key), ".last-used")

    # -- size and recency ------------------------------------------------
    def entry_size(self, key: str) -> int:
        """Bytes on disk for one entry."""
        total = 0
        d = self.entry_dir(key)
        try:
            for name in os.listdir(d):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    total += os.path.getsize(p)
        except OSError:
            return 0
        return total

    def total_size(self) -> int:
        """Bytes on disk for the whole store."""
        if not os.path.isdir(self.root):
            return 0
        total = 0
        for name in os.listdir(self.root):
            d = os.path.join(self.root, name)
            if os.path.isdir(d):
                total += self.entry_size(name)
        return total

    def touch(self, key: str) -> None:
        """Record that this entry was just served.

        Failures are ignored on purpose: a read-only store, or a full disk, must
        still be able to *serve* a cached build.  The cost of not recording a hit
        is that the entry looks staler than it is and may be evicted early —
        which is a far better outcome than refusing to serve it.
        """
        try:
            path = self._stamp_path(key)
            with open(path, "a"):
                os.utime(path, None)
        except OSError:
            pass

    def free_bytes(self) -> Optional[int]:
        """Free space on the filesystem holding the cache, or ``None`` if unknown."""
        try:
            return shutil.disk_usage(self.root).free
        except OSError:
            return None

    def has_headroom(self) -> bool:
        """True if there is enough free disk to be worth writing another entry."""
        free = self.free_bytes()
        return free is None or free > MIN_FREE_BYTES

    def last_used(self, key: str) -> float:
        """When this entry was last served, as a POSIX timestamp.

        Falls back to the entry directory's own mtime — roughly its creation
        time — so an entry that has never been hit still has a sensible age
        rather than sorting as epoch zero.
        """
        for path in (self._stamp_path(key), self.entry_dir(key)):
            try:
                return os.path.getmtime(path)
            except OSError:
                continue
        return 0.0

    # -- read -----------------------------------------------------------
    def lookup(self, source: str, params: PrintParams) -> Optional[dict]:
        """Return the stored metadata for a complete entry, else ``None``."""
        if not is_cacheable(source):
            return None
        return self.lookup_key(key_for(source, params))

    def lookup_key(self, key: str, touch: bool = True) -> Optional[dict]:
        """Metadata for a complete entry, else ``None``.

        ``touch=False`` inspects an entry without counting it as a use — needed
        by anything that walks the whole store, such as :meth:`index`, which
        would otherwise mark every entry as freshly used and destroy the LRU
        ordering the moment the index was written.
        """
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
        # An entry written by a different CACHE_VERSION is not this build's
        # geometry. In practice the version is already part of the key, so such
        # an entry can never be found by name -- but nothing checked, and that
        # made the invisible half of the problem worse: see enforce_limit.
        # Entries predating the field have no version and are treated as
        # current, which is what they are.
        if meta.get("cache_version", CACHE_VERSION) != CACHE_VERSION:
            return None
        meta["key"] = key
        meta["dir"] = d
        if touch:
            self.touch(key)
        return meta

    def index(self) -> List[dict]:
        """Every complete entry's metadata, newest first where dates exist."""
        if not os.path.isdir(self.root):
            return []
        out = []
        for name in sorted(os.listdir(self.root)):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue
            meta = self.lookup_key(name, touch=False)
            if meta:
                meta["size_bytes"] = self.entry_size(name)
                meta["last_used"] = self.last_used(name)
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

        # Free the space first if we are over the cap, so a store on a nearly
        # full disk succeeds by making room rather than failing. Only if there
        # is still no headroom afterwards do we decline.
        self.enforce_limit()
        if not self.has_headroom():
            return None

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
                "source": canonical_pdb_id(source) or source.strip().upper(),
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
        self.touch(key)
        self.enforce_limit()
        return key

    # -- eviction --------------------------------------------------------
    def evict(self, key: str) -> int:
        """Delete one entry; returns the bytes reclaimed."""
        size = self.entry_size(key)
        shutil.rmtree(self.entry_dir(key), ignore_errors=True)
        return size

    def enforce_limit(self) -> List[str]:
        """Drop least-recently-used entries until the store fits the cap.

        Returns the keys evicted.  Called after every store, so the cap holds
        continuously rather than needing a scheduled sweep.

        Least-recently-*used*, not least-recently-created: the point is to keep
        whatever people actually keep asking for.  A structure built once six
        months ago and never revisited is exactly what should go; one built long
        ago and fetched daily should not.

        Sorting the whole store on each store is affordable — the work is a
        ``stat`` per entry against a build that took tens of seconds, and it
        only ever runs on a miss.
        """
        if self.read_only or not self.max_bytes:
            return []
        total = self.total_size()
        if total <= self.max_bytes:
            return []

        # ``.tmp`` is a store() in progress (see :meth:`store`), not an entry.
        # It has no meta.json yet, so the stale-first ordering below would put it
        # at the very front of the eviction queue and delete it mid-write —
        # os.replace would then fail and the store would silently do nothing.
        keys = [n for n in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, n))
                and not n.endswith(".tmp")]
        # Stale-version entries first, then least-recently-used.
        #
        # A CACHE_VERSION bump makes every existing entry unreachable -- the
        # version is baked into the key -- but they stay on disk and keep
        # counting against the cap, so the next eviction pass deleted *live*
        # entries to make room for dead ones. That turns a version bump from a
        # cheap operation into one that quietly costs you the whole cache twice.
        keys.sort(key=lambda k: (not self._is_stale(k), self.last_used(k)))

        evicted = []
        for key in keys:
            if total <= self.max_bytes:
                break
            # Never evict the entry down to an empty store; if a single entry is
            # bigger than the whole cap, the cap is misconfigured and deleting
            # everything would not help.
            if len(evicted) >= len(keys) - 1:
                break
            total -= self.evict(key)
            evicted.append(key)
        return evicted

    def _is_stale(self, key: str) -> bool:
        """True if this entry was written by a different ``CACHE_VERSION``.

        Its key was derived with that version baked in, so nothing will ever
        look it up again. Unreadable metadata counts as stale for the same
        reason: an entry nobody can interpret is an entry nobody can serve.

        Only consulted when the store is over its cap, so the extra read per
        entry costs nothing in the ordinary case.
        """
        try:
            with open(os.path.join(self.entry_dir(key), "meta.json"),
                      "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            return True
        return meta.get("cache_version", CACHE_VERSION) != CACHE_VERSION

    def write_index(self) -> str:
        """Write ``index.json`` — a human-readable listing of what is cached."""
        entries = [{k: v for k, v in m.items() if k != "dir"} for m in self.index()]
        path = self._index_path()
        os.makedirs(self.root, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"cache_version": CACHE_VERSION,
                       "count": len(entries),
                       "total_bytes": self.total_size(),
                       "max_bytes": self.max_bytes,
                       "entries": entries}, fh, indent=2, sort_keys=True)
        return path
