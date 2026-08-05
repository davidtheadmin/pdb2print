"""Structure input: fetch by PDB ID from RCSB, or load an uploaded file.

Only biotite is used here so the parsing layer stays lightweight and portable.
"""

from __future__ import annotations

import contextlib
import os
import re
import socket
import tempfile

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.database import rcsb


#: The 4-character accession code every PDB entry has carried since 1971: a
#: digit then three alphanumerics — ``1UBQ``, ``6UV8``.
_LEGACY_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")

#: The extended 12-character ID: the prefix ``pdb_`` then eight alphanumerics —
#: ``pdb_00001ubq``.  On **21 July 2027** the wwPDB stops issuing 4-character
#: IDs and everything deposited after that has one of these and nothing else.
#:
#: Matched case-insensitively because people paste what they are handed, but the
#: prefix is *specified* lowercase and is written back out that way — the wwPDB
#: asks specifically that it not be altered and that the zeros not be dropped.
#: https://www.wwpdb.org/documentation/new-format-for-pdb-ids
_EXTENDED_ID_RE = re.compile(r"^pdb_[A-Za-z0-9]{8}$", re.IGNORECASE)

#: Extended IDs beginning ``pdb_0000`` are the 4-character entries wearing the
#: new format — ``1ABC`` is exactly ``pdb_00001abc``.  That block is reserved
#: for them; newly issued IDs come from elsewhere in the space, the wwPDB's own
#: example being ``pdb_1000axyz``.  The membership test below checks the tail
#: looks like a real 4-character ID as well, so a future ID that landed in this
#: block anyway would be left alone rather than silently truncated.
_LEGACY_BLOCK = "0000"


def canonical_pdb_id(text: str):
    """One spelling per structure, or ``None`` if this is not a PDB ID at all.

    Both formats are accepted, and both always will be: the 4-character codes
    stay valid for the entries that have them, and everything deposited after
    July 2027 is 12 characters.  They are not two kinds of thing — ``1UBQ`` and
    ``pdb_00001ubq`` are one entry under two names — so exactly one spelling
    leaves this function and nothing downstream has to know there were two.

    **The 4-character form wins wherever an entry has one.**  That is a cache
    decision rather than a matter of taste: ``cache.key_for`` hashes the source,
    so canonicalising the other way would rename every key in existence — the
    2.2 GB of pre-generated entries shipped in the repo, and everything a
    running deployment has ever built — over a change of spelling that produces
    a byte-identical mesh.  Collapsing instead means today's input still lands
    on today's key and only genuinely new IDs mint new ones.  Same rule as an
    empty ``joint_overrides``.

    The form returned is also the form to **fetch** with.  RCSB serves both, so
    a 4-character ID needs no translation on the way out, and there is no second
    function that could come to disagree with this one about what an ID means.
    """
    text = (text or "").strip()
    if _LEGACY_ID_RE.match(text):
        return text.upper()
    if _EXTENDED_ID_RE.match(text):
        body = text[len("pdb_"):]
        tail = body[len(_LEGACY_BLOCK):]
        if body[:len(_LEGACY_BLOCK)] == _LEGACY_BLOCK and _LEGACY_ID_RE.match(tail):
            return tail.upper()
        return "pdb_" + body.lower()
    return None


def looks_like_pdb_id(text: str) -> bool:
    """True if ``text`` is a PDB accession code, in either format."""
    return canonical_pdb_id(text) is not None


#: Seconds to allow one RCSB request.
#:
#: biotite's ``rcsb.fetch`` calls ``requests`` with no timeout at all, and this
#: runs on the worker thread holding the only build slot -- so one hung upstream
#: connection stalls every build on the site, permanently, with no recovery.
#: There is no timeout argument to pass through, so the socket default is set
#: around the call instead: requests and urllib3 both fall back to it for the
#: connect *and* the read when no explicit timeout is given.
_FETCH_TIMEOUT_S = 30.0


@contextlib.contextmanager
def _socket_timeout(seconds: float):
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def fetch_pdb_id(pdb_id: str, target_dir: str | None = None) -> str:
    """Download a structure from RCSB and return the local file path.

    Tries mmCIF first (always available, handles large/modern entries) and
    falls back to legacy PDB format.
    """
    pdb_id = canonical_pdb_id(pdb_id) or pdb_id.strip()
    target_dir = target_dir or tempfile.mkdtemp(prefix="pdb2print_")
    # mmCIF first because it is the format that will still be there.  An entry
    # issued an extended ID has no legacy PDB file and never will — the wwPDB
    # stops producing them — so asking is a guaranteed miss and a wasted round
    # trip against a 30 s timeout on the one thread holding the build slot.
    formats = ("cif",) if pdb_id.startswith("pdb_") else ("cif", "pdb")
    for fmt in formats:
        try:
            with _socket_timeout(_FETCH_TIMEOUT_S):
                return rcsb.fetch(pdb_id, fmt, target_dir)
        except Exception:
            continue
    raise ValueError(f"Could not fetch '{pdb_id}' from RCSB as "
                     f"{' or '.join(formats)}.")


def load_file(path: str) -> struc.AtomArray:
    """Load a ``.pdb``/``.cif``/``.mmcif``/``.bcif`` file into an AtomArray.

    Returns the first model only (structures for printing are single-model).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pdb", ".ent"):
        f = pdb.PDBFile.read(path)
        atoms = pdb.get_structure(f, model=1)
    elif ext in (".cif", ".mmcif"):
        f = pdbx.CIFFile.read(path)
        atoms = pdbx.get_structure(f, model=1)
    elif ext == ".bcif":
        f = pdbx.BinaryCIFFile.read(path)
        atoms = pdbx.get_structure(f, model=1)
    else:
        # Unknown extension: try mmCIF then PDB by content.
        try:
            f = pdbx.CIFFile.read(path)
            atoms = pdbx.get_structure(f, model=1)
        except Exception:
            f = pdb.PDBFile.read(path)
            atoms = pdb.get_structure(f, model=1)
    return _clean(atoms)


def load_any(source: str) -> struc.AtomArray:
    """Accept either a PDB ID or a path and return a cleaned AtomArray."""
    return load_with_names(source)[0]


#: PDB IDs already fetched by this process, and where they landed.
#:
#: ``fetch_pdb_id`` makes a fresh temp directory per call, so without this every
#: caller that wanted anything from a structure — the build, the plaque's title,
#: a display stand generated an hour later — downloaded the whole entry again.
#: Three round trips to RCSB to read one line of header is not a cache miss, it
#: is a bug that happens to work.
_FETCHED: dict = {}


def resolve_source(source: str) -> str:
    """Return a local file path for ``source`` (fetching a PDB ID if needed)."""
    if os.path.exists(source):
        return source
    # Keyed on the canonical id, so the two spellings of one entry share a
    # download rather than fetching it twice.
    pdb_id = canonical_pdb_id(source)
    if pdb_id is not None:
        path = _FETCHED.get(pdb_id)
        if path and os.path.exists(path):
            return path
        path = fetch_pdb_id(pdb_id)
        _FETCHED[pdb_id] = path
        return path
    raise ValueError(
        f"'{source}' is neither an existing file nor a PDB ID. An ID is four "
        f"characters (1UBQ) or the extended form (pdb_00001ubq)."
    )


def load_with_names(source: str):
    """Load ``source`` and parse per-chain subunit names from its header.

    Returns ``(atoms, names)`` where ``names`` maps chain id -> subunit name
    (possibly empty).  Parsing names from the same on-disk file avoids a second
    RCSB fetch and keeps header metadata that biotite drops from the AtomArray.
    """
    from .names import chain_names

    path = resolve_source(source)
    atoms = load_file(path)
    try:
        names = chain_names(path)
    except Exception:
        names = {}
    return atoms, names


def _clean(atoms: struc.AtomArray) -> struc.AtomArray:
    """Drop solvent and keep a single alternate-location set."""
    atoms = atoms[~struc.filter_solvent(atoms)]
    # Keep the first altloc where present ('' or 'A').
    if "altloc_id" in atoms.get_annotation_categories():
        alt = atoms.altloc_id
        atoms = atoms[(alt == "") | (alt == "A") | (alt == ".")]
    return atoms
