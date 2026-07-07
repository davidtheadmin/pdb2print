"""Structure input: fetch by PDB ID from RCSB, or load an uploaded file.

Only biotite is used here so the parsing layer stays lightweight and portable.
"""

from __future__ import annotations

import os
import re
import tempfile

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.database import rcsb


_PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


def looks_like_pdb_id(text: str) -> bool:
    """True if ``text`` is a 4-character PDB accession code (e.g. ``1UBQ``)."""
    return bool(_PDB_ID_RE.match(text.strip()))


def fetch_pdb_id(pdb_id: str, target_dir: str | None = None) -> str:
    """Download a structure from RCSB and return the local file path.

    Tries mmCIF first (always available, handles large/modern entries) and
    falls back to legacy PDB format.
    """
    pdb_id = pdb_id.strip()
    target_dir = target_dir or tempfile.mkdtemp(prefix="pdb2print_")
    for fmt in ("cif", "pdb"):
        try:
            return rcsb.fetch(pdb_id, fmt, target_dir)
        except Exception:
            continue
    raise ValueError(f"Could not fetch '{pdb_id}' from RCSB in cif or pdb format.")


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
    if os.path.exists(source):
        return load_file(source)
    if looks_like_pdb_id(source):
        return load_file(fetch_pdb_id(source))
    raise ValueError(
        f"'{source}' is neither an existing file nor a 4-character PDB ID."
    )


def _clean(atoms: struc.AtomArray) -> struc.AtomArray:
    """Drop solvent and keep a single alternate-location set."""
    atoms = atoms[~struc.filter_solvent(atoms)]
    # Keep the first altloc where present ('' or 'A').
    if "altloc_id" in atoms.get_annotation_categories():
        alt = atoms.altloc_id
        atoms = atoms[(alt == "") | (alt == "A") | (alt == ".")]
    return atoms
