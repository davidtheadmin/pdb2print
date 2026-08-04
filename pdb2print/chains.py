"""Split a structure into printable objects: polymer chains and bound ligands.

Each polymer chain is classified as protein or nucleic acid.  Each bound ligand
becomes an object of its own (one per residue) so it can take its own filament,
which is what makes a drug-bound structure worth printing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import biotite.structure as struc

from .config import (
    MoleculeType, LIGAND_BLOCKLIST, LIGAND_MIN_HEAVY_ATOMS,
)


# Canonical + common modified nucleotide residue names, used as a robust
# fallback across biotite versions (some lack ``filter_nucleotides``).
_NUCLEOTIDE_RESNAMES = {
    "DA", "DC", "DG", "DT", "DU", "DI",
    "A", "C", "G", "U", "I", "N",
    "5MC", "7MG", "PSU", "1MA", "2MG", "H2U", "M2G", "OMC", "OMG",
}

#: Water, under every name it is written with.  ``io._clean`` already drops
#: solvent through ``struc.filter_solvent``, so in the normal path nothing here
#: is ever reached; it is kept because the ligand filter must be safe to call on
#: an arbitrary AtomArray (a test, a script, a future caller that skips
#: ``_clean``), and "water became an object" is not a failure anyone should have
#: to debug twice.
_WATER_RESNAMES = {"HOH", "DOD", "WAT", "H2O", "SOL", "TIP", "TIP3", "TP3"}


@dataclass
class Chain:
    """One printable object's atoms plus its classification.

    Despite the name this is also how a **ligand** travels through the pipeline:
    a single-residue "chain" with ``mtype == MoleculeType.LIGAND``.  Reusing the
    one carrier is the point — every stage downstream (mesh dispatch, min-wall,
    interference, connections, export) already takes a ``Chain`` and branches on
    ``mtype``, so a ligand needs no parallel code path anywhere.

    ``name`` is the human-readable subunit/molecule name parsed from the
    structure header (PDB ``COMPND`` or mmCIF ``_entity``); it is ``None`` when
    the header carries no name for this chain.  For a ligand it is filled in from
    the residue instead (``"Ligand STI 1"``).  It is display metadata only —
    geometry never depends on it.

    ``res_name``/``res_id`` are set for a ligand only, and identify the residue
    it was built from: the CCD chemical-component code and the residue number.

    ``index`` is this chain's position in the *full* list :func:`split_chains`
    found in the file, assigned before anything is dropped.  It is the only
    stable name a chain has: chain ids are not unique (a homodimer repeats one,
    and a ligand carries its host's), and a position in the built list moves the
    moment a chain is excluded or fails to mesh.  Everything that has to point
    at a particular chain from outside — the palette entry, a joint's two ends,
    the exclusion list itself — points at this.
    """

    chain_id: str
    atoms: struc.AtomArray
    mtype: MoleculeType
    name: Optional[str] = None
    res_name: Optional[str] = None
    res_id: Optional[int] = None
    index: Optional[int] = None

    @property
    def n_atoms(self) -> int:
        return self.atoms.array_length()

    @property
    def n_residues(self) -> int:
        return struc.get_residue_count(self.atoms)

    @property
    def is_ligand(self) -> bool:
        return self.mtype == MoleculeType.LIGAND

    def ligand_tag(self) -> str:
        """``"STI1"``-style short id for a ligand: CCD code plus residue number."""
        return f"{self.res_name or 'LIG'}{'' if self.res_id is None else self.res_id}"

    def label(self) -> str:
        """Internal id, and the STL filename inside the zip — must be unique.

        A ligand carries its host chain id even though the tag alone is what gets
        shown, because two copies of the same ligand in two chains can share both
        code *and* residue number, and two files called ``ligand_HEM141.stl``
        would silently overwrite each other in the zip.
        """
        if self.is_ligand:
            return f"ligand_{self.ligand_tag()}_{self.chain_id}"
        return f"chain_{self.chain_id}_{self.mtype.value}"

    def display_name(self) -> str:
        """The subunit name if known, else ``"Chain <id>"`` as a fallback."""
        return self.name if self.name else f"Chain {self.chain_id}"

    def object_name(self) -> str:
        """Human-readable object name for exports (subunit name + chain id).

        e.g. ``"Ubiquitin (A)"``; falls back to ``"Chain A"`` with no name.
        This is what PrusaSlicer shows for each 3MF object.

        A ligand gets ``"ligand_STI1"`` instead: in the slicer's object list what
        you want is the thing you can look up — the CCD code — not a prettified
        description, and the ``ligand_`` prefix keeps every bound molecule
        together when the list is sorted.
        """
        if self.is_ligand:
            return f"ligand_{self.ligand_tag()}"
        return f"{self.name} ({self.chain_id})" if self.name else f"Chain {self.chain_id}"


def _nucleotide_mask(atoms: struc.AtomArray) -> np.ndarray:
    """Boolean mask of nucleic-acid atoms, resilient to biotite version."""
    try:
        mask = struc.filter_nucleotides(atoms)
        if mask.any():
            return mask
    except AttributeError:
        pass
    return np.isin(atoms.res_name, list(_NUCLEOTIDE_RESNAMES))


def classify(atoms: struc.AtomArray):
    """Return (MoleculeType, filtered_atoms) for a chain's *polymer*, or ``None``.

    Filtering to the classified atom set keeps the polymer mesh clean: whatever
    is bound to the chain would otherwise be fused into its surface and read as a
    bulge rather than as a separate molecule.  Those residues are not thrown away
    any more — :func:`ligand_chains` picks them back up as objects of their own —
    so this function is now "which polymer is this chain", not "what survives".
    """
    aa_mask = struc.filter_amino_acids(atoms)
    nuc_mask = _nucleotide_mask(atoms)
    n_aa = int(aa_mask.sum())
    n_nuc = int(nuc_mask.sum())

    if n_nuc == 0 and n_aa == 0:
        return None
    if n_nuc > n_aa:
        return MoleculeType.NUCLEIC, atoms[nuc_mask]
    return MoleculeType.PROTEIN, atoms[aa_mask]


# --------------------------------------------------------------------------
# Bound ligands
# --------------------------------------------------------------------------
def _residue_bounds(atoms: struc.AtomArray):
    """Yield ``(start, end)`` index pairs, one per residue, in file order."""
    starts = struc.get_residue_starts(atoms)
    total = atoms.array_length()
    for i, start in enumerate(starts):
        end = int(starts[i + 1]) if i + 1 < len(starts) else total
        yield int(start), end


def heavy_atom_mask(atoms: struc.AtomArray) -> np.ndarray:
    """Boolean mask of the non-hydrogen atoms in ``atoms``.

    Hydrogens have to be excluded from the ligand size floor, or the floor means
    something different for an NMR/neutron entry (every hydrogen present) than for
    an X-ray one (none) — a 4-atom acetate with its hydrogens on would sail past a
    6-atom threshold.  The ligand mesh drops them for a second reason: a
    ball-and-stick that draws hydrogens at the same bead size as carbons is an
    unreadable thicket, and at print scale they fuse the whole thing solid.

    ``element`` is used when the parser filled it in, with the PDB atom-name
    convention (the element is the leading letter, after any digit) as fallback.
    """
    try:
        if "element" in atoms.get_annotation_categories():
            return np.array([str(e).strip().upper() not in ("H", "D")
                             for e in atoms.element], dtype=bool)
    except Exception:
        pass
    return np.array(
        [str(n).strip().lstrip("0123456789")[:1].upper() not in ("H", "D")
         for n in atoms.atom_name], dtype=bool)


def _heavy_atom_count(res: struc.AtomArray) -> int:
    """Number of non-hydrogen atoms in one residue."""
    return int(np.count_nonzero(heavy_atom_mask(res)))


def is_ligand_residue(res: struc.AtomArray) -> bool:
    """True if one residue should be exported as a bound ligand.

    The residue must be none of amino acid / nucleotide / water, and then pass
    both filters described in :data:`config.LIGAND_MIN_HEAVY_ATOMS` and
    :data:`config.LIGAND_BLOCKLIST`.

    Modified polymer residues (selenomethionine, phosphoserine, a methylated
    base) are handled by the amino-acid/nucleotide test rather than by name:
    biotite resolves those against the chemical-component dictionary, so a
    phosphoserine reads as an amino acid and stays part of its chain instead of
    being cut out of it and offered as a "ligand" sitting in a hole of its own.
    """
    return _is_ligand_residue(res, None, None)


def _is_ligand_residue(res: struc.AtomArray, aa_mask, nuc_mask) -> bool:
    """:func:`is_ligand_residue` with the polymer masks optionally passed in.

    ``aa_mask``/``nuc_mask`` are this residue's slice of masks computed once for
    the whole chain.  They are worth threading through because
    ``filter_amino_acids`` resolves names against the chemical-component
    dictionary, and paying for that per residue on a 3000-residue chain turns a
    free check into a visible one.
    """
    if res.array_length() == 0:
        return False
    name = str(res.res_name[0]).strip().upper()
    if name in _WATER_RESNAMES or name in LIGAND_BLOCKLIST:
        return False
    if aa_mask is None:
        aa_mask = struc.filter_amino_acids(res)
    if nuc_mask is None:
        nuc_mask = _nucleotide_mask(res)
    if bool(np.any(aa_mask)) or bool(np.any(nuc_mask)):
        return False
    return _heavy_atom_count(res) >= LIGAND_MIN_HEAVY_ATOMS


def ligand_chains(atoms: struc.AtomArray, chain_id: str) -> List["Chain"]:
    """Every bound ligand in ``atoms`` (one chain's worth), as its own object.

    One object per *residue*, not per residue type: two copies of the same
    inhibitor in two pockets are two separate things to print, and a haemoglobin
    has four haems that are nowhere near each other.
    """
    out: List[Chain] = []
    if atoms.array_length() == 0:
        return out
    aa_all = struc.filter_amino_acids(atoms)
    nuc_all = _nucleotide_mask(atoms)
    for start, end in _residue_bounds(atoms):
        res = atoms[start:end]
        if not _is_ligand_residue(res, aa_all[start:end], nuc_all[start:end]):
            continue
        res_name = str(res.res_name[0]).strip().upper()
        try:
            res_id = int(res.res_id[0])
        except Exception:
            res_id = None
        out.append(Chain(
            chain_id=str(chain_id), atoms=res, mtype=MoleculeType.LIGAND,
            # A ligand has no header name of its own that is worth parsing (the
            # mmCIF ``_chem_comp.name`` is usually an IUPAC mouthful), so the
            # legend gets the code and the number, which is what a structural
            # biologist would say out loud anyway.
            name=f"Ligand {res_name}" + ("" if res_id is None else f" {res_id}"),
            res_name=res_name, res_id=res_id,
        ))
    return out


def split_chains(atoms: struc.AtomArray, min_residues: int = 2,
                 include_ligands: bool = True):
    """Yield a :class:`Chain` per polymer chain plus one per bound ligand.

    Chain ids are de-duplicated first: ``biotite.get_chains`` reports a chain id
    once per *contiguous* block, so a chain whose atoms are interrupted in the
    file — e.g. a protein followed later by its bound metal/ligand HETATM under
    the same chain id (1TUP's zinc-bound p53 copies A/B/C) — would otherwise be
    iterated, and therefore meshed and exported, twice.  Selecting all atoms of
    each unique id exactly once builds each chain a single time.

    Ligands are collected from the same per-id atom selection, and deliberately
    *outside* the polymer's own guards: a chain whose polymer is missing or too
    short to build (in many mmCIF files the ligand carries a chain id of its own,
    with no polymer under it at all) must still give up its ligands, and a ligand
    is a single residue so it can never satisfy ``min_residues``.

    Polymers come first in the returned list and ligands after, which is also the
    colour order — chains keep the palette entries they had before ligands
    existed, so adding a drug to the print does not recolour the protein.
    """
    chains = []
    ligands: List[Chain] = []
    seen = set()
    for cid in struc.get_chains(atoms):
        if cid in seen:
            continue
        seen.add(cid)
        sub = atoms[atoms.chain_id == cid]
        if include_ligands:
            ligands.extend(ligand_chains(sub, str(cid)))
        result = classify(sub)
        if result is None:
            continue
        mtype, filtered = result
        if filtered.array_length() == 0:
            continue
        if struc.get_residue_count(filtered) < min_residues:
            continue
        chains.append(Chain(chain_id=str(cid), atoms=filtered, mtype=mtype))
    # Numbered once, here, over the whole list: this is the position everything
    # downstream points at, so it must be settled before anyone can drop a
    # chain. Polymers first and ligands after, which is also the palette order.
    out = chains + ligands
    for i, chain in enumerate(out):
        chain.index = i
    return out


def parse_excluded(raw: str) -> set:
    """Parse a comma-separated list of chain indices into a set.

    Best-effort, like every other free-text field that reaches the builder: a
    malformed entry is skipped rather than raised on. An index naming a chain
    that is not in this structure is kept and simply never matches, so a list
    carried over from another file costs nothing.
    """
    out = set()
    for part in (raw or "").replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            i = int(part)
        except ValueError:
            continue
        if i >= 0:
            out.add(i)
    return out
