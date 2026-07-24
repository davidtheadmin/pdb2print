"""Split a structure into chains and classify each as protein or nucleic acid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import biotite.structure as struc

from .config import MoleculeType


# Canonical + common modified nucleotide residue names, used as a robust
# fallback across biotite versions (some lack ``filter_nucleotides``).
_NUCLEOTIDE_RESNAMES = {
    "DA", "DC", "DG", "DT", "DU", "DI",
    "A", "C", "G", "U", "I", "N",
    "5MC", "7MG", "PSU", "1MA", "2MG", "H2U", "M2G", "OMC", "OMG",
}


@dataclass
class Chain:
    """One chain's atoms plus its classification.

    ``name`` is the human-readable subunit/molecule name parsed from the
    structure header (PDB ``COMPND`` or mmCIF ``_entity``); it is ``None`` when
    the header carries no name for this chain.  It is display metadata only —
    geometry never depends on it.
    """

    chain_id: str
    atoms: struc.AtomArray
    mtype: MoleculeType
    name: Optional[str] = None

    @property
    def n_atoms(self) -> int:
        return self.atoms.array_length()

    @property
    def n_residues(self) -> int:
        return struc.get_residue_count(self.atoms)

    def label(self) -> str:
        return f"chain_{self.chain_id}_{self.mtype.value}"

    def display_name(self) -> str:
        """The subunit name if known, else ``"Chain <id>"`` as a fallback."""
        return self.name if self.name else f"Chain {self.chain_id}"

    def object_name(self) -> str:
        """Human-readable object name for exports (subunit name + chain id).

        e.g. ``"Ubiquitin (A)"``; falls back to ``"Chain A"`` with no name.
        This is what PrusaSlicer shows for each 3MF object.
        """
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
    """Return (MoleculeType, filtered_atoms) for a chain, or ``None`` to skip.

    Filtering to the classified atom set drops bound ligands/ions so they do
    not distort the mesh.
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


def split_chains(atoms: struc.AtomArray, min_residues: int = 2):
    """Yield a :class:`Chain` per polymer chain, skipping tiny/empty ones.

    Chain ids are de-duplicated first: ``biotite.get_chains`` reports a chain id
    once per *contiguous* block, so a chain whose atoms are interrupted in the
    file — e.g. a protein followed later by its bound metal/ligand HETATM under
    the same chain id (1TUP's zinc-bound p53 copies A/B/C) — would otherwise be
    iterated, and therefore meshed and exported, twice.  Selecting all atoms of
    each unique id exactly once builds each chain a single time.
    """
    chains = []
    seen = set()
    for cid in struc.get_chains(atoms):
        if cid in seen:
            continue
        seen.add(cid)
        sub = atoms[atoms.chain_id == cid]
        result = classify(sub)
        if result is None:
            continue
        mtype, filtered = result
        if filtered.array_length() == 0:
            continue
        if struc.get_residue_count(filtered) < min_residues:
            continue
        chains.append(Chain(chain_id=str(cid), atoms=filtered, mtype=mtype))
    return chains
