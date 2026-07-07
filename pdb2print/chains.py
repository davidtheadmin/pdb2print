"""Split a structure into chains and classify each as protein or nucleic acid."""

from __future__ import annotations

from dataclasses import dataclass

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
    """One chain's atoms plus its classification."""

    chain_id: str
    atoms: struc.AtomArray
    mtype: MoleculeType

    @property
    def n_atoms(self) -> int:
        return self.atoms.array_length()

    @property
    def n_residues(self) -> int:
        return struc.get_residue_count(self.atoms)

    def label(self) -> str:
        return f"chain_{self.chain_id}_{self.mtype.value}"


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
    """Yield a :class:`Chain` per polymer chain, skipping tiny/empty ones."""
    chains = []
    for cid in struc.get_chains(atoms):
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
