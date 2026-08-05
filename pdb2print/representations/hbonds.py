"""Backbone hydrogen bonds, for bracing a printed cartoon.

A cartoon ribbon is the one representation with no thickness lever: its
thickness is locked to its width by ``cartoon._RIBBON_ASPECT``, so
``connections._inflate_growth`` returns ``None`` for it and the only way to make
one stronger is to print it bigger.  A strut across each backbone hydrogen bond
is the lever it was missing — and it is not an invented brace, it is the bond
that holds the real fold together.

**Kabsch–Sander, not a distance cut.**  This is the electrostatic criterion DSSP
uses (Kabsch & Sander, *Biopolymers* 1983): the interaction energy of the
partial charges on the donor's N-H and the acceptor's C=O,

    E = 0.42 · 0.20 · 332 · (1/r(ON) + 1/r(CH) − 1/r(OH) − 1/r(CN))   kcal/mol

with a bond declared below :data:`_E_CUT`.  A plain N···O distance cut is the
obvious alternative and it is worse: it accepts pairs whose geometry is
side-on, which is exactly where a strut would run through open space at an angle
that reads as a mistake.

**The hydrogen is placed, not read.**  Crystal structures almost never carry
hydrogens, so the amide H is put where the geometry says it must be: one
ångström from N along the *reverse* of the preceding residue's C=O.  That is
DSSP's own rule, and it means this works on any structure rather than only on
the NMR entries that ship protons.

**Vectorised on purpose.**  The residue-by-residue double loop is the obvious
way to write this and costs 375 ms on a 196-residue chain — the same order as
the geometry it feeds.  The four distance matrices below are 19 ms for an
identical answer.
"""

from __future__ import annotations

import numpy as np

from .tube_slab import _residue_iter, _atom_coord


#: Kabsch–Sander's ``f · q1 · q2`` in kcal·Å/mol.
_Q = 0.42 * 0.20 * 332.0
#: Energy below which a pair counts as bonded (kcal/mol).  DSSP's own cutoff.
_E_CUT = -0.5
#: N···O further apart than this cannot bond; keeps the energy matrix honest
#: where the 1/r terms would otherwise go small-but-negative on distant pairs.
_MAX_ON_ANG = 5.5
#: Residues closer together than this in sequence are skipped.
#:
#: ``|i-j| = 2`` bonds are real, and about 13% of the total, but the ribbon is
#: already continuous between two residues two apart — a strut there lies along
#: the ribbon instead of bracing anything, so it is a lump and not a brace.
_MIN_SEPARATION = 3
#: N-H bond length used to place the amide hydrogen (ångström).
_NH_ANG = 1.0


def _backbone_arrays(chain):
    """Per-residue ``N``, ``C``, ``O`` coordinates and residue names.

    Missing atoms come back as ``NaN`` rows rather than ``None`` so the whole
    thing stays one array and the masks below do the filtering.
    """
    nan = np.full(3, np.nan)

    def coord(res, name):
        c = _atom_coord(res, name)
        return nan if c is None else np.asarray(c, float)

    N, C, O, names = [], [], [], []
    for res_name, res in _residue_iter(chain.atoms):
        names.append(res_name)
        N.append(coord(res, "N"))
        C.append(coord(res, "C"))
        O.append(coord(res, "O"))
    return (np.asarray(N, float), np.asarray(C, float), np.asarray(O, float),
            names)


def _amide_hydrogens(N, C, O):
    """Geometric amide H positions, ``NaN`` where they cannot be placed.

    ``H(i) = N(i) + unit(C(i-1) - O(i-1)) · 1 Å`` — the N-H points opposite the
    previous residue's carbonyl, which is what the peptide plane forces.  The
    first residue of a chain has no preceding carbonyl and so never donates.
    """
    H = np.full_like(N, np.nan)
    if len(N) < 2:
        return H
    d = C[:-1] - O[:-1]
    norms = np.linalg.norm(d, axis=1)
    ok = norms > 1e-6
    unit = np.zeros_like(d)
    unit[ok] = d[ok] / norms[ok, None]
    H[1:] = N[1:] + unit * _NH_ANG
    H[1:][~ok] = np.nan
    return H


def backbone_hbonds(chain, min_separation: int = _MIN_SEPARATION):
    """Backbone hydrogen bonds in one chain as ``(donor_i, acceptor_j)`` pairs.

    Indices are positions in the chain's residue order — the same order
    ``cartoon`` builds its control points in, so a pair indexes straight into
    them.  Intra-chain only: measured across whole structures, inter-chain
    backbone bonds are 1% of the total (4 of 351 on 1TUP, none on 1UBQ), and the
    chains are meshed independently — in separate processes when
    ``PDB2PRINT_WORKERS`` is set — so reaching across one would be a rewrite of
    the pipeline for a rounding error.
    """
    N, C, O, names = _backbone_arrays(chain)
    n = len(N)
    if n < min_separation + 1:
        return []
    H = _amide_hydrogens(N, C, O)

    # Proline's nitrogen is in the ring and carries no hydrogen, so it is never
    # a donor.  Every other residue donates if its backbone survived the file.
    is_pro = np.array([nm == "PRO" for nm in names])
    donor = np.isfinite(N).all(1) & np.isfinite(H).all(1) & ~is_pro
    acceptor = np.isfinite(C).all(1) & np.isfinite(O).all(1)
    if not donor.any() or not acceptor.any():
        return []

    def pair_dist(A, B):
        """``(n, n)`` distances, donor down the rows and acceptor across."""
        return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)

    r_ON = pair_dist(N, O)
    r_CH = pair_dist(H, C)
    r_OH = pair_dist(H, O)
    r_CN = pair_dist(N, C)

    with np.errstate(invalid="ignore", divide="ignore"):
        energy = _Q * (1.0 / r_ON + 1.0 / r_CH - 1.0 / r_OH - 1.0 / r_CN)

    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    # The 0.5 Å floor rejects the degenerate case where two atoms sit on top of
    # each other, which a broken file can produce and which would otherwise
    # dominate the energy with a huge negative term.
    closest = np.minimum.reduce([r_ON, r_CH, r_OH, r_CN])
    ok = (donor[:, None] & acceptor[None, :]
          & (sep >= int(min_separation))
          & (r_ON <= _MAX_ON_ANG)
          & (closest >= 0.5)
          & np.isfinite(energy) & (energy < _E_CUT))

    return [(int(i), int(j)) for i, j in zip(*np.nonzero(ok))]
