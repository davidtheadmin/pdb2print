"""Ball-and-stick representation for a bound ligand.

A ligand is small — a drug is 20-50 heavy atoms against a protein's thousands —
and that changes what a representation has to do.  A surface would render it as
an anonymous pebble; a backbone tube has nothing to trace.  Ball-and-stick is the
only style that survives being printed at this size and still reads as *that
molecule*: the ring count, the linker, the substituent hanging off the side.

The geometry is the same machinery the nucleic "molecule" styles use — spheres at
the atoms, capsules between anything within bonding distance, fused by one
manifold union — reused unchanged from :mod:`.tube_slab`.  Two things are ours:

**Its own size controls.**  ``ligand_atom_mm`` / ``ligand_bond_mm`` rather than
the nucleic ``atom_radius_mm`` / ``bond_radius_mm``, because those are tuned for
DNA and the right bead size for a base rung is the wrong one for a drug.  They
have to be adjustable rather than constants: they are absolute millimetre sizes
while the atom positions scale with ``scale_mm_per_angstrom``, so the pair only
agrees at one scale — turn the scale up and a fixed bead size leaves a cage of
pinheads on long thin sticks.  They are independent of each other because the
ratio between them is what decides whether the model reads as atoms joined by
bonds or as a smooth worm, and on a printer the bonds are what snap.

**One body, guaranteed.**  Distance-based bonding is a good enough chemistry
substitute for organic ligands, whose bonds all sit near 1.4 Å, but it has a hard
edge at the cutoff: a metal coordinated at 2.0-2.2 Å (a haem's iron, a metal
cluster's bridging atoms) falls outside it and would come off the printer as a
loose bead rattling inside the pocket.  :func:`_island_links` finds any such
disconnected group and ties it back with one extra stick, so the exported object
is always a single solid.  It works on the coordinates rather than on the mesh
because it has to run *before* the union, and it deliberately does not widen the
cutoff itself: a wider cutoff would also start drawing bonds that are not there
across a folded-back ligand.
"""

from __future__ import annotations

import numpy as np

from ..config import (
    PrintParams, LigandStyle, VDW_RADII_ANG, DEFAULT_VDW_ANG,
)
from . import _manifold
from .tube_slab import _ball_and_stick, _BOND_CUTOFF_ANG


def _radii(params: PrintParams):
    """``(atom_radius, bond_radius)`` in mm from the two size controls.

    Both controls are *diameters*, because that is what they mean to the person
    moving them; the halving belongs here.  Both are then grown to satisfy min
    wall by the same parametric offset the other analytic representations use: a
    solid of radius ``r`` is ``2r`` thick, so the guarantee is ``r >=
    min_wall/2``, applied to the primitives rather than by re-voxelising the
    finished mesh.
    """
    # Floored well below either slider's own minimum: a zero radius makes a
    # degenerate sphere, and the honest failure for that is a clamp here rather
    # than a mesh that fails the watertight gate for reasons nobody can read.
    atom_r = max(float(params.ligand_atom_mm), 0.2) / 2.0
    bond_r = max(float(params.ligand_bond_mm), 0.2) / 2.0
    if params.min_wall_mm > 0:
        half = params.min_wall_mm / 2.0
        atom_r = max(atom_r, half)
        bond_r = max(bond_r, half)
    return atom_r, bond_r


def _spacefill_solids(coords_ang, scale, params: PrintParams, elements):
    """Van der Waals spheres, one per heavy atom.

    The radii are the real ones, scaled with the structure rather than set in
    millimetres, because that is the whole claim a spacefill model makes: the
    atoms are the size they are relative to their spacing.  ``ligand_vdw_scale``
    exists to walk that back when the fused lump hides too much of the shape.

    ``ligand_atom_mm`` is still honoured as a *floor*, so a hydrogen-sized atom
    at a small scale cannot come out below the nozzle.
    """
    floor = max(float(params.ligand_atom_mm), 0.2) / 2.0
    if params.min_wall_mm > 0:
        floor = max(floor, params.min_wall_mm / 2.0)
    factor = float(params.ligand_vdw_scale) * float(scale)
    solids = []
    for coord, element in zip(coords_ang, elements):
        vdw = VDW_RADII_ANG.get(str(element).strip().upper(), DEFAULT_VDW_ANG)
        solids.append(_manifold.sphere(coord * scale, max(vdw * factor, floor)))
    return solids


def _sticks_solids(coords_ang, scale, bond_r, extra_links):
    """Bonds only, at a uniform radius — the licorice model.

    Capsules rather than plain cylinders, so each bond is rounded at both ends
    and a terminal atom finishes in a dome instead of a cut face. That rounding
    is also the only thing standing in for the atoms, which is why this style
    wants a larger bond size than ball-and-stick does.
    """
    solids = []
    count = len(coords_ang)
    for i in range(count):
        for j in range(i + 1, count):
            if np.linalg.norm(coords_ang[i] - coords_ang[j]) <= _BOND_CUTOFF_ANG:
                solids.append(_manifold.capsule(
                    coords_ang[i] * scale, coords_ang[j] * scale, bond_r))
    for i, j in extra_links:
        solids.append(_manifold.capsule(
            coords_ang[i] * scale, coords_ang[j] * scale, bond_r))
    if not solids:
        # A single atom, or a set with no bond inside the cutoff: there is no
        # stick to draw, so fall back to marking the atoms rather than failing.
        solids = [_manifold.sphere(c * scale, bond_r) for c in coords_ang]
    return solids


def _island_links(coords_ang: np.ndarray, cutoff: float):
    """Extra ``(i, j)`` atom pairs needed to make the bond graph connected.

    Groups the atoms by the same distance cutoff :func:`_ball_and_stick` bonds on,
    then repeatedly joins the two closest groups until one remains — a minimum
    spanning tree over the groups, so the number of added sticks is exactly the
    number of gaps and no more.  Returns ``[]`` for an already-connected ligand,
    which is the overwhelmingly common case.
    """
    n = len(coords_ang)
    if n < 2:
        return []

    # Union-find over the bonds _ball_and_stick will draw.
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    d = np.linalg.norm(coords_ang[:, None, :] - coords_ang[None, :, :], axis=-1)
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] <= cutoff:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    if len(groups) < 2:
        return []

    links = []
    members = list(groups.values())
    while len(members) > 1:
        # Closest pair of atoms between any two remaining groups.
        best, pair, which = np.inf, None, None
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                sub = d[np.ix_(members[a], members[b])]
                k = int(np.argmin(sub))
                dist = float(sub.flat[k])
                if dist < best:
                    ia = members[a][k // sub.shape[1]]
                    ib = members[b][k % sub.shape[1]]
                    best, pair, which = dist, (ia, ib), (a, b)
        if pair is None:
            break
        links.append(pair)
        a, b = which
        members[a] = members[a] + members[b]
        members.pop(b)
    return links


def build(chain, params: PrintParams):
    """Return a watertight trimesh ball-and-stick model of one ligand.

    Raises ``ValueError`` if the residue has no heavy atoms to draw.  The caller
    (``pipeline.build_all``) treats a failure here as "skip this ligand" rather
    than as a failed build, so a single awkward cofactor never costs someone the
    whole complex.
    """
    from ..chains import heavy_atom_mask

    s = float(params.scale_mm_per_angstrom)
    atom_r, bond_r = _radii(params)

    atoms = chain.atoms
    coords = np.asarray(atoms.coord, float)
    keep = heavy_atom_mask(atoms)
    if keep.any():
        coords = coords[keep]
    if len(coords) == 0:
        raise ValueError(f"{chain.label()} has no heavy atoms to build.")

    links = _island_links(coords, _BOND_CUTOFF_ANG)
    style = params.ligand_style

    if style == LigandStyle.SURFACE:
        # The protein's own solvent-excluded surface builder, handed a ligand.
        # It asks for nothing but atoms and radii, so this needs no ligand
        # variant at all — and a drug rendered the same way as the pocket it sits
        # in is the whole reason to want the style.
        from . import surface
        return surface.build(chain, params)

    if style == LigandStyle.SPACEFILL:
        solids = _spacefill_solids(coords, s, params, _elements(atoms, keep))
        # Van der Waals spheres normally overlap into one solid unaided, but a
        # metal coordinated beyond the cutoff can still stand apart — the same
        # loose-bead problem, so the same fix.
        for i, j in links:
            solids.append(_manifold.capsule(coords[i] * s, coords[j] * s, bond_r))
    elif style == LigandStyle.STICKS:
        solids = _sticks_solids(coords, s, bond_r, links)
    else:
        solids = _ball_and_stick(coords, s, atom_r, bond_r)
        for i, j in links:
            solids.append(_manifold.capsule(coords[i] * s, coords[j] * s, bond_r))

    fused = _manifold.union(solids)
    return _manifold.to_trimesh(fused)


def _elements(atoms, keep) -> list:
    """Element symbols for the kept atoms, falling back to the atom name.

    Same convention as ``chains.heavy_atom_mask``: the parser fills ``element``
    in most of the time, and where it does not the PDB atom-name rule — leading
    letter after any digit — is what is left.
    """
    try:
        if "element" in atoms.get_annotation_categories():
            values = [str(e) for e in atoms.element]
            return ([values[i] for i in np.flatnonzero(keep)]
                    if keep.any() else values)
    except Exception:
        pass
    names = [str(n).strip().lstrip("0123456789")[:1] for n in atoms.atom_name]
    return [names[i] for i in np.flatnonzero(keep)] if keep.any() else names
