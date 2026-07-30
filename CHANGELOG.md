# Changelog

All notable changes to pdb2print are recorded here.

This project follows [Semantic Versioning](https://semver.org/). "Mesh-affecting"
below means the exported geometry changed, so cached builds from an earlier
version are not interchangeable with new ones.

## [1.1.0] — 2026-07-30

### Added

- **Display stands** — a generated stand with an editable plaque, real outline
  fonts, and a live sketch that updates as you drag. The stand arrives in the
  3MF as a single object with parts, so one click moves the whole thing in the
  slicer while each part still takes its own filament.
- **Ligands** — bound ligands are built as their own objects with their own
  styles. The fit pass carves the host into an exact negative, so the ligand
  lifts out and drops back in; friction is the joint, no connector needed.
- **Editable legend labels**, anchored to the right margin.
- **Magnet panel** ordered to match the sequence you actually decide things in,
  with the default option first in both assembly controls.
- **Branding and discoverability** — favicon set, social sharing card, and
  structured metadata for search engines. The wordmark is outlined rather than
  depending on a font being installed.
- **Print photographs** in the README, with the originals kept out of the
  history.
- **Build cache**, bounded, with the temp-directory leak that fed it fixed.

### Changed

- **Settings UI rebuilt** around three stages instead of one long form.
- **model-viewer is vendored and served from this origin** rather than fetched
  from unpkg on every visit.
- **README** brought up to what the app actually does.
- **`manifold3d` pinned to 3.5.2.** The nucleic path now hands the kernel one
  flat union per chain, and the result is not identical across kernel versions —
  an unpinned floor would let a rebuild produce different meshes from the ones
  the cache already holds. See the note in `requirements.txt` before moving it.

### Performance

Measured on a 2-core container; ratios transfer, absolute times do not.

- **Skip the closing interference audit when the resolve sweep found nothing.** A
  clean multi-chain build ran the same O(n²) sweep three times to reach the same
  answer. Output identical.
- **Rank overlap lobes by volume before measuring them.** A 50-base-pair duplex
  interferes at every rung, so most lobes were fully measured — a mesh conversion
  and an SVD each — and then discarded. Output identical.
- **Fuse `tube_slab` primitives in one flat union** instead of nesting a boolean
  per spline segment, and emit one sphere per spline sample instead of two.
  **Mesh-affecting** — see below.
- **Separable squared distance and in-place EDT in the SES rasteriser.** Surface
  pass 25–35% faster; output bit-identical.
- **Optional parallel per-chain meshing** behind `PDB2PRINT_WORKERS`
  (unset/`0`/`1`/`off` keeps the serial path, an integer is taken as given,
  `auto` sizes from free RAM). Off by default. Meshing is 9–35% of a build
  depending on shape, so the realistic ceiling is around 25% — not the 2–3× a
  per-chain speedup might suggest.

Combined effect on whole builds: 1BNA 2.97s → 1.22s, 1ZAA 3.26s → 1.03s, a
4-object complex 20.07s → 8.83s, 1TUP 11.97s → 9.85s.

### Mesh-affecting change

Flattening the `tube_slab` union alters the solid slightly: on 1BNA one strand
went from 632.59 mm³ to 628.59 mm³ (−0.63%) with 11% fewer triangles. That is
roughly 4 µm of wall on a 1.2 mm tube radius — below what an FDM nozzle
resolves, and visually and functionally the same model.

The operational consequence: **cache entries written by 1.0.0 are not
interchangeable with 1.1.0 builds.** Clear the build cache when upgrading. Build
*warnings* on complexes may also differ, because the interference pass is
threshold-sensitive.

### Fixed

- Chain names are recovered for models reopened from an old cache entry.
- The display stand finds and reuses a model that came from the disk cache
  instead of rebuilding it.
- The stand's Advanced drawers collapse when a new model is generated.
- A stand solve that outruns a second says so, as a warning rather than a
  caption.

## [1.0.0] — 2026-07-26

First tagged release, archived on Zenodo
([10.5281/zenodo.21599702](https://doi.org/10.5281/zenodo.21599702)): the
PDB-to-multi-object-3MF pipeline, the surface / tube-slab / cartoon
representations, the interference and connector passes, and the deployed site.

The repository history was rewritten after this tag was published, so the v1.0.0
commit shares no ancestry with `main` and the two cannot be diffed. Everything
listed under 1.1.0 above is what separates that archived tree from this one.

[1.1.0]: https://github.com/davidtheadmin/pdb2print/releases/tag/v1.1.0
[1.0.0]: https://github.com/davidtheadmin/pdb2print/releases/tag/v1.0.0

<!-- No compare link between 1.0.0 and 1.1.0 on purpose: the repository history
     was rewritten after v1.0.0 was tagged (to keep handover notes and original
     photographs out of the public tree), so the v1.0.0 commit shares no
     ancestry with main and GitHub cannot diff the two. The tag and its Zenodo
     archive are still valid as a snapshot of what 1.0.0 was. -->
