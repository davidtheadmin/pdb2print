# Changelog

All notable changes to pdb2print are recorded here.

This project follows [Semantic Versioning](https://semver.org/). "Mesh-affecting"
below means the exported geometry changed, so cached builds from an earlier
version are not interchangeable with new ones.

## [Unreleased]

Not mesh-affecting on its own: with nothing switched off and no override set, a
build hashes and comes out exactly as it did at 1.2.0, so `CACHE_VERSION` stays
at 5. Bridge builds are the one exception — see below.

### Added

- **Chains you do not want are not built.** A Chains & joints button appears
  under Create display stand and opens a panel listing every chain the structure
  offered. Remove one and it is not meshed, not exported, and not there for the
  others to be carved to fit — the point is to not do the work, and to get a
  model of the part of the complex you actually want to hold. A removed chain
  stays in the list so it can be put back, and the last one cannot be removed.
- **Per-pair joint overrides**, in the second half of the same panel. Each joint
  the build made is set to Default, None or Join. None leaves that pair carved
  apart with nothing joining it. Join leaves the pair fused: its overlap is kept
  out of the carve, so the two parts stay welded without any new geometry. A
  pair with no override follows the Assembly setting.
- **A Regenerate button in the panel**, because the panel is a window away from
  Generate. Nothing is rebuilt on a click; one Regenerate applies every chain
  and every joint you have changed. The panel and the stand panel share the
  right-hand column, and opening one closes the other.
- **Zero magnets per interface is a real answer.** Setting either count to zero
  now vetoes every interface of that kind instead of silently placing one.
- **A joint reports how many connectors went in.** `count` was left at its
  default of 1 however many magnets were seated; the real number was only ever
  in the free-text note.
- **Every chain has a stable identity.** `Chain.index` is a chain's position in
  the structure, assigned before anything is dropped, and it is what the palette,
  a joint's two ends (`ai`, `bi` in the connections payload) and the exclusion
  list all point at. Chain ids could never do this job — a homodimer repeats one
  and a ligand carries its host's.

### Fixed

- **Leaving a chain out does not recolour the others.** Palette entries followed
  a chain's position in the *build*, so removing one shifted every chain after
  it onto the next colour — a bad surprise for anyone who has already printed
  half a model in matching filament. They follow the position in the structure
  now. The display stand's legend dots follow with them.
- **The stand sheet no longer covers the button under it.** It was pinned to a
  top offset sized for exactly one viewer button; it sits in the button stack's
  flow instead, so it lands below however many there are.
- **The bridge count reaches the cache key.** Both magnet counts were dropped
  from the key whenever magnets were off, which is right for inflate and wrong
  for the bridge: it reads exactly those two fields to decide how many rods to
  drop, so two bridge builds asking for different numbers of rods hashed the
  same. Bridge entries already in `cache/` are unreachable as a result;
  everything else keeps hitting.
- **A vetoed joint does not report a refusal.** "No magnet placed — every
  candidate was refused" used to repeat once per interface about something the
  user could not act on. Once a pair is set to None that is the plan, and the
  report comes out clean.

## [1.2.0] — 2026-08-03

Mesh-affecting: cartoon arrowheads, column tops and the plaque layout all
changed shape, so `CACHE_VERSION` goes 4 → 5 and every earlier entry is
unreachable.

### Fixed

- **Columns no longer bore a tunnel through themselves.** The seat was a plain
  boolean difference against the model, which left whatever the column had above
  the cut still standing — a hole through the shaft with a lid on it, worst
  against tube and cartoon models, and the lid is something the model cannot be
  lowered past. The tool is now swept upward and that is cut too, so a column
  stops where the model starts and never resumes above it. The exact difference
  is still applied, so the seat is still the model's own surface.
- **The tube leaving a sheet arrow is the same thickness as every other tube.**
  The arrowhead override ran across the whole of a strand's last segment rather
  than ending at the point, so the section sat frozen at the tip and then jumped
  to the coil tube in one step. It tapers into it now, like every other
  secondary-structure boundary.
- **Emptying a chain-legend box removes that row.** It used to put the header's
  own name back, which meant there was no way to leave a chain off. The rows
  below move up; the box stays, so the name can be typed back in.
- **Probe radius and Surface padding no longer appear under a cartoon.** They
  showed whenever *either* molecule was set to Surface but lived permanently in
  the protein's drawer. They now move to whichever card is asking for a surface.
- **The white tile stays on the plate.** At a large corner radius the lettering
  is moved inboard and the plate widened to pay for it, rather than the tile
  being cut off at the round.
- **Rounded corners are round.** The corner arcs took a fixed segment count, so
  the larger the radius the coarser it looked; the count now follows the radius.

### Added

- **Columns are nudged off splinters.** After a column is sited, its top is
  checked in plan for pieces too small to print and the column is walked up to
  2.5 mm to a position without them, keeping at least 60% of its contact.

### Changed

- **The stand panel is three panels** — Style (columns and plate), Plaque, and
  Advanced, which is a panel now rather than a drawer inside one. Margin round
  the model moved into Advanced; the tilt and the chain-name boxes moved into
  the Plaque body.
- **The Obelisk column style was withdrawn.** An incoming `taper` is served a
  square column rather than an error.
- **Thinnest printable stroke is no longer a control**, fixed at 0.45 mm.
- **Presets:** Museum is fluted with a flared foot, Classical a plain round
  column, and neither tilts the plaque any more.
- **Printability** puts Assembly before Size, with grid spacing behind Advanced.
- **Magnets is a switch**, and everything it governs sits inside it. There is no
  longer a segmented control whose off position is a button labelled Nothing.
- **The panels say a great deal less.** Around twenty blocks of explanatory text
  came out of the two panels; what was worth keeping is behind the help markers
  that were already there, and the rest was describing controls that describe
  themselves.

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
