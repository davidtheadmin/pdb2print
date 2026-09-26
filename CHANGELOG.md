# Changelog

All notable changes to pdb2print are recorded here.

This project follows [Semantic Versioning](https://semver.org/). "Mesh-affecting"
below means the exported geometry changed, so cached builds from an earlier
version are not interchangeable with new ones.

## [Unreleased]

Not mesh-affecting, and `CACHE_VERSION` stays at 6: every new setting is dropped
from the key when it is off, so the entries already in `cache/` — including the
2.2 GB shipped in the repo — stay reachable.

**Share links minted before this release no longer open.** `SHARE_FORMAT` goes
1 → 2. A per-joint count needs three bits where there was one, and the magnet
thickness slider now has twenty steps where it had twelve, so a format-1 code
reads both against fields that have moved under it. Either would decode to
something structurally valid and wrong, which is the one failure a share link
must not have, so an old code is refused with a message instead of read.

### Added

- **Magnets per joint, one joint at a time.** Each row in Chains & joints keeps
  Default, None and Join, and gains a small arrow that opens a count. Set it and
  that interface uses that number instead of the one under Magnets. Protein
  interfaces offer 1–5, DNA 1–2, a ligand one. The arrow shows the number when
  one is set, so a row still reads without opening it.
- **A magnet on a ligand.** A switch in the Magnets well, shown when ligands are
  on, **whichever style they are drawn in**. The pocket cut to fit the ligand is
  unchanged and is still what holds it; this adds a magnet on top of that where
  one fits. Whether one fits is a question about this ligand at this scale, and
  the seat search answers it from the real geometry, per interface: where there
  is not enough material no seat passes, the row says so in Chains & joints, and
  the friction fit is left exactly as it was. Ball & stick and Sticks are
  usually too thin — but only usually, and the earlier style gate refused the
  thick ones too, at any scale, with no way to ask.
- **Magnets up to 10 mm thick**, up from 6. A thick magnet needs a thick model:
  it wants roughly its own thickness plus the collar in solid plastic behind
  each face, and past that the seat search runs out of material and reports it.
- **A surface ligand has its own probe radius and surface padding**, under
  Advanced settings in the Ligands card. They used to be the protein's, which
  meant a ligand could only be tuned by detuning its host — and on a cartoon
  protein they were nowhere on the page at all, while the tooltip pointed at a
  card that did not have them. The probe is an absolute size in ångström and a
  ligand is a fraction of the size of what it is bound to: the radius that
  rounds off a protein's crevices closes a drug's rings, and the padding that
  saves a hairline gap on a surface a hundred ångström across is a visible bulge
  on one twelve across. Both start at the protein's own defaults, so a build
  that leaves them alone is the build it always was.

### Fixed

- **A joint set by hand keeps its row in Chains & joints.** Setting a pair to
  Join and regenerating made the row disappear, and the panel drops overrides
  for pairs the build did not list — so the setting went with it and the next
  Generate quietly put the magnet back. Each pass reported the pairs *it*
  handled, and a joined pair reached none of them when the fit pass was off
  (interference set to None, or an overlap-only assembly) or when it was a
  ligand, which Join refuses by design. Reconciled once at the end of the pass
  instead of at each site, so a new mode cannot lose one. Leaving a chain out of
  the build still drops its rows, which is the one case where forgetting is
  right.
- **Two temp-directory leaks that filled the server's disk and took the site
  down on 2026-09-02.** `OUTPUT_ROOT` was a fresh `mkdtemp` per process, and the
  TTL sweep only ever looks inside the root the *current* process made — so
  every restart orphaned a whole tree of build outputs that nothing could
  reclaim. Separately, `export.write_stl_zip` never removed the scratch
  directory holding the loose STLs, leaking one per export. In the container
  the system temp directory is part of the writable layer, i.e. the host disk,
  where 45.6 GB of orphans had collected: `docker system df` was the only place
  it was visible. `OUTPUT_ROOT` now has a fixed name, so the sweep also collects
  what a previous process left; `write_stl_zip` cleans up in a `finally`; and a
  startup sweep clears the backlog, including the per-structure directories
  `io.fetch_pdb_id` leaves behind.
- **The cache says so when it stops storing.** Below `MIN_FREE_BYTES` of free
  disk it silently declines to write. Every build then becomes a cold build and
  the only symptom is that the site feels slow — which is what hid the disk
  filling up for weeks. The transition is logged in both directions now.
- **A surface ligand's probe radius reached the cache key.** With neither
  polymer set to Surface the probe radius and surface padding were dropped from
  the key — but a surface ligand read both. Two builds differing only in probe
  radius shared one entry, and the second was served the first one's geometry.
  Both pairs are now keyed exactly where they are read.
- **A chain exclusion or joint override in a share link survives the build.**
  The panel wipes both lists whenever the structure changes and checks at the
  top of every build; a page opening a link is the largest change there is, so
  it cleared the two lists the link had just restored, before anything read
  them. No shared veto had ever survived a link.
- Magnet thickness is clamped to the slider's range on the way in. It was the
  one slider that reached a bare `float()` on a public endpoint.
- The "the magnet is large for this model" warning no longer measures a ligand.
  A small ligand is nearly always the narrowest part in a build, so the line
  named a part that could not be given a magnet under any setting.
- `ligand_bond` defaulted to 1.2 in `server.py` against 1.4 in the config and
  the slider. Invisible from the web UI, which always submits a value.
- The empty Advanced settings drawer under Ligands is hidden rather than left
  open on nothing.

### Documentation

- **The guide is rebuilt around screenshots.** The **?** button opens a guide
  with annotated pictures of the page and of each step, sections organised by
  what you are trying to do (magnets, Chains & joints, ligands, slicing, sharing,
  the stand), a troubleshooting list and a short glossary. Light and dark
  pictures follow the theme. `scripts/guide_screens/` retakes them and redraws
  the numbered boxes after a layout change.
- **The welcome card is a quick start.** Four steps over a marked-up picture of
  the page, a button that builds 1ZAA straight away, and a link into the guide.
  "New since July" moved to the bottom. `WELCOME_VERSION` is `2026-09`, so it
  shows once more for everyone who dismissed it.
- **Tooltips are shorter.** The long ones are down to one or two sentences, with
  the detail moved into the guide; an **i** that has more to say links to its
  section. The download, share, stand and Chains & joints buttons use the same
  popover, so they explain themselves on touch screens too. The protein Tubes
  slider is labelled *Tube radius*, which is what it sets.
- README brought up to date: no Fetch button or preset chips any more, ligand
  magnets and sizes, the Magnets switch, Overlap, Chains & joints, share links,
  hydrogen bonds, 12-character IDs, the 40 GB cache default, and a bio.tools badge.
- `server.py` serves `.webp` as `image/webp` (the slim image has no MIME table).

### Display stand

Only the stand changed; stands are never cached.

#### Changed

- **Stand presets removed.** The stand opens on one default instead: round
  columns, no flared foot, 7 mm columns floating 5 mm, 5 mm margin, 9 mm text,
  chain legend off.
- **Plaque order.** The structure name is the headline; "PDB ID: 1UBQ" sits
  under it, smaller. The PDB ID switch controls only that line.
- **Raised lettering is lifted clear of what it sits on** (the white tile stays sunk into the plate),
  onto a 0.4 mm plinth of its own outline in the colour below, so the colour
  change happens on a clean layer instead of inside the surface.
- **Plaque heights are whole 0.2 mm layers**: plate thickness snaps to 0.2 mm,
  tile 0.6 mm, letters 0.6 mm.
- The support button reads "Support the project".

#### Fixed

- A long structure name no longer squeezes the chain legend into an ellipsis:
  the legend gets its full width and the name wraps in what is left.
- Extended PDB IDs print as issued (`pdb_1000axyz`) and get their chain names on
  a stand raised from the cache.

## [1.3.0] — 2026-08-05

Not mesh-affecting: with nothing switched off and no override set, a build
meshes exactly as it did at 1.2.0. `CACHE_VERSION` goes 5 → 6 anyway, because
the *payload* changed — an entry stores the finished result verbatim, and an
older one carries no chain list and no joint indices for the new panel to read.

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
- **One piece, Overlap.** A third way to fuse a model, beside Inflate and
  Bridges: leave the chains where the structure puts them and carve nothing.
  They are built overlapping wherever they touch and the fit pass is the only
  thing that pulls them apart, so skipping it is a one-piece model with nothing
  invented for it — nothing grown, nothing cut, nothing bridged. Parts that only
  come close cannot be joined this way and are named in the report instead of
  being left quietly loose.
- **Share links.** A square button beside the two downloads copies a link that
  carries every setting: `pdb2print.org/#1bna.EcMIRdQi-JA6E4AGceAU`. Open one and
  the sliders go back where they were and the model builds itself. About twenty
  characters, because the code encodes only what you changed away from the
  defaults, quantised against each control's own range.

  The code is the settings, not a key into the cache. A key is a one-way hash of
  an already-pruned parameter set — it could fetch files but never put a slider
  back — and it would die every time `CACHE_VERSION` moves, which is every
  release that changes geometry. This way the cache hit falls out of the ordinary
  Generate path, and a link that misses rebuilds and warms the entry for whoever
  clicks next. A link that has been cut off or altered is refused rather than
  decoded into a different model.

  The field table is generated from the markup by `scripts/build_share_table.py`
  and a test fails if it drifts — a hand-written copy of every slider's range is
  the `presets.py` bug wearing a different name.
- **Select all** in the chains list, to put every chain back in one click.
- **A Regenerate button in the panel**, because the panel is a window away from
  Generate. Nothing is rebuilt on a click; one Regenerate applies every chain
  and every joint you have changed. The panel and the stand panel share the
  right-hand column, and opening one closes the other.
- **Zero magnets per interface is a real answer.** Setting either count to zero
  now vetoes every interface of that kind instead of silently placing one.
  Protein–protein offers 1–5 and DNA–protein 0–2: a backbone is a narrow thing
  to seat a magnet on, and nobody was asking for four of them.
- **A joint reports how many connectors went in.** `count` was left at its
  default of 1 however many magnets were seated; the real number was only ever
  in the free-text note.
- **Every chain has a stable identity.** `Chain.index` is a chain's position in
  the structure, assigned before anything is dropped, and it is what the palette,
  a joint's two ends (`ai`, `bi` in the connections payload) and the exclusion
  list all point at. Chain ids could never do this job — a homodimer repeats one
  and a ligand carries its host's.
- **Hydrogen bonds, printed as struts, on cartoon models.** A control in the
  cartoon Advanced drawer: None, Helices, Sheets, Both, All. It lays a strut
  across each backbone hydrogen bond, which is the only way to stiffen a cartoon
  short of printing it bigger — ribbon thickness is locked to width by
  `_RIBBON_ASPECT`, so the inflate pass has nothing it can grow. Sheets stop
  hinging and a helix stops behaving like a spring. Loops have no backbone
  hydrogen bonds of their own and do not change.

  Bonds are found with the Kabsch–Sander criterion DSSP uses, with the amide
  hydrogen placed geometrically rather than read, so it works on structures that
  ship no protons — which is nearly all of them. One end in the named structure
  is enough to qualify, so the bonds that anchor a sheet to the rest of the fold
  are included and not only the ones inside it. Bonds between residues fewer
  than three apart are skipped: the ribbon is already continuous there, so a
  strut is a lump rather than a brace.

  A strut is swept with the same `_section` and `_loft` the ribbon itself is, so
  it is a short piece of the same kind of object rather than a primitive bolted
  on. Its ends carry the ribbon's own profile — a flat-sided oval lying in the
  plane of a sheet, a circle on a round coil tube — eased into a plain rod over
  the middle. It leaves by the edge facing its partner rather than out of the
  middle of a plank, anchored short of the edge roll so no end cap can surface.

  Off is the default and returns the identical mesh, so `canonical_params` drops
  the field and every pre-generated cartoon entry stays reachable.
  `CACHE_VERSION` did not move for this.

- **Extended PDB IDs.** `pdb_00001ubq` is accepted everywhere a 4-character ID
  was, and both keep working — permanently. On 21 July 2027 the wwPDB stops
  issuing 4-character IDs and everything deposited after that has a 12-character
  one and nothing else, so a tool that only reads four characters stops being
  able to open new structures on that date.

  The two spellings are one entry, not two, and exactly one leaves
  `io.canonical_pdb_id`: the **4-character form wherever an entry has one**.
  That is a cache decision. `cache.key_for` hashes the source, so canonicalising
  the other way would rename every key in existence — the pre-generated entries
  shipped in the repo included — over a spelling change that produces a
  byte-identical mesh. Collapsing means today's input still lands on today's
  key, the two spellings share one cache entry instead of building the same
  model twice, and the plaque keeps engraving `1UBQ` rather than
  `PDB_00001UBQ`.

  The rule lives in one place. The front end no longer validates IDs at all —
  it decides how to *show* one and nothing else — because a second copy of the
  rule in JavaScript is the `presets.py` bug wearing a different name, and here
  the drift would be invisible until somebody's share link decoded to another
  structure. Share links need no format change: the ID rides as a literal before
  the `.`, not in the bitstream, so every link already out there still works.

  Also: an entry with an extended ID has no legacy `.pdb` file and never will,
  so the fetch no longer asks for one — that was a guaranteed miss against a
  30 s timeout on the thread holding the build slot.

- **The layout stacks on a phone.** Below 820px the left-to-right arrangement
  becomes top-to-bottom — settings, then the viewer, then whichever panel is
  open. Nothing is hidden: every control that exists on a desktop exists on a
  phone, because the point is that a visitor on a phone can browse, preview,
  share a link and download later rather than being handed a cut-down site.

  The grid was the easy half. This page is an app shell — `html, body` at
  `height: 100%` with `body { overflow: hidden }`, and the settings column
  scrolling inside a viewport-height frame — which is right for a two-column
  desktop tool and impossible where three stacked sections have to share one
  screen height. Below the breakpoint the shell is dismantled: the page grows
  to its content and the document scrolls, and the settings column stops being
  its own scroll region, because a scroller inside a scroller swallows the
  gesture and the page underneath will not move.

  All of it is inside one media query, so the desktop stylesheet is unchanged.
  The viewer and the panels are given heights back, since each collapses to
  nothing once it is no longer stretched by a full-height column. Generate and
  the panel buttons scroll their result into view, because stacked it happens
  below the fold and otherwise looks like nothing happened. Touch targets go to
  roughly 44px. The viewer's two overlay button groups stop sharing the top
  edge — the stand and joints buttons drop below the downloads instead of
  drawing on top of them.

  New controls now append to the **end** of the share-code field table
  (`LATE_SEG_FIELDS` in `scripts/build_share_table.py`). A code stores a 6-bit
  field index, so adding one to `SEG_FIELDS` would have shifted every checkbox
  and slider after it and silently re-pointed every link already shared.

### Fixed

- **Leaving a chain out does not recolour the others.** Palette entries followed
  a chain's position in the *build*, so removing one shifted every chain after
  it onto the next colour — a bad surprise for anyone who has already printed
  half a model in matching filament. They follow the position in the structure
  now. The display stand's legend dots follow with them.
- **The stand sheet no longer covers the button under it.** It was pinned to a
  top offset sized for exactly one viewer button; it sits in the button stack's
  flow instead, so it lands below however many there are.
- **Bridges weld rather than meet on a plane.** Both halves of a peg were built
  against the shared mid-plane, which is right for a magnet — that joint comes
  apart in the hand — and wrong for a bridge, which is a one-piece joint. Two
  flat discs meeting exactly on a plane have no contact area for the slicer to
  weld, and any sliver of numerical overlap between them was found by the
  closing sweep and carved out with a fit clearance on top. Each half now runs a
  short way past the plane; the overlap is entirely inside the peg's own
  cylinder, so nothing about the shape changes.
- **Base-pair rungs are welded again.** Each rung's two halves run past the
  midline on purpose, so the pair shares a real volume — two round ends that
  merely touch have nothing for the slicer to weld. That overlap is between two
  objects, so the closing sweep after connecting found it, carved it out and
  added a fit clearance on top: the deliberate weld came out of every build as a
  gap of air in the middle of every rung.
- **The download buttons grey out while a build runs.** They kept the previous
  build's URLs the whole way through, which is worst exactly when it is most
  tempting: the early preview swaps the new model into the viewer well before
  its files are written, so the model on screen was the new one and the button
  under it handed you the old one, with nothing saying so.
- **The display stand describes the model on screen, not the form.** The stand
  routes posted whatever the settings said at the moment you clicked, which was
  the same thing until the chains and joints panel arrived — a control meant to
  be left set and unbuilt until the next Generate, one click from the stand
  button. A pending veto built a cache key for a model that had never been
  built: the sketch fell back to its generic drawing after a visible wait, and a
  real stand rebuilt from scratch *with the pending change applied*, standing up
  a model the viewer was not showing. The stand lookup now uses what the build
  token recorded, which is what that token has always been for.
- **A pending change no longer relabels the plaque.** The plaque's legend boxes
  are keyed on built position — the one index left that still moves — and always
  send a line each, so a set left over from before a chain was removed named
  chain B with chain A's label and printed it. They are dropped when the built
  set changes and named afresh; a name you typed still survives an ordinary
  rebuild.
- **Chains and joints do not travel between structures.** Both lists are
  indices, and carried to a different structure they do not fail, they *match* —
  so a chain switched off in the last structure came out of the next one.
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
