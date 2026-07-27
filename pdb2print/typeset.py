"""Typefaces for the plaque: one stroke font, and real outline fonts.

Two kinds of lettering, one interface.

The **stroke** font (:mod:`pdb2print.strokefont`) is 64 hand-written glyphs as
centre lines, swept into ridges by the caller. It has no thickness of its own,
which is exactly what you want at two millimetres: the caller picks a stroke
width the nozzle can actually draw, and the letter is that wide everywhere. It
has no lowercase either — it sets it as small capitals — and no counters to
speak of, because at 2 mm the counter of an 'e' closes up whatever you do.

The **outline** fonts are the real thing: subset TrueType faces read with
``fontTools``, flattened to polygons and triangulated into filled solids. They
have proper lowercase, proper counters and real letterfitting, and at a 5 mm
headline they look like typography instead of like a plotter. What they do not
have is a guaranteed minimum stroke: a 2 mm cap in a regular weight has stems
about a fifth of a millimetre wide, which a 0.4 mm nozzle cannot draw and a
slicer will silently drop. Both bundled faces are therefore **bold**, and each
one measures its own stem so :func:`grow_for` can say how much the outline has
to be fattened to survive the nozzle it is going to.

Everything here is in millimetres and cap heights. Cap height rather than point
size because a plaque is specified by how tall the letters are, and because the
two font kinds have nothing else in common to measure by.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import strokefont


FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

#: Bundled outline faces. Subset to the couple of hundred characters a plaque
#: can print, which takes DejaVu's 700 KB down to 20 — small enough to live in
#: the repository, which is the only way the deployment is sure to have it.
_FILES = {
    "sans": "pdb2print-sans.ttf",
    "serif": "pdb2print-serif.ttf",
}

#: Segments per Bézier when flattening an outline. Eight is invisible at a 5 mm
#: cap and still under a hundred points for the busiest glyph.
_CURVE_STEPS = 8


# --------------------------------------------------------------------------
# The stroke font, behind the common interface
# --------------------------------------------------------------------------
class StrokeFace:
    """The hand-written centre-line font. Filled by the caller, not by itself."""

    key = "line"
    outline = False
    name = "Line"
    #: Which face was *asked* for, when this one is standing in for it, and why.
    #: A fallback that says nothing is indistinguishable from a setting that does
    #: nothing — which is exactly how this first presented: the typeface menu
    #: worked, the lettering never changed, and there was no way to tell that
    #: ``fontTools`` was simply not installed.
    requested = None
    fallback_reason = ""

    def text_width(self, text: str, cap_mm: float) -> float:
        return strokefont.text_width(text, cap_mm)

    def strokes(self, text: str, cap_mm: float, origin=(0.0, 0.0)):
        return strokefont.layout(text, cap_mm, origin)

    def stem_mm(self, cap_mm: float) -> float:
        # The caller chooses it, so there is nothing to measure and nothing to
        # grow: a stroke font is already exactly as thick as it was asked to be.
        return float("inf")


# --------------------------------------------------------------------------
# Outline fonts
# --------------------------------------------------------------------------
class OutlineFace:
    """A real typeface, as filled contours.

    Held lazily: ``fontTools`` is only imported, and the file only read, if
    somebody actually asks for an outline face. An install without it still
    prints stroke-font plaques rather than failing to start.
    """

    outline = True
    fallback_reason = ""

    def __init__(self, key: str, path: str, name: str):
        from fontTools.ttLib import TTFont

        self.key = key
        self.requested = key
        self.name = name
        self._font = TTFont(path, lazy=True)
        self._glyphs = self._font.getGlyphSet()
        self._cmap = self._font.getBestCmap()
        self._hmtx = self._font["hmtx"]
        self.upem = float(self._font["head"].unitsPerEm)
        self._rings: Dict[str, List[np.ndarray]] = {}
        self.cap_units = self._measure_cap()
        self.stem_units = self._measure_stem()

    # -- metrics ---------------------------------------------------------
    def _measure_cap(self) -> float:
        """Cap height in font units, measured rather than trusted.

        ``OS/2.sCapHeight`` is optional and routinely absent or wrong, and
        subsetting does not add it. The top of an 'H' is the definition anyway.
        """
        rings = self._glyph_rings("H") or self._glyph_rings("I")
        if rings:
            top = max(float(r[:, 1].max()) for r in rings)
            if top > 0:
                return top
        return self.upem * 0.7

    def _measure_stem(self) -> float:
        """Width of the 'H' left stem in font units.

        Scanned off the outline rather than taken from any table, because no
        table records it and because this is the number that decides whether the
        letters survive the nozzle.

        Scanned at 82% of cap height, which on an 'H' is above the crossbar and
        below the serifs. Mid-height — the obvious choice — is exactly where the
        crossbar is, so the scanline crosses the outline twice instead of four
        times and reports the whole letter as one stem.
        """
        default = self.upem * 0.13
        rings = self._glyph_rings("H")
        if not rings:
            return default
        for fraction in (0.82, 0.72, 0.20):
            y = self.cap_units * fraction
            xs: List[float] = []
            for ring in rings:
                n = len(ring)
                for i in range(n):
                    (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % n]
                    if (y0 <= y < y1) or (y1 <= y < y0):
                        xs.append(x0 + (x1 - x0) * (y - y0) / (y1 - y0))
            xs.sort()
            if len(xs) >= 4:
                stem = xs[1] - xs[0]
                # Four crossings and a plausible width: that is a stem. Anything
                # wider than a third of the cap is the whole letter, and means
                # the scanline found a bar rather than a pair of uprights.
                if 0 < stem < self.cap_units * 0.34:
                    return stem
        return default

    def stem_mm(self, cap_mm: float) -> float:
        return self.stem_units * (float(cap_mm) / self.cap_units)

    # -- glyph access ----------------------------------------------------
    def _glyph_name(self, ch: str) -> Optional[str]:
        name = self._cmap.get(ord(ch))
        if name is None and ch.upper() != ch:
            name = self._cmap.get(ord(ch.upper()))
        return name

    def _glyph_rings(self, ch: str) -> List[np.ndarray]:
        """Closed contours of one character, in font units, cached."""
        if ch in self._rings:
            return self._rings[ch]
        name = self._glyph_name(ch)
        rings: List[np.ndarray] = []
        if name is not None and name in self._glyphs:
            pen = _FlatteningPen(self._glyphs)
            try:
                self._glyphs[name].draw(pen)
                rings = [np.asarray(r, float) for r in pen.rings if len(r) >= 3]
            except Exception:
                rings = []
        self._rings[ch] = rings
        return rings

    def _advance(self, ch: str) -> float:
        name = self._glyph_name(ch)
        if name is None:
            return self.upem * 0.4
        try:
            return float(self._hmtx[name][0])
        except Exception:
            return self.upem * 0.4

    # -- public ----------------------------------------------------------
    def text_width(self, text: str, cap_mm: float) -> float:
        scale = float(cap_mm) / self.cap_units
        return sum(self._advance(ch) for ch in _fold(text)) * scale

    def contours(self, text: str, cap_mm: float, origin=(0.0, 0.0)):
        """``[(outer, [hole, ...]), ...]`` in millimetres, baseline-left.

        Rings are grouped by containment rather than by winding direction.
        Winding is the conventional signal and it is not reliable across fonts
        or across a subsetting pass; whether one closed curve is inside another
        is a fact about the shape.
        """
        scale = float(cap_mm) / self.cap_units
        ox, oy = float(origin[0]), float(origin[1])
        out = []
        pen_x = 0.0
        for ch in _fold(text):
            rings = self._glyph_rings(ch)
            if rings:
                placed = [np.column_stack([ox + (r[:, 0] + pen_x) * scale,
                                           oy + r[:, 1] * scale]) for r in rings]
                out.extend(_group_rings(placed))
            pen_x += self._advance(ch)
        return out


class _FlatteningPen:
    """A pen that records closed polygons, flattening every curve.

    Subclassed off ``BasePen`` so composite glyphs (an 'Å' is an 'A' and a ring)
    decompose themselves and quadratics arrive already converted to cubics.
    """

    def __new__(cls, glyph_set):
        from fontTools.pens.basePen import BasePen

        class _Pen(BasePen):
            def __init__(self, gs):
                BasePen.__init__(self, gs)
                self.rings: List[List[Tuple[float, float]]] = []
                self._cur: List[Tuple[float, float]] = []

            def _moveTo(self, pt):
                self._flush()
                self._cur = [tuple(pt)]

            def _lineTo(self, pt):
                self._cur.append(tuple(pt))

            def _curveToOne(self, p1, p2, p3):
                if not self._cur:
                    self._cur = [tuple(p3)]
                    return
                p0 = self._cur[-1]
                for i in range(1, _CURVE_STEPS + 1):
                    t = i / _CURVE_STEPS
                    s = 1.0 - t
                    self._cur.append((
                        s * s * s * p0[0] + 3 * s * s * t * p1[0]
                        + 3 * s * t * t * p2[0] + t * t * t * p3[0],
                        s * s * s * p0[1] + 3 * s * s * t * p1[1]
                        + 3 * s * t * t * p2[1] + t * t * t * p3[1]))

            def _closePath(self):
                self._flush()

            def _endPath(self):
                self._flush()

            def _flush(self):
                if len(self._cur) >= 3:
                    # A closing point identical to the first is redundant here:
                    # every ring is treated as closed.
                    if (abs(self._cur[0][0] - self._cur[-1][0]) < 1e-9
                            and abs(self._cur[0][1] - self._cur[-1][1]) < 1e-9):
                        self._cur.pop()
                    if len(self._cur) >= 3:
                        self.rings.append(self._cur)
                self._cur = []

        return _Pen(glyph_set)


# --------------------------------------------------------------------------
# Ring topology
# --------------------------------------------------------------------------
def _signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_in_ring(point, ring: np.ndarray) -> bool:
    """Crossing-number test. Rings here are small, so nothing cleverer pays."""
    px, py = float(point[0]), float(point[1])
    inside = False
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        if (y0 > py) != (y1 > py):
            xc = x0 + (x1 - x0) * (py - y0) / (y1 - y0)
            if px < xc:
                inside = not inside
    return inside


def _group_rings(rings: List[np.ndarray]):
    """Sort closed rings into ``(outer, [holes])`` groups by containment.

    A ring inside an odd number of others is a hole, and belongs to the
    smallest ring that contains it — which is what makes the dot of an 'i'
    inside the bowl of an 'o' come out solid rather than inverted.
    """
    if not rings:
        return []
    areas = [abs(_signed_area(r)) for r in rings]
    depth = []
    for i, ring in enumerate(rings):
        count = 0
        for j, other in enumerate(rings):
            if i != j and areas[j] > areas[i] and _point_in_ring(ring[0], other):
                count += 1
        depth.append(count)

    groups = []
    index_of = {}
    for i, ring in enumerate(rings):
        if depth[i] % 2 == 0:
            index_of[i] = len(groups)
            groups.append((_wound(ring, ccw=True), []))
    for i, ring in enumerate(rings):
        if depth[i] % 2 == 0:
            continue
        best, best_area = None, None
        for j in index_of:
            if areas[j] > areas[i] and _point_in_ring(ring[0], rings[j]):
                if best_area is None or areas[j] < best_area:
                    best, best_area = j, areas[j]
        if best is not None:
            groups[index_of[best]][1].append(_wound(ring, ccw=False))
    return groups


def _wound(ring: np.ndarray, ccw: bool) -> np.ndarray:
    """``ring`` re-wound to the requested direction."""
    if (_signed_area(ring) > 0) == ccw:
        return ring
    return ring[::-1].copy()


# --------------------------------------------------------------------------
# Faces, and the layout helpers that work on any of them
# --------------------------------------------------------------------------
_CACHE: Dict[str, object] = {}


def face(name):
    """The face for ``name``, falling back to the stroke font — and saying so.

    A missing ``fontTools``, a missing ``mapbox_earcut``, a missing file or a
    corrupt one all end up here as the stroke font and a plaque that still
    prints, because lettering is not worth refusing to build a stand over. But
    the substitution is recorded on the face it hands back, and
    :func:`unavailable` turns that into a sentence the user actually sees.
    Falling back in silence is worse than failing: the control appears to do
    nothing, and there is nothing anywhere to say why.
    """
    key = str(getattr(name, "value", name) or "line")
    if key in _CACHE:
        return _CACHE[key]

    made, reason = None, ""
    if key in _FILES:
        path = os.path.join(FONT_DIR, _FILES[key])
        try:
            # Checked here rather than where it is used. An outline face without
            # a triangulator is not a degraded face, it is a face that raises
            # halfway through building a plaque — so the time to find out is
            # before anything has been promised.
            import mapbox_earcut  # noqa: F401
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            if not os.path.isfile(path):
                reason = f"{_FILES[key]} is not in {FONT_DIR}"
            else:
                try:
                    made = OutlineFace(key, path, key.title())
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"

    if made is None:
        made = StrokeFace()
        if key in _FILES:
            made.requested = key
            made.fallback_reason = reason or "unavailable"
    _CACHE[key] = made
    return made


def unavailable(face_obj) -> str:
    """Why the face in use is not the one asked for, as a sentence, or ``""``."""
    reason = getattr(face_obj, "fallback_reason", "")
    if not reason:
        return ""
    want = getattr(face_obj, "requested", None) or "that"
    return (f"The {want} typeface could not be loaded, so the plaque is set in "
            f"the built-in stroke font instead ({reason}). Run "
            f"`pip install -r requirements.txt` — it needs fonttools and "
            f"mapbox_earcut.")


def grow_for(face_obj, cap_mm: float, min_stroke_mm: float) -> float:
    """How far to fatten an outline so its thinnest stem survives the nozzle.

    Returned as a *radius*: the caller sweeps the contour with a disc of this
    size, which grows the letter by it on every side. Zero for a stroke font,
    which has no stem of its own to be too thin.

    A 0.4 mm nozzle drawing a 0.3 mm stem does not draw a thin stem; it drops
    it, or lays one under-extruded bead with nothing either side to hold it
    down. Growing is not free — it closes counters and softens the letterform —
    so it is only ever applied by the shortfall, and at a 5 mm headline in a
    bold face the shortfall is nil and nothing happens at all.
    """
    if not getattr(face_obj, "outline", False):
        return 0.0
    stem = face_obj.stem_mm(cap_mm)
    return max(0.0, (float(min_stroke_mm) - stem) * 0.5)


def text_width(face_obj, text: str, cap_mm: float) -> float:
    return face_obj.text_width(text, cap_mm)


def wrap(face_obj, text: str, cap_mm: float, max_width_mm: float,
         max_lines: Optional[int] = 3, truncate: bool = True) -> List[str]:
    """Greedy word wrap to ``max_width_mm``.

    ``max_lines`` of ``None`` means as many as it takes.

    ``truncate`` decides what happens when the text does not fit in the lines
    allowed: an ellipsis on the last line, or nothing — the caller gets fewer
    words than it gave.  Both are lossy, which is why the *title* uses neither:
    it asks for unlimited lines instead and lets the apron get deeper, because a
    plaque with a deeper apron is a plaque, and a plaque reading "Crystal
    Structure Of Human Deoxyhaemo..." is a mistake somebody has to explain.

    A legend row is different and still truncates: it is one line by design, and
    a chain's name that will not fit on it has nowhere else to go.
    """
    words = str(text).split()
    if not words:
        return []
    lines: List[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if current and face_obj.text_width(trial, cap_mm) > max_width_mm:
            lines.append(current)
            current = word
            if max_lines and len(lines) == max_lines:
                break
        else:
            current = trial
    if (not max_lines or len(lines) < max_lines) and current:
        lines.append(current)

    if truncate and max_lines and len(lines) == max_lines:
        consumed = sum(len(l.split()) for l in lines)
        if consumed < len(words):
            last = lines[-1]
            while last and face_obj.text_width(last + "...", cap_mm) > max_width_mm:
                last = last[:-1]
            lines[-1] = last.rstrip() + "..."
    return lines


def wrap_balanced(face_obj, text: str, cap_mm: float, max_width_mm: float,
                  max_lines: int = 3, truncate: bool = True) -> List[str]:
    """Wrap into the fewest lines that fit, split as evenly as possible.

    Greedy wrapping fills each line to the brim and leaves whatever is left on
    the last one, so a two-line title reads as a full line and a stub — which on
    a plaque looks like the text ran out rather than like it was set.  Choosing
    the split that makes the lines most nearly equal costs nothing here (a title
    is a dozen words and at most three lines, so every split can simply be
    tried) and is the difference between a label and a leftover.
    """
    words = str(text).split()
    if not words:
        return []

    for count in range(1, max_lines + 1):
        best, best_cost = None, None
        for cuts in _splits(len(words), count):
            groups = []
            start = 0
            for cut in list(cuts) + [len(words)]:
                groups.append(words[start:cut])
                start = cut
            if any(not g for g in groups):
                continue
            widths = [face_obj.text_width(" ".join(g), cap_mm) for g in groups]
            if max(widths) > max_width_mm:
                continue
            cost = (max(widths) - min(widths)) + 0.15 * max(widths)
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, [" ".join(g) for g in groups]
        if best is not None:
            return best

    # Nothing fitted in the lines allowed. Either shorten it, or — for anything
    # that must not lose its ending — take as many lines as it needs.
    if truncate:
        return wrap(face_obj, text, cap_mm, max_width_mm, max_lines)
    return wrap(face_obj, text, cap_mm, max_width_mm, None, truncate=False)


def _splits(n_words: int, n_groups: int):
    """Every way to cut ``n_words`` into ``n_groups`` non-empty runs."""
    from itertools import combinations
    if n_groups <= 1:
        yield ()
        return
    if n_words < n_groups:
        return
    for cuts in combinations(range(1, n_words), n_groups - 1):
        yield cuts


def fit_cap_height(face_obj, lines: Sequence[str], max_width_mm: float,
                   preferred_cap_mm: float, min_cap_mm: float = 1.6) -> float:
    """Largest cap height at or below ``preferred_cap_mm`` that fits every line."""
    widest = max([face_obj.text_width(l, preferred_cap_mm) for l in lines] or [0.0])
    if widest <= max_width_mm or widest <= 0.0:
        return preferred_cap_mm
    return max(min_cap_mm, preferred_cap_mm * max_width_mm / widest)


def _fold(text: str) -> str:
    """Fold the handful of characters a subset face may still not carry."""
    return strokefont._expand(text)
