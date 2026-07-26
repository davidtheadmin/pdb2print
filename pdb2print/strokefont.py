"""A single-stroke vector font, in-tree and dependency-free.

The plaque needs letters as *geometry*, and every off-the-shelf route to that
adds a dependency the deployment does not have: matplotlib for ``TextPath``,
freetype-py for glyph outlines, fonttools plus a triangulator for raw contours.
Each of those is a large addition to a pinned ``requirements.txt`` and a Docker
image, for a feature that draws a few dozen characters.

So the font lives here.  Each glyph is a handful of **polylines** — the centre
lines of the strokes, not filled outlines — in the manner of the Hershey
engraving fonts.  ``stand`` sweeps each polyline into a rounded raised ridge,
which is what a router or an engraver would cut and reads cleanly at the 3–6 mm
cap heights a plaque uses.  A filled typeface would not: at 0.4 mm nozzle width
the counters of a 4 mm 'e' close up anyway, so the outline would be doing work
the printer throws away.

Coordinates are in a **7-unit cap-height grid** with the baseline at ``y = 0``
and ``x`` increasing right.  :func:`layout` converts to millimetres.

Lowercase is rendered as small capitals at :data:`SMALL_CAP` of the cap height
rather than as its own glyph set.  That halves the data, and on a plaque it
reads as a deliberate typographic choice instead of a compromise.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# A glyph is (advance_width, [polyline, ...]); a polyline is [(x, y), ...].
Glyph = Tuple[float, List[List[Tuple[float, float]]]]

#: Cap height of the design grid.  All glyph coordinates are in these units.
CAP = 7.0

#: Lowercase is drawn as capitals at this fraction of the cap height.
SMALL_CAP = 0.74

#: Gap between glyphs, in grid units.
LETTER_SPACING = 1.4

#: Width of a space character, in grid units.
SPACE_WIDTH = 3.0


def _ring(cx: float, cy: float, r: float, n: int = 16) -> List[Tuple[float, float]]:
    """A closed circular polyline — used for 'O'-like bowls and the ring of 'Å'."""
    import math
    pts = [(cx + r * math.cos(2.0 * math.pi * i / n),
            cy + r * math.sin(2.0 * math.pi * i / n)) for i in range(n)]
    pts.append(pts[0])
    return pts


# --------------------------------------------------------------------------
# Glyph table
# --------------------------------------------------------------------------
# Letterforms are deliberately plain: single-storey, uniform stroke, no serifs
# or optical corrections.  They are cut as physical ridges a few tenths of a
# millimetre wide, where any refinement would be finer than the extrusion.
_GLYPHS: Dict[str, Glyph] = {
    " ": (SPACE_WIDTH, []),

    "A": (6.0, [[(0, 0), (3, 7), (6, 0)], [(1.0, 2.33), (5.0, 2.33)]]),
    "B": (6.0, [[(0, 0), (0, 7), (4, 7), (5.6, 5.8), (5.6, 4.7), (4.2, 3.7), (0, 3.7)],
                [(4.2, 3.7), (5.8, 2.5), (5.8, 1.2), (4, 0), (0, 0)]]),
    "C": (6.0, [[(6, 5.6), (4.6, 7), (2, 7), (0, 5), (0, 2), (2, 0), (4.6, 0), (6, 1.4)]]),
    "D": (6.0, [[(0, 0), (0, 7), (3.2, 7), (5.6, 5.4), (5.6, 1.6), (3.2, 0), (0, 0)]]),
    "E": (5.8, [[(5.8, 7), (0, 7), (0, 0), (5.8, 0)], [(0, 3.6), (4.2, 3.6)]]),
    "F": (5.6, [[(5.6, 7), (0, 7), (0, 0)], [(0, 3.6), (4.0, 3.6)]]),
    "G": (6.2, [[(6.2, 5.6), (4.8, 7), (2, 7), (0, 5), (0, 2), (2, 0), (4.8, 0),
                 (6.2, 1.4), (6.2, 3.2), (4.0, 3.2)]]),
    "H": (6.0, [[(0, 0), (0, 7)], [(6, 0), (6, 7)], [(0, 3.6), (6, 3.6)]]),
    "I": (3.6, [[(0, 7), (3.6, 7)], [(1.8, 7), (1.8, 0)], [(0, 0), (3.6, 0)]]),
    "J": (5.4, [[(5.4, 7), (5.4, 2), (3.6, 0), (1.6, 0), (0, 1.8)]]),
    "K": (5.8, [[(0, 0), (0, 7)], [(5.4, 7), (0, 3.2)], [(1.9, 4.3), (5.8, 0)]]),
    "L": (5.2, [[(0, 7), (0, 0), (5.2, 0)]]),
    "M": (6.6, [[(0, 0), (0, 7), (3.3, 2.6), (6.6, 7), (6.6, 0)]]),
    "N": (6.2, [[(0, 0), (0, 7), (6.2, 0), (6.2, 7)]]),
    "O": (6.4, [[(2.2, 7), (4.2, 7), (6.4, 5), (6.4, 2), (4.2, 0), (2.2, 0),
                 (0, 2), (0, 5), (2.2, 7)]]),
    "P": (6.0, [[(0, 0), (0, 7), (4.2, 7), (6, 5.7), (6, 4.1), (4.2, 2.9), (0, 2.9)]]),
    "Q": (6.4, [[(2.2, 7), (4.2, 7), (6.4, 5), (6.4, 2), (4.2, 0), (2.2, 0),
                 (0, 2), (0, 5), (2.2, 7)], [(3.9, 1.7), (6.4, -0.9)]]),
    "R": (6.0, [[(0, 0), (0, 7), (4.2, 7), (6, 5.7), (6, 4.1), (4.2, 2.9), (0, 2.9)],
                [(3.2, 2.9), (6.0, 0)]]),
    "S": (5.8, [[(5.8, 5.8), (4.2, 7), (1.8, 7), (0, 5.8), (0, 4.6), (1.2, 3.8),
                 (4.6, 3.3), (5.8, 2.4), (5.8, 1.2), (4.0, 0), (1.6, 0), (0, 1.2)]]),
    "T": (5.8, [[(0, 7), (5.8, 7)], [(2.9, 7), (2.9, 0)]]),
    "U": (6.0, [[(0, 7), (0, 2), (2, 0), (4, 0), (6, 2), (6, 7)]]),
    "V": (6.0, [[(0, 7), (3, 0), (6, 7)]]),
    "W": (8.0, [[(0, 7), (2, 0), (4, 4.6), (6, 0), (8, 7)]]),
    "X": (6.0, [[(0, 0), (6, 7)], [(0, 7), (6, 0)]]),
    "Y": (6.0, [[(0, 7), (3, 3.4), (6, 7)], [(3, 3.4), (3, 0)]]),
    "Z": (5.8, [[(0, 7), (5.8, 7), (0, 0), (5.8, 0)]]),

    "0": (6.0, [[(2.1, 7), (3.9, 7), (6, 5), (6, 2), (3.9, 0), (2.1, 0),
                 (0, 2), (0, 5), (2.1, 7)], [(1.1, 1.4), (4.9, 5.6)]]),
    "1": (5.0, [[(0.6, 5.4), (2.6, 7), (2.6, 0)], [(0.5, 0), (4.7, 0)]]),
    "2": (6.0, [[(0, 5.6), (1.4, 7), (4.2, 7), (6, 5.6), (6, 4.2), (0, 0), (6, 0)]]),
    "3": (6.0, [[(0.2, 7), (6, 7), (3.0, 4.2)], [(3.0, 4.2), (4.8, 4.2), (6, 3.0),
                 (6, 1.4), (4.2, 0), (1.6, 0), (0, 1.2)]]),
    "4": (6.0, [[(4.3, 0), (4.3, 7), (0, 2.4), (6, 2.4)]]),
    "5": (6.0, [[(6, 7), (1.0, 7), (0.3, 4.0), (1.8, 4.6), (3.6, 4.7), (5.4, 4.0),
                 (6, 2.4), (5.2, 0.7), (3.2, 0), (1.2, 0.3), (0, 1.3)]]),
    "6": (6.0, [[(5.8, 6.0), (4.2, 7), (2.0, 7), (0, 5.0), (0, 2.0), (2.0, 0),
                 (4.0, 0), (6, 1.5), (6, 2.6), (4.0, 4.1), (2.0, 4.1), (0, 2.7)]]),
    "7": (5.8, [[(0, 7), (5.8, 7), (2.2, 0)]]),
    "8": (6.0, [[(2.2, 3.9), (0.2, 4.9), (0.2, 6.0), (2.2, 7), (3.9, 7), (5.8, 6.0),
                 (5.8, 4.9), (3.9, 3.9), (2.2, 3.9)],
                [(3.9, 3.9), (6, 2.7), (6, 1.2), (3.9, 0), (2.2, 0), (0, 1.2),
                 (0, 2.7), (2.2, 3.9)]]),
    "9": (6.0, [[(0.2, 1.0), (1.8, 0), (3.8, 0), (6, 2.0), (6, 5.0), (4.0, 7),
                 (2.0, 7), (0, 5.6), (0, 4.5), (2.0, 3.0), (4.0, 3.0), (6, 4.4)]]),

    ".": (2.6, [[(1.0, 0.25), (1.35, 0.25)]]),
    ",": (2.6, [[(1.3, 0.5), (0.7, -1.1)]]),
    ":": (2.6, [[(1.0, 0.9), (1.35, 0.9)], [(1.0, 4.4), (1.35, 4.4)]]),
    ";": (2.6, [[(1.3, 0.5), (0.7, -1.1)], [(1.0, 4.4), (1.35, 4.4)]]),
    "-": (4.6, [[(0.5, 3.5), (4.1, 3.5)]]),
    "–": (5.8, [[(0.5, 3.5), (5.3, 3.5)]]),      # en dash
    "—": (7.0, [[(0.5, 3.5), (6.5, 3.5)]]),      # em dash
    "_": (5.8, [[(0, -0.8), (5.8, -0.8)]]),
    "/": (4.6, [[(0, -0.4), (4.6, 7.2)]]),
    "\\": (4.6, [[(0, 7.2), (4.6, -0.4)]]),
    "(": (3.4, [[(3.0, 7.3), (0.9, 5.0), (0.9, 2.0), (3.0, -0.3)]]),
    ")": (3.4, [[(0.4, 7.3), (2.5, 5.0), (2.5, 2.0), (0.4, -0.3)]]),
    "[": (3.2, [[(2.9, 7.3), (0.8, 7.3), (0.8, -0.3), (2.9, -0.3)]]),
    "]": (3.2, [[(0.3, 7.3), (2.4, 7.3), (2.4, -0.3), (0.3, -0.3)]]),
    "+": (5.4, [[(0.4, 3.5), (5.0, 3.5)], [(2.7, 1.2), (2.7, 5.8)]]),
    "=": (5.4, [[(0.4, 4.6), (5.0, 4.6)], [(0.4, 2.4), (5.0, 2.4)]]),
    "×": (5.0, [[(0.7, 1.8), (4.3, 5.2)], [(0.7, 5.2), (4.3, 1.8)]]),   # ×
    "'": (2.2, [[(1.1, 7), (1.1, 5.3)]]),
    "’": (2.2, [[(1.1, 7), (1.1, 5.3)]]),
    '"': (3.8, [[(1.1, 7), (1.1, 5.3)], [(2.7, 7), (2.7, 5.3)]]),
    "!": (2.6, [[(1.2, 7), (1.2, 2.0)], [(1.2, 0.25), (1.5, 0.25)]]),
    "?": (5.2, [[(0.3, 5.6), (1.6, 7), (3.6, 7), (5.0, 5.6), (5.0, 4.4),
                 (2.6, 2.9), (2.6, 2.0)], [(2.6, 0.25), (2.9, 0.25)]]),
    "#": (6.4, [[(1.8, 0), (2.8, 7)], [(4.0, 0), (5.0, 7)],
                [(0.4, 2.2), (6.0, 2.2)], [(0.7, 4.8), (6.3, 4.8)]]),
    "%": (7.0, [[(0.8, 0), (6.2, 7)],
                [(1.4, 7.0), (0.4, 6.2), (1.4, 5.4), (2.4, 6.2), (1.4, 7.0)],
                [(5.6, 1.6), (4.6, 0.8), (5.6, 0.0), (6.6, 0.8), (5.6, 1.6)]]),
    "°": (3.6, [_ring(1.8, 6.1, 0.9, 8)]),
    # The ring of 'Å' sits above the cap height, so the A itself is drawn short
    # to keep the whole glyph inside the line box the layout reserved for it.
    "Å": (6.0, [[(0, 0), (3, 5.6), (6, 0)], [(0.9, 1.9), (5.1, 1.9)],
                _ring(3.0, 6.7, 0.85, 8)]),
    "å": (6.0, [[(0, 0), (3, 5.6), (6, 0)], [(0.9, 1.9), (5.1, 1.9)],
                _ring(3.0, 6.7, 0.85, 8)]),
}

# --------------------------------------------------------------------------
# Curve smoothing
# --------------------------------------------------------------------------
#: Glyphs whose strokes are meant to read as curves.  Drawn as a handful of
#: straight segments they come out visibly faceted — a 'C' with three flat sides
#: and two hard corners — which at plaque size looks like a rendering fault
#: rather than a typeface.  Corner-cutting rounds them for free.
#:
#: Letters made only of straight strokes (A E F H I K L M N T V W X Y Z, 1, 4, 7
#: and the punctuation) are *not* listed: rounding those would put a bend in an
#: upright, which is the one thing that would look worse.
_CURVED = set("BCDGJOPQRSU0235689åÅ°")

#: Chaikin passes.  Two is enough to hide the facets and mild enough that the
#: letterforms keep their proportions; more starts visibly shrinking the bowls.
_SMOOTH_PASSES = 2


def _chaikin(points: List[Tuple[float, float]], passes: int = _SMOOTH_PASSES):
    """Round a polyline's corners by repeated corner-cutting.

    Endpoints are held: a stroke has to still start and end where the letter
    says it does, and only the corners between are cut. A closed loop (first
    point equal to last) is cut all the way round instead, so the join does not
    stay a visible kink.
    """
    if len(points) < 3:
        return points
    closed = (abs(points[0][0] - points[-1][0]) < 1e-9
              and abs(points[0][1] - points[-1][1]) < 1e-9)
    pts = list(points[:-1]) if closed else list(points)
    for _ in range(max(0, passes)):
        if len(pts) < 3:
            break
        out = []
        if not closed:
            out.append(pts[0])
        span = range(len(pts)) if closed else range(len(pts) - 1)
        for i in span:
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            out.append((a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25))
            out.append((a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75))
        if not closed:
            out.append(pts[-1])
        pts = out
    return pts + [pts[0]] if closed else pts


def _smooth_table() -> None:
    """Round the curved glyphs once, at import, rather than per character."""
    for ch in list(_GLYPHS):
        if ch not in _CURVED:
            continue
        advance, strokes = _GLYPHS[ch]
        _GLYPHS[ch] = (advance, [_chaikin(s) for s in strokes])


_smooth_table()


#: Substituted before lookup so a name from a PDB header cannot punch a hole in
#: the plaque.  Anything still missing after this falls back to a hyphen.
_FOLD = {
    "‐": "-", "‑": "-", "‒": "-", "−": "-",
    "“": '"', "”": '"', "‘": "'",
    " ": " ", "\t": " ",
    "α": "a", "β": "b", "γ": "g", "κ": "k", "μ": "u",
    "é": "e", "è": "e", "ê": "e", "ü": "u", "ö": "o",
    "ä": "a", "ß": "ss", "í": "i", "ó": "o", "ú": "u",
}


def _lookup(ch: str):
    """``(glyph, scale)`` for one character, or ``None`` if it draws nothing."""
    if ch in _GLYPHS:
        return _GLYPHS[ch], 1.0
    up = ch.upper()
    if up in _GLYPHS:
        # Lowercase becomes a small capital.
        return _GLYPHS[up], SMALL_CAP
    return None


def text_width(text: str, cap_mm: float) -> float:
    """Advance width of ``text`` in millimetres at cap height ``cap_mm``."""
    unit = cap_mm / CAP
    total = 0.0
    for ch in _expand(text):
        got = _lookup(ch)
        if got is None:
            got = _GLYPHS["-"], 1.0
        (adv, _strokes), scale = got
        total += adv * scale + LETTER_SPACING
    return max(0.0, total - LETTER_SPACING) * unit


def _expand(text: str) -> str:
    """Fold characters the table cannot draw into ones it can."""
    return "".join(_FOLD.get(ch, ch) for ch in str(text))


def layout(text: str, cap_mm: float, origin=(0.0, 0.0)):
    """Polylines for ``text``, in millimetres, baseline-left at ``origin``.

    Returns ``[[(x, y), ...], ...]`` in a right-handed 2D frame with ``y`` up.
    The caller sweeps these into solids — see ``stand._stroke_solids`` — so the
    stroke *width* is not decided here; these are centre lines.
    """
    unit = cap_mm / CAP
    ox, oy = float(origin[0]), float(origin[1])
    out: List[List[Tuple[float, float]]] = []
    pen = 0.0
    for ch in _expand(text):
        got = _lookup(ch)
        if got is None:
            got = _GLYPHS["-"], 1.0
        (adv, strokes), scale = got
        for stroke in strokes:
            out.append([(ox + (pen + x * scale) * unit, oy + y * scale * unit)
                        for (x, y) in stroke])
        pen += adv * scale + LETTER_SPACING
    return out


def wrap(text: str, cap_mm: float, max_width_mm: float,
         max_lines: int = 3) -> List[str]:
    """Greedy word wrap to ``max_width_mm``, truncated with an ellipsis.

    A structure title out of a PDB header is frequently a full sentence, and a
    plaque has one width.  Overflowing it silently would push text off the end of
    the panel where it is not merely ugly but *gone*, so the last line is
    shortened until it fits and marked with a '...' that says so.
    """
    words = str(text).split()
    if not words:
        return []
    lines: List[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if current and text_width(trial, cap_mm) > max_width_mm:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                break
        else:
            current = trial
    if len(lines) < max_lines and current:
        lines.append(current)

    if len(lines) == max_lines:
        # Anything left over has to be signalled rather than dropped in silence.
        consumed = sum(len(l.split()) for l in lines)
        if consumed < len(words):
            last = lines[-1]
            while last and text_width(last + "...", cap_mm) > max_width_mm:
                last = last[:-1]
            lines[-1] = last.rstrip() + "..."
    return lines


def wrap_balanced(text: str, cap_mm: float, max_width_mm: float,
                  max_lines: int = 3) -> List[str]:
    """Wrap into the fewest lines that fit, split as evenly as possible.

    Greedy wrapping fills each line to the brim and leaves whatever is left on
    the last one, so a two-line title reads as a full line and a stub — which on
    a plaque looks like the text ran out rather than like it was set.  Choosing
    the split that makes the lines most nearly equal costs nothing here (a title
    is a dozen words and at most three lines, so every split can simply be
    tried) and is the difference between a label and a leftover.

    Returns ``[]`` for empty input, and falls back to :func:`wrap` — which
    truncates with an ellipsis — when even ``max_lines`` cannot hold it.
    """
    words = str(text).split()
    if not words:
        return []

    def widths(groups):
        return [text_width(" ".join(g), cap_mm) for g in groups]

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
            line_widths = widths(groups)
            if max(line_widths) > max_width_mm:
                continue
            # Evenness: the spread between the longest and shortest line, with
            # the longest line as a tie-break so a tidy block is preferred to a
            # merely even one.
            cost = (max(line_widths) - min(line_widths)) + 0.15 * max(line_widths)
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, [" ".join(g) for g in groups]
        if best is not None:
            return best

    return wrap(text, cap_mm, max_width_mm, max_lines)


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


def fit_cap_height(lines: Sequence[str], max_width_mm: float,
                   preferred_cap_mm: float, min_cap_mm: float = 1.6) -> float:
    """Largest cap height at or below ``preferred_cap_mm`` that fits every line.

    Never returns less than ``min_cap_mm``: below roughly a millimetre and a half
    a stroke is thinner than the extrusion that has to draw it, and shrinking
    further makes the text *less* legible rather than more. A line that still
    does not fit at the floor is left to :func:`wrap` to shorten.
    """
    widest = max([text_width(l, preferred_cap_mm) for l in lines] or [0.0])
    if widest <= max_width_mm or widest <= 0.0:
        return preferred_cap_mm
    return max(min_cap_mm, preferred_cap_mm * max_width_mm / widest)
