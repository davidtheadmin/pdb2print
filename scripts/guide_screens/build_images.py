"""Turn the raw captures from capture.mjs into the guide's pictures.

    python scripts/guide_screens/build_images.py

Writes frontend/img/guide/<name>-<theme>.webp and redraws the numbered boxes
over each picture in frontend/index.html. The boxes are computed from where the
controls actually were in the capture, so they follow a layout change without
anyone moving them by hand. The numbers are in the same order as the lists under
each picture in the guide: change one, change the other.

Needs Pillow.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / "raw"
OUT = ROOT / "frontend" / "img" / "guide"
INDEX = ROOT / "frontend" / "index.html"
DPR = 2   # capture.mjs shoots at deviceScaleFactor 2


def union(*bs):
    x0 = min(b[0] for b in bs); y0 = min(b[1] for b in bs)
    x1 = max(b[0] + b[2] for b in bs); y1 = max(b[1] + b[3] for b in bs)
    return [x0, y0, x1 - x0, y1 - y0]


def figures(B):
    """name -> (raw file, crop in CSS px or None, output width px, marks, alt).

    A mark is (number, box in CSS px of the raw capture, 'tl' or 'tr' for the
    corner its number sits on)."""
    ov, jo, sl, sp = B["overview"], B["joints"], B["stand_lock"], B["stand_panel"]
    es, ep = B["el_style"], B["el_print"]
    v, sh = sl["viewer"], sl["sheet"]
    model = [v[0] + 30, v[1] + 70, sh[0] - v[0] - 60, v[3] - 150]   # left of the prompt
    dp = es["dnaparts"]
    return {
        "overview": ("overview.png", None, 1600, [
            (1, union(ov["pdb"], ov["examples"]), "tr"), (2, ov["style"], "tr"),
            (3, ov["print"], "tr"), (4, ov["generate"], "tr"),
            (5, union(ov["dl3mf"], ov["dlstl"]), "tl"), (6, ov["share"], "tr"),
            (7, ov["stand"], "tl"), (8, ov["joints"], "tl"), (9, ov["legend"], "tr"),
            (10, ov["report"], "tl"), (11, union(ov["theme"], ov["help"], ov["legal"]), "tr")]),
        "joints": ("joints.png", [890, 0, 390, 800], 780, [
            (1, jo["chains"], "tl"), (2, jo["seg"], "tl"), (3, jo["arrow"], "tr"), (4, jo["go"], "tl")]),
        "stand-lock": ("stand_lock.png", v, 1254, [
            (1, model, "tl"), (2, [sl["roll"][0] - 6, sl["roll"][1] - 26, sl["roll"][2] + 12, 36], "tl"),
            (3, sl["lock"], "tl")]),
        "stand-panel": ("stand_panel.png", [890, 0, 390, 800], 780, [
            (1, sp["sketch"], "tl"), (2, union(sp["sec1"], sp["sec3"]), "tl"),
            (3, sp["unlock"], "tr"), (4, sp["go"], "tl")]),
        "style": ("el_style.png", None, 794, [
            (1, es["protein"], "tr"), (2, es["adv"], "tr"), (3, es["dna"], "tr"),
            (4, [dp[0], dp[1] + 14, dp[2], dp[3] - 14], "tr"),
            (5, union(es["ligsw"], es["ligstyle"]), "tr")]),
        "print": ("el_print.png", None, 794, [
            (1, ep["scale"], "tr"), (2, ep["adv"], "tr"), (3, ep["assembly"], "tr"),
            (4, ep["magsw"], "tr"), (5, ep["size"], "tr"), (6, ep["count"], "tr"),
            (7, ep["socket"], "tr"), (8, ep["bp"], "tr")]),
        "tooltip": ("el_tooltip.png", None, 722, []),
    }


def welcome_marks(B):
    """The welcome card's small copy of the overview: four steps, big numbers."""
    ov = B["overview"]
    return [(1, union(ov["pdb"], ov["examples"])), (2, ov["style"]), (3, ov["print"]),
            (4, ov["generate"]), (4, union(ov["dl3mf"], ov["dlstl"]))]


def svg_marks(marks, off, w, h, pad=5, r=14, rx=9):
    parts = []
    for n, b, *pos in marks:
        pos = pos[0] if pos else "tr"
        x, y, bw, bh = b[0] - off[0] - pad, b[1] - off[1] - pad, b[2] + 2 * pad, b[3] + 2 * pad
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}"/>')
        cx = min(max(x + bw if pos == "tr" else x, r + 3), w - r - 3)
        cy = min(max(y, r + 3), h - r - 3)
        parts.append(f'<g class="b"><circle cx="{cx}" cy="{cy}" r="{r}"/><text x="{cx}" y="{cy}">{n}</text></g>')
    return "".join(parts)


def main():
    B = json.loads((RAW / "light" / "boxes.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    html = INDEX.read_text(encoding="utf-8")
    for name, (src, crop, out_w, marks) in figures(B).items():
        for theme in ("light", "dark"):
            im = Image.open(RAW / theme / src)
            if crop:
                im = im.crop(tuple(int(c * DPR) for c in (crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3])))
            w, h = round(im.width / DPR), round(im.height / DPR)
            im = im.convert("RGB").resize((out_w, round(im.height * out_w / im.width)), Image.LANCZOS)
            im.save(OUT / f"{name}-{theme}.webp", "WEBP", quality=80, method=6)
        off = (crop[0], crop[1]) if crop else (0, 0)
        # Every figure showing this picture: the guide's, and for the overview
        # also the welcome card's, which gets its own marks.
        pat = re.compile(r'(<figure class="m-shot( wel-shot)?">\s*<img class="sh-l" src="img/guide/'
                         + re.escape(name) + r'-light\.webp".*?</figure>)', re.S)
        def redo(m):
            fig, wel = m.group(1), m.group(2)
            fig = re.sub(r'width="\d+" height="\d+"', f'width="{w}" height="{h}"', fig)
            if wel:
                svg = svg_marks(welcome_marks(B), off, w, h, pad=6, r=30, rx=12)
            else:
                svg = svg_marks(marks, off, w, h)
            if not svg:
                return fig
            return re.sub(r'<svg viewBox="[^"]*" aria-hidden="true">.*?</svg>',
                          lambda _: f'<svg viewBox="0 0 {w} {h}" aria-hidden="true">{svg}</svg>', fig, flags=re.S)
        html, n = pat.subn(redo, html)
        print(f"{name}: {w}x{h}, {n} figure(s) updated")
    INDEX.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
