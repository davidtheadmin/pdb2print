"""Per-chain subunit names parsed from a structure header.

The viewer legend is much more useful when it says "Hemoglobin alpha (A)" than
"Chain A".  The name lives in the file header, not the atom records, so it is
parsed here from the on-disk file (biotite drops header metadata when it builds
the :class:`AtomArray`):

* **PDB**   — the ``COMPND`` records: each ``MOL_ID`` block carries a
  ``MOLECULE`` name and a ``CHAIN:`` list, which we map to chain ids.
* **mmCIF** — ``_entity.pdbx_description`` mapped to author chain ids through
  ``_entity_poly.pdbx_strand_id`` (falling back to ``_struct_asym``).

Everything here is best-effort: any parse failure yields an empty mapping and
the pipeline simply falls back to "Chain <id>".  Names are display metadata and
never influence geometry.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional


def chain_names(path: str) -> Dict[str, str]:
    """Return a ``{chain_id: subunit_name}`` mapping for a structure file.

    Unknown extensions are tried as mmCIF then PDB.  Returns ``{}`` on any
    failure so callers can always fall back to the chain id.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".pdb", ".ent"):
            return _names_from_pdb(path)
        if ext in (".cif", ".mmcif", ".bcif"):
            return _names_from_cif(path)
        # Unknown: try CIF, then PDB.
        return _names_from_cif(path) or _names_from_pdb(path)
    except Exception:
        return {}


# --------------------------------------------------------------------------
# PDB COMPND
# --------------------------------------------------------------------------
def _names_from_pdb(path: str) -> Dict[str, str]:
    """Parse ``COMPND`` records into a ``{chain_id: molecule_name}`` mapping.

    The ``COMPND`` block is one logical, semicolon-delimited token list spread
    over continuation lines (text in columns 11-80).  Each ``MOL_ID`` starts a
    new molecule whose ``MOLECULE`` name applies to every id in its ``CHAIN``
    list.
    """
    chunks = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("COMPND"):
                # Columns 11-80 hold the continuation text (0-based slice 10:80).
                chunks.append(line[10:80].rstrip("\n").rstrip())
    if not chunks:
        return {}

    blob = " ".join(chunks)
    # Each MOL_ID starts a new molecule.  Some files omit the ';' terminator on
    # the token before a MOL_ID, which would otherwise merge two blocks; force a
    # separator before every interior MOL_ID so blocks always split cleanly.
    blob = re.sub(r"(?<!^)\bMOL_ID:", "; MOL_ID:", blob)
    entities: Dict[str, Dict[str, object]] = {}
    current: Optional[str] = None
    for token in blob.split(";"):
        key, sep, val = token.partition(":")
        if not sep:
            continue
        key = key.strip().upper()
        val = val.strip()
        if key == "MOL_ID":
            current = val
            entities.setdefault(current, {})
        elif key == "MOLECULE" and current is not None:
            entities.setdefault(current, {})["name"] = val
        elif key == "CHAIN" and current is not None:
            ids = [c.strip() for c in val.split(",") if c.strip()]
            entities.setdefault(current, {}).setdefault("chains", []).extend(ids)

    mapping: Dict[str, str] = {}
    for ent in entities.values():
        name = _prettify(ent.get("name"))  # type: ignore[arg-type]
        if not name:
            continue
        for cid in ent.get("chains", []):  # type: ignore[union-attr]
            mapping[cid] = name
    return mapping


# --------------------------------------------------------------------------
# mmCIF _entity
# --------------------------------------------------------------------------
def _names_from_cif(path: str) -> Dict[str, str]:
    """Map author chain ids to ``_entity.pdbx_description`` via ``_entity_poly``.

    biotite builds author chain ids (``auth_asym_id``) by default, so we prefer
    ``_entity_poly.pdbx_strand_id`` (author chains, comma-separated per entity)
    and fall back to ``_struct_asym`` (label chains) if that category is absent.
    """
    from biotite.structure.io import pdbx

    ext = os.path.splitext(path)[1].lower()
    if ext == ".bcif":
        cif = pdbx.BinaryCIFFile.read(path)
    else:
        cif = pdbx.CIFFile.read(path)
    block = cif.block

    # entity_id -> description
    descriptions: Dict[str, str] = {}
    if "entity" in block:
        entity = block["entity"]
        ids = entity["id"].as_array(str)
        descs = entity["pdbx_description"].as_array(str)
        for eid, desc in zip(ids, descs):
            pretty = _prettify(desc)
            if pretty:
                descriptions[str(eid)] = pretty
    if not descriptions:
        return {}

    mapping: Dict[str, str] = {}
    if "entity_poly" in block:
        poly = block["entity_poly"]
        eids = poly["entity_id"].as_array(str)
        strands = poly["pdbx_strand_id"].as_array(str)
        for eid, strand in zip(eids, strands):
            desc = descriptions.get(str(eid))
            if not desc:
                continue
            for cid in str(strand).split(","):
                cid = cid.strip()
                if cid:
                    mapping[cid] = desc
    if not mapping and "struct_asym" in block:
        asym = block["struct_asym"]
        asym_ids = asym["id"].as_array(str)
        asym_eids = asym["entity_id"].as_array(str)
        for cid, eid in zip(asym_ids, asym_eids):
            desc = descriptions.get(str(eid))
            if desc:
                mapping[str(cid)] = desc
    return mapping


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
_MAX_NAME_LEN = 48

#: Acronyms kept uppercase when re-casing an ALL-CAPS description.
_ACRONYMS = {"DNA", "RNA", "PNA", "TRNA", "MRNA", "RRNA", "TAR", "HIV", "ATP",
             "GTP", "NAD", "FAD"}

#: Markers that flag a raw nucleotide-sequence descriptor, e.g.
#: ``DNA (5'-D(*CP*GP...)-3')`` — noise in a legend, so everything from the
#: opening parenthesis on is stripped, leaving just the molecule type.
_SEQ_MARKER = re.compile(r"5'|3'|\*[A-Za-z]P|-[DR]\(")

#: A word run (letters/digits, incl. internal hyphen/apostrophe/slash) that the
#: ALL-CAPS re-caser operates on, so surrounding punctuation like "(" is left be.
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-/]*")


def _recase_word(match: "re.Match") -> str:
    """Title-case one word, keeping acronyms and mixed alphanumeric ids intact."""
    word = match.group(0)
    if any(ch.isdigit() for ch in word):
        return word  # identifier such as ZIF268 or a bare number
    parts = re.split(r"([-'/])", word)  # recase each hyphen/slash segment
    out = []
    for p in parts:
        if p in "-'/" or not p:
            out.append(p)
        elif p.upper() in _ACRONYMS:
            out.append(p.upper())
        else:
            out.append(p[:1].upper() + p[1:].lower())
    return "".join(out)


def _prettify(name: Optional[str]) -> Optional[str]:
    """Tidy a raw header name for display, or ``None`` if it is empty/uninformative.

    Strips raw nucleotide-sequence descriptors (so "DNA (5'-D(*CP*GP...)-3')"
    becomes "DNA"), collapses whitespace, title-cases ALL-CAPS descriptions (so
    "HEMOGLOBIN (ALPHA CHAIN)" reads as "Hemoglobin (Alpha Chain)" and
    "DNA-BINDING DOMAIN" as "DNA-binding Domain") while keeping known acronyms
    and identifiers, and ellipsis-truncates very long names.
    """
    if not name:
        return None
    if _SEQ_MARKER.search(name):
        # Sequence descriptor: keep only the molecule type before the sequence.
        name = name.split("(")[0]
    name = re.sub(r"\s+", " ", name).strip().strip(".")
    if not name or name.upper() in {"NULL", "?", "."}:
        return None
    if name == name.upper():  # biotite/PDB convention is ALL CAPS
        name = _WORD.sub(_recase_word, name)
    if len(name) > _MAX_NAME_LEN:
        name = name[: _MAX_NAME_LEN - 1].rstrip() + "…"
    return name
