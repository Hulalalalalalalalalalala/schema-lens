"""Core schema inference and validation logic.

A schema is a tree of nodes. Object nodes carry a mapping of field name to
field entry; each field entry records the set of observed types, how many
records carried the field, how many records were seen at that level, and
whether the field is optional. Array nodes merge their element types.

Inference is a streaming fold: records are folded one at a time and never
retained, so peak memory tracks the size of the statistics tree, not the
number of records. The tree itself can be bounded with two resource limits
given to ``Lens``: ``max_fields`` caps how many field entries one object
node tracks exactly (further distinct names fold into a shared overflow
entry kept under the node's ``"overflow"`` key), and ``max_depth`` caps how
deep object and array nodes are tracked (deeper subtrees collapse into
summary nodes that keep only counts and element kinds). Every statistic
that has been approximated is flagged with ``"approximate": true`` in the
schema tree and listed under the ``"approximate"`` key of ``Lens.stats``,
so precision loss is always visible and never silent. With no limits (the
default) no approximation is ever triggered and the folded schema is
exactly the schema produced by folding every record in one pass. The field
cap is also enforced after two batches are merged, so a merged object node
can never hold more exact field entries than the cap even when the two
batches named disjoint sets; the excess entries fold into that node's
overflow entry.

Persistence is multi-revision, incremental and crash safe. Each ``save``
is a commit serialized through a lock file inside the lens directory: it
folds the records folded since the last load or save into the schema of
the newest committed revision and writes the result as one brand new,
complete revision file in the ``versions`` directory, each revision
carrying its own monotonic number and checksum. Revision files are never
rewritten; a new revision only appears by an atomic rename of a fully
synced file and its number is one greater than the previous newest, so
readers see a total order and old revisions are pruned once more than
``max_versions`` complete revisions exist. A crash or an interrupted
commit can therefore only ever leave complete revisions, and any reader
observes either the previous complete revision or the next one, never a
half-written schema. A revision is
parsed, structurally validated and checksummed completely before any of
its content is handed out; a missing or corrupt requested revision raises
``SchemaConflict`` and leaves every other revision usable.

``check`` and ``schema`` can either target an explicit revision number or
read the snapshot the lens was opened (``load``-ed) at; inference keeps
appending batches as before. ``versions`` lists the committed revisions
oldest first and ``rollback`` expresses the target revision's schema as a
brand new commit, so history is appended to and never rewritten. A
revision that has already been pruned (or never existed) makes a rollback
raise ``SchemaConflict`` without touching any committed revision.

A lens directory holding only the old single-file format (``schema.json``)
is still read; the legacy file is migrated in place into the revisions
layout on first read, and a failed migration leaves the original file and
the in-memory schema untouched and raises ``SchemaConflict`` so it can be
retried later.

The oldest segment of the history can be compacted into one baseline
version. A baseline file carries the schema and statistics exactly as they
stood at the end of the merged range (the folded tree is checked
field-for-field against the last merged revision before the switch), and a
small manifest in the versions directory is the single pointer naming at
most one baseline file. Compaction stages a temp artifact, validates it
fully, only then atomically replaces the manifest, and deletes no merged
input until that switch has landed; a crash mid-compaction is reconciled
into "completely happened" or "never happened" the next time the lock is
taken. Reads never wait on a compaction: while it runs they route through
the old manifest to complete revisions, and afterwards the baseline is
readable as its anchor revision while the revisions merged into it answer
``SchemaConflict``.

``compat(old, new)`` reads two committed snapshots without changing memory
or any revision and reports how the evolution between them affects the
records each side accepts. One symmetric tree diff derives both directions
at once, judged by the same rules ``check`` enforces: ``backward`` says
whether the new revision still accepts records legal under the old one,
``forward`` the reverse. Adding an optional field, relaxing a required
field or widening a type set is non-breaking in the direction that accepts;
removing a field, tightening optionality, narrowing a type set or dropping
a nested type branch is breaking there. A branch either side only describes
approximately (an overflow entry, an approximate field entry, or a
depth-capped subtree) cannot be settled and makes both directions
``unknown``, listing its source in ``unknown_reasons`` with the same paths
``stats`` uses; a revision compared with itself is compatible in both
directions with no changes.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import itertools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows
    msvcrt = None  # type: ignore[assignment]

SCHEMA_FILENAME = "schema.json"
LOCK_FILENAME = "schema.lock"
VERSIONS_DIRNAME = "versions"
REVISION_PREFIX = "revision-"
REVISION_SUFFIX = ".json"
BASELINE_PREFIX = "baseline-"
BASELINE_SUFFIX = ".json"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_MAX_VERSIONS = 10
DEFAULT_COMPACT_KEEP = 5
SCHEMA_VERSION = 3
# Version 1 files predate approximation markers; version 2 is the single
# file layout. All three encodings carry the same node tree, so legacy
# files migrate to the revisions layout without re-interpreting the tree.
READABLE_VERSIONS = (1, 2, SCHEMA_VERSION)
# The overflow entry marks approximated locations in stats paths. It is a
# key no JSON object can literally carry, so it can never collide with a
# real field name (a field literally named "*" stays locatable as
# ``$["*"]`` while the marker renders as ``$.*``).
OVERFLOW_MARKER = "*\x00*"

# Each compaction attempt stages its artifact under a unique temp name;
# the file only takes its final baseline name by an atomic rename taken
# under the commit lock, so concurrent compactions never overwrite each
# other's staging file and a retried switch can never unlink a baseline a
# winning switch already committed.
_STAGING_COUNTER = itertools.count()


def _staging_name() -> str:
    return f"staging-{os.getpid()}-{next(_STAGING_COUNTER)}.tmp"


def _write_staging(directory: Path, text: str) -> Path:
    """Write a uniquely named, fully synced staging file in ``directory``."""
    for _ in range(100):
        path = directory / _staging_name()
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(path)
            raise
        return path
    raise SchemaConflict("cannot allocate a compaction staging file")

_KINDS = ("null", "boolean", "number", "string", "array", "object")


class SchemaConflict(Exception):
    """Raised when an operation conflicts with the stored schema state."""


class _RevisionMissing(SchemaConflict):
    """Internal: a mapped revision/baseline file is not on disk (a race)."""


def _kind_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise ValueError(f"unsupported value type: {type(value).__name__}")


def _new_node(kind: str) -> dict:
    node: dict[str, Any] = {"kind": kind}
    if kind == "object":
        node["fields"] = {}
        node["count"] = 0
    elif kind == "array":
        node["elements"] = {}
    return node


def _collapsed_node(kind: str) -> dict:
    """A depth-capped node: marked approximate, sub-structure not tracked.

    Collapsed objects keep only a record count; collapsed arrays keep only
    the kinds of their elements (each of which is itself collapsed or a
    scalar), so the memory a collapsed subtree can use is bounded.
    """
    node: dict[str, Any] = {"kind": kind, "approximate": True}
    if kind == "object":
        node["count"] = 0
    elif kind == "array":
        node["elements"] = {}
    return node


def _is_collapsed(node: dict) -> bool:
    """Whether an object node is a depth-capped summary without fields."""
    return (
        node["kind"] == "object"
        and bool(node.get("approximate"))
        and "fields" not in node
    )


def _new_field() -> dict:
    return {"types": {}, "observed": 0, "total": 0, "optional": False}


def _overflow_add_name(entry: dict, name: str) -> None:
    """Record that ``name`` really folded into an overflow entry.

    The list is kept sorted (and de-duplicated) so the schema tree stays
    canonical across folds, commits and reloads. An explicit ``names:
    None`` marks a pre-tracking overflow entry whose provenance is
    unknown; adding a known name cannot narrow that, so it stays unknown.
    """
    if "names" in entry and entry["names"] is None:
        return
    names = entry.setdefault("names", [])
    if name not in names:
        names.append(name)
        names.sort()


def _union_overflow_names(left: dict | None, right: dict | None) -> list[str] | None:
    """Union the folded-name provenance of two entries.

    A normal exact entry carries no ``names`` key (empty provenance); an
    explicit ``names: None`` marks an overflow entry loaded from a file
    written before provenance was tracked, whose folded names are unknown
    and must keep widening for every name.
    """
    if left is None or right is None:
        return None

    def provenance(entry: dict) -> list[str] | None:
        if "names" not in entry:
            return []
        return entry["names"]

    ln = provenance(left)
    rn = provenance(right)
    if ln is None or rn is None:
        return None
    return sorted(set(ln) | set(rn))


def _fold_object(
    node: dict,
    record: dict,
    depth: int = 0,
    max_fields: int | None = None,
    max_depth: int | None = None,
) -> None:
    if _is_collapsed(node):
        node["count"] += 1
        return
    node["count"] += 1
    fields = node["fields"]
    overflow = node.get("overflow")
    for name, value in record.items():
        entry = fields.get(name)
        if entry is None:
            if max_fields is not None and len(fields) >= max_fields:
                # The field cap is reached: fold this and every further new
                # name into the shared overflow entry. The entry is marked
                # approximate; its type set only widens and its observation
                # count only grows, exactly like an exact entry's.
                if overflow is None:
                    overflow = _new_field()
                    overflow["approximate"] = True
                    node["overflow"] = overflow
                _overflow_add_name(overflow, name)
                overflow["observed"] += 1
                _fold_value(overflow["types"], value, depth + 1, max_fields, max_depth)
                continue
            entry = _new_field()
            fields[name] = entry
        entry["observed"] += 1
        _fold_value(entry["types"], value, depth + 1, max_fields, max_depth)


def _fold_value(
    types: dict,
    value: Any,
    depth: int = 1,
    max_fields: int | None = None,
    max_depth: int | None = None,
) -> None:
    kind = _kind_of(value)
    node = types.get(kind)
    if node is None:
        if max_depth is not None and kind in ("object", "array") and depth > max_depth:
            node = _collapsed_node(kind)
        else:
            node = _new_node(kind)
        types[kind] = node
    if kind == "object":
        _fold_object(node, value, depth, max_fields, max_depth)
    elif kind == "array":
        for element in value:
            _fold_value(node["elements"], element, depth + 1, max_fields, max_depth)


def _refresh_field(entry: dict, total: int) -> None:
    entry["total"] = total
    entry["optional"] = entry["observed"] < total
    for child in entry["types"].values():
        _refresh_node(child)


def _refresh_node(node: dict) -> None:
    if node["kind"] == "object":
        if _is_collapsed(node):
            return
        for entry in node["fields"].values():
            _refresh_field(entry, node["count"])
        overflow = node.get("overflow")
        if overflow is not None:
            _refresh_field(overflow, node["count"])
    elif node["kind"] == "array":
        for child in node["elements"].values():
            _refresh_node(child)


def _cap_object_fields(node: dict, max_fields: int | None) -> None:
    """Fold exact fields beyond the cap into the overflow entry.

    Applied recursively after two batches merge: folding each batch under
    the cap bounds the names it introduces, but disjoint names across
    batches could otherwise leave a merged node with up to twice the cap
    exact entries. Entries are folded in name order, so which names stay
    exact is deterministic; folded entries widen the existing overflow
    entry (which may itself carry records seen under names neither batch
    tracked exactly) and the result is marked approximate.
    """
    if _is_collapsed(node):
        return
    if max_fields is not None and len(node["fields"]) > max_fields:
        # Keep the first names tracked exactly and fold the rest; merged
        # field order follows global first-seen order (left batch then
        # names new in the right batch), so this lands on exactly the
        # entries a single capped pass over every record would have kept.
        names = list(node["fields"])
        excess = names[max_fields:]
        fields = node["fields"]
        overflow = node.get("overflow")
        for name in excess:
            entry = fields.pop(name)
            overflow = (
                _merge_field(overflow, entry)
                if overflow is not None
                else copy.deepcopy(entry)
            )
            # The popped entry was an exact field at this node; its name is
            # the provenance it adds to the node's overflow. Overflow
            # provenance deeper in its type tree rides along in the merge.
            _overflow_add_name(overflow, name)
            overflow["approximate"] = True
            node["overflow"] = overflow
    for entry in node["fields"].values():
        for child in entry["types"].values():
            _cap_node_fields(child, max_fields)
    overflow = node.get("overflow")
    if overflow is not None:
        for child in overflow["types"].values():
            _cap_node_fields(child, max_fields)


def _cap_node_fields(node: dict, max_fields: int | None) -> None:
    if node["kind"] == "object":
        _cap_object_fields(node, max_fields)
    elif node["kind"] == "array":
        for child in node["elements"].values():
            _cap_node_fields(child, max_fields)


def _merge_node(left: dict, right: dict) -> dict:
    """Fold two schema nodes over disjoint batches into one new node."""
    kind = left["kind"]
    if kind == "object":
        if _is_collapsed(left) or _is_collapsed(right):
            # One side gave up per-field detail, so the union can only be
            # reported as a collapsed, marked summary.
            node = _collapsed_node("object")
            node["count"] = left["count"] + right["count"]
            return node
        node = _new_node("object")
        node["count"] = left["count"] + right["count"]
        left_overflow = left.get("overflow")
        right_overflow = right.get("overflow")
        names = list(left["fields"]) + [
            name for name in right["fields"] if name not in left["fields"]
        ]
        for name in names:
            le = left["fields"].get(name)
            re = right["fields"].get(name)
            if le is None:
                # The field is missing on the left; if the left capped its
                # fields, its occurrences of this field landed in the left
                # overflow entry, so the merged entry must absorb them and
                # be marked approximate. That field's own occurrences on
                # the right prove its name is folded too.
                if left_overflow is not None:
                    entry = _merge_field(left_overflow, re)
                    _overflow_add_name(entry, name)
                    entry["approximate"] = True
                else:
                    entry = copy.deepcopy(re)
            elif re is None:
                if right_overflow is not None:
                    entry = _merge_field(le, right_overflow)
                    _overflow_add_name(entry, name)
                    entry["approximate"] = True
                else:
                    entry = copy.deepcopy(le)
            else:
                entry = _merge_field(le, re)
            node["fields"][name] = entry
        if left_overflow is not None or right_overflow is not None:
            if left_overflow is None:
                overflow = copy.deepcopy(right_overflow)
            elif right_overflow is None:
                overflow = copy.deepcopy(left_overflow)
            else:
                overflow = _merge_field(left_overflow, right_overflow)
            overflow["approximate"] = True
            node["overflow"] = overflow
        return node
    if kind == "array":
        node = _new_node("array")
        if left.get("approximate") or right.get("approximate"):
            node["approximate"] = True
        kinds = list(left["elements"]) + [
            element_kind
            for element_kind in right["elements"]
            if element_kind not in left["elements"]
        ]
        for element_kind in kinds:
            lc = left["elements"].get(element_kind)
            rc = right["elements"].get(element_kind)
            if lc is None:
                node["elements"][element_kind] = copy.deepcopy(rc)
            elif rc is None:
                node["elements"][element_kind] = copy.deepcopy(lc)
            else:
                node["elements"][element_kind] = _merge_node(lc, rc)
        return node
    return _new_node(kind)


def _merge_field(left: dict, right: dict) -> dict:
    """Fold two field entries over disjoint batches into one new entry."""
    entry = _new_field()
    entry["observed"] = left["observed"] + right["observed"]
    if left.get("approximate") or right.get("approximate"):
        entry["approximate"] = True
    # Only overflow-ish entries carry provenance; attaching an empty list
    # to ordinary exact entries would make the merged tree differ from a
    # single one-shot fold over the same records.
    if "names" in left or "names" in right:
        entry["names"] = _union_overflow_names(left, right)
    kinds = list(left["types"]) + [
        kind for kind in right["types"] if kind not in left["types"]
    ]
    for kind in kinds:
        lc = left["types"].get(kind)
        rc = right["types"].get(kind)
        if lc is None:
            entry["types"][kind] = copy.deepcopy(rc)
        elif rc is None:
            entry["types"][kind] = copy.deepcopy(lc)
        else:
            entry["types"][kind] = _merge_node(lc, rc)
    return entry


def _subtract_node(mem: dict, base: dict) -> dict:
    """Return the part of ``mem`` folded after ``base`` was taken."""
    kind = mem["kind"]
    if kind == "object":
        if _is_collapsed(mem):
            node = _collapsed_node("object")
            node["count"] = mem["count"] - base.get("count", 0)
            return node
        node = _new_node("object")
        node["count"] = mem["count"] - base.get("count", 0)
        base_fields = base.get("fields") or {}
        for name, entry in mem["fields"].items():
            base_entry = base_fields.get(name)
            if base_entry is None:
                node["fields"][name] = copy.deepcopy(entry)
            else:
                node["fields"][name] = _subtract_field(entry, base_entry)
        mem_overflow = mem.get("overflow")
        if mem_overflow is not None:
            base_overflow = base.get("overflow")
            if base_overflow is None:
                node["overflow"] = copy.deepcopy(mem_overflow)
            else:
                node["overflow"] = _subtract_field(mem_overflow, base_overflow)
        return node
    if kind == "array":
        node = _new_node("array")
        if mem.get("approximate"):
            node["approximate"] = True
        base_elements = base.get("elements") or {}
        for element_kind, child in mem["elements"].items():
            base_child = base_elements.get(element_kind)
            if base_child is None:
                node["elements"][element_kind] = copy.deepcopy(child)
            else:
                node["elements"][element_kind] = _subtract_node(child, base_child)
        return node
    return _new_node(kind)


def _subtract_field(mem: dict, base: dict) -> dict:
    entry = _new_field()
    entry["observed"] = mem["observed"] - base["observed"]
    if mem.get("approximate"):
        entry["approximate"] = True
    mem_names = mem.get("names")
    if mem_names is not None:
        # The delta cannot attribute provenance per name (only aggregate
        # counts are kept), so it carries every name its memory ever folded
        # in; the downstream merge is a union anyway, and over-inclusion
        # only ever widens validation for genuinely folded names.
        entry["names"] = sorted(mem_names)
    for kind, child in mem["types"].items():
        base_child = base["types"].get(kind)
        if base_child is None:
            entry["types"][kind] = copy.deepcopy(child)
        else:
            entry["types"][kind] = _subtract_node(child, base_child)
    return entry


def _join(path: str, name: str) -> str:
    # An identifier is joined with a dot; every other key is quoted in
    # brackets so a name containing a dot or a backslash can never be read
    # as a path separator: $["a.b"] is distinct from $.a.b.
    if name.isidentifier():
        return f"{path}.{name}"
    return f"{path}[{json.dumps(name)}]"


def _marker_path(path: str) -> str:
    """Render the overflow-marker location beneath an object node.

    The marker can never be a real key (see ``OVERFLOW_MARKER``), so a bare
    ``.*`` segment cannot be produced by ``_join`` for any real field name;
    a field literally named "*" quotes as ``$["*"]`` and stays distinct.
    """
    return f"{path}.*"


def _check_object(node: dict, record: dict, path: str, reports: list[str]) -> None:
    if _is_collapsed(node):
        # A depth-capped node tracked no fields, so nothing inside the
        # record's value can be checked; the approximation is marked in
        # the schema instead of guessed about here.
        return
    fields = node["fields"]
    overflow = node.get("overflow")
    overflow_names = None if overflow is None else overflow.get("names")
    for name in record:
        if name not in fields:
            if overflow is not None and (
                overflow_names is None or name in overflow_names
            ):
                # Only names that really folded into the overflow entry are
                # checked against its widened type set; ``names is None``
                # marks a legacy overflow entry with unknown provenance. A
                # name never seen before is still unexpected.
                _check_value(overflow["types"], record[name], _join(path, name), reports)
            else:
                reports.append(f"{_join(path, name)}: unexpected field")
    for name, entry in fields.items():
        if name not in record:
            if not entry["optional"]:
                reports.append(f"{_join(path, name)}: missing required field")
            continue
        _check_value(entry["types"], record[name], _join(path, name), reports)


def _check_value(types: dict, value: Any, path: str, reports: list[str]) -> None:
    kind = _kind_of(value)
    node = types.get(kind)
    if node is None:
        expected = ", ".join(sorted(types))
        reports.append(f"{path}: type {kind} not in field types [{expected}]")
        return
    if kind == "object":
        _check_object(node, value, path, reports)
    elif kind == "array":
        for index, element in enumerate(value):
            _check_value(node["elements"], element, f"{path}[{index}]", reports)


# -- cross-revision compatibility -------------------------------------------
#
# Compatibility is read straight off the two committed schema trees, using
# exactly what ``check`` decides with: a record is accepted when every
# field it carries is a known (or genuinely overflow-folded) field whose
# value kind is in the entry's type set, every required field is present,
# and nested branches recurse. One symmetric walk therefore derives both
# directions at once: ``backward`` asks whether the new tree accepts every
# record the old tree accepts, ``forward`` the reverse.
#
# Each branch returns a ``(backward, forward)`` verdict pair. A branch
# either tree can only describe approximately - an overflow entry that may
# route the field, an entry flagged approximate, or a depth-capped subtree
# - is unknown in both directions and lists its source, in the same paths
# ``stats`` uses for approximated statistics (``$.*`` for an overflow
# entry, the subtree's own path for a collapsed node). The one exception
# is comparing a revision with itself: that revision is known to accept its
# own records, so ``Lens.compat`` answers compatible in both directions
# with no reasons after the revision has been read and validated.

_COMPATIBLE = "compatible"
_BREAKING = "breaking"
_UNKNOWN = "unknown"


def _combine_pair(pair: tuple[str, str], other: tuple[str, str]) -> tuple[str, str]:
    return (_worst(pair[0], other[0]), _worst(pair[1], other[1]))


def _worst(left: str, right: str) -> str:
    """Aggregate branch verdicts for one direction.

    A branch that demonstrably rejects a legal record settles the
    direction as breaking even when another branch is unknowable; without
    one of those an unknown branch degrades it to unknown.
    """
    if left == _BREAKING or right == _BREAKING:
        return _BREAKING
    if left == _UNKNOWN or right == _UNKNOWN:
        return _UNKNOWN
    return _COMPATIBLE


def _overflow_live(entry: dict) -> bool:
    """Whether an overflow entry can route a field name during check.

    An entry with known but empty provenance routes nothing; an explicit
    ``names: None`` (a pre-tracking overflow entry) routes any name.
    """
    names = entry.get("names", [])
    return names is None or bool(names)


def _overflow_routes(entry: dict | None, name: str) -> bool:
    if entry is None or not _overflow_live(entry):
        return False
    names = entry.get("names")
    return names is None or name in names


class _CompatDiffer:
    """Symmetric diff of two validated snapshots, aligned with ``check``."""

    def __init__(self) -> None:
        self.changes: list[dict] = []
        self.reasons: set[str] = set()

    def diff(self, old_root: dict, new_root: dict) -> dict:
        backward, forward = self.compare_object(old_root, new_root, "$")
        self.changes.sort(key=lambda change: (change["path"], change["change"]))
        return {
            "backward": backward,
            "forward": forward,
            "changes": self.changes,
            "unknown_reasons": sorted(self.reasons),
        }

    def change(self, path: str, category: str, breaking: bool) -> None:
        self.changes.append(
            {"path": path, "change": category, "breaking": breaking}
        )

    def compare_object(self, old: dict, new: dict, path: str) -> tuple[str, str]:
        if _is_collapsed(old) or _is_collapsed(new):
            # A depth-capped object lets anything nested through check, so
            # neither direction can be settled field by field.
            self.reasons.add(path)
            return _UNKNOWN, _UNKNOWN
        verdict = _COMPATIBLE, _COMPATIBLE
        old_fields = old["fields"]
        new_fields = new["fields"]
        old_overflow = old.get("overflow")
        new_overflow = new.get("overflow")
        for name in sorted(set(old_fields) | set(new_fields)):
            field_path = _join(path, name)
            old_entry = old_fields.get(name)
            new_entry = new_fields.get(name)
            if old_entry is not None and new_entry is not None:
                verdict = _combine_pair(
                    verdict, self.compare_entry(old_entry, new_entry, field_path)
                )
            elif old_entry is not None:
                # The field disappears from the new exact fields. New check
                # only accepts it when the name really folded into the new
                # overflow entry, whose merged type set is approximate.
                if _overflow_routes(new_overflow, name):
                    self.reasons.add(_marker_path(path))
                    verdict = _combine_pair(verdict, (_UNKNOWN, _UNKNOWN))
                else:
                    # New check rejects records carrying it; old records
                    # omitting it additionally fail an old required field
                    # under the forward direction.
                    self.change(field_path, "field_removed", True)
                    verdict = _combine_pair(
                        verdict,
                        (_BREAKING,
                         _BREAKING if not old_entry["optional"] else _COMPATIBLE),
                    )
            else:
                # The field appears only among the new exact fields. Old
                # check accepts it only through its own overflow entry.
                if _overflow_routes(old_overflow, name):
                    self.reasons.add(_marker_path(path))
                    verdict = _combine_pair(verdict, (_UNKNOWN, _UNKNOWN))
                else:
                    # Backward: an old record omits it, so a new required
                    # field rejects that record while a new optional one
                    # lets it through. Forward: old check calls any record
                    # carrying it unexpected.
                    self.change(
                        field_path, "field_added", not new_entry["optional"]
                    )
                    verdict = _combine_pair(
                        verdict,
                        (_BREAKING if not new_entry["optional"] else _COMPATIBLE,
                         _BREAKING),
                    )
        # A live overflow entry on either side folds names whose accepted
        # branches cannot be attributed exactly; every branch it governs is
        # unknown both ways. The marker can never be a real field name.
        for entry in (old_overflow, new_overflow):
            if entry is not None and _overflow_live(entry):
                self.reasons.add(_marker_path(path))
                verdict = _combine_pair(verdict, (_UNKNOWN, _UNKNOWN))
        return verdict

    def compare_entry(
        self, old: dict, new: dict, path: str
    ) -> tuple[str, str]:
        if old.get("approximate") or new.get("approximate"):
            # An entry flagged approximate absorbed overflow statistics in
            # an exact field position; its accepted value set cannot be
            # attributed precisely, so no change beneath it is asserted.
            self.reasons.add(path)
            return _UNKNOWN, _UNKNOWN
        verdict = _COMPATIBLE, _COMPATIBLE
        if old["optional"] != new["optional"]:
            if old["optional"]:
                # Optional becomes required: old records may omit it.
                self.change(path, "optional_to_required", True)
                verdict = (_BREAKING, _COMPATIBLE)
            else:
                # Required becomes optional: new records may omit it.
                self.change(path, "required_to_optional", False)
                verdict = (_COMPATIBLE, _BREAKING)
        verdict = _combine_pair(
            verdict, self.compare_types(old["types"], new["types"], path, "type")
        )
        return verdict

    def compare_types(
        self, old_types: dict, new_types: dict, path: str, prefix: str
    ) -> tuple[str, str]:
        verdict = _COMPATIBLE, _COMPATIBLE
        for kind in _KINDS:
            if kind not in old_types and kind not in new_types:
                continue
            old_node = old_types.get(kind)
            new_node = new_types.get(kind)
            if old_node is not None and new_node is not None:
                verdict = _combine_pair(
                    verdict, self.compare_kind(old_node, new_node, path)
                )
            elif old_node is not None:
                # Records of this kind are legal on the old side but the
                # new type set no longer contains it (narrowing).
                self.change(path, f"{prefix}_{kind}_removed", True)
                verdict = _combine_pair(verdict, (_BREAKING, _COMPATIBLE))
            else:
                # The new side widened with this kind; old check rejects
                # new records that take it (forward breaking), while old
                # records still pass the new, wider check.
                self.change(path, f"{prefix}_{kind}_added", False)
                verdict = _combine_pair(verdict, (_COMPATIBLE, _BREAKING))
        return verdict

    def compare_kind(
        self, old_node: dict, new_node: dict, path: str
    ) -> tuple[str, str]:
        kind = old_node["kind"]
        if kind in ("null", "boolean", "number", "string"):
            return _COMPATIBLE, _COMPATIBLE
        if kind == "object":
            return self.compare_object(old_node, new_node, path)
        return self.compare_arrays(old_node, new_node, path)

    def compare_arrays(
        self, old: dict, new: dict, path: str
    ) -> tuple[str, str]:
        """Compare array element branches under one schema-level path.

        check walks every element against the single merged element type
        set (a schema has no concrete indices), and it enforces that set
        even for a depth-capped array - only a collapsed *child* hides
        structure. So scalar element kinds gate exactly (rendered as
        ``path[]``) and collapsed element objects taint beneath them;
        element objects then join fields normally (``path[].name``).
        """
        element_path = f"{path}[]"
        verdict = _COMPATIBLE, _COMPATIBLE
        for kind in _KINDS:
            if kind not in old["elements"] and kind not in new["elements"]:
                continue
            old_child = old["elements"].get(kind)
            new_child = new["elements"].get(kind)
            if old_child is not None and new_child is not None:
                verdict = _combine_pair(
                    verdict, self.compare_kind(old_child, new_child, element_path)
                )
            elif old_child is not None:
                self.change(element_path, f"element_type_{kind}_removed", True)
                verdict = _combine_pair(verdict, (_BREAKING, _COMPATIBLE))
            else:
                self.change(element_path, f"element_type_{kind}_added", False)
                verdict = _combine_pair(verdict, (_COMPATIBLE, _BREAKING))
        return verdict


def _compat_report(
    old_revision: int, new_revision: int, old_root: dict, new_root: dict
) -> dict:
    report = _CompatDiffer().diff(old_root, new_root)
    return {"from": old_revision, "to": new_revision, **report}


def _normalize_overflow_names(node: Any) -> None:
    """Mark pre-tracking overflow entries with explicit ``names: None``.

    Files written before folded names were tracked carry overflow entries
    without a ``names`` key; unlike an exact entry (which also omits the
    key), their provenance is unknown and must stay permissive through
    merges. Exact entries are never reached here.
    """
    if not isinstance(node, dict):
        return
    kind = node.get("kind")
    if kind == "object" and not _is_collapsed(node):
        overflow = node.get("overflow")
        if overflow is not None and "names" not in overflow:
            overflow["names"] = None
        fields = node.get("fields") or {}
        for entry in fields.values():
            for child in (entry.get("types") or {}).values():
                _normalize_overflow_names(child)
        if overflow is not None:
            for child in (overflow.get("types") or {}).values():
                _normalize_overflow_names(child)
    elif kind == "array":
        for child in (node.get("elements") or {}).values():
            _normalize_overflow_names(child)


def _validate_node(node: Any) -> None:
    if not isinstance(node, dict) or node.get("kind") not in _KINDS:
        raise SchemaConflict("corrupt schema revision: malformed node")
    approximate = node.get("approximate", False)
    if not isinstance(approximate, bool):
        raise SchemaConflict("corrupt schema revision: malformed node")
    kind = node["kind"]
    if kind == "object":
        fields = node.get("fields")
        count = node.get("count")
        if isinstance(count, bool) or not isinstance(count, int):
            raise SchemaConflict("corrupt schema revision: malformed object node")
        if fields is None:
            # Only a collapsed (approximate) object node may omit fields.
            if not approximate:
                raise SchemaConflict("corrupt schema revision: malformed object node")
            fields = {}
        if not isinstance(fields, dict):
            raise SchemaConflict("corrupt schema revision: malformed object node")
        for entry in fields.values():
            _validate_field(entry)
        overflow = node.get("overflow")
        if overflow is not None:
            _validate_field(overflow)
    elif kind == "array":
        elements = node.get("elements")
        if not isinstance(elements, dict):
            raise SchemaConflict("corrupt schema revision: malformed array node")
        for child in elements.values():
            _validate_node(child)


def _validate_field(entry: Any) -> None:
    if not isinstance(entry, dict):
        raise SchemaConflict("corrupt schema revision: malformed field entry")
    types = entry.get("types")
    observed = entry.get("observed")
    total = entry.get("total")
    optional = entry.get("optional")
    approximate = entry.get("approximate", False)
    names = entry.get("names")
    if (
        not isinstance(types, dict)
        or isinstance(observed, bool)
        or not isinstance(observed, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or not isinstance(optional, bool)
        or not isinstance(approximate, bool)
        or (
            names is not None
            and (
                not isinstance(names, list)
                or any(not isinstance(name, str) for name in names)
            )
        )
    ):
        raise SchemaConflict("corrupt schema revision: malformed field entry")
    for child in types.values():
        _validate_node(child)


def _checksum(root: dict) -> str:
    """Checksum of the whole schema in its canonical JSON encoding."""
    canonical = json.dumps(root, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_payload(payload: Any) -> tuple[int, dict]:
    if not isinstance(payload, dict):
        raise SchemaConflict("corrupt schema revision: not a JSON object")
    version = payload.get("version")
    if version not in READABLE_VERSIONS:
        raise SchemaConflict(f"unsupported schema version: {version!r}")
    checksum = payload.get("checksum")
    if not isinstance(checksum, str):
        raise SchemaConflict("corrupt schema revision: missing checksum")
    if "root" not in payload:
        raise SchemaConflict("corrupt schema revision: missing root")
    root = payload["root"]
    _validate_node(root)
    if not isinstance(root, dict) or root.get("kind") != "object":
        raise SchemaConflict("corrupt schema revision: root must be an object")
    if _checksum(root) != checksum:
        raise SchemaConflict("corrupt schema revision: checksum mismatch")
    _normalize_overflow_names(root)
    return version, root


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(fd)


def _write_atomic(file_path: Path, text: str) -> None:
    """Write text atomically: a crash leaves only the old or new file."""
    tmp = file_path.with_name(file_path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, file_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(file_path.parent)


def _revision_path(directory: Path, revision: int) -> Path:
    return directory / f"{REVISION_PREFIX}{revision:010d}{REVISION_SUFFIX}"


def _baseline_path(directory: Path, anchor: int) -> Path:
    return directory / f"{BASELINE_PREFIX}{anchor:010d}{BASELINE_SUFFIX}"


def _baseline_anchor(name: str) -> int | None:
    if not (
        name.startswith(BASELINE_PREFIX)
        and name.endswith(BASELINE_SUFFIX)
        and len(name)
        == len(BASELINE_PREFIX) + 10 + len(BASELINE_SUFFIX)
    ):
        return None
    digits = name[len(BASELINE_PREFIX) : -len(BASELINE_SUFFIX)]
    if not digits.isdigit():
        return None
    return int(digits)


def _revision_number(name: str) -> int | None:
    if not (
        name.startswith(REVISION_PREFIX)
        and name.endswith(REVISION_SUFFIX)
        and len(name)
        == len(REVISION_PREFIX) + 10 + len(REVISION_SUFFIX)
    ):
        return None
    digits = name[len(REVISION_PREFIX) : -len(REVISION_SUFFIX)]
    if not digits.isdigit():
        return None
    return int(digits)


def _encode_revision(revision: int, root: dict) -> str:
    payload = {
        "version": SCHEMA_VERSION,
        "revision": revision,
        "checksum": _checksum(root),
        "root": root,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _encode_baseline(anchor: int, root: dict) -> str:
    payload = {
        "version": SCHEMA_VERSION,
        "kind": "baseline",
        "anchor": anchor,
        "checksum": _checksum(root),
        "root": root,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _validate_baseline_payload(payload: Any) -> tuple[int, dict]:
    if not isinstance(payload, dict):
        raise SchemaConflict("corrupt schema baseline: not a JSON object")
    version = payload.get("version")
    if version != SCHEMA_VERSION:
        raise SchemaConflict(f"unsupported schema baseline version: {version!r}")
    if payload.get("kind") != "baseline":
        raise SchemaConflict("corrupt schema baseline: wrong payload kind")
    anchor = payload.get("anchor")
    if isinstance(anchor, bool) or not isinstance(anchor, int) or anchor < 1:
        raise SchemaConflict("corrupt schema baseline: malformed anchor")
    checksum = payload.get("checksum")
    if not isinstance(checksum, str):
        raise SchemaConflict("corrupt schema baseline: missing checksum")
    if "root" not in payload:
        raise SchemaConflict("corrupt schema baseline: missing root")
    root = payload["root"]
    _validate_node(root)
    if not isinstance(root, dict) or root.get("kind") != "object":
        raise SchemaConflict("corrupt schema baseline: root must be an object")
    if _checksum(root) != checksum:
        raise SchemaConflict("corrupt schema baseline: checksum mismatch")
    _normalize_overflow_names(root)
    return anchor, root


def _decode_baseline(file_path: Path) -> tuple[int, dict]:
    """Read one baseline file and validate it completely before use."""
    try:
        raw = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise _RevisionMissing(
            f"schema baseline not found: {file_path.name}"
        ) from exc
    except OSError as exc:
        raise SchemaConflict(f"cannot read schema baseline {file_path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaConflict(f"corrupt schema baseline {file_path.name}: {exc}") from exc
    anchor, root = _validate_baseline_payload(payload)
    number_from_name = _baseline_anchor(file_path.name)
    if number_from_name is not None and number_from_name != anchor:
        raise SchemaConflict(
            f"corrupt schema baseline {file_path.name}: anchor mismatch"
        )
    return anchor, root


def _decode_revision(file_path: Path) -> tuple[int, dict]:
    """Read one revision file and validate it completely before use."""
    try:
        raw = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise _RevisionMissing(
            f"schema revision not found: {file_path.name}"
        ) from exc
    except OSError as exc:
        raise SchemaConflict(f"cannot read schema revision {file_path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaConflict(f"corrupt schema revision {file_path.name}: {exc}") from exc
    _version, root = _validate_payload(payload)
    revision = payload.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise SchemaConflict("corrupt schema revision: malformed revision number")
    number_from_name = _revision_number(file_path.name)
    if number_from_name is not None and number_from_name != revision:
        raise SchemaConflict(
            f"corrupt schema revision {file_path.name}: revision number mismatch"
        )
    return revision, root


class _DirectoryLock:
    """Non-blocking commit lock shared by threads and processes."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._thread_lock = threading.Lock()
        self._fd: int | None = None

    def acquire(self) -> None:
        if not self._thread_lock.acquire(blocking=False):
            raise SchemaConflict(
                "cannot commit schema: another thread holds the lens lock"
            )
        try:
            fd = os.open(
                self._directory / LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o644
            )
        except BaseException:
            self._thread_lock.release()
            raise
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise SchemaConflict(
                        "cannot commit schema: another process holds the lens lock"
                    ) from exc
            elif msvcrt is not None:  # pragma: no cover - Windows
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise SchemaConflict(
                        "cannot commit schema: another process holds the lens lock"
                    ) from exc
        except BaseException:
            os.close(fd)
            self._thread_lock.release()
            raise
        self._fd = fd

    def release(self) -> None:
        fd = self._fd
        self._fd = None
        if fd is not None:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            os.close(fd)
        self._thread_lock.release()

    @contextlib.contextmanager
    def held(self) -> Iterator[None]:
        self.acquire()
        try:
            yield
        finally:
            self.release()


class Lens:
    """Opens the lens directory ``path`` and manages its stored revisions.

    ``max_fields`` caps how many field entries each object node tracks
    exactly; further distinct names fold into a shared overflow entry
    flagged ``"approximate": true`` under the node's ``"overflow"`` key.
    ``max_depth`` caps how deep object and array nodes are tracked (the
    root is depth 0); deeper subtrees collapse into summary nodes flagged
    ``"approximate": true``. Both default to ``None``, which keeps every
    statistic exact and the folded schema identical to folding all records
    in one pass. Records themselves are never retained, so peak memory
    tracks these limits rather than the number of records folded.

    ``max_versions`` bounds how many complete revisions the lens directory
    keeps; commits beyond the bound prune the oldest revisions, newest
    first never being touched. It defaults to 10 and must be a positive
    integer. A revision pruned by retention is no longer readable or a
    valid rollback target.

    ``compact(keep=...)`` merges the oldest segment of the history into one
    baseline version, keeping at least the newest ``keep`` revisions
    separate (default 5); ``compact(background=True)`` runs the merge in a
    background thread without blocking the caller. ``compaction_status()``
    reports whether such a run is in progress and the range it covers.
    """

    def __init__(
        self,
        path: os.PathLike | str,
        *,
        max_fields: int | None = None,
        max_depth: int | None = None,
        max_versions: int = DEFAULT_MAX_VERSIONS,
        compact_keep: int = DEFAULT_COMPACT_KEEP,
    ) -> None:
        for label, value in (("max_fields", max_fields), ("max_depth", max_depth)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{label} must be a non-negative integer or None")
        if (
            isinstance(max_versions, bool)
            or not isinstance(max_versions, int)
            or max_versions < 1
        ):
            raise ValueError("max_versions must be a positive integer")
        if (
            isinstance(compact_keep, bool)
            or not isinstance(compact_keep, int)
            or compact_keep < 1
        ):
            raise ValueError("compact_keep must be a positive integer")
        self.path = Path(path)
        self.max_fields = max_fields
        self.max_depth = max_depth
        self.max_versions = max_versions
        self.compact_keep = compact_keep
        self._schema: dict | None = None
        self._base: dict | None = None
        self._revision: int | None = None
        self._lock = _DirectoryLock(self.path)
        self._bg_thread: threading.Thread | None = None
        self._bg_error: Exception | None = None
        self._bg_range: tuple[int, int] | None = None
        self._bg_lock = threading.Lock()

    @property
    def _legacy_file(self) -> Path:
        return self.path / SCHEMA_FILENAME

    @property
    def _versions_dir(self) -> Path:
        return self.path / VERSIONS_DIRNAME

    def _require(self) -> dict:
        if self._schema is None:
            raise SchemaConflict("no schema has been inferred or loaded yet")
        return self._schema

    def infer(self, records: Iterable[dict]) -> dict:
        """Fold a sequence of records into the stored schema.

        ``records`` may be any iterable, including a generator over a
        record stream; records are folded one at a time and never kept.
        """
        if self._schema is None:
            self._schema = _new_node("object")
            self._base = _new_node("object")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("records must be JSON objects")
            _fold_object(self._schema, record, 0, self.max_fields, self.max_depth)
        _refresh_node(self._schema)
        return self.schema()

    def check(self, record: Any, *, version: int | None = None) -> list[str]:
        """Report every way one record departs from a schema.

        With ``version`` omitted the record is checked against the snapshot
        this lens was opened at (the schema loaded by the last ``load`` or
        adopted by the last ``save``); with ``version`` given it is checked
        against that committed revision, read and fully validated from
        disk first, without changing the open snapshot or memory. A
        missing or corrupt revision raises ``SchemaConflict``.
        """
        schema = self._select(version)
        if not isinstance(record, dict):
            return [f"$: type {_kind_of(record)} not in field types [object]"]
        reports: list[str] = []
        _check_object(schema, record, "$", reports)
        return reports

    def schema(self, *, version: int | None = None) -> dict:
        """Return the stored schema.

        With ``version`` omitted this is the open snapshot; with a number
        given it is that committed revision, read and fully validated from
        disk without changing memory. A missing or corrupt revision raises
        ``SchemaConflict``.
        """
        return copy.deepcopy(self._select(version))

    def _select(self, version: int | None) -> dict:
        if version is None:
            return self._require()
        return self._read_revision(version)

    def compat(self, old_revision: int, new_revision: int) -> dict:
        """Compare two committed revisions for cross-version compatibility.

        Reads the two complete snapshots (old first, then new), each fully
        validated from disk, without changing the open snapshot, memory or
        any committed revision, and returns one report::

            {
              "from": <old revision>,
              "to": <new revision>,
              "backward": "compatible" | "breaking" | "unknown",
              "forward": "compatible" | "breaking" | "unknown",
              "changes": [{"path": ..., "change": ..., "breaking": ...}],
              "unknown_reasons": [<paths that could not be compared>],
            }

        ``backward`` says whether the new revision accepts every record
        legal under the old one and ``forward`` the reverse, judged by the
        same rules ``check`` enforces. A missing, pruned/compacted-away or
        corrupt revision raises ``SchemaConflict``. Repeating the analysis
        for the same pair is stable, and a revision compared with itself is
        compatible in both directions with no changes.
        """
        if (
            isinstance(old_revision, bool)
            or not isinstance(old_revision, int)
            or old_revision < 1
            or isinstance(new_revision, bool)
            or not isinstance(new_revision, int)
            or new_revision < 1
        ):
            raise SchemaConflict(
                f"unknown schema revision: {old_revision!r}, {new_revision!r}"
            )
        old_root = self._read_revision(old_revision)
        new_root = self._read_revision(new_revision)
        if old_revision == new_revision:
            # A revision is known to accept its own records, so the
            # self-comparison is compatible even when the snapshot itself
            # carries approximated branches.
            return {
                "from": old_revision,
                "to": new_revision,
                "backward": _COMPATIBLE,
                "forward": _COMPATIBLE,
                "changes": [],
                "unknown_reasons": [],
            }
        return _compat_report(old_revision, new_revision, old_root, new_root)

    def stats(self) -> dict:
        """Report fields, optional fields, observed types and counts.

        The ``"approximate"`` key lists the paths of every statistic that
        has been approximated under the resource limits: collapsed
        subtrees by their own path and overflow entries as ``$.*`` under
        their object node's path. The marker can never be a real key, so
        it never collides with a field literally named ``*`` (that field
        quotes as ``$["*"]``). It is empty when nothing was approximated.
        """
        schema = self._require()
        approximate: set[str] = set()
        summary: dict[str, Any] = {
            "records": schema["count"],
            "fields": 0,
            "optional": 0,
            "types": {},
            "observations": 0,
            "approximate": [],
        }

        def walk_entry(entry: dict, path: str) -> None:
            summary["fields"] += 1
            summary["observations"] += entry["observed"]
            if entry["optional"]:
                summary["optional"] += 1
            if entry.get("approximate"):
                approximate.add(path)
            for kind, child in entry["types"].items():
                summary["types"][kind] = summary["types"].get(kind, 0) + 1
                walk_node(child, path)

        def walk_object(node: dict, path: str) -> None:
            if _is_collapsed(node):
                approximate.add(path)
                return
            for name, entry in node["fields"].items():
                walk_entry(entry, _join(path, name))
            overflow = node.get("overflow")
            if overflow is not None:
                walk_entry(overflow, _marker_path(path))

        def walk_node(node: dict, path: str) -> None:
            if node["kind"] == "object":
                walk_object(node, path)
            elif node["kind"] == "array":
                if node.get("approximate"):
                    approximate.add(path)
                for child in node["elements"].values():
                    walk_node(child, path)

        walk_object(schema, "$")
        summary["approximate"] = sorted(approximate)
        return summary

    # -- revision storage -------------------------------------------------

    def _manifest_path(self) -> Path:
        return self._versions_dir / MANIFEST_FILENAME

    def _read_manifest(self) -> dict | None:
        """Read the manifest pointer, or ``None`` when no compaction ran.

        The manifest is the single pointer naming at most one baseline
        file; it only ever appears by an atomic replace, so a present file
        is always complete, though its content and every reference in it
        are still validated here. The named baseline file must exist as
        well (a cheap stat, no parsing); a pointer that names a missing
        baseline is corruption rather than a silent fallback.
        """
        data = self._load_manifest_file()
        if data is not None and data["baseline"] is not None:
            if not (self._versions_dir / data["baseline"]["file"]).exists():
                # A newer compaction may have switched and removed the old
                # baseline between the read and the stat; re-read once.
                fresh = self._load_manifest_file()
                if fresh != data:
                    return fresh
                raise SchemaConflict(
                    f"compaction manifest names missing baseline "
                    f"{data['baseline']['file']}"
                )
        return data

    def _load_manifest_file(self) -> dict | None:
        try:
            raw = self._manifest_path().read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SchemaConflict(
                f"cannot read compaction manifest {self._manifest_path()}: {exc}"
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaConflict(f"corrupt compaction manifest: {exc}") from exc
        return self._validate_manifest(payload)

    def _validate_manifest(self, payload: Any) -> dict:
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise SchemaConflict("corrupt compaction manifest: malformed payload")
        baseline = payload.get("baseline")
        if baseline is None:
            return {"baseline": None}
        if not isinstance(baseline, dict):
            raise SchemaConflict("corrupt compaction manifest: malformed baseline")
        anchor = baseline.get("anchor")
        merged = baseline.get("merged")
        file_name = baseline.get("file")
        expected_file = f"{BASELINE_PREFIX}{anchor:010d}{BASELINE_SUFFIX}"
        if (
            isinstance(anchor, bool)
            or not isinstance(anchor, int)
            or anchor < 1
            or not isinstance(merged, list)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in merged)
            or sorted(set(merged)) != sorted(merged)
            or not merged
            or merged[-1] != anchor
            or file_name != expected_file
        ):
            raise SchemaConflict("corrupt compaction manifest: malformed baseline")
        return {
            "baseline": {
                "anchor": anchor,
                "merged": sorted(merged),
                "file": file_name,
            }
        }

    def _write_manifest_locked(self, baseline: dict | None) -> None:
        payload = {"version": 1, "baseline": baseline}
        _write_atomic(self._manifest_path(), json.dumps(payload, indent=2,
                                                         sort_keys=True) + "\n")

    def _scan_revisions(self) -> list[int]:
        """Numbers of complete-looking revision files, oldest first.

        Listing alone makes no promise about content; every revision is
        validated from start to finish before anything uses it.
        """
        try:
            names = os.listdir(self._versions_dir)
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise SchemaConflict(
                f"cannot read versions directory {self._versions_dir}: {exc}"
            ) from exc
        numbers = [number for name in names if (number := _revision_number(name))]
        return sorted(set(numbers))

    def _logical_versions(self, manifest: dict | None) -> list[int]:
        """Ordered logical history through the manifest pointer.

        A directory listing plus the manifest is enough; no revision file
        is parsed to build the list.
        """
        numbers = self._scan_revisions()
        baseline = manifest.get("baseline") if manifest else None
        if baseline is None:
            return numbers
        anchor = baseline["anchor"]
        return [anchor] + [n for n in numbers if n > anchor]

    def _resolve_version(self, revision: int, manifest: dict | None) -> Path:
        """Map a logical revision number to its file without parsing files.

        The baseline anchor routes to the baseline file; revisions merged
        into it are gone and raise; everything else routes by its number.
        """
        baseline = manifest.get("baseline") if manifest else None
        if baseline is not None:
            if revision == baseline["anchor"]:
                return self._versions_dir / baseline["file"]
            if revision in baseline["merged"]:
                raise SchemaConflict(
                    f"schema revision {revision} was compacted into baseline "
                    f"revision {baseline['anchor']}"
                )
        return _revision_path(self._versions_dir, revision)

    def _read_version(self, revision: int) -> dict:
        """Read and fully validate one logical revision (revision or anchor)."""
        manifest = self._read_manifest()
        try:
            return self._decode_at(revision, manifest)
        except SchemaConflict as exc:
            # The pointer may have switched (or retention pruned) between
            # the manifest read and opening the file; re-resolve once
            # against the current pointer so a merged-away revision names
            # itself as compacted instead of missing.
            fresh = self._read_manifest()
            if fresh == manifest:
                raise
            return self._decode_at(revision, fresh)

    def _decode_at(self, revision: int, manifest: dict | None) -> dict:
        path = self._resolve_version(revision, manifest)
        info = manifest.get("baseline") if manifest else None
        if info is not None and revision == info["anchor"]:
            return _decode_baseline(path)[1]
        return _decode_revision(path)[1]

    def _read_revision(self, revision: int) -> dict:
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise SchemaConflict(f"unknown schema revision: {revision!r}")
        manifest = self._read_manifest()
        if manifest is None and not self._scan_revisions():
            # No revision store and no pointer yet: an old single-file
            # layout may carry the schema; migrate it in place first.
            self._migrate_legacy()
        return self._read_version(revision)

    def _read_newest(self) -> tuple[int, dict] | None:
        manifest = self._read_manifest()
        history = self._logical_versions(manifest)
        if not history:
            return None
        revision = history[-1]
        try:
            return self._decode_head(revision, manifest)
        except SchemaConflict:
            # A compaction switch or retention prune can land between
            # listing and opening the file; re-resolve once.
            fresh = self._read_manifest()
            fresh_history = self._logical_versions(fresh)
            if not fresh_history or (
                fresh == manifest and fresh_history == history
            ):
                raise
            revision = fresh_history[-1]
            return self._decode_head(revision, fresh)

    def _decode_head(self, revision: int, manifest: dict | None) -> tuple[int, dict]:
        path = self._resolve_version(revision, manifest)
        info = manifest.get("baseline") if manifest else None
        # A revision or baseline file only appears by an atomic rename of a
        # fully synced file, so the newest logical version is complete or
        # absent; if its content does not validate the stored state is
        # corrupt and the caller gets SchemaConflict instead of an older
        # schema.
        if info is not None and revision == info["anchor"]:
            return _decode_baseline(path)
        return _decode_revision(path)

    def _migrate_legacy(self) -> tuple[int, dict] | None:
        """Migrate the old single-file layout into one revision.

        The legacy file is read and fully validated first, then revision 1
        is written and the directory synced, and only then is the legacy
        file removed. Any failure before that removal leaves the original
        file byte for byte untouched, so the migration can simply be
        retried later.
        """
        try:
            raw = self._legacy_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SchemaConflict(
                f"cannot read schema file {self._legacy_file}: {exc}"
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaConflict(
                f"corrupt schema file {self._legacy_file}: {exc}"
            ) from exc
        _version, root = _validate_payload(payload)
        try:
            self._versions_dir.mkdir(parents=True, exist_ok=True)
            revision_file = _revision_path(self._versions_dir, 1)
            _write_atomic(revision_file, _encode_revision(1, root))
        except OSError as exc:
            raise SchemaConflict(
                f"cannot migrate schema file {self._legacy_file} into "
                f"{self._versions_dir}: {exc}"
            ) from exc
        try:
            os.unlink(self._legacy_file)
        except OSError:
            # The revision is committed and complete; a leftover legacy
            # file must not shadow it on the next open.
            pass
        _fsync_dir(self.path)
        return 1, root

    def _read_head(self) -> tuple[int, dict] | None:
        """The newest complete revision, migrating a legacy layout once."""
        newest = self._read_newest()
        if newest is not None:
            return newest
        legacy = self._migrate_legacy()
        if legacy is not None:
            return legacy
        return None

    def versions(self) -> list[int]:
        """Logical version numbers, oldest first; empty before any commit.

        The baseline anchor is the first number after a compaction;
        revisions merged into it are not listed. Building the list takes a
        directory listing and a manifest read - no revision file is
        parsed.
        """
        return self._logical_versions(self._read_manifest())

    def save(self) -> None:
        """Commit the stored schema into the lens directory.

        The commit is serialized by a non-blocking lock file in the
        directory; contention raises SchemaConflict and the in-memory
        batch stays retryable. Under the lock leftover artifacts of an
        interrupted compaction are reconciled, the newest complete
        revision is read and fully validated, the records folded since the
        last load or save are merged in, the field cap is re-imposed on
        the merged tree, and the result is written as a brand new complete
        revision file. The newest-revision pointer only advances after the
        file is fully on disk, revisions older than ``max_versions`` are
        pruned, and memory is replaced only after the commit lands, so a
        failed commit leaves memory and every committed revision untouched.
        """
        self._require()
        self.path.mkdir(parents=True, exist_ok=True)
        with self._lock.held():
            self._reconcile_locked()
            head = self._read_head()
            if head is None:
                revision = 1
                merged = copy.deepcopy(self._schema)
            else:
                head_revision, disk = head
                revision = head_revision + 1
                delta = _subtract_node(self._schema, self._base)
                merged = _merge_node(disk, delta)
            _refresh_node(merged)
            _cap_node_fields(merged, self.max_fields)
            _refresh_node(merged)
            self._versions_dir.mkdir(parents=True, exist_ok=True)
            target = _revision_path(self._versions_dir, revision)
            _write_atomic(target, _encode_revision(revision, merged))
            self._prune(revision)
        self._schema = merged
        self._base = copy.deepcopy(merged)
        self._revision = revision

    def _prune(self, newest: int) -> None:
        """Remove complete revisions beyond the retention bound.

        Only numbered revision files older than ``newest - max_versions +
        1`` are removed; the baseline file, the newest ``max_versions``
        revisions and any other file in the directory are left alone.
        """
        cutoff = newest - self.max_versions
        if cutoff < 1:
            return
        for revision in self._scan_revisions():
            if revision <= cutoff:
                try:
                    os.unlink(_revision_path(self._versions_dir, revision))
                except FileNotFoundError:
                    pass
        _fsync_dir(self._versions_dir)

    def load(self, *, version: int | None = None) -> None:
        """Read committed schema, replacing memory only on success.

        With ``version`` omitted the newest complete revision is read (and
        a legacy single-file layout migrated in place); with a number
        given that exact revision is read and fully validated. A missing,
        compacted-away or corrupt revision raises ``SchemaConflict`` and
        leaves memory exactly as it was.
        """
        if version is None:
            head = self._read_head()
            if head is None:
                raise SchemaConflict(
                    f"no schema revisions in {self._versions_dir}"
                )
            revision, root = head
        else:
            revision = version
            root = self._read_revision(revision)
        self._schema = root
        self._base = copy.deepcopy(root)
        self._revision = revision

    def rollback(self, version: int) -> int:
        """Commit the schema of revision ``version`` as a new revision.

        History is never rewritten: the target revision stays put and a
        brand new revision carrying an equivalent schema is committed on
        top of it, so the revision list keeps growing. Rolling back to a
        revision that is missing, compacted away or fails validation
        raises ``SchemaConflict`` and changes no committed revision.
        Returns the new revision number; the in-memory snapshot adopts it.
        """
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise SchemaConflict(f"unknown schema revision: {version!r}")
        self.path.mkdir(parents=True, exist_ok=True)
        with self._lock.held():
            self._reconcile_locked()
            target_root = self._read_revision(version)
            head = self._read_newest()
            new_revision = 1 if head is None else head[0] + 1
            merged = copy.deepcopy(target_root)
            self._versions_dir.mkdir(parents=True, exist_ok=True)
            target = _revision_path(self._versions_dir, new_revision)
            _write_atomic(target, _encode_revision(new_revision, merged))
            self._prune(new_revision)
        self._schema = merged
        self._base = copy.deepcopy(merged)
        self._revision = new_revision
        return new_revision

    # -- compaction -------------------------------------------------------

    def _reconcile_locked(self, *, preserve: str | None = None) -> None:
        """Resolve a crash or interruption into one finished state.

        With a committed manifest, its baseline must exist and validate;
        merged inputs left on disk by a crash after the switch are removed.
        Without one, every staged or stray baseline artifact is an
        interrupted compaction that never switched and is removed. Staging
        temp files are likewise leftovers of attempts that never switched
        (a concurrent attempt whose file is removed here simply retries and
        re-stages); the one file named by ``preserve`` belongs to the
        attempt driving this reconcile and is left alone.
        """
        try:
            names = os.listdir(self._versions_dir)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise SchemaConflict(
                f"cannot read versions directory {self._versions_dir}: {exc}"
            ) from exc
        baselines = {
            anchor: name
            for name in names
            if (anchor := _baseline_anchor(name)) is not None
        }
        temps = {
            name for name in names if name.endswith(".tmp") and name != preserve
        }
        manifest = self._read_manifest()
        changed = False
        info = manifest.get("baseline") if manifest else None
        if info is not None:
            anchor = info["anchor"]
            if anchor not in baselines or baselines[anchor] != info["file"]:
                raise SchemaConflict(
                    f"compaction manifest names missing baseline "
                    f"{info['file']}"
                )
            _decode_baseline(self._versions_dir / info["file"])
            for other in list(baselines):
                if other != anchor:
                    os.unlink(self._versions_dir / baselines[other])
                    changed = True
            for merged_revision in info["merged"]:
                path = _revision_path(self._versions_dir, merged_revision)
                try:
                    os.unlink(path)
                    changed = True
                except FileNotFoundError:
                    pass
        else:
            for anchor, name in baselines.items():
                os.unlink(self._versions_dir / name)
                changed = True
        for name in temps:
            os.unlink(self._versions_dir / name)
            changed = True
        if changed:
            _fsync_dir(self._versions_dir)

    def compact(
        self,
        *,
        keep: int | None = None,
        background: bool = False,
    ) -> dict | None:
        """Merge the oldest history segment into one baseline version.

        At least the newest ``keep`` revisions (default
        ``compact_keep``) stay separate; everything older folds into a
        single baseline version whose schema and statistics are exactly
        those of the last merged revision. The merged tree is built from
        immutable files without taking the commit lock, staged as a temp
        artifact, and switched in only after a full self-check against the
        anchor; the revisions being merged are not deleted until the
        manifest pointer has atomically switched. A crash therefore leaves
        either the whole compaction or none of it. Returns a description
        of the new baseline, or ``None`` when the history is already at or
        below ``keep`` versions.

        With ``background=True`` the merge runs in a background thread
        (neither commits nor reads wait on it) and the call returns
        immediately; poll ``compaction_status`` for progress.
        """
        keep = self.compact_keep if keep is None else keep
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise ValueError("keep must be a positive integer")
        if background:
            with self._bg_lock:
                if self._bg_thread is not None and self._bg_thread.is_alive():
                    raise SchemaConflict("a background compaction is already running")
                self._bg_error = None
                self._bg_range = None
                thread = threading.Thread(
                    target=self._compact_in_background,
                    args=(keep,),
                    name="schema-lens-compaction",
                    daemon=True,
                )
                self._bg_thread = thread
            thread.start()
            return None
        return self._compact_loop(keep)

    def _compact_in_background(self, keep: int) -> None:
        try:
            worker = Lens(
                self.path,
                max_fields=self.max_fields,
                max_depth=self.max_depth,
                max_versions=self.max_versions,
                compact_keep=keep,
            )
            worker._compact_loop(keep, range_sink=self._note_bg_range)
        except BaseException as exc:  # reported through compaction_status
            with self._bg_lock:
                self._bg_error = exc

    def _note_bg_range(self, merged: list[int] | None) -> None:
        with self._bg_lock:
            self._bg_range = None if not merged else [merged[0], merged[-1]]

    def compaction_status(self) -> dict:
        """Report the committed baseline and any in-progress compaction.

        Keys: ``"running"`` (a background compaction started on this lens
        is still going), ``"range"`` (the ``[first, last]`` merged range
        of that run), ``"baseline"`` (anchor of the committed baseline or
        ``None``), ``"merged"`` (every revision folded into it),
        ``"versions"`` (the logical history), and ``"error"`` (last
        background failure message, if any).
        """
        with self._bg_lock:
            thread = self._bg_thread
            running = bool(thread is not None and thread.is_alive())
            bg_range = list(self._bg_range) if self._bg_range is not None else None
            error = None if self._bg_error is None else str(self._bg_error)
        manifest = self._read_manifest()
        info = manifest.get("baseline") if manifest else None
        return {
            "running": running,
            "range": bg_range,
            "baseline": None if info is None else info["anchor"],
            "merged": [] if info is None else list(info["merged"]),
            "versions": self._logical_versions(manifest),
            "error": error,
        }

    def _compact_loop(
        self,
        keep: int,
        *,
        range_sink: Any = None,
        attempts: int = 50,
    ) -> dict | None:
        """Run compaction attempts until one switches or the budget is spent.

        Building and self-checking the baseline never touches the commit
        lock; only the final pointer switch is serialized with commits, so
        a history that moved under a concurrent commit simply retries on a
        fresh range instead of blocking readers or writers.
        """
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                staged = self._stage_compaction(keep, range_sink)
                if staged is None:
                    return None
                return self._switch_compaction(staged)
            except _CompactionFatal:
                raise
            except (_CompactionRetry, _RevisionMissing) as exc:
                # The commit lock was busy, the pointer moved, or a file
                # vanished under a concurrent commit; a fresh attempt on
                # the current history settles it.
                last = exc
                time.sleep(min(0.02 * (attempt + 1), 0.1))
            except OSError as exc:
                last = exc
                time.sleep(min(0.02 * (attempt + 1), 0.1))
        raise SchemaConflict(f"compaction could not settle: {last}")

    def _stage_compaction(
        self, keep: int, range_sink: Any
    ) -> dict | None:
        """Build and self-check the baseline artifact without the lock."""
        manifest = self._read_manifest()
        history = self._logical_versions(manifest)
        if len(history) <= keep:
            if range_sink is not None:
                range_sink(None)
            return None
        merged = history[: len(history) - keep]
        anchor = merged[-1]
        info = manifest.get("baseline") if manifest else None
        if info is not None and anchor == info["anchor"]:
            # The segment folds no new revision into the existing
            # baseline, so there is nothing to compact.
            if range_sink is not None:
                range_sink(None)
            return None

        # Fold the segment exactly the way commits build on each other:
        # merge the previous folded tree with the next revision's delta,
        # refresh and re-impose the cap. A revision that cannot be derived
        # this way (one produced by ``rollback``) is a reset point - the
        # fold restarts there, so the result still ends exactly equal to
        # the anchor revision rather than drifting.
        acc: dict | None = None
        previous: dict | None = None
        anchor_root: dict | None = None
        for number in merged:
            path = self._resolve_version(number, manifest)
            try:
                if info is not None and number == info["anchor"]:
                    _anchor_number, root = _decode_baseline(path)
                else:
                    _number, root = _decode_revision(path)
            except _RevisionMissing as exc:
                raise _CompactionRetry(str(exc)) from exc
            if acc is None:
                acc = copy.deepcopy(root)
            else:
                candidate = _merge_node(acc, _subtract_node(root, previous))
                _refresh_node(candidate)
                _cap_node_fields(candidate, self.max_fields)
                _refresh_node(candidate)
                acc = candidate if candidate == root else copy.deepcopy(root)
            previous = root
            anchor_root = root
        assert acc is not None and anchor_root is not None

        # The fold must reproduce the anchor revision's schema and
        # statistics exactly; nothing switches unless it does.
        if acc != anchor_root:
            raise _CompactionFatal(
                "compaction self-check failed: merged schema does not match "
                f"revision {anchor}"
            )

        self._versions_dir.mkdir(parents=True, exist_ok=True)
        staging_path = _write_staging(
            self._versions_dir, _encode_baseline(anchor, acc)
        )
        try:
            disk_anchor, disk_root = _decode_baseline(staging_path)
            if disk_anchor != anchor or disk_root != anchor_root:
                raise _CompactionFatal(
                    "compaction self-check failed: staged baseline does not "
                    f"match revision {anchor}"
                )
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(staging_path)
            raise
        return {
            "anchor": anchor,
            "merged": list(merged),
            "staging": staging_path.name,
            "old_anchor": None if info is None else info["anchor"],
        }

    def _switch_compaction(self, staged: dict) -> dict:
        """Rename the staged artifact in under the lock and switch pointer."""
        anchor = staged["anchor"]
        merged = staged["merged"]
        old_anchor = staged["old_anchor"]
        staging_name = staged["staging"]
        acquired = False
        try:
            try:
                self._lock.acquire()
            except SchemaConflict:
                raise _CompactionRetry("commit lock busy")
            acquired = True
            self._reconcile_locked(preserve=staging_name)
            staging_path = self._versions_dir / staging_name
            if not staging_path.exists():
                # A concurrent lock holder reconciled this attempt's
                # staging file away; a fresh attempt re-stages.
                raise _CompactionRetry("staging file collected before switch")
            manifest = self._read_manifest()
            info = manifest.get("baseline") if manifest else None
            current_anchor = None if info is None else info["anchor"]
            if current_anchor != old_anchor:
                # Another compaction switched while this one was staging;
                # let the next attempt fold from the new floor.
                raise _CompactionRetry(
                    "baseline pointer changed during compaction"
                )
            # The immutable inputs must still be exactly as staged; if
            # retention or a crash removed any of them the range moved.
            for number in merged:
                if number == old_anchor:
                    continue
                if not _revision_path(self._versions_dir, number).exists():
                    raise _CompactionRetry(
                        f"revision {number} disappeared during compaction"
                    )
            final_path = _baseline_path(self._versions_dir, anchor)
            # The artifact takes its real baseline name inside the lock.
            # Its unique O_EXCL staging name means it could only have been
            # removed (handled above), never replaced, so the full
            # validation done right after staging still holds.
            os.replace(staging_path, final_path)
            _fsync_dir(self._versions_dir)
            self._write_manifest_locked(
                {
                    "anchor": anchor,
                    "file": final_path.name,
                    "merged": merged,
                }
            )
            # Pointer switched: only now may the merged inputs go away -
            # the old baseline included, so exactly one baseline remains.
            for number in merged:
                if number == old_anchor:
                    continue
                try:
                    os.unlink(_revision_path(self._versions_dir, number))
                except FileNotFoundError:
                    pass
            if old_anchor is not None:
                try:
                    os.unlink(_baseline_path(self._versions_dir, old_anchor))
                except FileNotFoundError:
                    pass
            _fsync_dir(self._versions_dir)
        except _CompactionRetry:
            with contextlib.suppress(OSError):
                os.unlink(self._versions_dir / staging_name)
            if acquired:
                _fsync_dir(self._versions_dir)
            raise
        finally:
            if acquired:
                self._lock.release()
        return {"anchor": anchor, "merged": merged}


class _CompactionRetry(SchemaConflict):
    """Internal: the history moved; retry the compaction on a fresh range."""


class _CompactionFatal(SchemaConflict):
    """Internal: a compaction self-check failed; retrying cannot help."""
