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
exactly the schema produced by folding every record in one pass.

Persistence keeps multiple complete versions inside the lens directory's
``versions`` subdirectory, one immutable file per commit named by its
sequential revision number. ``Lens.save`` serializes commits through a
lock file, folds the records folded since the last load or save into
whatever complete revision is stored at the head, and writes the merged
schema as a brand-new revision file atomically (temp file, fsync, rename,
directory fsync). A crash or interrupted commit can therefore only ever
leave a complete earlier revision or a complete later one; a partial temp
file is never published under a revision name, so any read during
concurrent commits sees some complete revision and never a half-written
one. Each revision carries a format version and a checksum of the whole
schema and is validated completely before anything is handed out or used
as a merge base; a missing or corrupt requested revision raises
``SchemaConflict`` while the other revisions stay usable.

Only the most recent ``keep_versions`` complete revisions are retained;
older ones are compacted away (oldest first) after a successful commit.
``Lens.versions`` lists the revisions still present, oldest first, and
``Lens.rollback`` expresses a rollback as a fresh commit whose schema
equals an earlier revision's -- history is appended to, never rewritten,
and rolling back to a revision that is absent (or corrupt) raises
``SchemaConflict`` without touching any committed revision. Reads may
pin a revision number or take a snapshot of the head at open/load time.

The pre-multi-version single-file layout (``schema.json`` beside the
lock) is still read; the first read migrates it in place into revision 1
under ``versions`` and removes the legacy file, all under the commit
lock. A failed migration leaves the legacy file exactly as it was and
raises ``SchemaConflict`` so the read can be retried later.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

LEGACY_FILENAME = "schema.json"
VERSIONS_DIRNAME = "versions"
LOCK_FILENAME = "schema.lock"
# Kept as SCHEMA_FILENAME for backwards-compatible imports; it names the
# legacy single-file layout that migration consumes.
SCHEMA_FILENAME = LEGACY_FILENAME
SCHEMA_VERSION = 2
# Version 1 files predate approximation markers; their schemas are valid
# version 2 schemas as-is, so migration only re-encodes the payload.
READABLE_VERSIONS = (1, SCHEMA_VERSION)

_DEFAULT_KEEP_VERSIONS = 10
_REVISION_SUFFIX = ".json"
_REVISION_RE = re.compile(r"(\d+)" + re.escape(_REVISION_SUFFIX) + r"$")
# Overflow statistics are located by the bracketed, quoted marker $["*"];
# it uses the same quoting as a literal field name and therefore collides
# with no join that the dot form $.* could, while a real field named "*"
# renders identically and is still located unambiguously (the fields
# mapping and the overflow entry are distinct in the tree).
OVERFLOW_MARKER = '*'

_KINDS = ("null", "boolean", "number", "string", "array", "object")


class SchemaConflict(Exception):
    """Raised when an operation conflicts with the stored schema state."""


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


def _enforce_field_cap(node: dict, max_fields: int | None) -> None:
    """Rebuild one object node so it tracks at most ``max_fields`` entries.

    Folding only ever adds exact field entries, so a node merged from two
    batches can hold more exact names than this lens's cap. The names that
    no longer fit (folded in first-seen order, matching the fold's own
    overflow policy) have their entries merged into the node's overflow
    entry and are flagged approximate; capped subtrees are rebuilt the same
    way recursively.
    """
    if node["kind"] == "object":
        if _is_collapsed(node):
            return
        for child in node["fields"].values():
            for subnode in child["types"].values():
                _enforce_field_cap(subnode, max_fields)
        overflow = node.get("overflow")
        if max_fields is not None and len(node["fields"]) > max_fields:
            names = list(node["fields"])
            keep = names[:max_fields]
            spill = names[max_fields:]
            fields = {name: node["fields"][name] for name in keep}
            for name in spill:
                entry = node["fields"][name]
                if overflow is None:
                    overflow = _new_field()
                else:
                    overflow = _merge_field(overflow, entry)
                overflow["approximate"] = True
            node["fields"] = fields
            if overflow is not None:
                node["overflow"] = overflow
        if overflow is not None:
            for subnode in overflow["types"].values():
                _enforce_field_cap(subnode, max_fields)
    elif node["kind"] == "array":
        for child in node["elements"].values():
            _enforce_field_cap(child, max_fields)


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
                # be marked approximate.
                if left_overflow is not None:
                    entry = _merge_field(left_overflow, re)
                    entry["approximate"] = True
                else:
                    entry = copy.deepcopy(re)
            elif re is None:
                if right_overflow is not None:
                    entry = _merge_field(le, right_overflow)
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
    for kind, child in mem["types"].items():
        base_child = base["types"].get(kind)
        if base_child is None:
            entry["types"][kind] = copy.deepcopy(child)
        else:
            entry["types"][kind] = _subtract_node(child, base_child)
    return entry


def _join(path: str, name: str) -> str:
    # Every key is bracketed and JSON-quoted, including ordinary
    # identifiers: $["a.b"] stays distinct from $["a"]["b"], and the
    # approximate-overflow marker $["*"] is the same shape as a literal
    # field named "*" and so can never be confused with a path separator or
    # a real name -- the overflow entry lives under "overflow", not in
    # "fields", and is located by its marker alone.
    return f"{path}[{json.dumps(name)}]"


def _check_object(node: dict, record: dict, path: str, reports: list[str]) -> None:
    if _is_collapsed(node):
        # A depth-capped node tracked no fields, so nothing inside the
        # record's value can be checked; the approximation is marked in
        # the schema instead of guessed about here.
        return
    fields = node["fields"]
    overflow = node.get("overflow")
    for name in record:
        if name not in fields:
            if overflow is not None:
                # Names beyond the field cap were folded into the overflow
                # entry; check the value against its widened type set
                # rather than reporting the field as unexpected.
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


def _validate_node(node: Any) -> None:
    if not isinstance(node, dict) or node.get("kind") not in _KINDS:
        raise SchemaConflict("corrupt schema: malformed node")
    approximate = node.get("approximate", False)
    if not isinstance(approximate, bool):
        raise SchemaConflict("corrupt schema: malformed node")
    kind = node["kind"]
    if kind == "object":
        fields = node.get("fields")
        count = node.get("count")
        if isinstance(count, bool) or not isinstance(count, int):
            raise SchemaConflict("corrupt schema: malformed object node")
        if fields is None:
            # Only a collapsed (approximate) object node may omit fields.
            if not approximate:
                raise SchemaConflict("corrupt schema: malformed object node")
            fields = {}
        if not isinstance(fields, dict):
            raise SchemaConflict("corrupt schema: malformed object node")
        for entry in fields.values():
            _validate_field(entry)
        overflow = node.get("overflow")
        if overflow is not None:
            _validate_field(overflow)
    elif kind == "array":
        elements = node.get("elements")
        if not isinstance(elements, dict):
            raise SchemaConflict("corrupt schema: malformed array node")
        for child in elements.values():
            _validate_node(child)


def _validate_field(entry: Any) -> None:
    if not isinstance(entry, dict):
        raise SchemaConflict("corrupt schema: malformed field entry")
    types = entry.get("types")
    observed = entry.get("observed")
    total = entry.get("total")
    optional = entry.get("optional")
    approximate = entry.get("approximate", False)
    if (
        not isinstance(types, dict)
        or isinstance(observed, bool)
        or not isinstance(observed, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or not isinstance(optional, bool)
        or not isinstance(approximate, bool)
    ):
        raise SchemaConflict("corrupt schema: malformed field entry")
    for child in types.values():
        _validate_node(child)


def _checksum(root: dict) -> str:
    """Checksum of the whole schema in its canonical JSON encoding."""
    canonical = json.dumps(root, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_payload(payload: Any) -> tuple[int, dict]:
    if not isinstance(payload, dict):
        raise SchemaConflict("corrupt schema: not a JSON object")
    version = payload.get("version")
    if version not in READABLE_VERSIONS:
        raise SchemaConflict(f"unsupported schema version: {version!r}")
    checksum = payload.get("checksum")
    if not isinstance(checksum, str):
        raise SchemaConflict("corrupt schema: missing checksum")
    if "root" not in payload:
        raise SchemaConflict("corrupt schema: missing root")
    root = payload["root"]
    _validate_node(root)
    if not isinstance(root, dict) or root.get("kind") != "object":
        raise SchemaConflict("corrupt schema: root must be an object")
    if _checksum(root) != checksum:
        raise SchemaConflict("corrupt schema: checksum mismatch")
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


def _encode(root: dict) -> str:
    payload = {"version": SCHEMA_VERSION, "checksum": _checksum(root), "root": root}
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_atomic(file_path: Path, text: str) -> None:
    """Write a file atomically: a crash leaves only the old or new file."""
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


def _write_payload(file_path: Path, root: dict) -> None:
    _write_atomic(file_path, _encode(root))


def _revision_path(directory: Path, revision: int) -> Path:
    return directory / VERSIONS_DIRNAME / f"{revision:08d}{_REVISION_SUFFIX}"


def _revision_number(name: str) -> int | None:
    match = _REVISION_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


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
    """Opens the lens directory ``path`` and manages its stored versions.

    ``max_fields`` caps how many field entries each object node tracks
    exactly; further distinct names fold into a shared overflow entry
    flagged ``"approximate": true`` under the node's ``"overflow"`` key.
    ``max_depth`` caps how deep object and array nodes are tracked (the
    root is depth 0); deeper subtrees collapse into summary nodes flagged
    ``"approximate": true``. Both default to ``None``, which keeps every
    statistic exact and the folded schema identical to folding all records
    in one pass. Records themselves are never retained, so peak memory
    tracks these limits rather than the number of records folded.

    ``keep_versions`` bounds how many complete revisions the lens
    directory retains; after each commit the oldest revisions beyond the
    bound are compacted away. It must be a positive integer.
    """

    def __init__(
        self,
        path: os.PathLike | str,
        *,
        max_fields: int | None = None,
        max_depth: int | None = None,
        keep_versions: int = _DEFAULT_KEEP_VERSIONS,
    ) -> None:
        for label, value in (("max_fields", max_fields), ("max_depth", max_depth)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{label} must be a non-negative integer or None")
        if isinstance(keep_versions, bool) or not isinstance(keep_versions, int) or keep_versions < 1:
            raise ValueError("keep_versions must be a positive integer")
        self.path = Path(path)
        self.max_fields = max_fields
        self.max_depth = max_depth
        self.keep_versions = keep_versions
        self._schema: dict | None = None
        self._base: dict | None = None
        # The revision the in-memory schema was snapshotted from, used to
        # resolve "the snapshot taken at open/load time". None until a
        # revision has been loaded or committed.
        self._revision: int | None = None
        self._lock = _DirectoryLock(self.path)

    @property
    def _versions_dir(self) -> Path:
        return self.path / VERSIONS_DIRNAME

    @property
    def _legacy_file(self) -> Path:
        return self.path / LEGACY_FILENAME

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

        With ``version=None`` (the default) the in-memory schema is used,
        which is either the snapshot loaded at open/``load`` time extended
        by batches inferred in this process, or the last committed
        snapshot. With a revision number the check runs against that
        stored complete revision without altering memory; a missing or
        corrupt revision raises ``SchemaConflict``.
        """
        if version is None:
            schema = self._require()
        else:
            schema = self.read_version(version)
        if not isinstance(record, dict):
            return [f"$: type {_kind_of(record)} not in field types [object]"]
        reports: list[str] = []
        _check_object(schema, record, "$", reports)
        return reports

    def schema(self, *, version: int | None = None) -> dict:
        """Return the stored schema.

        ``version=None`` returns the in-memory schema (the open/load
        snapshot plus inferred batches). A revision number returns a deep
        copy of that complete stored revision and leaves memory untouched;
        a missing or corrupt revision raises ``SchemaConflict``.
        """
        if version is None:
            return copy.deepcopy(self._require())
        return copy.deepcopy(self.read_version(version))

    def stats(self, *, version: int | None = None) -> dict:
        """Report fields, optional fields, observed types and counts.

        The ``"approximate"`` key lists the paths of every statistic that
        has been approximated under the resource limits: collapsed
        subtrees by their own path and overflow entries as ``$["*"]`` under
        their object node's path. It is empty when nothing was approximated.

        With ``version=None`` the in-memory snapshot is summarized; a
        revision number summarizes that complete stored revision without
        altering memory.
        """
        schema = self._require() if version is None else self.read_version(version)
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
                walk_entry(overflow, _join(path, OVERFLOW_MARKER))

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

    # ------------------------------------------------------------------
    # Stored revisions
    # ------------------------------------------------------------------

    def _list_revision_files(self) -> list[tuple[int, Path]]:
        try:
            names = os.listdir(self._versions_dir)
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise SchemaConflict(
                f"cannot read versions directory {self._versions_dir}: {exc}"
            ) from exc
        revisions: list[tuple[int, Path]] = []
        for name in names:
            number = _revision_number(name)
            if number is not None:
                revisions.append((number, self._versions_dir / name))
        revisions.sort(key=lambda item: item[0])
        return revisions

    def _read_revision_file(self, path: Path) -> dict:
        """Read and fully validate one revision file.

        The whole payload (format version, structure and checksum) is
        checked before the schema is returned; nothing partial is ever
        handed out.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SchemaConflict(f"schema version not found: {path.name}") from exc
        except OSError as exc:
            raise SchemaConflict(f"cannot read schema version {path.name}: {exc}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaConflict(f"corrupt schema version {path.name}: {exc}") from exc
        try:
            _format_version, root = _validate_payload(payload)
        except SchemaConflict as exc:
            raise SchemaConflict(
                f"corrupt schema version {path.name}: {exc}"
            ) from exc
        return root

    def read_version(self, revision: int) -> dict:
        """Return a validated copy of one stored revision.

        Raises ``SchemaConflict`` when the revision is absent, unreadable
        or corrupt; other revisions are unaffected.
        """
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise SchemaConflict(f"no such schema version: {revision!r}")
        path = _revision_path(self.path, revision)
        return copy.deepcopy(self._read_revision_file(path))

    def versions(self) -> list[int]:
        """List the complete revisions still retained, oldest first.

        Returns an empty list when the lens directory holds no history.
        Revision files are only ever published by an atomic rename of a
        fully written temp file, so every listed revision is complete; a
        stray temp file is never listed.
        """
        return [number for number, _path in self._list_revision_files()]

    def head(self) -> tuple[int, dict] | None:
        """Return ``(revision, root)`` for the newest complete revision."""
        revisions = self._list_revision_files()
        if not revisions:
            return None
        number, path = revisions[-1]
        return number, self._read_revision_file(path)

    def _migrate_legacy_file(self) -> dict | None:
        """Consume the old single-file layout under the caller's lock.

        Reads and fully validates ``schema.json`` (migrating its format
        version in memory), publishes it as revision 1 with an atomic
        rename, then unlinks the legacy file and fsyncs the directory. Any
        failure leaves the legacy file byte for byte untouched and raises
        ``SchemaConflict``; no revision file is left behind on failure.
        Returns the migrated root, or ``None`` when no legacy file exists.
        """
        legacy = self._legacy_file
        try:
            raw = legacy.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SchemaConflict(f"cannot read schema file {legacy}: {exc}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaConflict(f"corrupt schema file {legacy}: {exc}") from exc
        _format_version, root = _validate_payload(payload)

        versions_dir = self._versions_dir
        versions_dir.mkdir(parents=True, exist_ok=True)
        target = _revision_path(self.path, 1)
        try:
            _write_atomic(target, _encode(root))
            os.unlink(legacy)
        except OSError as exc:
            # The migration write is atomic: a failed rename/publish means
            # the legacy file is still the only schema. Drop any revision 1
            # temp artifact so a later read starts clean, and leave the
            # legacy file exactly as it was.
            try:
                if target.exists():
                    os.unlink(target)
            except OSError:
                pass
            raise SchemaConflict(
                f"cannot migrate schema file {legacy} into {versions_dir}: {exc}"
            ) from exc
        _fsync_dir(self.path)
        return root

    def _ensure_history(self) -> tuple[int, dict] | None:
        """Return the head after migrating any legacy single-file store.

        Must be called holding the commit lock. When neither a legacy file
        nor any revision exists, returns ``None``.
        """
        legacy = self._migrate_legacy_file()
        head_state = self.head()
        if head_state is not None:
            return head_state
        if legacy is not None:
            return 1, legacy
        return None

    def _prune(self, retained: list[tuple[int, Path]]) -> None:
        """Delete the oldest revisions beyond ``keep_versions``."""
        excess = len(retained) - self.keep_versions
        if excess <= 0:
            return
        for _number, path in retained[:excess]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise SchemaConflict(
                    f"cannot compact old schema version {path.name}: {exc}"
                ) from exc
        _fsync_dir(self._versions_dir)

    def save(self) -> None:
        """Commit the stored schema as a new complete revision.

        The commit is serialized by a non-blocking lock file in the
        directory; contention raises ``SchemaConflict`` and the in-memory
        batch is kept so the same ``save`` can be retried. Under the lock
        the head revision is re-read and fully validated (a legacy
        single-file store is migrated to revision 1 first), the records
        folded since the last load or save are merged in, the merged tree
        is re-capped to this lens's field limit, and the result is written
        to a brand-new revision file with an atomic rename and a checksum.
        Memory is replaced only after the revision lands, so a failed
        commit leaves both memory and every existing revision untouched.
        Revisions beyond ``keep_versions`` are compacted afterwards,
        oldest first.
        """
        self._require()
        self.path.mkdir(parents=True, exist_ok=True)
        with self._lock.held():
            head_state = self._ensure_history()
            if head_state is None:
                merged = copy.deepcopy(self._schema)
                next_revision = 1
            else:
                head_revision, disk = head_state
                delta = _subtract_node(self._schema, self._base)
                merged = _merge_node(disk, delta)
                next_revision = head_revision + 1
            _refresh_node(merged)
            _enforce_field_cap(merged, self.max_fields)
            _refresh_node(merged)
            target = _revision_path(self.path, next_revision)
            self._versions_dir.mkdir(parents=True, exist_ok=True)
            _write_atomic(target, _encode(merged))
            # The rename above published a complete revision; only now
            # adopt it in memory and compact the oldest extras.
            retained = self._list_revision_files()
            self._prune(retained)
        self._schema = merged
        self._base = copy.deepcopy(merged)
        self._revision = next_revision

    def load(self, *, version: int | None = None) -> None:
        """Read a complete revision, replacing memory only on success.

        With ``version=None`` the newest retained revision is read (and a
        legacy single-file store migrated into it); pin a revision number
        to open that exact snapshot instead. Either way the revision is
        fully validated before memory changes, and the pinned revision
        becomes the snapshot that later appends build on. A missing or
        corrupt revision -- or an empty history -- raises
        ``SchemaConflict`` and leaves memory untouched.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        with self._lock.held():
            if version is None:
                head_state = self._ensure_history()
                if head_state is None:
                    raise SchemaConflict(
                        f"no schema versions in {self._versions_dir}"
                    )
                revision, root = head_state
            else:
                if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                    raise SchemaConflict(f"no such schema version: {version!r}")
                root = self._read_revision_file(_revision_path(self.path, version))
                revision = version
        self._schema = root
        self._base = copy.deepcopy(root)
        self._revision = revision

    def rollback(self, revision: int) -> int:
        """Roll back to an earlier revision by committing a new one.

        The stored schema of ``revision`` is validated and republished as
        the next revision, so history is only ever appended to -- no
        existing revision is rewritten or deleted. Returns the new
        revision number. Rolling back to a revision that is missing or
        corrupt raises ``SchemaConflict`` and changes no committed
        revision. The in-memory snapshot becomes the new revision.
        """
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise SchemaConflict(f"no such schema version: {revision!r}")
        self.path.mkdir(parents=True, exist_ok=True)
        with self._lock.held():
            head_state = self._ensure_history()
            target = _revision_path(self.path, revision)
            root = self._read_revision_file(target)
            if head_state is None:
                next_revision = 1
            else:
                next_revision = head_state[0] + 1
            published = _revision_path(self.path, next_revision)
            self._versions_dir.mkdir(parents=True, exist_ok=True)
            _write_atomic(published, _encode(root))
            retained = self._list_revision_files()
            self._prune(retained)
        self._schema = root
        self._base = copy.deepcopy(root)
        self._revision = next_revision
        return next_revision
