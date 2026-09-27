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
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import threading
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
DEFAULT_MAX_VERSIONS = 10
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
    if (
        not isinstance(types, dict)
        or isinstance(observed, bool)
        or not isinstance(observed, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or not isinstance(optional, bool)
        or not isinstance(approximate, bool)
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


def _decode_revision(file_path: Path) -> tuple[int, dict]:
    """Read one revision file and validate it completely before use."""
    try:
        raw = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SchemaConflict(f"schema revision not found: {file_path.name}") from exc
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
    """

    def __init__(
        self,
        path: os.PathLike | str,
        *,
        max_fields: int | None = None,
        max_depth: int | None = None,
        max_versions: int = DEFAULT_MAX_VERSIONS,
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
        self.path = Path(path)
        self.max_fields = max_fields
        self.max_depth = max_depth
        self.max_versions = max_versions
        self._schema: dict | None = None
        self._base: dict | None = None
        self._revision: int | None = None
        self._lock = _DirectoryLock(self.path)

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

    def _read_revision(self, revision: int) -> dict:
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise SchemaConflict(f"unknown schema revision: {revision!r}")
        target = _revision_path(self._versions_dir, revision)
        if not target.exists() and not self._scan_revisions():
            # No revision store yet: an old single-file layout may carry
            # the schema; migrate it in place, then honor the request.
            self._migrate_legacy()
        return _decode_revision(target)[1]

    def _read_newest(self) -> tuple[int, dict] | None:
        numbers = self._scan_revisions()
        if not numbers:
            return None
        # A revision file only appears by an atomic rename of a fully
        # synced file, so the newest numbered file is complete or absent;
        # if its content does not validate the stored state is corrupt and
        # the caller gets SchemaConflict instead of an older schema.
        revision = numbers[-1]
        return _decode_revision(_revision_path(self._versions_dir, revision))

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
        """Committed revision numbers, oldest first; empty before any commit."""
        return self._scan_revisions()

    def save(self) -> None:
        """Commit the stored schema into the lens directory.

        The commit is serialized by a non-blocking lock file in the
        directory; contention raises SchemaConflict and the in-memory
        batch stays retryable. Under the lock the newest complete
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
        1`` are removed; the newest ``max_versions`` revisions and any
        other file in the directory are left alone.
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
        given that exact revision is read and fully validated. A missing
        revision or a corrupt file raises ``SchemaConflict`` and leaves
        memory exactly as it was.
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
        revision that does not exist or fails validation raises
        ``SchemaConflict`` and changes no committed revision. Returns the
        new revision number; the in-memory snapshot adopts it.
        """
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise SchemaConflict(f"unknown schema revision: {version!r}")
        self.path.mkdir(parents=True, exist_ok=True)
        with self._lock.held():
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
