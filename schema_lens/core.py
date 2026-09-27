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

Persistence is incremental and crash safe. ``Lens.save`` serializes commits
through a lock file inside the lens directory, folds the records folded
since the last load or save into whatever schema is already stored, and
writes the result as one atomic file replacement carrying a version and a
checksum of the whole schema. ``Lens.load`` validates version, structure and
checksum completely before replacing memory, so a crashed or interrupted
write can only ever leave the previous complete schema or the next complete
one, and concurrent committers never lose each other's batches. Files
written by an older version are migrated to the current version
atomically on read; a failed migration leaves the original file and the
in-memory schema untouched and raises ``SchemaConflict`` so it can be
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
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

SCHEMA_FILENAME = "schema.json"
LOCK_FILENAME = "schema.lock"
SCHEMA_VERSION = 2
# Version 1 files predate approximation markers; their schemas are valid
# version 2 schemas as-is, so migration only re-encodes the payload.
READABLE_VERSIONS = (1, SCHEMA_VERSION)

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
        raise SchemaConflict("corrupt schema file: malformed node")
    approximate = node.get("approximate", False)
    if not isinstance(approximate, bool):
        raise SchemaConflict("corrupt schema file: malformed node")
    kind = node["kind"]
    if kind == "object":
        fields = node.get("fields")
        count = node.get("count")
        if isinstance(count, bool) or not isinstance(count, int):
            raise SchemaConflict("corrupt schema file: malformed object node")
        if fields is None:
            # Only a collapsed (approximate) object node may omit fields.
            if not approximate:
                raise SchemaConflict("corrupt schema file: malformed object node")
            fields = {}
        if not isinstance(fields, dict):
            raise SchemaConflict("corrupt schema file: malformed object node")
        for entry in fields.values():
            _validate_field(entry)
        overflow = node.get("overflow")
        if overflow is not None:
            _validate_field(overflow)
    elif kind == "array":
        elements = node.get("elements")
        if not isinstance(elements, dict):
            raise SchemaConflict("corrupt schema file: malformed array node")
        for child in elements.values():
            _validate_node(child)


def _validate_field(entry: Any) -> None:
    if not isinstance(entry, dict):
        raise SchemaConflict("corrupt schema file: malformed field entry")
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
        raise SchemaConflict("corrupt schema file: malformed field entry")
    for child in types.values():
        _validate_node(child)


def _checksum(root: dict) -> str:
    """Checksum of the whole schema in its canonical JSON encoding."""
    canonical = json.dumps(root, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_payload(payload: Any) -> tuple[int, dict]:
    if not isinstance(payload, dict):
        raise SchemaConflict("corrupt schema file: not a JSON object")
    version = payload.get("version")
    if version not in READABLE_VERSIONS:
        raise SchemaConflict(f"unsupported schema version: {version!r}")
    checksum = payload.get("checksum")
    if not isinstance(checksum, str):
        raise SchemaConflict("corrupt schema file: missing checksum")
    if "root" not in payload:
        raise SchemaConflict("corrupt schema file: missing root")
    root = payload["root"]
    _validate_node(root)
    if not isinstance(root, dict) or root.get("kind") != "object":
        raise SchemaConflict("corrupt schema file: root must be an object")
    if _checksum(root) != checksum:
        raise SchemaConflict("corrupt schema file: checksum mismatch")
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


def _write_payload(file_path: Path, root: dict) -> None:
    """Write the payload atomically: a crash leaves only the old or new file."""
    payload = {"version": SCHEMA_VERSION, "checksum": _checksum(root), "root": root}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
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
    """Opens the lens directory ``path`` and manages its stored schema.

    ``max_fields`` caps how many field entries each object node tracks
    exactly; further distinct names fold into a shared overflow entry
    flagged ``"approximate": true`` under the node's ``"overflow"`` key.
    ``max_depth`` caps how deep object and array nodes are tracked (the
    root is depth 0); deeper subtrees collapse into summary nodes flagged
    ``"approximate": true``. Both default to ``None``, which keeps every
    statistic exact and the folded schema identical to folding all records
    in one pass. Records themselves are never retained, so peak memory
    tracks these limits rather than the number of records folded.
    """

    def __init__(
        self,
        path: os.PathLike | str,
        *,
        max_fields: int | None = None,
        max_depth: int | None = None,
    ) -> None:
        for label, value in (("max_fields", max_fields), ("max_depth", max_depth)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{label} must be a non-negative integer or None")
        self.path = Path(path)
        self.max_fields = max_fields
        self.max_depth = max_depth
        self._schema: dict | None = None
        self._base: dict | None = None
        self._lock = _DirectoryLock(self.path)

    @property
    def _file(self) -> Path:
        return self.path / SCHEMA_FILENAME

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

    def check(self, record: Any) -> list[str]:
        """Report every way one record departs from the stored schema."""
        schema = self._require()
        if not isinstance(record, dict):
            return [f"$: type {_kind_of(record)} not in field types [object]"]
        reports: list[str] = []
        _check_object(schema, record, "$", reports)
        return reports

    def schema(self) -> dict:
        """Return the stored schema."""
        return copy.deepcopy(self._require())

    def stats(self) -> dict:
        """Report fields, optional fields, observed types and counts.

        The ``"approximate"`` key lists the paths of every statistic that
        has been approximated under the resource limits: collapsed
        subtrees by their own path and overflow entries as ``$["*"]`` under
        their object node's path. It is empty when nothing was approximated.
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
                walk_entry(overflow, _join(path, "*"))

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

    def _read_disk(self) -> dict | None:
        """Read, validate and migrate the stored schema.

        Returns ``None`` only when the lens directory holds no schema yet.
        A file written under an older supported version is migrated to the
        current version by one atomic replacement before the schema is
        handed out; if the migration write fails the original file is left
        exactly as it was and SchemaConflict is raised, so the caller's
        in-memory schema is never replaced by a half-migrated state and
        the read can simply be retried later.
        """
        try:
            raw = self._file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SchemaConflict(f"cannot read schema file {self._file}: {exc}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaConflict(f"corrupt schema file {self._file}: {exc}") from exc
        version, root = _validate_payload(payload)
        if version != SCHEMA_VERSION:
            try:
                _write_payload(self._file, root)
            except OSError as exc:
                raise SchemaConflict(
                    f"cannot migrate schema file {self._file} to version "
                    f"{SCHEMA_VERSION}: {exc}"
                ) from exc
        return root

    def save(self) -> None:
        """Commit the stored schema into the lens directory.

        The commit is serialized by a non-blocking lock file in the
        directory; contention raises SchemaConflict. Under the lock the
        stored schema is re-read and fully validated, the records folded
        since the last load or save are merged in, and the result is written
        as one atomic file replacement carrying a version and a checksum.
        Memory is replaced only after the write lands, so a failed commit
        leaves the in-memory schema untouched and earlier commits are never
        overwritten.
        """
        self._require()
        self.path.mkdir(parents=True, exist_ok=True)
        with self._lock.held():
            disk = self._read_disk()
            if disk is None:
                merged = copy.deepcopy(self._schema)
            else:
                delta = _subtract_node(self._schema, self._base)
                merged = _merge_node(disk, delta)
            _refresh_node(merged)
            _write_payload(self._file, merged)
        self._schema = merged
        self._base = copy.deepcopy(merged)

    def load(self) -> None:
        """Re-read the stored schema, replacing memory only on success."""
        root = self._read_disk()
        if root is None:
            raise SchemaConflict(
                f"cannot read schema file {self._file}: no such file"
            )
        self._schema = root
        self._base = copy.deepcopy(root)
