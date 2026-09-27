"""Core schema inference and validation logic.

A schema is a tree of nodes. Object nodes carry a mapping of field name to
field entry; each field entry records the set of observed types, how many
records carried the field, how many records were seen at that level, and
whether the field is optional. Array nodes merge their element types.

Inference is a bounded-resource streaming process. Records are folded one
at a time and never retained; memory holds only the per-field and per-node
statistics. Two limits bound those statistics: ``field_limit`` caps how
many distinct field names one object node tracks exactly (further names
fold into an ``"other"`` bucket of per-type counts) and ``depth_limit``
caps how deep composite values are unfolded (deeper values become
``"truncated"`` nodes). Both limits default high enough that ordinary
streams stay exact, and can be set per lens or through the environment
variables ``SCHEMA_LENS_FIELD_LIMIT`` and ``SCHEMA_LENS_DEPTH_LIMIT``.

Approximation is never silent. A node that absorbed overflowed fields or
truncated values carries ``"approximate": true`` plus the reason marker
(``"other"`` or ``"truncated"``), every field entry above it carries a
propagated ``"approximate"`` flag, and ``Lens.stats`` lists the dotted
paths of every approximated site. When no limit is ever hit the folded
schema is exactly the schema produced by folding every record in one pass;
when a limit is hit, only the marked fields differ from the exact result.
An approximated field's type set can only widen, never narrow, and
observation counts grow monotonically with the records seen.

Persistence is incremental and crash safe. ``Lens.save`` serializes commits
through a lock file inside the lens directory, folds the records folded
since the last load or save into whatever schema is already stored, and
writes the result as one atomic file replacement carrying a version and a
checksum of the whole schema. The on-disk format is version 2; version 1
files are still read, migrated in memory, and re-committed as version 2 on
the next save. A failed read or migration raises SchemaConflict, leaves
the original file untouched and memory unchanged, and can be retried.
``Lens.load`` validates version, structure and checksum completely before
replacing memory, so a crashed or interrupted write can only ever leave
the previous complete schema or the next complete one, and concurrent
committers never lose each other's batches.
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
SUPPORTED_VERSIONS = (1, 2)

DEFAULT_FIELD_LIMIT = 10000
DEFAULT_DEPTH_LIMIT = 64

_KINDS = ("null", "boolean", "number", "string", "array", "object")

# Constructor keyword aliases for the two statistics limits, so callers can
# name the caps in whichever vocabulary fits their pipeline.
_FIELD_LIMIT_KEYS = (
    "field_limit",
    "max_fields",
    "max_fields_per_object",
    "field_memory_limit",
)
_DEPTH_LIMIT_KEYS = (
    "depth_limit",
    "max_depth",
    "max_nesting",
    "nesting_limit",
)


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


def _check_limit(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _env_limit(variable: str) -> int | None:
    raw = os.environ.get(variable)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"environment variable {variable} must be a positive integer, "
            f"got {raw!r}"
        ) from None
    return _check_limit(variable, value)


def _resolve_limits(overrides: dict[str, Any]) -> tuple[int, int]:
    unknown = set(overrides) - set(_FIELD_LIMIT_KEYS) - set(_DEPTH_LIMIT_KEYS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unexpected Lens keyword argument(s): {names}")
    field_limit = _env_limit("SCHEMA_LENS_FIELD_LIMIT") or DEFAULT_FIELD_LIMIT
    depth_limit = _env_limit("SCHEMA_LENS_DEPTH_LIMIT") or DEFAULT_DEPTH_LIMIT
    for key in _FIELD_LIMIT_KEYS:
        if overrides.get(key) is not None:
            field_limit = _check_limit(key, overrides[key])
    for key in _DEPTH_LIMIT_KEYS:
        if overrides.get(key) is not None:
            depth_limit = _check_limit(key, overrides[key])
    return field_limit, depth_limit


def _new_node(kind: str) -> dict:
    node: dict[str, Any] = {"kind": kind}
    if kind == "object":
        node["fields"] = {}
        node["count"] = 0
    elif kind == "array":
        node["elements"] = {}
    return node


def _new_field() -> dict:
    return {"types": {}, "observed": 0, "total": 0, "optional": False}


def _new_other() -> dict:
    return {"observed": 0, "types": {}}


def _fold_other(other: dict, value: Any) -> None:
    """Fold one overflowed field value into the per-type count bucket."""
    other["observed"] += 1
    kind = _kind_of(value)
    other["types"][kind] = other["types"].get(kind, 0) + 1


def _fold_object(
    node: dict, record: dict, depth: int, field_limit: int, depth_limit: int
) -> None:
    node["count"] += 1
    if node.get("truncated"):
        # The whole node is a depth-truncated summary; only the record
        # count above is tracked, never the record's contents.
        return
    fields = node["fields"]
    other = node.get("other")
    for name, value in record.items():
        entry = fields.get(name)
        if entry is None:
            if len(fields) >= field_limit:
                # Field cap reached: merge this name into the overflow
                # bucket instead of growing the statistics without bound.
                if other is None:
                    other = _new_other()
                    node["other"] = other
                    node["approximate"] = True
                _fold_other(other, value)
                continue
            entry = _new_field()
            fields[name] = entry
        entry["observed"] += 1
        _fold_value(entry["types"], value, depth + 1, field_limit, depth_limit)


def _fold_value(
    types: dict, value: Any, depth: int, field_limit: int, depth_limit: int
) -> None:
    kind = _kind_of(value)
    node = types.get(kind)
    if node is None:
        node = _new_node(kind)
        types[kind] = node
    if kind == "object":
        if node.get("truncated"):
            node["count"] += 1
        elif depth >= depth_limit:
            # Depth cap reached: collapse this subtree into a truncated
            # summary node. Any exact interior folded so far is dropped;
            # the node is marked approximate and only counts from here on.
            node.pop("fields", None)
            node.pop("other", None)
            node["truncated"] = True
            node["approximate"] = True
            node["count"] += 1
        else:
            _fold_object(node, value, depth, field_limit, depth_limit)
    elif kind == "array":
        if node.get("truncated"):
            node["observed"] = node.get("observed", 0) + 1
        elif depth >= depth_limit:
            node.pop("elements", None)
            node["truncated"] = True
            node["approximate"] = True
            node["observed"] = node.get("observed", 0) + 1
        else:
            elements = node["elements"]
            for element in value:
                _fold_value(elements, element, depth + 1, field_limit, depth_limit)


def _refresh_node(node: dict) -> bool:
    """Refresh derived flags and report whether the subtree is approximate.

    Object nodes get their per-field ``total``/``optional`` recomputed and
    every field entry gets an ``"approximate"`` flag propagated bottom-up
    from any marked node beneath it. Exact subtrees carry no flags at all,
    so an exact schema is byte-identical to one folded without limits.
    """
    kind = node["kind"]
    if kind == "object":
        if node.get("other") is not None or node.get("truncated"):
            node["approximate"] = True
        else:
            node.pop("approximate", None)
        approximate = bool(node.get("approximate"))
        count = node["count"]
        for entry in node.get("fields", {}).values():
            entry["total"] = count
            entry["optional"] = entry["observed"] < count
            child_approximate = False
            for child in entry["types"].values():
                if _refresh_node(child):
                    child_approximate = True
            if child_approximate:
                entry["approximate"] = True
            else:
                entry.pop("approximate", None)
            approximate = approximate or child_approximate
        return approximate
    if kind == "array":
        if node.get("truncated"):
            node["approximate"] = True
            return True
        node.pop("approximate", None)
        approximate = False
        for child in node.get("elements", {}).values():
            if _refresh_node(child):
                approximate = True
        return approximate
    return False


def _merge_other(left: dict, right: dict) -> dict:
    other = _new_other()
    other["observed"] = left["observed"] + right["observed"]
    for kind in list(left["types"]) + [
        kind for kind in right["types"] if kind not in left["types"]
    ]:
        other["types"][kind] = left["types"].get(kind, 0) + right["types"].get(kind, 0)
    return other


def _enforce_field_cap(node: dict, field_limit: int) -> None:
    """Evict exact field entries into the overflow bucket past the cap.

    Only ever runs while merging committed schemas, so repeated appends
    cannot grow one object node beyond the configured statistics limit.
    Evicted fields are not lost silently: their observations are folded
    into the ``"other"`` bucket and the node is marked approximate.
    """
    fields = node["fields"]
    if len(fields) <= field_limit:
        return
    other = node.get("other")
    if other is None:
        other = _new_other()
        node["other"] = other
    node["approximate"] = True
    for name in list(fields)[field_limit:]:
        entry = fields.pop(name)
        other["observed"] += entry["observed"]
        for kind in entry["types"]:
            other["types"][kind] = other["types"].get(kind, 0) + entry["observed"]


def _merge_node(left: dict, right: dict, field_limit: int | None = None) -> dict:
    """Fold two schema nodes over disjoint batches into one new node."""
    kind = left["kind"]
    if kind == "object":
        node = _new_node("object")
        node["count"] = left["count"] + right["count"]
        if left.get("truncated") or right.get("truncated"):
            # A truncated summary merged with anything stays a widened
            # summary: the union of both sides' interior knowledge.
            node.pop("fields", None)
            node["count"] = left.get("count", 0) + right.get("count", 0)
            node["truncated"] = True
            node["approximate"] = True
            return node
        left_fields = left.get("fields", {})
        right_fields = right.get("fields", {})
        names = list(left_fields) + [
            name for name in right_fields if name not in left_fields
        ]
        for name in names:
            le = left_fields.get(name)
            re = right_fields.get(name)
            if le is None:
                node["fields"][name] = copy.deepcopy(re)
            elif re is None:
                node["fields"][name] = copy.deepcopy(le)
            else:
                node["fields"][name] = _merge_field(le, re, field_limit)
        left_other = left.get("other")
        right_other = right.get("other")
        if left_other is not None or right_other is not None:
            if left_other is None:
                node["other"] = copy.deepcopy(right_other)
            elif right_other is None:
                node["other"] = copy.deepcopy(left_other)
            else:
                node["other"] = _merge_other(left_other, right_other)
            node["approximate"] = True
        if field_limit is not None:
            _enforce_field_cap(node, field_limit)
        return node
    if kind == "array":
        node = _new_node("array")
        if left.get("truncated") or right.get("truncated"):
            node.pop("elements", None)
            node["truncated"] = True
            node["approximate"] = True
            node["observed"] = left.get("observed", 0) + right.get("observed", 0)
            return node
        left_elements = left.get("elements", {})
        right_elements = right.get("elements", {})
        kinds = list(left_elements) + [
            element_kind
            for element_kind in right_elements
            if element_kind not in left_elements
        ]
        for element_kind in kinds:
            lc = left_elements.get(element_kind)
            rc = right_elements.get(element_kind)
            if lc is None:
                node["elements"][element_kind] = copy.deepcopy(rc)
            elif rc is None:
                node["elements"][element_kind] = copy.deepcopy(lc)
            else:
                node["elements"][element_kind] = _merge_node(lc, rc, field_limit)
        return node
    return _new_node(kind)


def _merge_field(left: dict, right: dict, field_limit: int | None = None) -> dict:
    """Fold two field entries over disjoint batches into one new entry."""
    entry = _new_field()
    entry["observed"] = left["observed"] + right["observed"]
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
            entry["types"][kind] = _merge_node(lc, rc, field_limit)
    return entry


def _subtract_node(mem: dict, base: dict) -> dict:
    """Return the part of ``mem`` folded after ``base`` was taken."""
    kind = mem["kind"]
    if kind == "object":
        node = _new_node("object")
        node["count"] = mem["count"] - base["count"]
        if mem.get("truncated"):
            node.pop("fields", None)
            node["truncated"] = True
            if node["count"] > 0:
                node["approximate"] = True
            return node
        mem_fields = mem.get("fields", {})
        base_fields = base.get("fields", {})
        for name, entry in mem_fields.items():
            base_entry = base_fields.get(name)
            if base_entry is None:
                node["fields"][name] = copy.deepcopy(entry)
            else:
                node["fields"][name] = _subtract_field(entry, base_entry)
        mem_other = mem.get("other")
        if mem_other is not None:
            base_other = base.get("other")
            if base_other is None:
                node["other"] = copy.deepcopy(mem_other)
            else:
                other = _new_other()
                other["observed"] = mem_other["observed"] - base_other["observed"]
                for kind_name, count in mem_other["types"].items():
                    other["types"][kind_name] = count - base_other["types"].get(
                        kind_name, 0
                    )
                if other["observed"] > 0:
                    node["other"] = other
                    node["approximate"] = True
        return node
    if kind == "array":
        node = _new_node("array")
        if mem.get("truncated"):
            node.pop("elements", None)
            node["truncated"] = True
            node["observed"] = mem.get("observed", 0) - base.get("observed", 0)
            if node["observed"] > 0:
                node["approximate"] = True
            return node
        mem_elements = mem.get("elements", {})
        base_elements = base.get("elements", {})
        for element_kind, child in mem_elements.items():
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
    if node.get("truncated"):
        # A truncated summary accepts any object interior; its type set was
        # deliberately widened when the depth limit collapsed the subtree.
        return
    fields = node["fields"]
    has_overflow = node.get("other") is not None
    for name in record:
        if name not in fields and not has_overflow:
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
        if node.get("truncated"):
            return
        for index, element in enumerate(value):
            _check_value(node["elements"], element, f"{path}[{index}]", reports)


def _validate_other(other: Any) -> None:
    if not isinstance(other, dict):
        raise SchemaConflict("corrupt schema file: malformed overflow bucket")
    observed = other.get("observed")
    types = other.get("types")
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed < 0
        or not isinstance(types, dict)
    ):
        raise SchemaConflict("corrupt schema file: malformed overflow bucket")
    for kind, count in types.items():
        if (
            kind not in _KINDS
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise SchemaConflict("corrupt schema file: malformed overflow bucket")


def _validate_flag(node: dict, name: str) -> None:
    if name in node and node[name] is not True:
        raise SchemaConflict(f"corrupt schema file: malformed {name} flag")


def _validate_node(node: Any, version: int = SCHEMA_VERSION) -> None:
    if not isinstance(node, dict) or node.get("kind") not in _KINDS:
        raise SchemaConflict("corrupt schema file: malformed node")
    kind = node["kind"]
    if version >= 2:
        _validate_flag(node, "approximate")
        _validate_flag(node, "truncated")
    elif "approximate" in node or "truncated" in node or "other" in node:
        raise SchemaConflict("corrupt schema file: malformed node")
    if kind == "object":
        if node.get("truncated"):
            count = node.get("count")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                or "fields" in node
                or "other" in node
            ):
                raise SchemaConflict("corrupt schema file: malformed object node")
            return
        fields = node.get("fields")
        count = node.get("count")
        if isinstance(count, bool) or not isinstance(fields, dict) or not isinstance(count, int):
            raise SchemaConflict("corrupt schema file: malformed object node")
        for entry in fields.values():
            _validate_field(entry, version)
        if "other" in node:
            if version < 2:
                raise SchemaConflict("corrupt schema file: malformed object node")
            _validate_other(node["other"])
    elif kind == "array":
        if node.get("truncated"):
            observed = node.get("observed", 0)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, int)
                or observed < 0
                or "elements" in node
            ):
                raise SchemaConflict("corrupt schema file: malformed array node")
            return
        elements = node.get("elements")
        if not isinstance(elements, dict):
            raise SchemaConflict("corrupt schema file: malformed array node")
        for child in elements.values():
            _validate_node(child, version)


def _validate_field(entry: Any, version: int = SCHEMA_VERSION) -> None:
    if not isinstance(entry, dict):
        raise SchemaConflict("corrupt schema file: malformed field entry")
    types = entry.get("types")
    observed = entry.get("observed")
    total = entry.get("total")
    optional = entry.get("optional")
    if (
        not isinstance(types, dict)
        or isinstance(observed, bool)
        or not isinstance(observed, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or not isinstance(optional, bool)
    ):
        raise SchemaConflict("corrupt schema file: malformed field entry")
    if version >= 2:
        _validate_flag(entry, "approximate")
    elif "approximate" in entry:
        raise SchemaConflict("corrupt schema file: malformed field entry")
    for child in types.values():
        _validate_node(child, version)


def _migrate_node(node: dict) -> dict:
    """Deep-copy a version 1 node into the current in-memory shape."""
    kind = node["kind"]
    migrated = _new_node(kind)
    if kind == "object":
        migrated["count"] = node["count"]
        for name, entry in node["fields"].items():
            migrated["fields"][name] = _migrate_field(entry)
    elif kind == "array":
        for element_kind, child in node["elements"].items():
            migrated["elements"][element_kind] = _migrate_node(child)
    return migrated


def _migrate_field(entry: dict) -> dict:
    migrated = _new_field()
    migrated["observed"] = entry["observed"]
    migrated["total"] = entry["total"]
    migrated["optional"] = entry["optional"]
    for kind, child in entry["types"].items():
        migrated["types"][kind] = _migrate_node(child)
    return migrated


def _migrate_root(root: dict, from_version: int) -> dict:
    """Migrate a validated older-version root to the current version.

    Version 1 schemas carry no approximation markers, so migration is a
    structural copy; the migrated tree is exactly equivalent to the stored
    one and its version 2 checksum is self-consistent.
    """
    if from_version == SCHEMA_VERSION:
        return root
    if from_version == 1:
        return _migrate_node(root)
    raise SchemaConflict(f"cannot migrate schema version: {from_version!r}")


def _checksum(root: dict) -> str:
    """Checksum of the whole schema in its canonical JSON encoding."""
    canonical = json.dumps(root, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_payload(payload: Any) -> tuple[dict, int]:
    if not isinstance(payload, dict):
        raise SchemaConflict("corrupt schema file: not a JSON object")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise SchemaConflict(f"unsupported schema version: {version!r}")
    if version not in SUPPORTED_VERSIONS:
        raise SchemaConflict(f"unsupported schema version: {version!r}")
    checksum = payload.get("checksum")
    if not isinstance(checksum, str):
        raise SchemaConflict("corrupt schema file: missing checksum")
    if "root" not in payload:
        raise SchemaConflict("corrupt schema file: missing root")
    root = payload["root"]
    _validate_node(root, version)
    if not isinstance(root, dict) or root.get("kind") != "object":
        raise SchemaConflict("corrupt schema file: root must be an object")
    if _checksum(root) != checksum:
        raise SchemaConflict("corrupt schema file: checksum mismatch")
    return root, version


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

    ``field_limit`` caps how many distinct field names one object node
    tracks exactly; ``depth_limit`` caps how deep composite values are
    unfolded. Both default to ``SCHEMA_LENS_FIELD_LIMIT`` /
    ``SCHEMA_LENS_DEPTH_LIMIT`` from the environment, then to generous
    built-in defaults under which folds stay exact.
    """

    def __init__(
        self,
        path: os.PathLike | str,
        field_limit: int | None = None,
        depth_limit: int | None = None,
        **limit_overrides: Any,
    ) -> None:
        if field_limit is not None:
            limit_overrides["field_limit"] = field_limit
        if depth_limit is not None:
            limit_overrides["depth_limit"] = depth_limit
        self._field_limit, self._depth_limit = _resolve_limits(limit_overrides)
        self.path = Path(path)
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

        Records are consumed one at a time and never retained, so any
        iterable — including a generator over a multi-million-line file —
        folds in bounded memory.
        """
        if self._schema is None:
            self._schema = _new_node("object")
            self._base = _new_node("object")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("records must be JSON objects")
            _fold_object(self._schema, record, 0, self._field_limit, self._depth_limit)
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

        The ``"approximate"`` list names every field whose statistics were
        approximated — by overflow past the field limit or by truncation
        past the depth limit — using the same path syntax as check reports.
        """
        schema = self._require()
        summary: dict[str, Any] = {
            "records": schema["count"],
            "fields": 0,
            "optional": 0,
            "types": {},
            "observations": 0,
            "approximate": [],
        }

        def walk_object(node: dict, path: str) -> None:
            other = node.get("other")
            if other is not None:
                summary["observations"] += other["observed"]
                for kind in other["types"]:
                    summary["types"][kind] = summary["types"].get(kind, 0) + 1
            for name, entry in node.get("fields", {}).items():
                field_path = _join(path, name)
                summary["fields"] += 1
                summary["observations"] += entry["observed"]
                if entry["optional"]:
                    summary["optional"] += 1
                for kind, child in entry["types"].items():
                    summary["types"][kind] = summary["types"].get(kind, 0) + 1
                    walk_node(child, field_path)

        def walk_node(node: dict, path: str) -> None:
            # The node itself carrying the marker names the precise site of
            # approximation (its own overflow bucket or truncation); merely
            # having an approximate descendant is implied by the paths
            # beneath it and is not listed again at this level.
            if node.get("approximate") and path not in summary["approximate"]:
                summary["approximate"].append(path)
            if node["kind"] == "object":
                if node.get("truncated"):
                    return
                walk_object(node, path)
            elif node["kind"] == "array":
                if node.get("truncated"):
                    return
                for child in node.get("elements", {}).values():
                    walk_node(child, path + "[]")

        if schema.get("approximate"):
            summary["approximate"].append("$")
        walk_object(schema, "$")
        return summary

    def _read_disk(self) -> dict | None:
        """Read, fully validate and migrate the stored schema.

        Returns ``None`` only when the lens directory holds no schema yet.
        Any version, structure, checksum or migration failure raises
        SchemaConflict; the file on disk is never touched here.
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
        root, version = _validate_payload(payload)
        try:
            return _migrate_root(root, version)
        except SchemaConflict as exc:
            raise SchemaConflict(
                f"cannot migrate schema file {self._file}: {exc}"
            ) from exc

    def save(self) -> None:
        """Commit the stored schema into the lens directory.

        The commit is serialized by a non-blocking lock file in the
        directory; contention raises SchemaConflict. Under the lock the
        stored schema is re-read, fully validated and migrated to the
        current version, the records folded since the last load or save are
        merged in, and the result is written as one atomic file replacement
        carrying a version and a checksum. Memory is replaced only after
        the write lands, so a failed commit or migration leaves the
        in-memory schema untouched, the original file is never modified by
        a failed save, and the same save can simply be retried later.
        """
        self._require()
        self.path.mkdir(parents=True, exist_ok=True)
        with self._lock.held():
            disk = self._read_disk()
            if disk is None:
                merged = copy.deepcopy(self._schema)
            else:
                delta = _subtract_node(self._schema, self._base)
                merged = _merge_node(disk, delta, self._field_limit)
            _refresh_node(merged)
            _write_payload(self._file, merged)
        self._schema = merged
        self._base = copy.deepcopy(merged)

    def load(self) -> None:
        """Re-read the stored schema, replacing memory only on success.

        The file is parsed, fully validated against its version and
        checksum, and migrated to the current in-memory version before
        anything changes; any failure raises SchemaConflict and leaves
        memory exactly as it was.
        """
        root = self._read_disk()
        if root is None:
            raise SchemaConflict(
                f"cannot read schema file {self._file}: no such file"
            )
        self._schema = root
        self._base = copy.deepcopy(root)
