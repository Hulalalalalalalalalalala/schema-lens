"""Core schema inference and validation logic.

A schema is a tree of nodes. Object nodes carry a mapping of field name to
field entry; each field entry records the set of observed types, how many
records carried the field, how many records were seen at that level, and
whether the field is optional. Array nodes merge their element types.

Inference is incremental: every ``infer`` call appends one batch of records
to the stored schema, and ``save`` commits the pending batch to the lens
directory as a single atomic, checksummed write serialized through a lock
file, so concurrent processes never lose or overwrite committed batches.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
    import msvcrt

SCHEMA_FILENAME = "schema.json"
LOCK_FILENAME = SCHEMA_FILENAME + ".lock"
SCHEMA_VERSION = 1

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


def _new_field() -> dict:
    return {"types": {}, "observed": 0, "total": 0, "optional": False}


def _fold_object(node: dict, record: dict) -> None:
    node["count"] += 1
    fields = node["fields"]
    for name, value in record.items():
        entry = fields.get(name)
        if entry is None:
            entry = _new_field()
            fields[name] = entry
        entry["observed"] += 1
        _fold_value(entry["types"], value)


def _fold_value(types: dict, value: Any) -> None:
    kind = _kind_of(value)
    node = types.get(kind)
    if node is None:
        node = _new_node(kind)
        types[kind] = node
    if kind == "object":
        _fold_object(node, value)
    elif kind == "array":
        for element in value:
            _fold_value(node["elements"], element)


def _refresh_node(node: dict) -> None:
    if node["kind"] == "object":
        for entry in node["fields"].values():
            entry["total"] = node["count"]
            entry["optional"] = entry["observed"] < node["count"]
            for child in entry["types"].values():
                _refresh_node(child)
    elif node["kind"] == "array":
        for child in node["elements"].values():
            _refresh_node(child)


def _merge_types(left: dict, right: dict) -> dict:
    merged: dict[str, Any] = {}
    for kind in _KINDS:
        a = left.get(kind)
        b = right.get(kind)
        if a is None:
            if b is not None:
                merged[kind] = copy.deepcopy(b)
        elif b is None:
            merged[kind] = copy.deepcopy(a)
        else:
            merged[kind] = _merge_node(a, b)
    return merged


def _merge_field(left: dict, right: dict) -> dict:
    return {
        "types": _merge_types(left["types"], right["types"]),
        "observed": left["observed"] + right["observed"],
        "total": 0,
        "optional": False,
    }


def _merge_node(left: dict, right: dict) -> dict:
    """Combine two schema nodes of the same kind into one.

    Merging is associative and commutative: counts and observations add up
    and type sets union, so batches committed by different processes fold
    together exactly as if all records had been inferred in one pass.
    ``total``/``optional`` are derived and recomputed by ``_refresh_node``.
    """
    kind = left["kind"]
    node = _new_node(kind)
    if kind == "object":
        node["count"] = left["count"] + right["count"]
        fields = node["fields"]
        for name, entry in left["fields"].items():
            other = right["fields"].get(name)
            fields[name] = (
                _merge_field(entry, other)
                if other is not None
                else copy.deepcopy(entry)
            )
        for name, entry in right["fields"].items():
            if name not in left["fields"]:
                fields[name] = copy.deepcopy(entry)
    elif kind == "array":
        node["elements"] = _merge_types(left["elements"], right["elements"])
    return node


def _join(path: str, name: str) -> str:
    if name.isidentifier():
        return f"{path}.{name}"
    return f"{path}[{json.dumps(name)}]"


def _check_object(node: dict, record: dict, path: str, reports: list[str]) -> None:
    fields = node["fields"]
    for name in record:
        if name not in fields:
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
    kind = node["kind"]
    if kind == "object":
        fields = node.get("fields")
        count = node.get("count")
        if not isinstance(fields, dict) or not isinstance(count, int):
            raise SchemaConflict("corrupt schema file: malformed object node")
        for entry in fields.values():
            _validate_field(entry)
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
    if (
        not isinstance(types, dict)
        or not isinstance(observed, int)
        or not isinstance(total, int)
        or not isinstance(optional, bool)
    ):
        raise SchemaConflict("corrupt schema file: malformed field entry")
    for child in types.values():
        _validate_node(child)


def _checksum(root: dict) -> str:
    canonical = json.dumps(root, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_payload(payload: Any) -> dict:
    if not isinstance(payload, dict) or "root" not in payload:
        raise SchemaConflict("corrupt schema file: missing root")
    version = payload.get("version")
    if version != SCHEMA_VERSION:
        raise SchemaConflict(f"unsupported schema version: {version!r}")
    root = payload["root"]
    _validate_node(root)
    if not isinstance(root, dict) or root.get("kind") != "object":
        raise SchemaConflict("corrupt schema file: root must be an object")
    checksum = payload.get("checksum")
    if not isinstance(checksum, str) or checksum != _checksum(root):
        raise SchemaConflict("corrupt schema file: checksum mismatch")
    return root


class _CommitLock:
    """Non-blocking advisory lock serializing commits to a lens directory."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def __enter__(self) -> "_CommitLock":
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            os.close(fd)
            raise SchemaConflict(
                "cannot commit schema: another process holds the lens lock"
            ) from exc
        self._fd = fd
        return self

    def __exit__(self, *exc_info: Any) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            else:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)


class Lens:
    """Opens the lens directory ``path`` and manages its stored schema."""

    def __init__(self, path: os.PathLike | str) -> None:
        self.path = Path(path)
        self._schema: dict | None = None
        # Batch folded since the last load/save, committed by the next save.
        self._pending: dict = _new_node("object")

    @property
    def _file(self) -> Path:
        return self.path / SCHEMA_FILENAME

    def _require(self) -> dict:
        if self._schema is None:
            raise SchemaConflict("no schema has been inferred or loaded yet")
        return self._schema

    def infer(self, records: Iterable[dict]) -> dict:
        """Fold a sequence of records into the stored schema."""
        if self._schema is None:
            self._schema = _new_node("object")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("records must be JSON objects")
            _fold_object(self._schema, record)
            _fold_object(self._pending, record)
        _refresh_node(self._schema)
        _refresh_node(self._pending)
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
        """Report fields, optional fields, observed types and counts."""
        schema = self._require()
        summary: dict[str, Any] = {
            "records": schema["count"],
            "fields": 0,
            "optional": 0,
            "types": {},
            "observations": 0,
        }

        def walk_object(node: dict) -> None:
            for entry in node["fields"].values():
                summary["fields"] += 1
                summary["observations"] += entry["observed"]
                if entry["optional"]:
                    summary["optional"] += 1
                for kind, child in entry["types"].items():
                    summary["types"][kind] = summary["types"].get(kind, 0) + 1
                    walk_node(child)

        def walk_node(node: dict) -> None:
            if node["kind"] == "object":
                walk_object(node)
            elif node["kind"] == "array":
                for child in node["elements"].values():
                    walk_node(child)

        walk_object(schema)
        return summary

    def _read_committed(self) -> dict | None:
        """Return the fully validated on-disk schema, or None if absent."""
        try:
            raw = self._file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SchemaConflict(
                f"cannot read schema file {self._file}: {exc}"
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaConflict(
                f"corrupt schema file {self._file}: {exc}"
            ) from exc
        return _validate_payload(payload)

    def _write_file(self, schema: dict) -> None:
        payload = {
            "version": SCHEMA_VERSION,
            "checksum": _checksum(schema),
            "root": schema,
        }
        tmp = self.path / (SCHEMA_FILENAME + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._file)
        try:
            dir_fd = os.open(self.path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def save(self) -> None:
        """Commit the pending batch into the lens directory atomically.

        The commit is serialized through a lock file: a caller that cannot
        take the lock raises SchemaConflict instead of risking a lost batch.
        The pending batch is merged into whatever schema is currently
        committed, so batches saved by other processes are preserved.
        """
        schema = self._require()
        self.path.mkdir(parents=True, exist_ok=True)
        with _CommitLock(self.path / LOCK_FILENAME):
            committed = self._read_committed()
            if committed is None:
                merged = copy.deepcopy(schema)
                self._write_file(merged)
            elif self._pending["count"] == 0:
                merged = committed
            else:
                merged = _merge_node(committed, self._pending)
                _refresh_node(merged)
                self._write_file(merged)
        self._schema = merged
        self._pending = _new_node("object")

    def load(self) -> None:
        """Re-read the stored schema, replacing memory only on success."""
        committed = self._read_committed()
        if committed is None:
            raise SchemaConflict(
                f"cannot read schema file {self._file}: file is missing"
            )
        self._schema = committed
        self._pending = _new_node("object")
