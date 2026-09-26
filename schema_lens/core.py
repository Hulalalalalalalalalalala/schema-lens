"""Core schema inference and validation logic.

A schema is a tree of nodes. Object nodes carry a mapping of field name to
field entry; each field entry records the set of observed types, how many
records carried the field, how many records were seen at that level, and
whether the field is optional. Array nodes merge their element types.

Persistence is incremental and crash safe. ``Lens.save`` serializes commits
through a lock file inside the lens directory, folds the records folded
since the last load or save into whatever schema is already stored, and
writes the result as one atomic file replacement carrying a version and a
checksum of the whole schema. ``Lens.load`` validates version, structure and
checksum completely before replacing memory, so a crashed or interrupted
write can only ever leave the previous complete schema or the next complete
one, and concurrent committers never lose each other's batches.
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


def _merge_node(left: dict, right: dict) -> dict:
    """Fold two schema nodes over disjoint batches into one new node."""
    kind = left["kind"]
    if kind == "object":
        node = _new_node("object")
        node["count"] = left["count"] + right["count"]
        names = list(left["fields"]) + [
            name for name in right["fields"] if name not in left["fields"]
        ]
        for name in names:
            le = left["fields"].get(name)
            re = right["fields"].get(name)
            if le is None:
                node["fields"][name] = copy.deepcopy(re)
            elif re is None:
                node["fields"][name] = copy.deepcopy(le)
            else:
                node["fields"][name] = _merge_field(le, re)
        return node
    if kind == "array":
        node = _new_node("array")
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
        node = _new_node("object")
        node["count"] = mem["count"] - base["count"]
        for name, entry in mem["fields"].items():
            base_entry = base["fields"].get(name)
            if base_entry is None:
                node["fields"][name] = copy.deepcopy(entry)
            else:
                node["fields"][name] = _subtract_field(entry, base_entry)
        return node
    if kind == "array":
        node = _new_node("array")
        for element_kind, child in mem["elements"].items():
            base_child = base["elements"].get(element_kind)
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
        if isinstance(count, bool) or not isinstance(fields, dict) or not isinstance(count, int):
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
        or isinstance(observed, bool)
        or not isinstance(observed, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or not isinstance(optional, bool)
    ):
        raise SchemaConflict("corrupt schema file: malformed field entry")
    for child in types.values():
        _validate_node(child)


def _checksum(root: dict) -> str:
    """Checksum of the whole schema in its canonical JSON encoding."""
    canonical = json.dumps(root, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise SchemaConflict("corrupt schema file: not a JSON object")
    version = payload.get("version")
    if version != SCHEMA_VERSION:
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
    return root


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
    """Opens the lens directory ``path`` and manages its stored schema."""

    def __init__(self, path: os.PathLike | str) -> None:
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
        """Fold a sequence of records into the stored schema."""
        if self._schema is None:
            self._schema = _new_node("object")
            self._base = _new_node("object")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("records must be JSON objects")
            _fold_object(self._schema, record)
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

    def _read_disk(self) -> dict | None:
        """Read and fully validate the stored schema.

        Returns ``None`` only when the lens directory holds no schema yet.
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
        return _validate_payload(payload)

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
