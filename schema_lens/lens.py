"""Core lens: fold records into a schema, check records against it, persist it.

A schema node describes everything observed at one position of a record:

- ``type``: the set of JSON value types seen there;
- ``count``: how many values were observed at the node;
- ``records``: how many of those values were objects (the denominator used
  when deciding whether an object's fields are optional);
- ``fields``: object fields merged by name, each a node of its own;
- ``items``: the single node into which every array element is merged.

Only JSON-compatible values are understood: object, array, string, number,
boolean and null. There is no coercion and no date/decimal/binary handling.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

SCHEMA_FILE = "schema.json"

T_OBJECT = "object"
T_ARRAY = "array"
T_STRING = "string"
T_NUMBER = "number"
T_BOOLEAN = "boolean"
T_NULL = "null"

_VALID_TYPES = {T_OBJECT, T_ARRAY, T_STRING, T_NUMBER, T_BOOLEAN, T_NULL}
_SCALAR_TYPES = _VALID_TYPES - {T_OBJECT, T_ARRAY}


class SchemaConflict(Exception):
    """Raised when an operation needs a usable stored schema and none exists."""


def value_type(value: Any) -> str:
    """Return the JSON type name of a JSON-compatible Python value."""
    if value is None:
        return T_NULL
    # bool is a subclass of int, so it must be tested first.
    if isinstance(value, bool):
        return T_BOOLEAN
    if isinstance(value, (int, float)):
        return T_NUMBER
    if isinstance(value, str):
        return T_STRING
    if isinstance(value, list):
        return T_ARRAY
    if isinstance(value, dict):
        return T_OBJECT
    raise TypeError(f"value is not JSON compatible: {type(value).__name__}")


def _new_node() -> dict:
    return {
        "type": set(),
        "count": 0,
        "records": 0,
        "fields": {},
        "items": None,
    }


# --------------------------------------------------------------------------
# Folding records into a schema
# --------------------------------------------------------------------------


def _fold(node: dict, value: Any) -> None:
    """Merge one value into a node."""
    kind = value_type(value)
    node["type"].add(kind)
    node["count"] += 1

    if kind == T_OBJECT:
        node["records"] += 1
        # Fields are merged by name; a child node is folded once for every
        # record in which the field is present, so its count is its presence.
        for name, item in value.items():
            child = node["fields"].get(name)
            if child is None:
                child = _new_node()
                node["fields"][name] = child
            _fold(child, item)
    elif kind == T_ARRAY:
        items = node["items"]
        if items is None:
            items = _new_node()
            node["items"] = items
        # An array that has always been empty still owns an (empty) items
        # node, so the schema and its checks are identical after a reload.
        for item in value:
            _fold(items, item)


# --------------------------------------------------------------------------
# Checking a single record
# --------------------------------------------------------------------------

ROOT = "$"


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _type_list(node: dict) -> list[str]:
    return sorted(node["type"])


def _check(node: dict, value: Any, path: str, issues: list[str]) -> None:
    """Append every mismatch between value and node to issues."""
    kind = value_type(value)
    if kind not in node["type"]:
        issues.append(
            f"{path or ROOT}: type {kind!r} is not one of the observed "
            f"types {_type_list(node)}"
        )
        return

    if kind == T_OBJECT:
        # Missing required fields first, so a report names the exact gap.
        for name, child in node["fields"].items():
            if name not in value and child["count"] < node["records"]:
                continue  # observed absent at least once -> optional
            if name not in value:
                issues.append(f"{_join(path, name)}: required field is missing")
        for name, item in value.items():
            child_path = _join(path, name)
            child = node["fields"].get(name)
            if child is None:
                issues.append(f"{child_path}: field is not in schema")
            else:
                _check(child, item, child_path, issues)
    elif kind == T_ARRAY:
        items = node["items"]
        for index, item in enumerate(value):
            if items is None:
                continue
            _check(items, item, f"{path}[{index}]", issues)


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def _node_to_dict(node: dict, parent_records: int | None = None) -> dict:
    """Turn an in-memory node into a JSON-compatible dict.

    ``parent_records`` is set when serializing an object field: the number of
    objects observed on the parent, against which the field's presence count
    decides optionality.
    """
    data: dict = {"type": sorted(node["type"]), "count": node["count"]}
    if parent_records is not None:
        data["optional"] = node["count"] < parent_records
    if T_OBJECT in node["type"]:
        data["records"] = node["records"]
        data["fields"] = {
            name: _node_to_dict(child, node["records"])
            for name, child in sorted(node["fields"].items())
        }
    if T_ARRAY in node["type"]:
        data["items"] = _node_to_dict(node["items"]) if node["items"] is not None else {
            "type": [],
            "count": 0,
        }
    return data


def _require_int(data: dict, key: str) -> int:
    value = data.get(key)
    # bool is an int in Python; schema counters must be plain integers.
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaConflict(f"schema field {key!r} must be a non-negative integer")
    return value


def _node_from_dict(data: Any, parent_records: int | None = None) -> dict:
    """Rebuild a node from persisted data.

    Raises SchemaConflict on any structural problem; the caller only swaps the
    rebuilt node in after this returns, so corrupt files never touch memory.
    """
    if not isinstance(data, dict):
        raise SchemaConflict("schema node must be a JSON object")

    types = data.get("type")
    if not isinstance(types, list) or any(
        not isinstance(t, str) or t not in _VALID_TYPES for t in types
    ) or len(types) != len(set(types)):
        raise SchemaConflict(f"invalid observed type list: {types!r}")

    node = _new_node()
    node["type"] = set(types)
    node["count"] = _require_int(data, "count")

    if parent_records is not None:
        optional = data.get("optional")
        if not isinstance(optional, bool):
            raise SchemaConflict("schema field is missing its optional flag")
        if optional != (node["count"] < parent_records):
            raise SchemaConflict(
                f"optional flag disagrees with observation count "
                f"({node['count']} of {parent_records} records)"
            )

    if T_OBJECT in node["type"]:
        records = _require_int(data, "records")
        if records > node["count"]:
            raise SchemaConflict("object record count exceeds observation count")
        node["records"] = records
        fields = data.get("fields")
        if not isinstance(fields, dict):
            raise SchemaConflict("object node is missing its fields map")
        for name, raw in fields.items():
            if not isinstance(name, str):
                raise SchemaConflict("object field names must be strings")
            node["fields"][name] = _node_from_dict(raw, records)

    if T_ARRAY in node["type"]:
        items = data.get("items")
        if items is None:
            raise SchemaConflict("array node is missing its items node")
        node["items"] = _node_from_dict(items)

    return node


# --------------------------------------------------------------------------
# Lens
# --------------------------------------------------------------------------


class Lens:
    """A lens directory holding one inferred schema."""

    def __init__(self, path: str):
        self.path = path
        self._root: dict | None = None

    @property
    def schema_path(self) -> str:
        return os.path.join(self.path, SCHEMA_FILE)

    # -- inference ---------------------------------------------------------

    def infer(self, records: Iterable[Any]) -> dict:
        """Fold a sequence of records into the stored schema and return it."""
        if self._root is None:
            self._root = _new_node()
        for record in records:
            _fold(self._root, record)
        return self.schema()

    # -- checking ----------------------------------------------------------

    def check(self, record: Any) -> list[str]:
        """Report every way one record departs from the stored schema."""
        root = self._require_schema()
        issues: list[str] = []
        _check(root, record, "", issues)
        return issues

    # -- reading -----------------------------------------------------------

    def schema(self) -> dict:
        """Return the stored schema as a JSON-compatible dict."""
        return _node_to_dict(self._require_schema())

    def stats(self) -> dict:
        """Report fields, optional fields, observed types and observation counts."""
        root = self._require_schema()
        fields: list[str] = []
        optional: list[str] = []
        types: dict[str, list[str]] = {}
        counts: dict[str, int] = {}
        _collect_stats(root, "", fields, optional, types, counts)
        return {
            "fields": fields,
            "optional_fields": optional,
            "types": types,
            "counts": counts,
        }

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        """Persist the stored schema into the lens directory."""
        payload = json.dumps(self.schema(), indent=2, sort_keys=True) + "\n"
        os.makedirs(self.path, exist_ok=True)
        tmp_path = self.schema_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_path, self.schema_path)

    def load(self) -> None:
        """Re-read the persisted schema, replacing memory wholesale on success."""
        try:
            with open(self.schema_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaConflict(
                f"cannot read schema from {self.schema_path}: {exc}"
            ) from exc
        # Rebuild fully first; only swap it in once the file has proven valid.
        root = _node_from_dict(data)
        self._root = root

    # -- internals ---------------------------------------------------------

    def _require_schema(self) -> dict:
        if self._root is None:
            raise SchemaConflict("no schema has been inferred or loaded yet")
        return self._root


def _collect_stats(
    node: dict,
    prefix: str,
    fields: list[str],
    optional: list[str],
    types: dict[str, list[str]],
    counts: dict[str, int],
) -> None:
    for name, child in node["fields"].items():
        path = _join(prefix, name)
        fields.append(path)
        types[path] = sorted(child["type"])
        counts[path] = child["count"]
        if child["count"] < node["records"]:
            optional.append(path)
        _collect_stats(child, path, fields, optional, types, counts)
    if node["items"] is not None:
        item_prefix = f"{prefix or ROOT}[]"
        _collect_stats(
            node["items"], item_prefix, fields, optional, types, counts
        )
