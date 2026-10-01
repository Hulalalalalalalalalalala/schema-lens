"""schema_lens: infer a schema from a stream of JSON records and check later
records against it.

Public interface:

- ``Lens(path)`` opens the lens directory ``path``.
- ``SchemaConflict`` raised when records cannot share one schema, or when a
  loaded schema file is malformed.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Iterable

__all__ = ["Lens", "SchemaConflict"]

_SCHEMA_FILE = "schema.json"

# Allowed node kinds, in canonical (sorted) order.
_KINDS = ("null", "boolean", "integer", "number", "string", "array", "object")


class SchemaConflict(Exception):
    """Raised when two values cannot be folded into one schema node, or when a
    persisted schema fails validation."""


def _kind_of(value: Any) -> str:
    """Return the schema kind of a JSON value.

    Booleans are their own kind (they are not integers) and floats that carry
    integral values stay ``number``.
    """
    # bool must be tested before int: bool is a subclass of int in Python.
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    # Anything else (should be impossible for json.loads output) is not a
    # JSON-compatible value.
    raise SchemaConflict(f"unsupported value type: {type(value).__name__}")


def _node_from_value(value: Any) -> dict:
    """Build a fresh schema node describing one value."""
    kind = _kind_of(value)
    node: dict[str, Any] = {"kind": kind}
    if kind == "object":
        node["fields"] = {
            name: {"required": True, "schema": _node_from_value(sub)}
            for name, sub in value.items()
        }
    elif kind == "array":
        node["items"] = _merge_items(value)
    return node


def _merge_items(items: list) -> Any:
    """Fold the elements of one array into one items node.

    An empty array has no element shape, so its ``items`` is ``None``.
    """
    merged: Any = None
    for element in items:
        node = _node_from_value(element)
        merged = node if merged is None else _merge_node(merged, node)
    return merged


def _merge_node(existing: dict, incoming: dict) -> dict:
    """Merge ``incoming`` into ``existing`` in place and return it.

    Raises ``SchemaConflict`` when the two nodes have incompatible shapes.
    """
    old_kind = existing["kind"]
    new_kind = incoming["kind"]

    if old_kind == new_kind:
        if old_kind == "object":
            _merge_fields(existing["fields"], incoming["fields"])
        elif old_kind == "array":
            if existing["items"] is None:
                existing["items"] = incoming["items"]
            elif incoming["items"] is not None:
                existing["items"] = _merge_node(existing["items"], incoming["items"])
        return existing

    # integer widens to number; the reverse and every other pairing conflicts.
    if old_kind == "integer" and new_kind == "number":
        existing["kind"] = "number"
        return existing
    if old_kind == "number" and new_kind == "integer":
        return existing

    raise SchemaConflict(f"type conflict: {old_kind} vs {new_kind}")


def _merge_fields(existing: dict, incoming: dict) -> None:
    """Merge two object field maps in place.

    Fields seen in both objects keep their required flag (they are required if
    present in both); fields missing from either object become optional.
    """
    for name, spec in incoming.items():
        if name in existing:
            existing[name]["schema"] = _merge_node(
                existing[name]["schema"], spec["schema"]
            )
            # required stays whatever it was: still required if present here,
            # already optional if an earlier record omitted it.
        else:
            # A field appearing for the first time was absent from every
            # earlier record, so it is optional.
            existing[name] = {"required": False, "schema": spec["schema"]}
    for name, spec in existing.items():
        if name not in incoming:
            spec["required"] = False


def _validate_node(node: Any, path: str = "$") -> None:
    """Validate a loaded schema node. Raises SchemaConflict on any defect."""
    if not isinstance(node, dict):
        raise SchemaConflict(f"invalid schema node at {path}: expected object")
    kind = node.get("kind")
    if kind not in _KINDS:
        raise SchemaConflict(f"invalid schema node at {path}: bad kind {kind!r}")
    allowed = {"kind"}
    if kind == "object":
        allowed.add("fields")
    elif kind == "array":
        allowed.add("items")
    extra = set(node) - allowed
    if extra:
        raise SchemaConflict(
            f"invalid schema node at {path}: unexpected keys {sorted(extra)}"
        )
    if kind == "object":
        fields = node.get("fields")
        if not isinstance(fields, dict):
            raise SchemaConflict(f"invalid object schema at {path}: missing fields")
        for name in sorted(fields):
            spec = fields[name]
            if not isinstance(spec, dict) or not isinstance(
                spec.get("required"), bool
            ) or "schema" not in spec:
                raise SchemaConflict(
                    f"invalid field {name!r} at {path}: need required and schema"
                )
            _validate_node(spec["schema"], f"{path}.{name}")
    elif kind == "array":
        if "items" not in node:
            raise SchemaConflict(f"invalid array schema at {path}: missing items")
        items = node["items"]
        if items is not None:
            _validate_node(items, f"{path}[]")


class Lens:
    """A schema lens backed by a directory containing ``schema.json``."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._schema: dict | None = None
        self._records = 0
        # Per-field statistics, keyed by dotted path (e.g. "$" and "$.tags").
        self._field_count: dict[str, int] = {}
        self._field_types: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def schema(self) -> dict:
        """Return the current schema node, or an empty-object schema before any
        records have been folded in."""
        if self._schema is None:
            return {"kind": "object", "fields": {}}
        return self._schema

    def infer(self, records: Iterable[Any]) -> dict:
        """Fold a sequence of records into the stored schema and return it.

        A record that fails to merge leaves the schema untouched before the
        ``SchemaConflict`` propagates.
        """
        for record in records:
            if not isinstance(record, dict) or isinstance(record, bool):
                raise SchemaConflict("infer expects each record to be an object")
            incoming = _node_from_value(record)
            snapshot = copy.deepcopy(self._schema)
            try:
                if self._schema is None:
                    self._schema = incoming
                else:
                    _merge_node(self._schema, incoming)
            except SchemaConflict:
                self._schema = snapshot
                raise
            self._records += 1
            self._observe(record, "$")
        return self.schema()

    def _observe(self, value: Any, path: str) -> None:
        """Accumulate per-field counts and observed kinds from one raw value.

        Walking the value (rather than the merged node) counts every array
        element, keeping the totals aligned with the paths in ``stats()``.
        """
        kind = _kind_of(value)
        self._field_count[path] = self._field_count.get(path, 0) + 1
        self._field_types.setdefault(path, set()).add(kind)
        if kind == "object":
            for name, sub in value.items():
                self._observe(sub, f"{path}.{name}")
        elif kind == "array":
            for element in value:
                self._observe(element, f"{path}[]")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        fields: list[str] = []
        optional: list[str] = []

        def collect(node: dict, path: str) -> None:
            if node["kind"] == "object":
                for name in sorted(node["fields"]):
                    spec = node["fields"][name]
                    child = f"{path}.{name}"
                    fields.append(child)
                    if not spec["required"]:
                        optional.append(child)
                    collect(spec["schema"], child)
            elif node["kind"] == "array" and node["items"] is not None:
                collect(node["items"], f"{path}[]")

        root = self.schema()
        if root["kind"] == "object":
            collect(root, "$")

        observed_types = {
            path: sorted(types) for path, types in sorted(self._field_types.items())
        }
        field_observations = {
            path: self._field_count[path] for path in sorted(self._field_count)
        }
        return {
            "records": self._records,
            "fields": fields,
            "optional_fields": optional,
            "observed_types": observed_types,
            "field_observations": field_observations,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Write the schema as sorted, UTF-8 JSON to ``path/schema.json``."""
        payload = {
            "records": self._records,
            "schema": self.schema(),
            "stats": {
                "field_observations": dict(sorted(self._field_count.items())),
                "field_types": {
                    path: sorted(types)
                    for path, types in sorted(self._field_types.items())
                },
            },
        }
        os.makedirs(self.path, exist_ok=True)
        target = os.path.join(self.path, _SCHEMA_FILE)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=False, indent=2)
            handle.write("\n")

    def load(self) -> None:
        """Read ``path/schema.json``, validate it, and replace in-memory state.

        Malformed schemas raise ``SchemaConflict``; filesystem problems surface
        as ``OSError``.
        """
        target = os.path.join(self.path, _SCHEMA_FILE)
        with open(target, "r", encoding="utf-8") as handle:
            raw = handle.read()
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaConflict(f"invalid schema.json: {exc}") from exc

        # Accept both a bare schema node and the wrapped document save() writes.
        if isinstance(document, dict) and "kind" in document:
            node = document
            records = 0
            observations: dict = {}
            type_lists: dict = {}
        elif isinstance(document, dict) and "schema" in document:
            node = document["schema"]
            records = document.get("records", 0)
            stats = document.get("stats", {})
            observations = stats.get("field_observations", {}) if isinstance(
                stats, dict
            ) else {}
            type_lists = stats.get("field_types", {}) if isinstance(stats, dict) else {}
        else:
            raise SchemaConflict("invalid schema.json: missing schema")

        # Validate fully before touching in-memory state.
        _validate_node(node)
        if not isinstance(records, int) or isinstance(records, bool) or records < 0:
            raise SchemaConflict("invalid schema.json: bad records")
        field_count: dict[str, int] = {}
        for key, value in observations.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise SchemaConflict("invalid schema.json: bad field_observations")
            field_count[key] = value
        field_types: dict[str, set[str]] = {}
        for key, value in type_lists.items():
            if not isinstance(key, str) or not isinstance(value, list) or any(
                kind not in _KINDS for kind in value
            ):
                raise SchemaConflict("invalid schema.json: bad field_types")
            field_types[key] = set(value)

        self._schema = node
        self._records = records
        self._field_count = field_count
        self._field_types = field_types

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------
    def check(self, record: Any) -> list[str]:
        """Report every way one record departs from the stored schema."""
        deviations: list[str] = []
        self._check_value(record, self.schema(), "$", deviations)
        return deviations

    def _check_value(
        self, value: Any, node: dict, path: str, deviations: list[str]
    ) -> None:
        kind = node["kind"]
        actual = _kind_of(value)
        if not self._matches(actual, kind):
            deviations.append(f"type_mismatch:{path}:{kind}:{actual}")
            return
        if kind == "object":
            self._check_object(value, node, path, deviations)
        elif kind == "array":
            self._check_array(value, node, path, deviations)

    @staticmethod
    def _matches(actual: str, expected: str) -> bool:
        if actual == expected:
            return True
        # An integer satisfies a number schema (widened during inference).
        return expected == "number" and actual == "integer"

    def _check_object(
        self, value: dict, node: dict, path: str, deviations: list[str]
    ) -> None:
        fields = node["fields"]
        for name, spec in fields.items():
            child = f"{path}.{name}"
            if name not in value:
                if spec["required"]:
                    deviations.append(f"missing_required:{child}")
            else:
                self._check_value(value[name], spec["schema"], child, deviations)
        for name in value:
            if name not in fields:
                deviations.append(f"unexpected_field:{path}.{name}")

    def _check_array(
        self, value: list, node: dict, path: str, deviations: list[str]
    ) -> None:
        items = node["items"]
        if items is None:
            # Schema came from empty arrays only: accept any elements.
            return
        for index, element in enumerate(value):
            self._check_value(element, items, f"{path}[{index}]", deviations)
