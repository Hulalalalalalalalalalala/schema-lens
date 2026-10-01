"""Core schema folding, validation and persistence."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

SCHEMA_FILENAME = "schema.json"

LEAF_KINDS = ("null", "boolean", "integer", "number", "string")
ALL_KINDS = LEAF_KINDS + ("array", "object")


class SchemaConflict(Exception):
    """Raised when records cannot be folded into one schema, or a stored
    schema is invalid."""


def kind_of(value: Any) -> str:
    """Return the schema kind of a JSON value.

    Booleans are classified strictly: ``True``/``False`` are ``boolean``,
    never ``integer``.
    """
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
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _join(path: str, name: str) -> str:
    return f"{path}.{name}"


class Lens:
    """Folds JSON object records into a schema and checks records against it.

    Internal nodes carry bookkeeping keys (``_n`` on object nodes, ``obs``
    and ``types`` on field entries) that are stripped before the schema is
    exported or saved.
    """

    def __init__(self, path: str):
        self.path = path
        self._root: dict[str, Any] | None = None
        self._records = 0

    # ------------------------------------------------------------------ build

    @staticmethod
    def _build(value: Any, path: str) -> dict[str, Any]:
        kind = kind_of(value)
        if kind in LEAF_KINDS:
            return {"kind": kind}
        if kind == "array":
            node: dict[str, Any] = {"kind": "array", "items": None}
            for index, element in enumerate(value):
                element_path = f"{path}[{index}]"
                if node["items"] is None:
                    node["items"] = Lens._build(element, element_path)
                else:
                    Lens._merge(node["items"], element, element_path)
            return node
        node = {"kind": "object", "fields": {}, "_n": 0}
        Lens._fold_object(node, value, path)
        return node

    @staticmethod
    def _fold_object(node: dict[str, Any], value: dict[str, Any], path: str) -> None:
        fields = node["fields"]
        seen = node.get("_n", 0)
        for name, element in value.items():
            entry = fields.get(name)
            kind = kind_of(element)
            field_path = _join(path, name)
            if entry is None:
                # A field first seen after record one was missing earlier,
                # so it starts optional; otherwise it starts required.
                fields[name] = {
                    "required": seen == 0,
                    "schema": Lens._build(element, field_path),
                    "obs": 1,
                    "types": [kind],
                }
            else:
                Lens._merge(entry["schema"], element, field_path)
                entry["obs"] += 1
                if kind not in entry["types"]:
                    entry["types"].append(kind)
        for name, entry in fields.items():
            if name not in value:
                entry["required"] = False
        node["_n"] = seen + 1

    @staticmethod
    def _merge(node: dict[str, Any], value: Any, path: str) -> None:
        """Fold one JSON value into an existing node."""
        kind = kind_of(value)
        current = node["kind"]

        if current in ("null", "boolean", "string"):
            if kind != current:
                raise SchemaConflict(
                    f"{path}: expected {current}, observed {kind}"
                )
            return

        if current == "integer":
            if kind == "integer":
                return
            if kind == "number":
                node["kind"] = "number"  # integers widen to numbers
                return
            raise SchemaConflict(f"{path}: expected integer, observed {kind}")

        if current == "number":
            if kind in ("integer", "number"):
                return
            raise SchemaConflict(f"{path}: expected number, observed {kind}")

        if current == "array":
            if kind != "array":
                raise SchemaConflict(f"{path}: expected array, observed {kind}")
            items = node["items"]
            for index, element in enumerate(value):
                element_path = f"{path}[{index}]"
                if items is None:
                    node["items"] = Lens._build(element, element_path)
                    items = node["items"]
                else:
                    Lens._merge(items, element, element_path)
            return

        if current == "object":
            if kind != "object":
                raise SchemaConflict(f"{path}: expected object, observed {kind}")
            Lens._fold_object(node, value, path)
            return

        raise SchemaConflict(f"{path}: unknown schema kind {current!r}")

    def _fold(self, record: Any) -> None:
        if not isinstance(record, dict):
            raise TypeError("record must be a JSON object")
        if self._root is None:
            self._root = {"kind": "object", "fields": {}, "_n": 0}
        self._merge(self._root, record, "$")
        self._records += 1

    # --------------------------------------------------------------- export

    @staticmethod
    def _public(node: dict[str, Any]) -> dict[str, Any]:
        kind = node["kind"]
        if kind in LEAF_KINDS:
            return {"kind": kind}
        if kind == "array":
            items = node["items"]
            return {"kind": "array", "items": None if items is None else Lens._public(items)}
        fields = {
            name: {
                "required": entry["required"],
                "schema": Lens._public(entry["schema"]),
            }
            for name, entry in node["fields"].items()
        }
        return {"kind": "object", "fields": fields}

    @staticmethod
    def _from_public(node: Any, path: str = "$", seen: int = 1) -> dict[str, Any]:
        """Validate a public schema node and rebuild its internal form.

        ``seen`` is the number of records already folded into an object
        node; loaded schemas start at one so that fields appearing only in
        later batches are correctly treated as optional.
        """
        if not isinstance(node, dict):
            raise SchemaConflict(f"{path}: schema node must be an object")
        kind = node.get("kind")
        if kind not in ALL_KINDS:
            raise SchemaConflict(f"{path}: invalid kind {kind!r}")

        if kind in LEAF_KINDS:
            if set(node) != {"kind"}:
                raise SchemaConflict(f"{path}: unexpected keys for kind {kind}")
            return {"kind": kind}

        if kind == "array":
            if set(node) != {"kind", "items"}:
                raise SchemaConflict(f"{path}: unexpected keys for kind array")
            items = node["items"]
            if items is not None and not isinstance(items, dict):
                raise SchemaConflict(f"{path}.items: must be a node or null")
            return {
                "kind": "array",
                "items": None if items is None else Lens._from_public(items, path + "[]"),
            }

        if set(node) != {"kind", "fields"}:
            raise SchemaConflict(f"{path}: unexpected keys for kind object")
        fields = node["fields"]
        if not isinstance(fields, dict) or not all(isinstance(k, str) for k in fields):
            raise SchemaConflict(f"{path}.fields: must map string names to entries")
        internal_fields = {}
        for name, entry in fields.items():
            field_path = _join(path, name)
            if not isinstance(entry, dict) or set(entry) != {"required", "schema"}:
                raise SchemaConflict(f"{field_path}: invalid field entry")
            required = entry["required"]
            if not isinstance(required, bool):
                raise SchemaConflict(f"{field_path}.required: must be boolean")
            child = Lens._from_public(entry["schema"], field_path)
            internal_fields[name] = {
                "required": required,
                "schema": child,
                "obs": 0,
                "types": [child["kind"]],
            }
        return {"kind": "object", "fields": internal_fields, "_n": seen}

    # ------------------------------------------------------------- public API

    def infer(self, records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
        """Fold a sequence of records into the stored schema and save it.

        A schema already saved at ``path`` is loaded first so the fold
        continues across calls; a missing file starts from scratch.
        """
        if self._root is None:
            try:
                self.load()
            except FileNotFoundError:
                pass
        folded = 0
        for record in records:
            self._fold(record)
            folded += 1
        if folded:
            self.save()
        return self.schema()

    def schema(self) -> dict[str, Any] | None:
        """Return a deep copy of the stored schema, or None before infer."""
        if self._root is None:
            return None
        return self._public(self._root)

    def stats(self) -> dict[str, Any]:
        if self._root is None:
            return {
                "records": self._records,
                "fields": [],
                "optional_fields": [],
                "observed_types": {},
                "field_observations": {},
            }
        fields = self._root["fields"]
        names = sorted(fields)
        return {
            "records": self._records,
            "fields": names,
            "optional_fields": sorted(
                name for name in names if not fields[name]["required"]
            ),
            "observed_types": {
                name: sorted(fields[name]["types"]) for name in names
            },
            "field_observations": {
                name: fields[name]["obs"] for name in names
            },
        }

    def check(self, record: Any) -> list[str]:
        """Report every deviation of one record from the stored schema."""
        if self._root is None:
            raise SchemaConflict("no schema stored; run infer first")
        return self._check(self._root, record, "$")

    @staticmethod
    def _check(node: dict[str, Any], value: Any, path: str) -> list[str]:
        kind = kind_of(value)
        current = node["kind"]

        if current in ("null", "boolean", "string"):
            if kind == current:
                return []
            return [f"type_mismatch:{path}:{current}:{kind}"]

        if current == "integer":
            return [] if kind == "integer" else [
                f"type_mismatch:{path}:integer:{kind}"
            ]

        if current == "number":
            return [] if kind in ("integer", "number") else [
                f"type_mismatch:{path}:number:{kind}"
            ]

        if current == "array":
            if kind != "array":
                return [f"type_mismatch:{path}:array:{kind}"]
            items = node["items"]
            if items is None:
                return []
            reports: list[str] = []
            for index, element in enumerate(value):
                reports.extend(Lens._check(items, element, f"{path}[{index}]"))
            return reports

        if kind != "object":
            return [f"type_mismatch:{path}:object:{kind}"]
        fields = node["fields"]
        reports = []
        for name in sorted(fields):
            entry = fields[name]
            field_path = _join(path, name)
            if name not in value:
                if entry["required"]:
                    reports.append(f"missing_required:{field_path}")
            else:
                reports.extend(Lens._check(entry["schema"], value[name], field_path))
        for name in sorted(value):
            if name not in fields:
                reports.append(f"unexpected_field:{_join(path, name)}")
        return reports

    # ---------------------------------------------------------- persistence

    def _schema_path(self) -> str:
        return os.path.join(self.path, SCHEMA_FILENAME)

    def save(self) -> None:
        """Write the schema as sorted UTF-8 JSON to ``path/schema.json``."""
        if self._root is None:
            raise SchemaConflict("no schema to save")
        os.makedirs(self.path, exist_ok=True)
        encoded = json.dumps(
            self._public(self._root),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        with open(self._schema_path(), "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")

    def load(self) -> None:
        """Validate the stored schema and replace the in-memory schema with it."""
        with open(self._schema_path(), encoding="utf-8") as handle:
            text = handle.read()
        try:
            raw = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SchemaConflict(f"invalid schema JSON: {exc}") from exc
        root = self._from_public(raw)
        if root["kind"] != "object":
            raise SchemaConflict("$: root schema must be an object")
        self._root = root
        self._records = 0
