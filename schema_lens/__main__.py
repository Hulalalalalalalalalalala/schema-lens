"""Command-line entry point: ``python -m schema_lens``.

Reads JSONL from standard input (one JSON object per line, blank lines
ignored) and writes machine-readable reports to standard output:

- infer  fold records into the stored schema, then save it
- check  report each record's deviations from the stored schema
- show   print the stored schema as JSON

Exit codes: 0 = clean, 1 = deviations found, 2 = input or storage errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import Lens, SchemaConflict

_EXIT_OK = 0
_EXIT_DEVIATIONS = 1
_EXIT_ERROR = 2


def _parse_line(text: str) -> tuple[Any, str | None]:
    """Parse one JSONL line strictly.

    Returns ``(value, None)`` or ``(None, detail)``. Rejects syntax errors,
    non-standard constants (NaN/Infinity) and duplicate object fields.
    """

    def reject_constant(value: str) -> Any:
        raise ValueError(f"invalid JSON constant: {value}")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text, object_pairs_hook=pairs_hook, parse_constant=reject_constant
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)
    return value, None


def _iter_jsonl(stream: Any) -> tuple[list[Any], list[tuple[int, str]]]:
    """Read JSONL from ``stream``; return parsed records and parse errors."""
    records: list[Any] = []
    errors: list[tuple[int, str]] = []
    for line_number, raw in enumerate(stream, start=1):
        if not raw.strip():
            continue
        value, detail = _parse_line(raw)
        if detail is not None:
            errors.append((line_number, detail))
            continue
        if not isinstance(value, dict):
            errors.append((line_number, "record is not a JSON object"))
            continue
        records.append((line_number, value))
    return records, errors


def _report_input_errors(errors: list[tuple[int, str]], out: Any) -> None:
    for line_number, detail in errors:
        out.write(f"input_error:{line_number}:{detail}\n")


def _cmd_infer(lens: Lens, stream: Any, out: Any) -> int:
    records, errors = _iter_jsonl(stream)
    _report_input_errors(errors, out)
    try:
        for line_number, record in records:
            try:
                lens.infer([record])
            except SchemaConflict as exc:
                out.write(f"input_error:{line_number}:{exc}\n")
                errors.append((line_number, str(exc)))
        lens.save()
    except OSError as exc:
        out.write(f"storage_error:{lens.path}/schema.json:{exc}\n")
        return _EXIT_ERROR
    return _EXIT_ERROR if errors else _EXIT_OK


def _cmd_check(lens: Lens, stream: Any, out: Any) -> int:
    try:
        lens.load()
    except SchemaConflict as exc:
        out.write(f"storage_error:{lens.path}/schema.json:{exc}\n")
        return _EXIT_ERROR
    except OSError as exc:
        out.write(f"storage_error:{lens.path}/schema.json:{exc}\n")
        return _EXIT_ERROR

    records, errors = _iter_jsonl(stream)
    _report_input_errors(errors, out)
    deviations = 0
    for _, record in records:
        for deviation in lens.check(record):
            out.write(deviation + "\n")
            deviations += 1
    if errors:
        return _EXIT_ERROR
    return _EXIT_DEVIATIONS if deviations else _EXIT_OK


def _cmd_show(lens: Lens, out: Any) -> int:
    try:
        lens.load()
    except SchemaConflict as exc:
        out.write(f"storage_error:{lens.path}/schema.json:{exc}\n")
        return _EXIT_ERROR
    except OSError as exc:
        out.write(f"storage_error:{lens.path}/schema.json:{exc}\n")
        return _EXIT_ERROR
    out.write(
        json.dumps(lens.schema(), sort_keys=True, ensure_ascii=False, indent=2)
        + "\n"
    )
    return _EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="schema_lens")
    parser.add_argument("--path", required=True, help="lens directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("infer", help="infer a schema from JSONL on stdin")
    commands.add_parser("check", help="check JSONL records on stdin")
    commands.add_parser("show", help="print the stored schema")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    lens = Lens(args.path)
    out = sys.stdout
    if args.command == "infer":
        return _cmd_infer(lens, sys.stdin, out)
    if args.command == "check":
        return _cmd_check(lens, sys.stdin, out)
    return _cmd_show(lens, out)


if __name__ == "__main__":
    sys.exit(main())
