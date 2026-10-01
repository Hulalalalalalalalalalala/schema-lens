"""Command line interface: infer, check, show over stdin JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from typing import Any

from .lens import Lens, SchemaConflict, kind_of


def loads_strict(text: str) -> Any:
    """Parse one JSON document, rejecting duplicate object fields and the
    non-standard NaN/Infinity constants."""

    def reject_constant(constant: str) -> Any:
        raise ValueError(f"invalid JSON constant: {constant}")

    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise ValueError(f'duplicate field "{key}"')
            seen.add(key)
        return dict(pairs)

    return json.loads(
        text, object_pairs_hook=object_hook, parse_constant=reject_constant
    )


def read_jsonl(stream: Iterator[str]) -> Iterator[tuple[int, Any, str | None]]:
    """Yield ``(line_number, record, error_detail)`` triples.

    Blank lines are ignored. Parse failures, non-object records and
    duplicate fields come back as errors instead of raising.
    """
    for line_number, raw in enumerate(stream, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            value = loads_strict(line)
        except ValueError as exc:
            yield line_number, None, str(exc)
            continue
        if not isinstance(value, dict):
            yield line_number, None, f"record is not an object: {kind_of(value)}"
            continue
        yield line_number, value, None


def _storage_error(path: str, exc: Exception) -> int:
    print(f"storage_error:{path}:{exc}", file=sys.stderr)
    return 2


def _cmd_infer(lens: Lens, stream: Iterator[str]) -> int:
    try:
        lens.load()
    except FileNotFoundError:
        pass
    except (OSError, SchemaConflict) as exc:
        return _storage_error(lens._schema_path(), exc)

    input_failed = False
    folded = 0
    for line_number, record, detail in read_jsonl(stream):
        if detail is not None:
            print(f"input_error:{line_number}:{detail}", file=sys.stderr)
            input_failed = True
            continue
        try:
            lens._fold(record)
            folded += 1
        except SchemaConflict as exc:
            print(f"input_error:{line_number}:{exc}", file=sys.stderr)
            return 2

    if folded:
        try:
            lens.save()
        except OSError as exc:
            return _storage_error(lens._schema_path(), exc)

    return 2 if input_failed else 0


def _cmd_check(lens: Lens, stream: Iterator[str]) -> int:
    try:
        lens.load()
    except (OSError, SchemaConflict) as exc:
        return _storage_error(lens._schema_path(), exc)

    input_failed = False
    any_deviation = False
    for line_number, record, detail in read_jsonl(stream):
        if detail is not None:
            print(f"input_error:{line_number}:{detail}", file=sys.stderr)
            input_failed = True
            continue
        for report in lens.check(record):
            print(report)
            any_deviation = True

    if input_failed:
        return 2
    return 1 if any_deviation else 0


def _cmd_show(lens: Lens) -> int:
    try:
        lens.load()
    except (OSError, SchemaConflict) as exc:
        return _storage_error(lens._schema_path(), exc)
    print(json.dumps(lens.schema(), sort_keys=True, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="schema_lens")
    parser.add_argument("--path", required=True, help="lens directory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("infer")
    subparsers.add_parser("check")
    subparsers.add_parser("show")
    args = parser.parse_args(argv)

    lens = Lens(args.path)
    if args.command == "infer":
        return _cmd_infer(lens, sys.stdin)
    if args.command == "check":
        return _cmd_check(lens, sys.stdin)
    return _cmd_show(lens)


if __name__ == "__main__":
    raise SystemExit(main())
