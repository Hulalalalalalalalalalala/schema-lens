"""Command line entry point: ``python -m schema_lens``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Iterator

from .lens import SchemaConflict
from . import Lens


def _read_jsonl(path: str) -> Iterator[Any]:
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"{path}: cannot open records file: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc


def _cmd_infer(args: argparse.Namespace) -> int:
    # Fold onto whatever is already stored so batches accumulate like a stream.
    lens = Lens(args.path)
    if os.path.exists(lens.schema_path):
        lens.load()
    lens.infer(_read_jsonl(args.records))
    lens.save()
    json.dump(lens.stats(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    lens = Lens(args.path)
    lens.load()
    mismatches = 0
    for index, record in enumerate(_read_jsonl(args.records), start=1):
        issues = lens.check(record)
        for issue in issues:
            print(f"record {index}: {issue}")
        if issues:
            mismatches += 1
    return 1 if mismatches else 0


def _cmd_show(args: argparse.Namespace) -> int:
    lens = Lens(args.path)
    lens.load()
    json.dump(lens.schema(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="schema_lens",
        description="Infer a schema from JSON records and check records against it.",
    )
    parser.add_argument(
        "--path", required=True, help="lens directory holding the schema file"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    infer = sub.add_parser("infer", help="fold JSONL records into the schema")
    infer.add_argument("records", help="JSONL file of records")
    infer.set_defaults(func=_cmd_infer)

    check = sub.add_parser("check", help="report records departing from the schema")
    check.add_argument("records", help="JSONL file of records")
    check.set_defaults(func=_cmd_check)

    show = sub.add_parser("show", help="print the stored schema as JSON")
    show.set_defaults(func=_cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SchemaConflict as exc:
        print(f"schema conflict: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
