"""Command line entry point: python3 -m schema_lens --path <lens> <command>."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterator

from .core import Lens, SchemaConflict


def _iter_records(path: str) -> Iterator[dict]:
    """Yield records one at a time so streams of any size can be folded
    without ever holding the file's contents in memory."""
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{path}: cannot read records file: {exc}") from exc
    with handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: record is not a JSON object")
            yield record


def _read_records(path: str) -> list:
    return list(_iter_records(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="schema_lens",
        description="Infer a schema from JSON records and validate against it.",
    )
    parser.add_argument("--path", required=True, help="lens directory")
    parser.add_argument(
        "--max-fields",
        type=int,
        default=None,
        help="cap exactly tracked fields per object; overflow is approximated",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="cap nesting depth of tracked nodes; deeper subtrees are approximated",
    )
    parser.add_argument(
        "--keep-versions",
        type=int,
        default=10,
        help="retain at most this many complete revisions; older ones are compacted",
    )
    parser.add_argument(
        "--version",
        dest="revision",
        type=int,
        default=None,
        help="revision to read for check/show (default: the snapshot opened at load)",
    )
    parser.add_argument(
        "command", choices=["infer", "check", "show", "versions", "rollback"]
    )
    parser.add_argument(
        "records",
        nargs="?",
        help="JSONL records file (infer/check) or revision number (rollback)",
    )
    args = parser.parse_args(argv)

    if args.revision is not None and args.command not in ("check", "show"):
        parser.error("--version only applies to check and show")

    lens = Lens(
        args.path,
        max_fields=args.max_fields,
        max_depth=args.max_depth,
        keep_versions=args.keep_versions,
    )
    try:
        if args.command == "infer":
            if not args.records:
                parser.error("infer requires a records file")
            lens.infer(_iter_records(args.records))
            lens.save()
            json.dump(lens.stats(), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.command == "check":
            if not args.records:
                parser.error("check requires a records file")
            lens.load(version=args.revision)
            problems = 0
            for lineno, record in enumerate(_iter_records(args.records), 1):
                for report in lens.check(record):
                    problems += 1
                    print(f"line {lineno}: {report}")
            return 1 if problems else 0
        if args.command == "show":
            lens.load(version=args.revision)
            json.dump(lens.schema(), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.command == "versions":
            for revision in lens.versions():
                print(revision)
            return 0
        # rollback
        if args.records is None:
            parser.error("rollback requires a revision number")
        try:
            target = int(args.records)
        except ValueError:
            parser.error("rollback revision must be an integer")
        new_revision = lens.rollback(target)
        print(new_revision)
        return 0
    except SchemaConflict as exc:
        print(f"schema_lens: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"schema_lens: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
