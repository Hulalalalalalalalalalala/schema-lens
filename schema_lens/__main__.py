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
        "--max-versions",
        type=int,
        default=None,
        help="keep at most this many complete revisions; older ones are pruned",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        dest="revision",
        help="revision number to check against or show (default: the open snapshot)",
    )
    parser.add_argument(
        "command",
        choices=[
            "infer",
            "check",
            "show",
            "versions",
            "rollback",
            "compact",
            "compact-status",
        ],
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="JSONL records file (infer/check) or revision number (rollback)",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse already printed the usage error; keep the command's
        # exit-code contract instead of unwinding through the caller.
        return exc.code if isinstance(exc.code, int) else 2

    def fail(message: str) -> int:
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: error: {message}", file=sys.stderr)
        return 2

    if args.revision is not None and args.command not in ("check", "show"):
        return fail("--version is only valid with check and show")
    if args.command in ("infer", "check") and not args.target:
        return fail(f"{args.command} requires a records file")
    rollback_revision: int | None = None
    if args.command == "rollback":
        if args.target is None:
            return fail("rollback requires a revision number")
        try:
            rollback_revision = int(args.target)
        except ValueError:
            return fail("rollback revision must be an integer")

    lens_kwargs = {}
    if args.max_versions is not None:
        lens_kwargs["max_versions"] = args.max_versions
    lens = Lens(
        args.path,
        max_fields=args.max_fields,
        max_depth=args.max_depth,
        **lens_kwargs,
    )
    try:
        if args.command == "infer":
            lens.infer(_iter_records(args.target))
            lens.save()
            json.dump(lens.stats(), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.command == "check":
            if args.revision is None:
                lens.load()
            problems = 0
            for lineno, record in enumerate(_iter_records(args.target), 1):
                for report in lens.check(record, version=args.revision):
                    problems += 1
                    print(f"line {lineno}: {report}")
            return 1 if problems else 0
        if args.command == "show":
            if args.revision is None:
                lens.load()
            json.dump(
                lens.schema(version=args.revision),
                sys.stdout,
                indent=2,
                sort_keys=True,
            )
            sys.stdout.write("\n")
            return 0
        if args.command == "versions":
            json.dump(lens.versions(), sys.stdout)
            sys.stdout.write("\n")
            return 0
        if args.command == "compact-status":
            json.dump(lens.compact_status(), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.command == "compact":
            status = lens.compact()
            json.dump(status, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        # rollback: the target schema is committed again as a new revision.
        new_revision = lens.rollback(rollback_revision)
        json.dump(
            {"rolled_back_to": rollback_revision, "revision": new_revision},
            sys.stdout,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0
    except SchemaConflict as exc:
        print(f"schema_lens: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"schema_lens: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
