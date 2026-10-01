"""Command line entry point: python3 -m schema_lens --path <lens> <command>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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
        choices=["infer", "check", "show", "versions", "rollback", "compact",
                 "status", "compat", "matrix"],
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="JSONL records file (infer/check), revision number (rollback) "
             "or old revision number (compat)",
    )
    parser.add_argument(
        "target2",
        nargs="?",
        help="new revision number (compat only)",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=None,
        help="compact: keep at least this many newest revisions out of the baseline",
    )
    parser.add_argument(
        "--revisions",
        type=str,
        default=None,
        help="matrix: comma-separated ascending revision numbers "
             "(default: every committed revision)",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="compact: run the merge in the background",
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
    if args.keep is not None and args.command != "compact":
        return fail("--keep is only valid with compact")
    if args.background and args.command != "compact":
        return fail("--background is only valid with compact")
    if args.revisions is not None and args.command != "matrix":
        return fail("--revisions is only valid with matrix")
    if args.command == "compact" and args.target is not None:
        return fail("compact takes no positional argument")
    if args.command == "status" and args.target is not None:
        return fail("status takes no positional argument")
    if args.command == "matrix" and args.target is not None:
        return fail("matrix takes no positional argument")
    if args.command == "compat":
        if args.target is None or args.target2 is None:
            return fail("compat requires two revision numbers: <old> <new>")
        try:
            compat_old = int(args.target)
            compat_new = int(args.target2)
        except ValueError:
            return fail("compat revisions must be integers")
    else:
        if args.target2 is not None:
            return fail("only compat takes two positional arguments")
    matrix_revisions: list[int] | None = None
    if args.command == "matrix" and args.revisions is not None:
        try:
            matrix_revisions = [
                int(part) for part in args.revisions.split(",")
            ]
        except ValueError:
            return fail("matrix revisions must be integers")
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
    if args.keep is not None:
        lens_kwargs["compact_keep"] = args.keep
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
        if args.command == "compat":
            report = lens.compat(compat_old, compat_new)
            json.dump(report, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.command == "matrix":
            report = lens.compat_matrix(matrix_revisions)
            json.dump(report, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.command == "status":
            json.dump(lens.compaction_status(), sys.stdout, indent=2,
                      sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.command == "compact":
            if args.background:
                import subprocess

                log = open(Path(args.path) / "compaction.log", "a",
                           encoding="utf-8")
                child_argv = [
                    sys.executable, "-m", "schema_lens",
                    "--path", args.path,
                ]
                if args.max_fields is not None:
                    child_argv += ["--max-fields", str(args.max_fields)]
                if args.max_depth is not None:
                    child_argv += ["--max-depth", str(args.max_depth)]
                if args.max_versions is not None:
                    child_argv += ["--max-versions", str(args.max_versions)]
                if args.keep is not None:
                    child_argv += ["--keep", str(args.keep)]
                child_argv.append("compact")
                subprocess.Popen(
                    child_argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                log.close()
                json.dump({"started": True}, sys.stdout, sort_keys=True)
                sys.stdout.write("\n")
                return 0
            result = lens.compact()
            if result is None:
                payload = {"compacted": False}
            else:
                payload = {"compacted": True, **result}
            json.dump(payload, sys.stdout, sort_keys=True)
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
