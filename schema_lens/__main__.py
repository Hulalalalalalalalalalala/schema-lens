"""Command line entry point: python3 -m schema_lens --path <lens> <command>."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterator

from .core import Lens, SchemaConflict


class _RecordReader:
    """Streams JSON object records from a JSONL file, one line at a time.

    The file is opened eagerly so a missing or unreadable file raises
    ValueError at construction; the records themselves are parsed lazily as
    the reader is iterated, so arbitrarily large record streams fold
    without ever being held in memory.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        try:
            self._handle = open(path, "r", encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"{path}: cannot read records file: {exc}") from exc

    def __iter__(self) -> Iterator[dict]:
        with self._handle:
            for lineno, line in enumerate(self._handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self._path}:{lineno}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{self._path}:{lineno}: record is not a JSON object"
                    )
                yield record


def _read_records(path: str) -> _RecordReader:
    return _RecordReader(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="schema_lens",
        description="Infer a schema from JSON records and validate against it.",
    )
    parser.add_argument("--path", required=True, help="lens directory")
    parser.add_argument("command", choices=["infer", "check", "show"])
    parser.add_argument("records", nargs="?", help="JSONL records file")
    args = parser.parse_args(argv)

    lens = Lens(args.path)
    try:
        if args.command == "infer":
            if not args.records:
                parser.error("infer requires a records file")
            lens.infer(_read_records(args.records))
            lens.save()
            json.dump(lens.stats(), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0
        if args.command == "check":
            if not args.records:
                parser.error("check requires a records file")
            lens.load()
            problems = 0
            reader = _read_records(args.records)
            for lineno, record in enumerate(reader, 1):
                for report in lens.check(record):
                    problems += 1
                    print(f"line {lineno}: {report}")
            return 1 if problems else 0
        lens.load()
        json.dump(lens.schema(), sys.stdout, indent=2, sort_keys=True)
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
