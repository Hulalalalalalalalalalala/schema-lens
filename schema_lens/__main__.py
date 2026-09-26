"""Command line entry point: python3 -m schema_lens --path <lens> <command>."""

from __future__ import annotations

import argparse
import json
import sys

from .core import Lens, SchemaConflict


def _read_records(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return records


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
            for lineno, record in enumerate(_read_records(args.records), 1):
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
