# schema-lens

Infers a schema from a stream of JSON records and validates later records against it, so a pipeline can name and reject the records that would break its downstream consumers.

## Requirements

Python 3.11 or newer. Standard library only.

## Install

    python3 -m pip install -e .

## Run

    python3 -m schema_lens --path ./lens infer <records.jsonl>
    python3 -m schema_lens --path ./lens check <records.jsonl>
    python3 -m schema_lens --path ./lens show

## Public interface

`schema_lens.Lens(path)` opens the lens directory `path`.
- `infer(records) -> dict` folds a sequence of records into the stored schema.
- `check(record) -> list[str]` reports every way one record departs from the stored schema.
- `schema() -> dict` returns the stored schema.
- `save() -> None` and `load() -> None` persist it and re-read it before replacing memory.
- `stats() -> dict` reports fields, optional fields, observed types and observation counts.
- `SchemaConflict` exported exception.

## Streaming inference under resource limits

`infer` accepts any iterable, including a generator over a record stream,
and the command line reads records one at a time, so a single inference can
be fed millions of records in batches. Folded records are never retained:
once a record is folded, only the per-field and per-node statistics remain,
and peak memory tracks the statistics limits below rather than the number
of records seen.

`Lens(path, max_fields=..., max_depth=...)` (and the matching `--max-fields`
/ `--max-depth` command line options) bound the statistics memory:

- `max_fields` caps how many field entries each object node tracks
  exactly. Once the cap is reached, further distinct field names fold into
  a shared overflow entry kept under the node's `"overflow"` key and
  flagged `"approximate": true`.
- `max_depth` caps how deep object and array nodes are tracked (the root
  is depth 0). Deeper subtrees collapse into summary nodes flagged
  `"approximate": true` that keep only a record count (objects) or element
  type kinds (arrays).

Both default to `None`, meaning no limit. With no limit in effect no
approximation is ever triggered and the folded schema is exactly the
schema produced by folding every record in one pass. When approximation
does trigger, the difference against the exact result is confined to the
marked entries; every unmarked field stays exact. An approximated entry's
type set can only widen and its observation count only grows as more
records are folded, so it never silently drops a type it once reported.

Approximation is always visible, never silent: the flags appear in the
`schema()` tree and the `show` output, and `stats()` lists the path of
every approximated statistic under its `"approximate"` key — collapsed
subtrees by their own path, overflow entries as `$["*"]` under their
object node's path. When checking records, a field name that fell into an
overflow entry is validated against the overflow entry's widened type set
instead of being reported as unexpected, and collapsed subtrees are not
descended into.

## Incremental inference

Inference is an append process: call `infer` with more batches and `save`
again whenever you like. Only the records folded since the last `load` or
`save` are merged into the committed schema, so earlier batches stay folded
in and are never recomputed or lost. The schema read back after any number
of appended batches is exactly the schema produced by folding every record
once in one pass.

`save` commits through `schema.lock`, a non-blocking lock file inside the
lens directory. When another process holds the lock the call raises
`SchemaConflict` without changing the committed schema; retry the same
`save` later (in-memory state is kept) and its batch still lands on top of
everything committed in the meantime.

The schema is stored as `schema.json`, a single JSON object with a
`version`, a `checksum` (a SHA-256 of the whole schema in canonical form)
and the `root` schema. The current format version is 2; version 1 files
are still read, and are migrated in place by one atomic replacement the
first time they are read, so after migration the file's checksum matches
its content again. A failed migration (say, an unwritable directory)
leaves the original file byte for byte untouched, leaves the in-memory
schema exactly as it was, and raises `SchemaConflict`, so the read can
simply be retried later. Each commit writes a fresh file and atomically
replaces the old one, so a crash or interrupted write can only leave the
previous complete schema or the new complete schema — never a readable file
that fails its checksum. `load` parses and fully validates the version, the
structure and the checksum, and only then replaces the in-memory schema; a
missing or invalid file raises `SchemaConflict` and leaves memory exactly
as it was.

Field paths in check reports keep dotted and quoted keys apart: ordinary
identifier fields read as `$.user.name`, array elements as `$.tags[1]`, and
any key that is not an identifier is bracketed and JSON-quoted, so a field
literally named `a.b` is reported as `$["a.b"]` and can never be confused
with the nested path `$.a.b`. Backslashes are quoted the same way
(`$["a\\b"]`).

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

JSON compatible values only: no dates, decimals or binary.
Object fields are matched by name; array element types are merged.
Empty arrays contribute no element types.
No coercion: a record either matches the schema or is reported.
