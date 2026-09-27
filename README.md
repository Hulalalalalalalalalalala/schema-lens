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
- `stats() -> dict` reports fields, optional fields, observed types, observation counts and the paths of approximated fields.
- `SchemaConflict` exported exception.

`Lens(path, field_limit=…, depth_limit=…)` additionally bounds the
statistics; the limits can also be set with the `SCHEMA_LENS_FIELD_LIMIT`
and `SCHEMA_LENS_DEPTH_LIMIT` environment variables (constructor arguments
win). They default to `10000` and `64`, under which folds stay exact.

## Bounded streaming inference

`infer` consumes records one at a time — any iterable works, including a
generator over a multi-million-line JSONL file — and never retains record
contents. Memory holds only the per-field and per-node statistics, so peak
memory follows the statistics limits, not the number of records folded.

Two limits bound the statistics:

- **field limit** (`field_limit`): the number of distinct field names one
  object node tracks exactly. Once the cap is reached, further field names
  fold into an `"other"` overflow bucket that only keeps per-type
  observation counts.
- **depth limit** (`depth_limit`): how deep composite values (objects and
  arrays) are unfolded. A value past the limit collapses into a
  `"truncated"` summary node that only records how often it was seen.

Approximation is never silent: a node that absorbed overflow (`"other"`) or
truncated values (`"truncated"`) carries `"approximate": true`, the flag
propagates onto the field entries above it, and `stats()["approximate"]`
lists the dotted path of every approximated site (e.g. `$.user`,
`$.tags[]`; the root object itself is listed as `$`). Overflowed field
names pass `check` (their type set was widened), and a truncated node
accepts any interior; exactly tracked fields keep their exact semantics.

When neither limit is ever reached the result is exactly the schema
produced by folding every record once in one pass — no approximation
markers appear at all. After approximation, only the marked fields differ
from the exact result; every other field stays exact. An approximated
field's type set can only grow wider, never narrower, and observation
counts only grow as more records are folded.

## Incremental inference

Inference is an append process: call `infer` with more batches and `save`
again whenever you like. Only the records folded since the last `load` or
`save` are merged into the committed schema, so earlier batches stay folded
in and are never recomputed or lost. With no statistics limit reached the
schema read back after any number of appended batches is exactly the schema
produced by folding every record once in one pass; fields folded under a
limit are marked approximate as described above.

`save` commits through `schema.lock`, a non-blocking lock file inside the
lens directory. When another process holds the lock the call raises
`SchemaConflict` without changing the committed schema; retry the same
`save` later (in-memory state is kept) and its batch still lands on top of
everything committed in the meantime.

The schema is stored as `schema.json`, a single JSON object with a
`version`, a `checksum` (a SHA-256 of the whole schema in canonical form)
and the `root` schema. The current format is version 2; version 1 files
are still read and migrated in memory, their content exactly equivalent
after migration, and the next `save` re-commits them as version 2. The
original file is never modified by a failed migration, which raises
`SchemaConflict`, leaves the in-memory schema untouched, and can be
retried later. Each commit writes a fresh file and atomically replaces the
old one, so a crash or interrupted write can only leave the previous
complete schema or the new complete schema — never a readable file that
fails its checksum. `load` parses and fully validates the version, the
structure and the checksum, and only then migrates and replaces the
in-memory schema; a missing or invalid file raises `SchemaConflict` and
leaves memory exactly as it was. A failed migration on `save` likewise
raises `SchemaConflict` and leaves both the file and memory untouched.

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
