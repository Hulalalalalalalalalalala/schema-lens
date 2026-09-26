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
- `infer(records) -> dict` folds a sequence of records into the stored schema. Repeated calls append batches; inferring in several batches yields the same schema as folding all records at once.
- `check(record) -> list[str]` reports every way one record departs from the stored schema.
- `schema() -> dict` returns the stored schema.
- `save() -> None` commits the pending batch to `schema.json` as one atomic write carrying a version and a checksum of the whole schema. Commits are serialized through a lock file in the lens directory, so concurrent processes never lose or overwrite committed batches; a caller that cannot take the lock raises `SchemaConflict` and may retry.
- `load() -> None` re-reads the stored schema and replaces memory only after the file fully validates (JSON, version, structure, checksum); a missing, truncated or tampered file raises `SchemaConflict` and leaves memory untouched.
- `stats() -> dict` reports fields, optional fields, observed types and observation counts.
- `SchemaConflict` exported exception.

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

JSON compatible values only: no dates, decimals or binary.
Object fields are matched by name; array element types are merged.
No coercion: a record either matches the schema or is reported.
Field names containing dots or backslashes appear in reports in bracketed
form (`$["a.b"]`) so a report path always locates one field unambiguously.
