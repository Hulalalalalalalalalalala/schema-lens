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

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

JSON compatible values only: no dates, decimals or binary.
Object fields are matched by name; array element types are merged.
No coercion: a record either matches the schema or is reported.
