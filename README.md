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
    python3 -m schema_lens --path ./lens versions
    python3 -m schema_lens --path ./lens rollback <revision>
    python3 -m schema_lens --path ./lens compact
    python3 -m schema_lens --path ./lens compact-status

`check` and `show` accept `--version <revision>` to target one specific
committed revision; without it they use the snapshot loaded at open time.

## Public interface

`schema_lens.Lens(path)` opens the lens directory `path`.
- `infer(records) -> dict` folds a sequence of records into the stored schema.
- `check(record, *, version=None) -> list[str]` reports every way one record departs from the stored schema, or from one committed revision.
- `schema(*, version=None) -> dict` returns the stored schema, or one committed revision.
- `save() -> None` and `load(*, version=None) -> None` commit and read before replacing memory.
- `versions() -> list[int]` lists committed revision numbers, oldest first.
- `rollback(version) -> int` commits the target revision's schema again as a new revision and returns the new revision number.
- `compact(*, wait=True) -> dict` merges the oldest stretch of revisions into one baseline revision and returns the compaction status.
- `compact_status() -> dict` reports whether a compaction round is running, the current baseline, the merged-away revisions and the readable revisions.
- `stats() -> dict` reports fields, optional fields, observed types and observation counts.
- `SchemaConflict` exported exception.

`Lens(path, max_versions=...)` (and the matching `--max-versions` command
line option, default 10) bounds how many complete revisions the lens
directory keeps; every commit beyond the bound prunes the oldest
revisions. A pruned revision is no longer readable or a valid rollback
target.

## Revisions, snapshots and rollback

Every `save` is one commit: the records folded since the last `load` or
`save` are merged into the newest committed revision and the result is
written as a brand new, complete, immutable revision file under
`versions/` (`revision-0000000001.json`, `revision-0000000002.json`, ...),
each carrying its own monotonic `revision` number and a SHA-256
`checksum` of the whole schema. Revision files are never rewritten and
revision numbers never repeat. A revision file appears only by an atomic
rename of a fully synced file, so after a crash or an interrupted commit
each revision is either completely readable or absent, and concurrent
readers always observe one complete revision or another, never a
half-written schema.

Commits are serialized by `schema.lock`, a non-blocking lock file inside
the lens directory. When another thread or process holds the lock the
call raises `SchemaConflict` without changing any committed revision; the
in-memory batch is kept, so retrying the same `save` later lands on top of
everything committed in the meantime. Every revision is parsed,
structurally validated and checksummed in full before any of its content
is used. A missing or corrupt requested revision (an explicit
`version=`, a `rollback` target, or the newest revision on an unqualified
read) raises `SchemaConflict`; one damaged revision does not make the
other revisions unusable.

`load()` reads the newest revision; `load(version=N)` reads exactly
revision N. Memory is replaced only after the read validates completely.
Reads while other commits are in flight are snapshots: once a lens is
opened at a revision, `check`/`schema`/`stats` keep answering for that
revision no matter how many newer commits land, until the next `load`.
Inference still appends batches as before; the held delta merges onto the
newest committed revision at the next `save`. `check(record, version=N)`
and `schema(version=N)` read one committed revision directly without
moving the open snapshot.

`versions()` returns the committed revision numbers oldest first and is
empty on a directory with no history; reading any specific revision in an
empty history raises `SchemaConflict`. `rollback(N)` expresses the target
revision as a new commit: revision N is never rewritten or deleted, a new
revision carrying an equivalent schema is appended on top, and the
returned revision is what subsequent unqualified reads see. Rolling back
to a revision that is missing, pruned or corrupt raises `SchemaConflict`
and changes no committed revision.

A lens directory that only holds the old single-file format
(`schema.json`, format version 1 or 2) is still read. The legacy file is
validated completely and migrated in place into revision 1 the first time
it is read (including the first explicit revision read or rollback); after
migration the legacy file is removed. A failed migration (say, an
unwritable directory) leaves the original file byte for byte untouched and
raises `SchemaConflict`, so the read can simply be retried later.

## Background compaction

The oldest stretch of retained revisions can be merged into one baseline
revision with `compact()` (also the `compact` command); commits schedule
the same round in the background once the live history fills to the
compaction watermark, and `compact_status()` (the `compact-status`
command) reports the state:

- `running` — whether a round is in flight,
- `baseline` — the current baseline revision number (or null),
- `compacted` — the revision numbers merged into the baseline,
- `revisions` — the readable revision numbers, oldest first,
- `pending` — the readable revisions after the baseline.

Each revision is already a complete cumulative schema, so the merged
range's final pattern and statistics are exactly its boundary revision's:
that revision is fully validated and copied verbatim into the baseline
product, which keeps the boundary's revision number and reads as that
complete revision. The product is staged under a temp name, flushed and
fsynced, re-read from disk and structurally and checksum validated before
a single small pointer file (`versions/baseline.json`) atomically
switches to it; only after the switch are the merged originals deleted.
Up to the switch every original stays readable; afterwards requests for a
merged revision raise `SchemaConflict`, while the baseline and every later
revision read normally.

A crash or an interrupted round leaves no half state: a staging temp is
discarded, a baseline renamed but never pointed at is treated as
unpublished and removed, and a pointer that switched simply has its
pending file cleanup finished by the next lock-holding operation. The
pointer is the only switch, so a round is wholly present or wholly absent
and the directory never holds two baselines. Reads never take the lock, so
compaction and commits never block reads and every read lands on one
complete revision; a read that catches the instant one baseline is
switched for the next follows the new pointer once instead of failing.

Compaction does not change any other semantics: the revision list keeps
its oldest-first order, rollback still expresses itself as a new commit,
history is never rewritten, one damaged revision only affects that
revision, and type merging, optional-field judgment and mismatch reports
are unchanged with no coercion. Resolving a revision is a constant-time
pointer lookup, listing revisions is one directory read plus one small
pointer file, and a round reads one revision and writes one file, so
lookups do not scan unrelated files and a round does not re-read or
re-merge the whole history.

The merged schema also re-imposes `max_fields` after two batches merge:
disjoint field names across batches can never leave a merged object node
holding more exact entries than the cap — excess entries fold into that
node's overflow entry, and with the cap in force the committed result is
the same schema a single capped pass over every record would produce.

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
subtrees by their own path, overflow entries as `$.*` under their object
node's path. The marker segment `*` is not a quoted key and cannot be a
real field name, so a field literally named `*` is still reported as
`$["*"]` and never confused with the overflow marker. When checking
records, a field name that fell into an overflow entry is validated
against the overflow entry's widened type set instead of being reported
as unexpected, and collapsed subtrees are not descended into.

## Incremental inference

Inference is an append process: call `infer` with more batches and `save`
again whenever you like. Only the records folded since the last `load` or
`save` are merged into the committed schema, so earlier batches stay folded
in and are never recomputed or lost. The schema read back after any number
of appended batches is exactly the schema produced by folding every record
once in one pass; when a field cap is in force, the cap is re-imposed on
the merged tree after each commit so the bound holds across batches too.

Field paths in check reports keep dotted and quoted keys apart: ordinary
identifier fields read as `$.user.name`, array elements as `$.tags[1]`, and
any key that is not an identifier is bracketed and JSON-quoted, so a field
literally named `a.b` is reported as `$["a.b"]` and can never be confused
with the nested path `$.a.b`. Backslashes are quoted the same way
(`$["a\\b"]`), and a field literally named `*` reads as `$["*"]`, distinct
from the `$.*` overflow marker.

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

JSON compatible values only: no dates, decimals or binary.
Object fields are matched by name; array element types are merged.
Empty arrays contribute no element types.
No coercion: a record either matches the schema or is reported.
