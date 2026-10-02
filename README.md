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
    python3 -m schema_lens --path ./lens compact [--keep <revisions>] [--background]
    python3 -m schema_lens --path ./lens status
    python3 -m schema_lens --path ./lens compat <old-revision> <new-revision>
    python3 -m schema_lens --path ./lens witness <old-revision> <new-revision>
    python3 -m schema_lens --path ./lens matrix [--revisions 1,2,3]

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
- `compact(*, keep=None, background=False) -> dict | None` merges the oldest history segment into one baseline version and returns `{"anchor": ..., "merged": [...]}` (or `None` when the history is already within the keep bound).
- `compaction_status() -> dict` reports the committed baseline, the revisions merged into it, the live logical history and any background run in progress.
- `compat(old_revision, new_revision) -> dict` reads two committed snapshots and reports how the evolution between them affects records legal under each side, without changing memory or any revision.
- `compat_witness(old_revision, new_revision) -> dict` returns the same report as `compat` plus a `witnesses` object with a concrete, `check`-verified counterexample for each breaking direction (`null` for compatible/unknown, both `null` on a self-comparison), without changing memory or any revision.
- `compat_matrix(revisions=None) -> dict` reads committed snapshots for every revision (or a strictly increasing selection) and reports both directional verdicts for every ordered pair, self-comparisons included, in one matrix.
- `migrate(record, old_revision, new_revision, rules) -> dict` returns the `compat` report for the pair plus a `migration` object previewing an explicit field-level migration of one JSON object: check against the old revision, apply the rename/drop/default rules in order, check against the new revision. Read-only like `compat`.
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

## Background compaction

`compact(keep=N)` merges the oldest segment of the history into one
baseline version: at least the newest `N` revisions (default 5, or the
`compact_keep=` constructor argument / `--keep` option) stay separate and
every revision older than that folds into a single baseline whose schema
and statistics are exactly those of the last merged revision (its anchor).
The baseline is stored as `versions/baseline-<anchor>.json` with the same
checksum guarantees as a revision, and a small `versions/manifest.json` is
the single pointer naming it. After compaction:

- the anchor is readable as its own revision number (via `schema`,
  `check`, `load` and as a `rollback` target) and answers exactly as the
  original anchor revision did;
- the other revisions merged into the baseline are no longer readable -
  requesting them (explicitly, by rollback, or by pinning `--version`)
  raises `SchemaConflict`;
- every surviving revision and all subsequent commits behave exactly as
  before; `versions()` lists the anchor followed by the surviving tail.

Compaction never blocks a read and can run while commits land. The merge
reads only the files of the merged segment (the surviving tail is never
re-read, so one compaction round costs strictly less than re-reading the
whole history), builds the baseline in memory, stages it as a temp
artifact, and switches only after the staged tree has been checked
field-for-field against the anchor revision both from memory and again
after re-reading the staged file. The manifest pointer is replaced
atomically as the single switch; none of the merged revision files is
deleted until that replacement has landed. A crash or interruption is
therefore reconciled the next time the lock is taken into either "the
compaction fully happened" (manifest points at a complete baseline;
leftover merged files and stale baselines are removed) or "it never
happened" (no manifest; every staged baseline and temp file is removed),
never a half baseline or two baselines at once. While a compaction runs,
every read still observes one complete version through the old pointer;
if a commit moves the history under an in-flight compaction it retries on
the fresh range.

`compact(background=True)` (or `--background`, which detaches a worker
process) runs the merge without blocking the caller; `compaction_status()`
(or the `status` command) reports whether it is `running`, its `[first,
last]` merged `range`, the committed `baseline` anchor and `merged`
revisions, the live `versions`, and any background `error`. A background
failure changes no committed revision and is reported there.

Retention never deletes the baseline: the bound applies to numbered
revision files only, so the baseline plus the newest revisions always
remain readable. `compact` returns `None` (the command prints
`{"compacted": false}`) when the history already holds at most `keep`
versions.

A lens directory that only holds the old single-file format
(`schema.json`, format version 1 or 2) is still read. The legacy file is
validated completely and migrated in place into revision 1 the first time
it is read (including the first explicit revision read or rollback); after
migration the legacy file is removed. A failed migration (say, an
unwritable directory) leaves the original file byte for byte untouched and
raises `SchemaConflict`, so the read can simply be retried later.

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
records, a field name that actually folded into the overflow entry is
validated against its widened type set instead of being reported as
unexpected (the folded names are recorded in the entry's `names` list), a
name never seen before is still an unexpected field, and collapsed
subtrees are not descended into.

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

## Cross-revision compatibility

`compat(old, new)` (command: `compat <old-revision> <new-revision>`) reads
the two complete, fully validated snapshots and reports how the evolution
from `old` to `new` affects records that were legal under each side. It
never infers, commits, rolls back or changes the open snapshot, and it
writes nothing but its JSON report:

```json
{
  "from": 1,
  "to": 2,
  "backward": "compatible",
  "forward": "breaking",
  "changes": [
    {"path": "$.tags", "change": "element_type_number_added", "breaking": false}
  ],
  "unknown_reasons": []
}
```

`backward` says whether the new revision accepts every record the old
revision accepted (can old records be read by the new schema?); `forward`
says whether the old revision accepts every record the new revision
accepts. Each verdict is one of `compatible`, `breaking` or `unknown`,
judged by exactly what `check` does with the record shape, field
optionality and the observed type sets — no defaults, enums or migrations
are invented.

In the accepting direction these evolutions are non-breaking; their
inverses are breaking:

- adding an *optional* field (new records may omit it; old records always
  do), versus removing a field (records carrying it are now unexpected);
- a required field becoming optional, versus an optional field becoming
  required (records that omitted it are now rejected);
- a type set widening (e.g. `number` to `number|string`), versus narrowing
  or a type branch disappearing, including inside nested objects and array
  elements.

Objects and array elements are compared by the same rules. A schema has no
concrete array indices, so an element-type branch renders with `[]`
(`$.tags[]`) and fields of element objects join normally
(`$.items[].name`). `changes` lists each changed path once with its
category and a `breaking` flag, sorted by path then category. Field paths
use the same notation as `check`: ordinary identifiers as `$.name`,
non-identifier keys quoted as `$["a.b"]`, backslashes quoted
(`$["a\\b"]`), and a field literally named `*` quoted as `$["*"]`, never
confused with the `$.*` overflow marker.

When an overflow entry or a depth-capped subtree on either side governs a
branch, the accepted values there cannot be compared exactly, so both
directions are `unknown` and the source is listed once in
`unknown_reasons` (the same paths `stats()` uses: `$.*` for an overflow
entry, the subtree's own path for a collapsed node). A demonstrable
rejection elsewhere still reports that change and settles that direction
as `breaking`. A revision compared with itself is `compatible` in both
directions with no changes, and repeating an analysis of the same pair
returns the same report.

A missing, pruned/compacted-away or corrupt revision raises
`SchemaConflict` (the command exits `2` and prints no report); giving
fewer or more than two revision numbers is an argument error (`2`, no
report). A completed analysis exits `0` even when a verdict is `breaking`
or `unknown`.

## Breaking-change witnesses

`compat_witness(old, new)` (command: `witness <old-revision>
<new-revision>`) runs the same analysis as `compat` - identical fields,
content and ordering - and adds one `witnesses` object holding a
`backward` and a `forward` key. Each key is `null`, or an object with a
concrete counterexample:

```json
{
  "from": 1,
  "to": 2,
  "backward": "breaking",
  "forward": "compatible",
  "changes": [
    {"path": "$.b", "change": "field_removed", "breaking": true}
  ],
  "unknown_reasons": [],
  "witnesses": {
    "backward": {
      "record": {"a": 1, "b": "x"},
      "reports": ["$.b: unexpected field"]
    },
    "forward": null
  }
}
```

For a `backward` breaking direction, `record` is a JSON object the old
revision's `check` accepts (its report list is empty) and `reports` is the
exact, complete, non-empty list the new revision's `check` returns for it;
for `forward` breaking the old and new roles are swapped. Either key is
`null` when its direction is `compatible` or `unknown`, and a
self-comparison has both `null`.

Counterexamples are assembled from the accepting revision's own schema
tree while the compatibility diff walks the two trees (required sibling
fields filled, array branches reached through a single element), and each
candidate is run through `check` itself before it is reported, so a
witness can be re-checked directly against the lens. They cover added and
removed fields, optionality changes, type-set changes, nested objects,
array element objects and multi-level arrays; booleans are never used as
numbers, a missing field is never represented as `null`, and an empty
array is used where the tracked element set admits no element. Field
names containing dots, backslashes or asterisks keep their real key in
the record while the report paths stay quoted exactly as `check` prints
them (`$["a.b"]`, `$["a\\b"]`, `$["*"]`). A merely approximate branch
yields no witness, but when another branch elsewhere proves a direction
breaking its witness is still supplied; only `unknown` directions have no
counterexample.

The analysis is read-only - it reads and validates the two snapshots,
never infers, saves, rolls back, compacts or changes the open snapshot,
pending batches or committed history - and the same pair always produces
the same record, object key order, report list and command stdout bytes.
Invalid revision arguments (non-integers, booleans or non-positive
numbers) and missing, cleaned-up (pruned/compacted-away) or corrupt
revisions raise `SchemaConflict`; the command exits `2` with empty stdout
and the message on stderr, and exits `0` with only the report JSON on
success, including `breaking` or `unknown` conclusions.

## Cross-version compatibility matrix

`compat_matrix(revisions=None)` (command: `matrix [--revisions 1,2,3]`)
collects the pairwise reports of `compat` into one document for judging
upgrade and rollback risk across several revisions. It is strictly
read-only: it reads and validates the committed snapshots, never infers,
commits, rolls back, compacts, or changes the open snapshot:

```json
{
  "revisions": [1, 2],
  "pairs": [
    {"from": 1, "to": 1, "backward": "compatible", "forward": "compatible",
     "changes": [], "unknown_reasons": []},
    {"from": 1, "to": 2, "backward": "compatible", "forward": "breaking",
     "changes": [{"path": "$.b", "change": "field_added", "breaking": false}],
     "unknown_reasons": []},
    {"from": 2, "to": 1, "backward": "breaking", "forward": "compatible",
     "changes": [{"path": "$.b", "change": "field_removed", "breaking": true}],
     "unknown_reasons": []},
    {"from": 2, "to": 2, "backward": "compatible", "forward": "compatible",
     "changes": [], "unknown_reasons": []}
  ]
}
```

`revisions` lists the analysed revision numbers in analysis order and
`pairs` covers every ordered combination - including each revision
compared with itself - sorted by `from` then `to`. Every pair carries
its own `backward` and `forward` verdicts (`compatible`, `breaking` or
`unknown`, the two directions never merged, and `unknown` never confused
with another verdict), plus the same `changes` and `unknown_reasons` a
single `compat` report would give for that pair. A self-comparison is
`compatible` in both directions with empty `changes` and
`unknown_reasons`, even when the revision carries approximated branches;
pairs touching approximate fields or collapsed subtrees still report
fully, including their unknown reasons, and all other pairs are
unaffected.

With `revisions` omitted every currently committed logical revision is
analysed in ascending revision order; an explicit selection must contain
distinct positive integers in strictly increasing order. An empty
history with no selection raises `SchemaConflict`; an empty selection or
a boolean, non-positive, duplicated or non-increasing value raises
`ValueError`; a revision that does not exist, has been pruned or
compacted away, or is corrupt raises `SchemaConflict`. The command exits
`2` and prints no report in every one of those cases; on success it exits
`0` with only the matrix JSON on stdout (errors go to stderr), and
repeating the same inputs produces byte-identical content and ordering.

## Field migration preview

`migrate(record, old_revision, new_revision, rules)` (Python only)
previews an explicit field-level migration of one JSON object between
two committed revisions. It returns exactly the `compat` report for the
pair plus one `migration` object:

```json
{
  "from": 1, "to": 2,
  "backward": "compatible", "forward": "breaking",
  "changes": [{"path": "$.c", "change": "field_added", "breaking": false}],
  "unknown_reasons": [],
  "migration": {"stage": "done", "record": {"a": 1, "c": "x"}, "reports": []}
}
```

The record is first checked against the old revision, then transformed
by the rules in order (each rule applies to the result of the previous
one; an empty rule array validates without transforming), then checked
against the new revision - both checks judged exactly as `check` judges
optional fields, type unions and approximate branches. When both pass,
`stage` is `"done"`, `record` is the complete migrated record and
`reports` is empty. When the source or target check fails, `stage` is
`"source"` or `"target"`, `record` is `null` and `reports` is the full
list that revision's `check` returns. Migrating from a revision to
itself still applies the rules and runs both checks.

`rules` is a JSON array of rule objects. Every rule has an `op` and a
`path`; `rename` adds a string `to` naming the new sibling field,
`default` adds a JSON `value` to fill in, and `drop` takes nothing
else. No other keys are allowed. The path is a non-empty array whose
string segments are literal field names (a name containing a dot names
one field, never a descent) and whose `null` segments select every
element of the array at that position; the last segment is always a
string, so `["items", null, "a.b"]` names the field `a.b` inside every
element of `items`. `rename` keeps the field's value under the new
name, skipping objects where the source field is missing and raising
`ValueError` where the target name already exists; `drop` removes the
field, skipping it where missing; `default` fills only a genuinely
missing field and never replaces a present `null`. A missing parent
field skips the rule for that branch (parents are never created, and an
empty array simply selects nothing), while a string segment meeting a
non-object or a `null` segment meeting a non-array raises `ValueError`.

The record must be a JSON object; a non-JSON value, a malformed rule
(unknown op, missing or extra keys, an invalid path, a rename onto the
same name) raises `ValueError`. Invalid revision numbers and missing,
pruned/compacted-away or corrupt revisions raise `SchemaConflict`; a
compaction anchor stays readable, and a legacy single-file lens
directory is migrated in place on first read exactly as for the other
entry points. Nothing else is written and nothing shared is mutated:
the input record and rules, the open snapshot, uncommitted batches and
every committed revision are exactly as they were, and the same input
always returns the same content in the same order.

## Tests

    python3 -m unittest discover -s tests -t .

## Limits

JSON compatible values only: no dates, decimals or binary.
Object fields are matched by name; array element types are merged.
Empty arrays contribute no element types.
No coercion: a record either matches the schema or is reported.
