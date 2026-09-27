import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from schema_lens import Lens, SchemaConflict
from schema_lens.__main__ import _read_records, main as cli_main
from schema_lens.core import (
    DEFAULT_MAX_VERSIONS,
    LOCK_FILENAME,
    OVERFLOW_MARKER,
    SCHEMA_FILENAME,
    SCHEMA_VERSION,
    VERSIONS_DIRNAME,
)

try:
    import fcntl
except ImportError:
    fcntl = None


def revision_path(directory: str, revision: int) -> Path:
    return Path(directory, VERSIONS_DIRNAME, f"revision-{revision:010d}.json")


class InferTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)

    def test_merges_types_of_same_named_field(self):
        schema = self.lens.infer([{"a": 1}, {"a": "x"}, {"a": 2.5}])
        entry = schema["fields"]["a"]
        self.assertEqual(sorted(entry["types"]), ["number", "string"])
        self.assertEqual(entry["observed"], 3)
        self.assertEqual(entry["total"], 3)
        self.assertFalse(entry["optional"])

    def test_field_present_in_some_records_is_optional(self):
        schema = self.lens.infer([{"a": 1, "b": "x"}, {"a": 2}])
        self.assertFalse(schema["fields"]["a"]["optional"])
        self.assertTrue(schema["fields"]["b"]["optional"])
        self.assertEqual(schema["fields"]["b"]["observed"], 1)
        self.assertEqual(schema["fields"]["b"]["total"], 2)

    def test_optionality_accumulates_across_batches(self):
        self.lens.infer([{"a": 1}])
        schema = self.lens.infer([{"b": 2}])
        self.assertTrue(schema["fields"]["a"]["optional"])
        self.assertTrue(schema["fields"]["b"]["optional"])
        self.assertEqual(schema["count"], 2)

    def test_nested_objects_merge_by_field_name(self):
        schema = self.lens.infer(
            [
                {"user": {"name": "ann", "age": 3}},
                {"user": {"name": "bob"}},
            ]
        )
        user = schema["fields"]["user"]["types"]["object"]
        self.assertEqual(user["count"], 2)
        self.assertFalse(user["fields"]["name"]["optional"])
        self.assertTrue(user["fields"]["age"]["optional"])

    def test_array_elements_merge_by_type(self):
        schema = self.lens.infer([{"tags": ["a", 1, None]}, {"tags": ["b"]}])
        elements = schema["fields"]["tags"]["types"]["array"]["elements"]
        self.assertEqual(sorted(elements), ["null", "number", "string"])

    def test_array_of_objects_merges_element_fields(self):
        schema = self.lens.infer([{"items": [{"x": 1}, {"x": 2, "y": "s"}]}])
        items = schema["fields"]["items"]["types"]["array"]["elements"]["object"]
        self.assertEqual(items["count"], 2)
        self.assertFalse(items["fields"]["x"]["optional"])
        self.assertTrue(items["fields"]["y"]["optional"])

    def test_boolean_is_not_number(self):
        schema = self.lens.infer([{"a": True}, {"a": 1}])
        self.assertEqual(sorted(schema["fields"]["a"]["types"]), ["boolean", "number"])

    def test_rejects_non_json_values(self):
        with self.assertRaises(ValueError):
            self.lens.infer([{"a": object()}])

    def test_rejects_non_object_records(self):
        with self.assertRaises(ValueError):
            self.lens.infer([[1, 2]])


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)
        self.lens.infer(
            [
                {"a": 1, "b": "x", "user": {"name": "ann"}, "tags": ["t"]},
                {"a": 2, "b": "y", "user": {"name": "bob"}, "tags": []},
            ]
        )

    def test_conforming_record_reports_nothing(self):
        self.assertEqual(self.lens.check({"a": 3, "b": "z", "user": {"name": "c"}, "tags": []}), [])

    def test_reports_unexpected_field(self):
        reports = self.lens.check({"a": 1, "b": "x", "user": {"name": "n"}, "tags": [], "extra": 1})
        self.assertIn("$.extra: unexpected field", reports)

    def test_reports_missing_required_field(self):
        reports = self.lens.check({"a": 1, "user": {"name": "n"}, "tags": []})
        self.assertIn("$.b: missing required field", reports)

    def test_reports_type_outside_field_type_set(self):
        reports = self.lens.check({"a": True, "b": "x", "user": {"name": "n"}, "tags": []})
        self.assertIn("$.a: type boolean not in field types [number]", reports)

    def test_reports_nested_field_path(self):
        reports = self.lens.check({"a": 1, "b": "x", "user": {"name": 5}, "tags": []})
        self.assertIn("$.user.name: type number not in field types [string]", reports)

    def test_reports_array_element_path_with_index(self):
        reports = self.lens.check({"a": 1, "b": "x", "user": {"name": "n"}, "tags": ["ok", 7]})
        self.assertIn("$.tags[1]: type number not in field types [string]", reports)

    def test_reports_every_departure(self):
        reports = self.lens.check({"a": True, "extra": 1})
        self.assertEqual(len(reports), 5)

    def test_check_does_not_modify_schema(self):
        before = self.lens.schema()
        self.lens.check({"a": True, "extra": 1})
        self.assertEqual(self.lens.schema(), before)

    def test_check_without_schema_raises(self):
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).check({"a": 1})

    def test_schema_without_schema_raises(self):
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).schema()

    def test_pinned_version_without_any_schema_raises(self):
        # An explicit revision is read from disk, so it conflicts even
        # though no in-memory schema is required for that read shape.
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).check({"a": 1}, version=1)
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).schema(version=1)


class RevisionPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_save_and_load_roundtrip(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1, "b": "x"}, {"a": "s"}])
        lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.schema(), lens.schema())

    def test_every_commit_writes_a_new_complete_revision(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        lens.infer([{"b": "x"}])
        lens.save()
        lens.infer([{"c": True}])
        lens.save()
        self.assertEqual(lens.versions(), [1, 2, 3])
        for revision in (1, 2, 3):
            path = revision_path(self.dir.name, revision)
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], SCHEMA_VERSION)
            self.assertEqual(payload["revision"], revision)
            self.assertEqual(len(payload["checksum"]), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in payload["checksum"]))

    def test_committed_revisions_are_never_rewritten(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        first = revision_path(self.dir.name, 1).read_bytes()
        lens.infer([{"b": "x"}])
        lens.save()
        lens.infer([{"c": True}])
        lens.save()
        self.assertEqual(revision_path(self.dir.name, 1).read_bytes(), first)

    def test_versions_empty_before_any_commit(self):
        self.assertEqual(Lens(self.dir.name).versions(), [])

    def test_load_missing_revisions_raises(self):
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).load()

    def test_load_replaces_memory_wholesale(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        lens.infer([{"b": 2}])
        lens.load()
        self.assertEqual(sorted(lens.schema()["fields"]), ["a"])

    def test_load_replaces_memory_wholesale_with_pinned_revision(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        lens.infer([{"b": 2}])
        lens.save()
        lens.load(version=1)
        self.assertEqual(sorted(lens.schema()["fields"]), ["a"])

    def test_save_without_schema_raises(self):
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).save()

    def test_successful_commit_leaves_no_temp_file(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        names = os.listdir(Path(self.dir.name, VERSIONS_DIRNAME))
        self.assertEqual(names, ["revision-0000000001.json"])

    def test_explicit_revision_reads_that_version_only(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        lens.infer([{"b": "x"}])
        lens.save()
        reader = Lens(self.dir.name)
        reader.load()
        snapshot = reader.schema()
        self.assertEqual(set(reader.schema(version=1)["fields"]), {"a"})
        self.assertEqual(set(reader.schema(version=2)["fields"]), {"a", "b"})
        # Pinned reads do not move the open snapshot.
        self.assertEqual(reader.schema(), snapshot)

    def test_pinned_check_uses_the_requested_revision(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        lens.infer([{"b": "x"}])
        lens.save()
        reader = Lens(self.dir.name)
        reader.load()
        # b exists only in revision 2.
        self.assertEqual(reader.check({"a": 2, "b": "y"}, version=2), [])
        reports = reader.check({"a": 2, "b": "y"}, version=1)
        self.assertIn("$.b: unexpected field", reports)
        # Open snapshot is still revision 2.
        self.assertEqual(reader.check({"a": 2, "b": "y"}), [])

    def test_missing_explicit_revision_raises_and_keeps_others_usable(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        lens.infer([{"b": "x"}])
        lens.save()
        reader = Lens(self.dir.name)
        reader.load()
        before = reader.schema()
        with self.assertRaises(SchemaConflict):
            reader.schema(version=3)
        with self.assertRaises(SchemaConflict):
            reader.check({"a": 1}, version=99)
        # The open snapshot and surviving revisions stay usable.
        self.assertEqual(reader.schema(), before)
        self.assertEqual(reader.schema(version=1), reader.schema(version=1))
        self.assertEqual(set(reader.schema(version=2)["fields"]), {"a", "b"})

    def test_pruned_revision_reads_as_missing(self):
        lens = Lens(self.dir.name, max_versions=2)
        for index in range(4):
            lens.infer([{f"f{index}": index}])
            lens.save()
        self.assertEqual(lens.versions(), [3, 4])
        with self.assertRaises(SchemaConflict):
            lens.schema(version=1)
        with self.assertRaises(SchemaConflict):
            lens.load(version=2)
        self.assertEqual(lens.schema(version=4), lens.schema())

    def test_default_retention_binding(self):
        lens = Lens(self.dir.name)
        self.assertEqual(lens.max_versions, DEFAULT_MAX_VERSIONS)

    def test_invalid_max_versions_rejected(self):
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                Lens(self.dir.name, max_versions=bad)


_WORKER = """
import json
import sys

from schema_lens import Lens, SchemaConflict

lens_dir, records_path = sys.argv[1], sys.argv[2]
records = []
with open(records_path, encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            records.append(json.loads(line))
lens = Lens(lens_dir)
try:
    lens.load()
except SchemaConflict:
    pass
lens.infer(records)
lens.save()
print(json.dumps(lens.stats(), sort_keys=True))
"""

_LOCK_HOLDER = """
import os
import sys
import time

import fcntl

fd = os.open(os.path.join(sys.argv[1], "schema.lock"),
             os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
sys.stdout.write("locked\\n")
sys.stdout.flush()
time.sleep(float(sys.argv[2]))
"""


def _fold_all(records):
    lens = Lens(tempfile.mkdtemp())
    lens.infer(records)
    return lens.schema()


class IncrementalPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_repeated_appends_equal_one_shot_fold(self):
        batches = [
            [{"a": 1, "b": "x"}, {"a": 2}],
            [{"a": 3, "b": "y"}, {"b": "z", "c": True}],
            [{"a": None}, {"c": False}, {"a": 4, "b": "w"}],
        ]
        lens = Lens(self.dir.name)
        for batch in batches:
            lens.infer(batch)
            lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        all_records = [record for batch in batches for record in batch]
        self.assertEqual(reloaded.schema(), _fold_all(all_records))
        stats = reloaded.stats()
        self.assertEqual(stats["records"], 7)
        self.assertTrue(reloaded.schema()["fields"]["a"]["optional"])
        self.assertEqual(
            sorted(reloaded.schema()["fields"]["a"]["types"]),
            ["null", "number"],
        )

    def test_earlier_folded_records_survive_later_appends(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}, {"a": 2, "b": "x"}])
        lens.save()
        lens.infer([{"c": [1, 2]}, {"c": [None]}])
        lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        fields = reloaded.schema()["fields"]
        self.assertEqual(set(fields), {"a", "b", "c"})
        self.assertEqual(fields["a"]["observed"], 2)
        self.assertEqual(fields["a"]["total"], 4)
        elements = fields["c"]["types"]["array"]["elements"]
        self.assertEqual(set(elements), {"null", "number"})
        self.assertEqual(reloaded.check({}), [])
        self.assertEqual(
            reloaded.check({"a": "s", "b": "x", "c": []}),
            ["$.a: type string not in field types [number]"],
        )

    def test_separate_lens_instances_build_on_each_others_commits(self):
        first = Lens(self.dir.name)
        first.infer([{"a": 1}, {"a": 2}])
        first.save()
        second = Lens(self.dir.name)
        second.load()
        second.infer([{"a": 3, "b": "x"}])
        second.save()
        first.infer([{"b": "y"}])
        first.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.stats()["records"], 4)
        expected = _fold_all(
            [{"a": 1}, {"a": 2}, {"a": 3, "b": "x"}, {"b": "y"}]
        )
        self.assertEqual(reloaded.schema(), expected)

    def test_uncommitted_memory_is_not_published(self):
        first = Lens(self.dir.name)
        first.infer([{"a": 1}])
        first.save()
        second = Lens(self.dir.name)
        second.load()
        second.infer([{"secret": 1}])  # no save
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertNotIn("secret", reloaded.schema()["fields"])

    def test_separate_processes_appending_lose_no_batches(self):
        lens_dir = self.dir.name
        batch_files = []
        batches = [
            [{"a": 1, "b": "x"}, {"a": 2}],
            [{"a": 3}, {"b": "y", "c": True}],
            [{"a": None}, {"a": 4, "c": False}],
        ]
        for index, batch in enumerate(batches):
            path = Path(self.dir.name, f"batch-{index}.jsonl")
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in batch),
                encoding="utf-8",
            )
            batch_files.append(str(path))
        worker = Path(self.dir.name, "worker.py")
        worker.write_text(_WORKER, encoding="utf-8")
        env = dict(os.environ)
        repo_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        for path in batch_files:
            proc = subprocess.run(
                [sys.executable, str(worker), lens_dir, path],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
        reloaded = Lens(lens_dir)
        reloaded.load()
        all_records = [record for batch in batches for record in batch]
        self.assertEqual(reloaded.schema(), _fold_all(all_records))
        self.assertEqual(reloaded.stats()["records"], 6)
        self.assertEqual(reloaded.versions(), [1, 2, 3])


class RevisionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)
        self.lens.infer([{"a": 1, "b": "x"}, {"a": "s"}])
        self.lens.save()
        self.lens.infer([{"c": True}])
        self.lens.save()

    def test_checksum_mismatch_raises_and_preserves_memory(self):
        path = revision_path(self.dir.name, 2)
        payload = json.loads(path.read_text())
        payload["root"]["count"] += 100
        path.write_text(json.dumps(payload), encoding="utf-8")
        before = self.lens.schema()
        with self.assertRaises(SchemaConflict):
            self.lens.schema(version=2)
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).load()
        self.assertEqual(self.lens.schema(), before)

    def test_rewritten_checksum_raises(self):
        path = revision_path(self.dir.name, 2)
        payload = json.loads(path.read_text())
        payload["checksum"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            self.lens.schema(version=2)

    def test_unknown_format_version_raises(self):
        path = revision_path(self.dir.name, 2)
        payload = json.loads(path.read_text())
        payload["version"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            self.lens.schema(version=2)

    def test_revision_number_mismatch_raises(self):
        path = revision_path(self.dir.name, 2)
        payload = json.loads(path.read_text())
        payload["revision"] = 5
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            self.lens.schema(version=2)

    def test_truncated_half_file_raises(self):
        path = revision_path(self.dir.name, 2)
        raw = path.read_text(encoding="utf-8")
        path.write_text(raw[: len(raw) // 2], encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            self.lens.schema(version=2)

    def test_one_corrupt_revision_leaves_others_usable(self):
        path = revision_path(self.dir.name, 2)
        path.write_text("{broken", encoding="utf-8")
        # Revision 1 still reads cleanly.
        root = self.lens.schema(version=1)
        self.assertEqual(set(root["fields"]), {"a", "b"})
        # The newest revision being corrupt makes unqualified reads fail.
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).load()
        with self.assertRaises(SchemaConflict):
            self.lens.save()

    def test_stale_temp_file_never_masks_revisions(self):
        Path(self.dir.name, VERSIONS_DIRNAME,
             "revision-0000000002.json.tmp").write_text("{half", encoding="utf-8")
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.versions(), [1, 2])
        self.assertEqual(reloaded.schema(), self.lens.schema())

    def test_save_against_corrupt_head_raises_and_preserves_memory(self):
        from schema_lens.core import _checksum

        with tempfile.TemporaryDirectory() as clean:
            lens = Lens(clean)
            lens.infer([{"a": 1}])
            lens.save()
            first_batch = lens.schema()
            lens.infer([{"b": 2}])
            memory = lens.schema()
            revision_path(clean, 1).write_text("{broken", encoding="utf-8")
            with self.assertRaises(SchemaConflict):
                lens.save()
            self.assertEqual(lens.schema(), memory)
            revision_path(clean, 1).write_text(
                json.dumps(
                    {
                        "version": SCHEMA_VERSION,
                        "revision": 1,
                        "checksum": _checksum(first_batch),
                        "root": first_batch,
                    }
                ),
                encoding="utf-8",
            )
            lens.save()
            reloaded = Lens(clean)
            reloaded.load()
            self.assertEqual(reloaded.stats()["records"], 2)
            self.assertEqual(set(reloaded.schema()["fields"]), {"a", "b"})
            self.assertEqual(reloaded.versions(), [1, 2])


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_prunes_oldest_above_bound(self):
        lens = Lens(self.dir.name, max_versions=3)
        for index in range(6):
            lens.infer([{f"f{index}": index}])
            lens.save()
        self.assertEqual(lens.versions(), [4, 5, 6])
        names = os.listdir(Path(self.dir.name, VERSIONS_DIRNAME))
        self.assertEqual(
            sorted(names),
            [f"revision-{n:010d}.json" for n in (4, 5, 6)],
        )

    def test_bound_of_one_keeps_only_newest(self):
        lens = Lens(self.dir.name, max_versions=1)
        for index in range(4):
            lens.infer([{f"f{index}": index}])
            lens.save()
        self.assertEqual(lens.versions(), [4])

    def test_bound_does_not_delete_unrelated_files(self):
        lens = Lens(self.dir.name, max_versions=1)
        lens.infer([{"a": 1}])
        lens.save()
        keep = Path(self.dir.name, VERSIONS_DIRNAME, "notes.txt")
        keep.write_text("handmade", encoding="utf-8")
        lens.infer([{"b": 2}])
        lens.save()
        self.assertTrue(keep.exists())

    def test_new_commits_after_pruning_keep_monotonic_numbers(self):
        lens = Lens(self.dir.name, max_versions=2)
        for index in range(5):
            lens.infer([{f"f{index}": index}])
            lens.save()
        self.assertEqual(lens.versions(), [4, 5])
        lens.infer([{"f5": 5}])
        lens.save()
        self.assertEqual(lens.versions(), [5, 6])


class RollbackTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)
        self.lens.infer([{"a": 1}])
        self.lens.save()
        self.lens.infer([{"b": "x"}])
        self.lens.save()
        self.lens.infer([{"c": True}])
        self.lens.save()

    def test_rollback_appends_a_new_revision(self):
        new_revision = self.lens.rollback(1)
        self.assertEqual(new_revision, 4)
        self.assertEqual(self.lens.versions(), [1, 2, 3, 4])
        # History is untouched.
        self.assertEqual(
            set(self.lens.schema(version=3)["fields"]), {"a", "b", "c"}
        )
        # The new revision is equivalent to the target.
        target = self.lens.schema(version=1)
        self.assertEqual(self.lens.schema(version=4), target)
        self.assertEqual(self.lens.schema(), target)

    def test_rollback_schema_equivalent_to_target_under_load(self):
        self.lens.rollback(2)
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.schema(), reloaded.schema(version=2))

    def test_rollback_to_missing_revision_raises_and_changes_nothing(self):
        before = self.lens.versions()
        with self.assertRaises(SchemaConflict):
            self.lens.rollback(99)
        self.assertEqual(self.lens.versions(), before)
        newest_bytes = revision_path(self.dir.name, 3).read_bytes()
        with self.assertRaises(SchemaConflict):
            self.lens.rollback(0)
        self.assertEqual(revision_path(self.dir.name, 3).read_bytes(), newest_bytes)

    def test_rollback_to_pruned_revision_raises(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(directory,
                                                            ignore_errors=True))
        lens = Lens(directory, max_versions=2)
        for index in range(4):
            lens.infer([{f"f{index}": index}])
            lens.save()
        self.assertEqual(lens.versions(), [3, 4])
        with self.assertRaises(SchemaConflict):
            lens.rollback(1)
        self.assertEqual(lens.versions(), [3, 4])

    def test_rollback_to_corrupt_revision_raises_and_changes_nothing(self):
        path = revision_path(self.dir.name, 1)
        path.write_text("{broken", encoding="utf-8")
        before = self.lens.versions()
        with self.assertRaises(SchemaConflict):
            self.lens.rollback(1)
        self.assertEqual(self.lens.versions(), before)
        self.assertFalse((Path(self.dir.name, VERSIONS_DIRNAME) /
                          "revision-0000000004.json").exists())

    def test_rollback_then_inference_builds_on_target(self):
        self.lens.rollback(1)
        self.lens.infer([{"d": None}])
        self.lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(set(reloaded.schema()["fields"]), {"a", "d"})
        self.assertEqual(reloaded.versions(), [1, 2, 3, 4, 5])

    def test_rollback_respects_retention(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(directory,
                                                            ignore_errors=True))
        lens = Lens(directory, max_versions=3)
        for index in range(4):
            lens.infer([{f"f{index}": index}])
            lens.save()
        self.assertEqual(lens.versions(), [2, 3, 4])
        lens.rollback(2)
        self.assertEqual(lens.versions(), [3, 4, 5])


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_open_snapshot_is_not_affected_by_later_commits(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        reader = Lens(self.dir.name)
        reader.load()
        other = Lens(self.dir.name)
        other.load()
        other.infer([{"b": "x"}, {"c": True}])
        other.save()
        # The open snapshot still answers as revision 1.
        self.assertEqual(set(reader.schema()["fields"]), {"a"})
        self.assertEqual(reader.check({"a": 2, "b": "y"}),
                         ["$.b: unexpected field"])
        # Explicit reads reach the committed newer revision.
        self.assertEqual(set(reader.schema(version=2)["fields"]),
                         {"a", "b", "c"})
        reader.load()
        self.assertEqual(set(reader.schema()["fields"]), {"a", "b", "c"})

    def test_inference_appends_to_the_open_snapshot(self):
        writer = Lens(self.dir.name)
        writer.infer([{"a": 1}])
        writer.save()
        writer.infer([{"b": "x"}])
        writer.save()
        lens = Lens(self.dir.name)
        lens.load(version=1)
        lens.infer([{"z": None}])
        # Snapshot delta fold: z joins revision 1's fields, not revision 2's.
        self.assertEqual(set(lens.schema()["fields"]), {"a", "z"})
        lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        # Saving builds on the committed newest (revision 2), so the result
        # contains b from revision 2 as well.
        self.assertEqual(set(reloaded.schema()["fields"]), {"a", "b", "z"})
        self.assertEqual(reloaded.versions(), [1, 2, 3])


@unittest.skipIf(fcntl is None, "flock is only available on POSIX")
class LockTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.holder = Path(self.dir.name, "hold_lock.py")
        self.holder.write_text(_LOCK_HOLDER, encoding="utf-8")

    def test_contended_lock_raises_conflict_without_touching_commits(self):
        lens_dir = self.dir.name
        lens = Lens(lens_dir)
        lens.infer([{"a": 1}])
        lens.save()
        before = set(os.listdir(Path(lens_dir, VERSIONS_DIRNAME)))
        proc = subprocess.Popen(
            [sys.executable, str(self.holder), lens_dir, "3"],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(proc.stdout.readline().strip(), "locked")
        try:
            lens.infer([{"b": 2}])
            memory_at_commit = lens.schema()
            with self.assertRaises(SchemaConflict):
                lens.save()
            # The failed commit neither added a revision nor rolled back
            # (nor published) the in-memory schema.
            self.assertEqual(
                set(os.listdir(Path(lens_dir, VERSIONS_DIRNAME))), before
            )
            self.assertEqual(lens.schema(), memory_at_commit)
        finally:
            proc.wait()
            proc.stdout.close()
        lens.save()
        reloaded = Lens(lens_dir)
        reloaded.load()
        self.assertEqual(reloaded.stats()["records"], 2)
        self.assertEqual(set(reloaded.schema()["fields"]), {"a", "b"})

    def test_contended_rollback_raises_without_committing(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        proc = subprocess.Popen(
            [sys.executable, str(self.holder), self.dir.name, "3"],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(proc.stdout.readline().strip(), "locked")
        try:
            with self.assertRaises(SchemaConflict):
                lens.rollback(1)
        finally:
            proc.wait()
            proc.stdout.close()
        self.assertEqual(lens.versions(), [1])

    def test_two_threads_in_one_process_are_serialized(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()

        held = threading.Event()
        release = threading.Event()

        def hold():
            lens2 = Lens(self.dir.name)
            lens2.infer([{"held": 1}])
            lens2._lock.acquire()
            held.set()
            release.wait(timeout=5)
            lens2._lock.release()

        thread = threading.Thread(target=hold)
        thread.start()
        self.addCleanup(thread.join)
        held.wait(timeout=5)
        other = Lens(self.dir.name)
        other.infer([{"b": 2}])
        with self.assertRaises(SchemaConflict):
            other.save()
        release.set()
        thread.join()

    def test_lock_file_lives_inside_lens_directory(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        self.assertTrue((Path(self.dir.name) / LOCK_FILENAME).exists())


class ConcurrentReadTests(unittest.TestCase):
    """Concurrent readers always observe one complete revision or another."""

    def test_concurrent_readers_never_see_a_half_written_revision(self):
        lens_dir = self.dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(lens_dir,
                                                            ignore_errors=True))
        lens = Lens(lens_dir)
        lens.infer([{"a": 1}])
        lens.save()
        stop = threading.Event()
        bad_reads = []
        seen = set()

        def commit_forever():
            writer = Lens(lens_dir)
            writer.load()
            n = 0
            while not stop.is_set():
                n += 1
                writer.infer([{f"f{n}": n}])
                try:
                    writer.save()
                except SchemaConflict:
                    continue

        def read_forever():
            while not stop.is_set():
                reader = Lens(lens_dir)
                try:
                    reader.load()
                    root = reader.schema()
                except SchemaConflict as exc:
                    bad_reads.append(str(exc))
                    continue
                count = root["count"]
                # A complete revision has the field recorded for it.
                if f"f{count - 1}" not in root["fields"] and count > 1:
                    bad_reads.append(f"revision for {count} records missing field")
                seen.add(count)

        threads = [threading.Thread(target=commit_forever)]
        threads += [threading.Thread(target=read_forever) for _ in range(3)]
        for thread in threads:
            thread.start()
        timer = threading.Timer(2.0, stop.set)
        timer.start()
        for thread in threads:
            thread.join(timeout=10)
        timer.join()
        self.assertEqual(bad_reads, [])
        self.assertGreater(len(seen), 1)


class DottedKeyPathTests(unittest.TestCase):
    def test_dotted_key_path_is_distinguishable_from_nested_path(self):
        lens = Lens(tempfile.mkdtemp())
        lens.infer([{"a.b": 1, "a": {"b": 2}}])
        reports = lens.check({"a.b": "x", "a": {"b": "x"}})
        self.assertIn('$["a.b"]: type string not in field types [number]', reports)
        self.assertIn("$.a.b: type string not in field types [number]", reports)
        self.assertEqual(len(reports), 2)

    def test_backslash_key_path_is_quoted(self):
        lens = Lens(tempfile.mkdtemp())
        lens.infer([{"a\\b": 1}])
        reports = lens.check({"a\\b": "x"})
        self.assertEqual(
            reports, ['$["a\\\\b"]: type string not in field types [number]']
        )

    def test_missing_dotted_key_uses_quoted_path(self):
        lens = Lens(tempfile.mkdtemp())
        lens.infer([{"a.b": 1}, {"a.b": 2}])
        reports = lens.check({})
        self.assertEqual(reports, ['$["a.b"]: missing required field'])

    def test_unexpected_dotted_key_uses_quoted_path(self):
        lens = Lens(tempfile.mkdtemp())
        lens.infer([{"a": 1}])
        reports = lens.check({"a": 1, "x.y": 2})
        self.assertEqual(reports, ['$["x.y"]: unexpected field'])

    def test_literal_star_key_is_quoted_and_distinct_from_marker(self):
        lens = Lens(tempfile.mkdtemp())
        lens.infer([{"*": 1}, {"*": 2}])
        # The real key quotes; missing/type reports stay uniquely located.
        self.assertEqual(lens.check({}), ['$["*"]: missing required field'])
        self.assertEqual(
            lens.check({"*": "x"}),
            ['$["*"]: type string not in field types [number]'],
        )
        self.assertNotIn(OVERFLOW_MARKER, json.dumps(lens.schema()))


class ReadRecordsTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_non_object_line_raises_value_error_with_line_number(self):
        path = Path(self.dir.name, "records.jsonl")
        path.write_text('{"a": 1}\n[1, 2]\n', encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            _read_records(str(path))
        message = str(ctx.exception)
        self.assertIn(str(path), message)
        self.assertIn("2", message)

    def test_invalid_json_line_raises_value_error_with_line_number(self):
        path = Path(self.dir.name, "records.jsonl")
        path.write_text('{"a": 1}\n{broken\n', encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            _read_records(str(path))
        self.assertIn("2", str(ctx.exception))

    def test_missing_records_file_raises_value_error(self):
        path = Path(self.dir.name, "missing.jsonl")
        with self.assertRaises(ValueError):
            _read_records(str(path))

    def test_cli_non_object_line_exits_nonzero_with_line_number(self):
        records = Path(self.dir.name, "records.jsonl")
        records.write_text("[1, 2]\n", encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            code = cli_main(["--path", self.dir.name, "infer", str(records)])
        self.assertEqual(code, 2)


class StatsTests(unittest.TestCase):
    def test_stats_reports_fields_types_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            lens = Lens(d)
            lens.infer([{"a": 1, "b": "x"}, {"a": "s"}])
            stats = lens.stats()
            self.assertEqual(stats["records"], 2)
            self.assertEqual(stats["fields"], 2)
            self.assertEqual(stats["optional"], 1)
            self.assertEqual(stats["types"], {"number": 1, "string": 2})
            self.assertEqual(stats["observations"], 3)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens_dir = str(Path(self.dir.name, "lens"))
        self.records = str(Path(self.dir.name, "records.jsonl"))

    def _write(self, records):
        with open(self.records, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def _run(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli_main(list(argv))
        return code, out.getvalue()

    def test_infer_then_show_roundtrip(self):
        self._write([{"a": 1, "b": "x"}, {"a": 2}])
        code, out = self._run("--path", self.lens_dir, "infer", self.records)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["records"], 2)
        code, out = self._run("--path", self.lens_dir, "show")
        self.assertEqual(code, 0)
        schema = json.loads(out)
        self.assertTrue(schema["fields"]["b"]["optional"])

    def test_check_exit_codes_and_output(self):
        self._write([{"a": 1}])
        self._run("--path", self.lens_dir, "infer", self.records)
        self._write([{"a": 2}, {"a": "s", "extra": 1}])
        code, out = self._run("--path", self.lens_dir, "check", self.records)
        self.assertEqual(code, 1)
        self.assertIn("line 2: $.a: type string not in field types [number]", out)
        self.assertIn("line 2: $.extra: unexpected field", out)
        self._write([{"a": 3}])
        code, out = self._run("--path", self.lens_dir, "check", self.records)
        self.assertEqual(code, 0)
        self.assertEqual(out, "")

    def test_check_without_stored_schema_fails(self):
        self._write([{"a": 1}])
        code, _ = self._run("--path", self.lens_dir, "check", self.records)
        self.assertEqual(code, 2)

    def test_versions_command_lists_revisions_oldest_first(self):
        self._write([{"a": 1}])
        self._run("--path", self.lens_dir, "infer", self.records)
        self._write([{"b": "x"}])
        self._run("--path", self.lens_dir, "infer", self.records)
        code, out = self._run("--path", self.lens_dir, "versions")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), [1, 2])

    def test_versions_command_empty_on_fresh_lens(self):
        code, out = self._run("--path", self.lens_dir, "versions")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), [])

    def test_show_specific_revision(self):
        self._write([{"a": 1}])
        self._run("--path", self.lens_dir, "infer", self.records)
        self._write([{"b": "x"}])
        self._run("--path", self.lens_dir, "infer", self.records)
        code, out = self._run("--path", self.lens_dir,
                              "--version", "1", "show")
        self.assertEqual(code, 0)
        self.assertEqual(set(json.loads(out)["fields"]), {"a"})

    def test_check_specific_revision(self):
        self._write([{"a": 1}])
        self._run("--path", self.lens_dir, "infer", self.records)
        self._write([{"b": "x"}])
        self._run("--path", self.lens_dir, "infer", self.records)
        self._write([{"a": 2, "b": "y"}])
        code, out = self._run(
            "--path", self.lens_dir, "--version", "1", "check", self.records
        )
        self.assertEqual(code, 1)
        self.assertIn("line 1: $.b: unexpected field", out)

    def test_show_missing_revision_fails(self):
        code, _ = self._run("--path", self.lens_dir, "--version", "7", "show")
        self.assertEqual(code, 2)

    def test_rollback_command_appends_equivalent_revision(self):
        self._write([{"a": 1}])
        self._run("--path", self.lens_dir, "infer", self.records)
        self._write([{"b": "x"}])
        self._run("--path", self.lens_dir, "infer", self.records)
        code, out = self._run("--path", self.lens_dir, "rollback", "1")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), {"rolled_back_to": 1, "revision": 3})
        code, out = self._run("--path", self.lens_dir, "versions")
        self.assertEqual(json.loads(out), [1, 2, 3])
        code, latest = self._run("--path", self.lens_dir, "show")
        code, target = self._run("--path", self.lens_dir,
                                 "--version", "1", "show")
        self.assertEqual(json.loads(latest), json.loads(target))

    def test_rollback_missing_revision_fails(self):
        self._write([{"a": 1}])
        self._run("--path", self.lens_dir, "infer", self.records)
        code, _ = self._run("--path", self.lens_dir, "rollback", "5")
        self.assertEqual(code, 2)
        code, out = self._run("--path", self.lens_dir, "versions")
        self.assertEqual(json.loads(out), [1])

    def test_rollback_requires_a_revision(self):
        code, _ = self._run("--path", self.lens_dir, "rollback")
        self.assertEqual(code, 2)

    def test_version_flag_only_valid_on_check_and_show(self):
        code, _ = self._run("--path", self.lens_dir, "--version", "1", "versions")
        self.assertEqual(code, 2)

    def test_max_versions_option_prunes(self):
        for index in range(4):
            self._write([{f"f{index}": index}])
            code, _ = self._run(
                "--path", self.lens_dir, "--max-versions", "2",
                "infer", self.records,
            )
            self.assertEqual(code, 0)
        code, out = self._run("--path", self.lens_dir, "versions")
        self.assertEqual(json.loads(out), [3, 4])


class ResourceLimitTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _exact(self, records):
        lens = Lens(tempfile.mkdtemp())
        lens.infer(records)
        return lens.schema()

    def test_generous_limits_match_unlimited_fold(self):
        records = [
            {"a": 1, "b": {"c": [1, "x"], "d": True}},
            {"a": "s", "b": {"c": [], "e": None}},
        ]
        lens = Lens(self.dir.name, max_fields=50, max_depth=50)
        lens.infer(records)
        self.assertEqual(lens.schema(), self._exact(records))
        self.assertEqual(lens.stats()["approximate"], [])

    def test_no_limits_adds_no_marker_keys(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": {"b": [1]}}])
        self.assertNotIn("overflow", lens.schema())
        self.assertNotIn("approximate", json.dumps(lens.schema()))

    def test_field_cap_overflow_is_marked_and_widens(self):
        lens = Lens(self.dir.name, max_fields=1)
        lens.infer([{"a": 1, "b": 2}, {"a": 3, "b": "x", "c": True}])
        root = lens.schema()
        self.assertEqual(list(root["fields"]), ["a"])
        overflow = root["overflow"]
        self.assertTrue(overflow["approximate"])
        self.assertEqual(overflow["observed"], 3)
        self.assertEqual(sorted(overflow["types"]), ["boolean", "number", "string"])
        self.assertEqual(root["fields"]["a"]["observed"], 2)
        self.assertEqual(sorted(root["fields"]["a"]["types"]), ["number"])

    def test_overflow_observations_grow_monotonically_across_batches(self):
        lens = Lens(self.dir.name, max_fields=1)
        seen = 0
        for batch in [[{"a": 1, "x": 1}], [{"a": 2, "y": "s"}, {"a": 3, "z": None}]]:
            lens.infer(batch)
            overflow = lens.schema()["overflow"]
            self.assertGreater(overflow["observed"], seen)
            seen = overflow["observed"]
        self.assertEqual(
            sorted(lens.schema()["overflow"]["types"]), ["null", "number", "string"]
        )

    def test_overflow_fields_are_not_unexpected_in_check(self):
        lens = Lens(self.dir.name, max_fields=1)
        lens.infer([{"a": 1, "b": 2}])
        self.assertEqual(lens.check({"a": 1, "b": 2, "c": 3}), [])
        reports = lens.check({"a": 1, "b": "x"})
        self.assertIn('$.b: type string not in field types [number]', reports)

    def test_depth_cap_collapses_nested_nodes(self):
        lens = Lens(self.dir.name, max_depth=1)
        lens.infer(
            [
                {"user": {"name": "ann", "addr": {"city": "x"}}},
                {"user": {"name": "bob"}},
            ]
        )
        root = lens.schema()
        user = root["fields"]["user"]["types"]["object"]
        self.assertNotIn("approximate", user)
        self.assertEqual(user["count"], 2)
        self.assertEqual(user["fields"]["name"]["observed"], 2)
        addr = user["fields"]["addr"]["types"]["object"]
        self.assertTrue(addr["approximate"])
        self.assertNotIn("fields", addr)
        self.assertEqual(addr["count"], 1)
        self.assertEqual(lens.stats()["approximate"], ["$.user.addr"])
        self.assertEqual(
            lens.check({"user": {"name": "n", "addr": {"anything": [1, {}]}}}), []
        )

    def test_depth_cap_collapses_array_elements(self):
        lens = Lens(self.dir.name, max_depth=1)
        lens.infer([{"items": [{"x": 1}, {"x": 2}]}])
        items = lens.schema()["fields"]["items"]["types"]["array"]
        self.assertNotIn("approximate", items)
        element = items["elements"]["object"]
        self.assertTrue(element["approximate"])
        self.assertEqual(element["count"], 2)

    def test_stats_lists_overflow_location_with_non_colliding_marker(self):
        lens = Lens(self.dir.name, max_fields=1)
        lens.infer([{"a": 1, "b": 2}])
        self.assertEqual(lens.stats()["approximate"], ["$.*"])

    def test_marker_distinct_from_literal_star_key(self):
        lens = Lens(self.dir.name, max_fields=1)
        # The exact field "*" keeps its quoted path; overflow still gets
        # the bare marker path, and the two never compare equal.
        lens.infer([{"*": 1, "other": 2}, {"*": 3, "again": 4}])
        self.assertEqual(lens.stats()["approximate"], ["$.*"])
        reports = lens.check({})
        self.assertIn('$["*"]: missing required field', reports)

    def test_approximation_survives_commits_and_loads(self):
        lens = Lens(self.dir.name, max_fields=1, max_depth=1)
        lens.infer([{"a": 1, "b": {"c": 2}}])
        lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.schema(), lens.schema())
        self.assertEqual(reloaded.stats()["approximate"], lens.stats()["approximate"])
        reloaded2 = Lens(self.dir.name, max_fields=1, max_depth=1)
        reloaded2.load()
        reloaded2.infer([{"a": 2, "d": "x"}])
        reloaded2.save()
        final = Lens(self.dir.name)
        final.load()
        self.assertEqual(final.schema()["overflow"]["observed"], 2)

    def test_limited_batches_still_commit_incrementally(self):
        batches = [
            [{"a": 1, "b": 1}, {"a": 2, "c": "x"}],
            [{"a": 3, "b": 2, "d": None}],
        ]
        lens = Lens(self.dir.name, max_fields=2)
        for batch in batches:
            lens.infer(batch)
            lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.stats()["records"], 3)
        self.assertEqual(sorted(reloaded.schema()["fields"]), ["a", "b"])
        self.assertEqual(reloaded.schema()["overflow"]["observed"], 2)

    def test_invalid_limits_rejected(self):
        with self.assertRaises(ValueError):
            Lens(self.dir.name, max_fields=-1)
        with self.assertRaises(ValueError):
            Lens(self.dir.name, max_depth=1.5)
        with self.assertRaises(ValueError):
            Lens(self.dir.name, max_fields=True)

    def test_dotted_keys_still_located_with_limits(self):
        lens = Lens(self.dir.name, max_fields=5)
        lens.infer([{"a.b": 1, "a": {"b": 2}}])
        reports = lens.check({"a.b": "x", "a": {"b": "x"}})
        self.assertIn('$["a.b"]: type string not in field types [number]', reports)
        self.assertIn("$.a.b: type string not in field types [number]", reports)

    def test_merged_node_caps_exact_field_entries(self):
        # Two disjoint batches each stay under the cap on their own, but
        # their union would double it; the merge folds the excess entries
        # into the overflow entry.
        first = Lens(self.dir.name, max_fields=2)
        first.infer([{"a": 1, "b": 1}, {"a": 2, "b": 2}])
        first.save()
        second = Lens(self.dir.name, max_fields=2)
        second.load()
        second.infer([{"c": "x", "d": "y"}, {"c": "z", "d": "w"}])
        second.save()
        reloaded = Lens(self.dir.name, max_fields=2)
        reloaded.load()
        root = reloaded.schema()
        self.assertLessEqual(len(root["fields"]), 2)
        self.assertIn("overflow", root)
        self.assertTrue(root["overflow"]["approximate"])
        # Every folded record still counts.
        self.assertEqual(root["count"], 4)

    def test_merged_cap_matches_single_capped_pass(self):
        batches = [
            [{"a": 1}, {"b": 2}, {"c": 3}],
            [{"d": 4}, {"e": 5}, {"f": 6}],
        ]
        lens = Lens(self.dir.name, max_fields=3)
        for batch in batches:
            lens.infer(batch)
            lens.save()
        reloaded = Lens(self.dir.name, max_fields=3)
        reloaded.load()
        one_shot = Lens(tempfile.mkdtemp(), max_fields=3)
        one_shot.infer([r for batch in batches for r in batch])
        self.assertEqual(reloaded.schema(), one_shot.schema())

    def test_record_stream_is_not_retained(self):
        lens = Lens(self.dir.name, max_fields=3, max_depth=2)

        def stream():
            for index in range(20000):
                yield {
                    f"field{index % 40}": index,
                    "nested": {"deep": {"deeper": {"leaf": index}}},
                }

        lens.infer(stream())
        root = lens.schema()
        self.assertLessEqual(len(root["fields"]), 3)
        self.assertEqual(root["count"], 20000)
        nested = root["fields"]["nested"]["types"]["object"]
        self.assertNotIn("approximate", nested)
        deeper = nested["fields"]["deep"]["types"]["object"]["fields"]["deeper"]
        self.assertTrue(deeper["types"]["object"]["approximate"])


class LegacyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        lens = Lens(tempfile.mkdtemp())
        lens.infer([{"a": 1, "b": "x"}, {"a": "s"}])
        self.root = lens.schema()

    def _write_legacy(self, version=2):
        from schema_lens.core import _checksum

        path = Path(self.dir.name, SCHEMA_FILENAME)
        payload = {"version": version, "checksum": _checksum(self.root),
                   "root": self.root}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    def test_legacy_file_migrates_on_load(self):
        path = self._write_legacy()
        lens = Lens(self.dir.name)
        lens.load()
        self.assertEqual(lens.schema(), self.root)
        self.assertEqual(lens.versions(), [1])
        # The legacy file is gone and revision 1 carries the schema.
        self.assertFalse(path.exists())
        payload = json.loads(revision_path(self.dir.name, 1).read_text())
        self.assertEqual(payload["version"], SCHEMA_VERSION)
        self.assertEqual(payload["revision"], 1)
        self.assertEqual(payload["root"], self.root)

    def test_version_one_file_migrates_too(self):
        self._write_legacy(version=1)
        lens = Lens(self.dir.name)
        lens.load()
        self.assertEqual(lens.schema(), self.root)
        self.assertEqual(lens.versions(), [1])

    def test_pinned_read_migrates_legacy_in_place(self):
        path = self._write_legacy()
        lens = Lens(self.dir.name)
        self.assertEqual(lens.schema(version=1), self.root)
        self.assertFalse(path.exists())
        self.assertEqual(lens.versions(), [1])
        # A pin to anything other than revision 1 still conflicts.
        with self.assertRaises(SchemaConflict):
            lens.schema(version=2)

    def test_rollback_migrates_legacy_then_appends(self):
        self._write_legacy()
        lens = Lens(self.dir.name)
        new_revision = lens.rollback(1)
        self.assertEqual(new_revision, 2)
        self.assertEqual(lens.versions(), [1, 2])
        self.assertEqual(lens.schema(version=2), self.root)

    def test_legacy_file_migrates_on_save_and_keeps_batches(self):
        self._write_legacy()
        lens = Lens(self.dir.name)
        lens.load()
        lens.infer([{"c": True}])
        lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.stats()["records"], 3)
        self.assertEqual(sorted(reloaded.schema()["fields"]), ["a", "b", "c"])
        self.assertEqual(reloaded.versions(), [1, 2])
        self.assertFalse(Path(self.dir.name, SCHEMA_FILENAME).exists())

    def test_failed_migration_preserves_file_and_memory(self):
        from unittest import mock

        path = self._write_legacy()
        before = path.read_bytes()
        lens = Lens(self.dir.name)
        with mock.patch(
            "schema_lens.core._write_atomic", side_effect=OSError("disk full")
        ):
            with self.assertRaises(SchemaConflict):
                lens.load()
        # The original file is untouched, no revision landed, and memory
        # holds no schema.
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(
            list(Path(self.dir.name, VERSIONS_DIRNAME).glob("*.json")), []
        )
        with self.assertRaises(SchemaConflict):
            lens.schema()
        # The read can be retried later and migrates cleanly.
        lens.load()
        self.assertEqual(lens.schema(), self.root)
        self.assertEqual(lens.versions(), [1])

    def test_corrupt_legacy_file_raises_and_preserves_memory(self):
        path = self._write_legacy()
        path.write_text("{not json", encoding="utf-8")
        lens = Lens(self.dir.name)
        lens.infer([{"z": 1}])
        memory = lens.schema()
        with self.assertRaises(SchemaConflict):
            lens.load()
        self.assertEqual(lens.schema(), memory)

    def test_migrated_schema_equivalent_to_fresh_fold(self):
        self._write_legacy()
        migrated = Lens(self.dir.name)
        migrated.load()
        fresh = Lens(tempfile.mkdtemp())
        fresh.infer([{"a": 1, "b": "x"}, {"a": "s"}])
        self.assertEqual(migrated.schema(), fresh.schema())

    def test_legacy_file_is_ignored_once_revisions_exist(self):
        # A leftover legacy file never masks committed revisions.
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        Path(self.dir.name, SCHEMA_FILENAME).write_text(
            json.dumps({"version": 2, "root": {}}), encoding="utf-8"
        )
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(set(reloaded.schema()["fields"]), {"a"})


class StreamingCliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens_dir = str(Path(self.dir.name, "lens"))
        self.records = str(Path(self.dir.name, "records.jsonl"))

    def _write(self, records):
        with open(self.records, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def _run(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli_main(list(argv))
        return code, out.getvalue()

    def test_cli_infer_with_limits_marks_approximation(self):
        self._write([{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}])
        code, out = self._run(
            "--path", self.lens_dir, "--max-fields", "2", "infer", self.records
        )
        self.assertEqual(code, 0)
        stats = json.loads(out)
        self.assertEqual(stats["records"], 2)
        self.assertEqual(stats["approximate"], ["$.*"])
        code, out = self._run("--path", self.lens_dir, "show")
        self.assertEqual(code, 0)
        schema = json.loads(out)
        self.assertTrue(schema["overflow"]["approximate"])
        self.assertEqual(sorted(schema["fields"]), ["a", "b"])

    def test_cli_infer_without_limits_has_no_approximation(self):
        self._write([{"a": 1}, {"a": 2, "b": "x"}])
        code, out = self._run("--path", self.lens_dir, "infer", self.records)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["approximate"], [])

    def test_cli_check_exit_codes_unchanged_with_limits(self):
        self._write([{"a": 1, "b": 2}])
        self._run("--path", self.lens_dir, "--max-fields", "1", "infer", self.records)
        self._write([{"a": 2, "b": 3, "c": 4}])
        code, _ = self._run("--path", self.lens_dir, "check", self.records)
        self.assertEqual(code, 0)
        self._write([{"a": "bad"}])
        code, out = self._run("--path", self.lens_dir, "check", self.records)
        self.assertEqual(code, 1)
        self.assertIn("$.a: type string not in field types [number]", out)


class CompactionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _committed(self, n, *, max_fields=None, max_versions=100, records=None):
        lens = Lens(
            self.dir.name, max_fields=max_fields, max_versions=max_versions
        )
        # Only explicit compact() calls run in these tests.
        lens._schedule_compaction = lambda **_: None
        for index in range(n):
            record = records(index) if records else {"a": index, f"f{index}": index}
            lens.infer([record])
            lens.save()
        return lens

    def _reference_schema(self, n, *, max_fields=None, records=None):
        lens = Lens(tempfile.mkdtemp(), max_fields=max_fields)
        for index in range(n):
            record = records(index) if records else {"a": index, f"f{index}": index}
            lens.infer([record])
        return lens.schema()

    def test_status_before_any_compaction(self):
        self._committed(4)
        status = Lens(self.dir.name).compact_status()
        self.assertFalse(status["running"])
        self.assertIsNone(status["baseline"])
        self.assertEqual(status["compacted"], [])
        self.assertEqual(status["revisions"], [1, 2, 3, 4])
        self.assertEqual(status["pending"], [1, 2, 3, 4])

    def test_compact_preserves_boundary_pattern_and_stats_exactly(self):
        lens = self._committed(8)
        boundary_schema = lens.schema(version=4)
        head_schema = lens.schema(version=8)
        status = lens.compact()
        self.assertEqual(status["baseline"], 4)
        self.assertEqual(status["compacted"], [1, 2, 3])
        self.assertEqual(status["revisions"], [4, 5, 6, 7, 8])
        # The baseline is the boundary revision, pattern and stats verbatim.
        self.assertEqual(lens.schema(version=4), boundary_schema)
        self.assertEqual(lens.schema(version=4), self._reference_schema(4))
        # Survivors, including the head, are untouched.
        self.assertEqual(lens.schema(version=8), head_schema)
        self.assertEqual(lens.schema(version=5), lens.schema(version=5))
        self.assertEqual(lens.versions(), [4, 5, 6, 7, 8])

    def test_merged_revisions_raise_schema_conflict(self):
        lens = self._committed(8)
        lens.compact()
        for gone in (1, 2, 3):
            with self.assertRaises(SchemaConflict):
                lens.schema(version=gone)
            with self.assertRaises(SchemaConflict):
                lens.check({"a": 1}, version=gone)
            with self.assertRaises(SchemaConflict):
                lens.rollback(gone)
            with self.assertRaises(SchemaConflict):
                lens.load(version=gone)
        # Other revisions keep working.
        self.assertEqual(lens.check({"a": 9}, version=8)[:1], [])

    def test_no_temp_or_double_files_after_compaction(self):
        lens = self._committed(8)
        lens.compact()
        names = set(os.listdir(Path(self.dir.name, VERSIONS_DIRNAME)))
        self.assertIn("baseline.json", names)
        self.assertIn("revision-0000000004.baseline.json", names)
        self.assertNotIn("revision-0000000004.json", names)
        self.assertNotIn("revision-0000000001.json", names)
        self.assertNotIn("revision-0000000002.json", names)
        self.assertNotIn("revision-0000000003.json", names)
        self.assertEqual(
            [n for n in names if n.endswith(".tmp") or n.endswith(".tmp.json")],
            [],
        )
        self.assertEqual(
            [n for n in names if n.endswith(".baseline.json")],
            ["revision-0000000004.baseline.json"],
        )

    def test_repeated_compaction_extends_one_baseline(self):
        lens = self._committed(12)
        first = lens.compact()
        self.assertEqual(first["baseline"], 6)
        self.assertEqual(first["compacted"], [1, 2, 3, 4, 5])
        second = lens.compact()
        self.assertEqual(second["baseline"], 9)
        self.assertEqual(second["compacted"], [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(second["revisions"], [9, 10, 11, 12])
        self.assertEqual(lens.schema(version=9), self._reference_schema(9))
        self.assertEqual(lens.schema(version=12)["count"], 12)
        for gone in range(1, 9):
            with self.assertRaises(SchemaConflict):
                lens.schema(version=gone)
        names = set(os.listdir(Path(self.dir.name, VERSIONS_DIRNAME)))
        self.assertNotIn("revision-0000000006.baseline.json", names)
        self.assertIn("revision-0000000009.baseline.json", names)

    def test_compaction_survives_reopen(self):
        lens = self._committed(10)
        lens.compact()
        reloaded = Lens(self.dir.name)
        status = reloaded.compact_status()
        self.assertEqual(status["baseline"], 5)
        self.assertEqual(reloaded.versions(), [5, 6, 7, 8, 9, 10])
        reloaded.load()
        self.assertEqual(reloaded.schema()["count"], 10)
        self.assertEqual(reloaded.schema(version=5)["count"], 5)
        with self.assertRaises(SchemaConflict):
            reloaded.schema(version=4)

    def test_commits_after_compaction_keep_full_history(self):
        lens = self._committed(8)
        lens.compact()
        lens.infer([{"newfield": 1}])
        lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.versions(), [4, 5, 6, 7, 8, 9])
        root = reloaded.schema()
        self.assertEqual(root["count"], 9)
        self.assertIn("newfield", root["fields"])
        self.assertIn("f0", root["fields"])
        # "a" was folded in the first eight records only.
        self.assertEqual(root["fields"]["a"]["observed"], 8)
        self.assertTrue(root["fields"]["a"]["optional"])

    def test_rollback_after_compaction_appends_and_keeps_order(self):
        lens = self._committed(8)
        lens.compact()
        new_revision = lens.rollback(5)
        self.assertEqual(new_revision, 9)
        self.assertEqual(lens.versions(), [4, 5, 6, 7, 8, 9])
        self.assertEqual(lens.schema(version=9), lens.schema(version=5))
        lens.infer([{"z": None}])
        lens.save()
        self.assertEqual(lens.versions(), [4, 5, 6, 7, 8, 9, 10])

    def test_rollback_to_baseline_revision_works(self):
        lens = self._committed(8)
        lens.compact()
        new_revision = lens.rollback(4)
        self.assertEqual(new_revision, 9)
        self.assertEqual(lens.schema(version=9), lens.schema(version=4))

    def test_capped_schema_and_overflow_stats_preserved(self):
        def records(index):
            return {"a": 1, f"f{index}": index, "b": index}

        lens = self._committed(10, max_fields=2, records=records)
        boundary = lens.schema(version=5)
        head = lens.schema(version=10)
        lens.compact()
        self.assertEqual(lens.schema(version=5), boundary)
        self.assertEqual(lens.schema(version=10), head)
        # Overflow checking semantics are unchanged by compaction.
        reports = lens.check({"a": 1, "f3": "x", "b": 2}, version=10)
        self.assertIn(
            "$.f3: type string not in field types [number]", reports
        )

    def test_compact_with_two_revisions_merges_the_oldest(self):
        lens = self._committed(2)
        status = lens.compact()
        self.assertEqual(status["baseline"], 2)
        self.assertEqual(status["revisions"], [2])
        with self.assertRaises(SchemaConflict):
            lens.schema(version=1)
        lens.infer([{"m": 1}])
        lens.save()
        self.assertEqual(lens.versions(), [2, 3])

    def test_compact_with_one_revision_is_a_noop(self):
        lens = self._committed(1)
        status = lens.compact()
        self.assertIsNone(status["baseline"])
        self.assertEqual(lens.versions(), [1])

    def test_stale_temp_products_are_recovered(self):
        lens = self._committed(6)
        vdir = Path(self.dir.name, VERSIONS_DIRNAME)
        (vdir / "revision-0000000003.baseline.tmp").write_text("{half", encoding="utf-8")
        (vdir / "baseline.json.tmp").write_text("{half", encoding="utf-8")
        # Lockless reads ignore the temps and keep full history.
        self.assertEqual(Lens(self.dir.name).versions(), [1, 2, 3, 4, 5, 6])
        lens.compact()
        names = os.listdir(vdir)
        self.assertEqual([n for n in names if n.endswith(".tmp")], [])

    def test_switch_without_cleanup_is_finished_by_next_locked_op(self):
        lens = self._committed(6)
        lens.compact()
        vdir = Path(self.dir.name, VERSIONS_DIRNAME)
        # Simulate a crash between the pointer switch and cleanup: the
        # merged revisions' ordinary files are present again while the
        # pointer already names the baseline.
        donor_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(donor_dir, ignore_errors=True))
        donor = Lens(donor_dir, max_versions=100)
        donor._schedule_compaction = lambda **_: None
        for index in range(6):
            donor.infer([{"a": index, f"f{index}": index}])
            donor.save()
        for number in (1, 2):
            shutil.copy(
                Path(donor_dir, VERSIONS_DIRNAME, f"revision-{number:010d}.json"),
                vdir,
            )
        self.assertEqual(Lens(self.dir.name).versions(), [3, 4, 5, 6])
        again = Lens(self.dir.name)
        again.infer([{"q": 1}])
        again.save()
        names = set(os.listdir(vdir))
        self.assertNotIn("revision-0000000001.json", names)
        self.assertNotIn("revision-0000000002.json", names)
        self.assertEqual(Lens(self.dir.name).versions()[0], 3)

    def test_unpublished_baseline_is_discarded_on_recovery(self):
        lens = self._committed(6)
        lens.compact()
        vdir = Path(self.dir.name, VERSIONS_DIRNAME)
        donor = Lens(tempfile.mkdtemp(), max_versions=100)
        donor._schedule_compaction = lambda **_: None
        for index in range(6):
            donor.infer([{"a": index, f"f{index}": index}])
            donor.save()
        for number in (1, 2, 3):
            shutil.copy(
                Path(donor.path, VERSIONS_DIRNAME, f"revision-{number:010d}.json"),
                vdir,
            )
        os.unlink(vdir / "baseline.json")
        again = Lens(self.dir.name)
        again.infer([{"q": 1}])
        again.save()
        names = set(os.listdir(vdir))
        self.assertNotIn("revision-0000000003.baseline.json", names)
        self.assertIn("revision-0000000001.json", names)
        self.assertEqual(
            Lens(self.dir.name).versions(), [1, 2, 3, 4, 5, 6, 7]
        )

    def test_corrupt_surviving_revision_only_affects_that_revision(self):
        lens = self._committed(6)
        lens.compact()
        path = revision_path(self.dir.name, 5)
        path.write_text("{broken", encoding="utf-8")
        self.assertEqual(lens.schema(version=3)["count"], 3)
        self.assertEqual(lens.schema(version=4)["count"], 4)
        self.assertEqual(lens.schema(version=6)["count"], 6)
        with self.assertRaises(SchemaConflict):
            lens.schema(version=5)
        # The listing names the file (content is only validated on read),
        # but the damaged version blocks no other revision.
        self.assertEqual(lens.versions(), [3, 4, 5, 6])

    def test_status_and_versions_do_not_parse_revision_files(self):
        from unittest import mock

        lens = self._committed(10)
        lens.compact()
        with mock.patch("schema_lens.core._decode_revision") as decode:
            Lens(self.dir.name).versions()
            Lens(self.dir.name).compact_status()
            decode.assert_not_called()

    def test_manual_compact_is_foreground_and_serialized(self):
        lens = self._committed(6)
        other = Lens(self.dir.name)
        other._lock.acquire()
        self.addCleanup(other._lock.release)
        with self.assertRaises(SchemaConflict):
            lens.compact()
        # Nothing was published under the failed attempt.
        self.assertIsNone(lens.compact_status()["baseline"])

    def test_cli_compact_and_status(self):
        for index in range(6):
            records = Path(self.dir.name, f"r{index}.jsonl")
            records.write_text(json.dumps({"a": index, f"f{index}": index}) + "\n")
            lens_dir = str(Path(self.dir.name, "lens"))
            with redirect_stdout(io.StringIO()):
                code = cli_main(["--path", lens_dir, "infer", str(records)])
            self.assertEqual(code, 0)
        lens_dir = str(Path(self.dir.name, "lens"))
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli_main(["--path", lens_dir, "compact-status"])
        self.assertEqual(code, 0)
        self.assertIsNone(json.loads(out.getvalue())["baseline"])
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli_main(["--path", lens_dir, "compact"])
        self.assertEqual(code, 0)
        status = json.loads(out.getvalue())
        self.assertEqual(status["baseline"], 3)
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli_main(["--path", lens_dir, "versions"])
        self.assertEqual(json.loads(out.getvalue()), [3, 4, 5, 6])


class BackgroundCompactionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_background_round_runs_after_watermark_without_blocking_commit(self):
        from schema_lens.core import COMPACT_WATERMARK

        lens = Lens(self.dir.name)
        lens.infer([{"a": 0}]); lens.save()
        deadline = time.time() + 5
        for index in range(1, COMPACT_WATERMARK + 1):
            lens.infer([{f"f{index}": index}])
            lens.save()
        while time.time() < deadline:
            if lens.compact_status()["baseline"] is not None:
                break
            time.sleep(0.02)
        status = lens.compact_status()
        self.assertIsNotNone(status["baseline"])
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.schema()["count"], COMPACT_WATERMARK + 1)

    def test_concurrent_commits_compactions_and_reads_stay_complete(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 0}]); lens.save()
        stop = threading.Event()
        bad = []
        seen = set()

        def writer():
            worker = Lens(self.dir.name)
            worker.load()
            n = 0
            while not stop.is_set():
                n += 1
                worker.infer([{"a": n, f"f{n % 40}": n}])
                try:
                    worker.save()
                except SchemaConflict:
                    continue
                if n % 4 == 0:
                    try:
                        worker.compact()
                    except SchemaConflict:
                        pass

        def reader():
            while not stop.is_set():
                r = Lens(self.dir.name)
                try:
                    r.load()
                    root = r.schema()
                    nums = r.versions()
                    r.schema(version=nums[-1])
                except SchemaConflict as exc:
                    bad.append(str(exc))
                    continue
                seen.add(root["count"])
                if "a" not in root["fields"] or root["count"] != root["fields"]["a"]["observed"]:
                    bad.append("incomplete revision observed")

        threads = [threading.Thread(target=writer)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        time.sleep(3)
        stop.set()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(bad, [])
        self.assertGreater(len(seen), 5)
        final = Lens(self.dir.name)
        final.load()
        self.assertIsNotNone(final.compact_status()["baseline"])


if __name__ == "__main__":
    unittest.main()
