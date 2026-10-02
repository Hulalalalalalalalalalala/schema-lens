import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
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
        # b really folded into the overflow entry, so it is checked
        # against its widened type set rather than called unexpected.
        self.assertEqual(lens.check({"a": 1, "b": 2}), [])
        reports = lens.check({"a": 1, "b": "x"})
        self.assertIn('$.b: type string not in field types [number]', reports)
        # c never appeared and therefore never folded into overflow; a
        # never-seen name is still an unexpected field.
        self.assertEqual(
            lens.check({"a": 1, "b": 2, "c": 3}),
            ["$.c: unexpected field"],
        )

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

    def test_overflow_records_which_names_actually_folded(self):
        lens = Lens(self.dir.name, max_fields=2)
        lens.infer([{"a": 1, "b": 1, "c": 2, "d": 3}])
        self.assertEqual(lens.schema()["overflow"]["names"], ["c", "d"])
        lens.infer([{"a": 2, "e": 4}])
        self.assertEqual(lens.schema()["overflow"]["names"], ["c", "d", "e"])

    def test_overflow_provenance_survives_commits_and_delta_merges(self):
        lens = Lens(self.dir.name, max_fields=2)
        lens.infer([{"a": 1, "b": 1, "c": 3}])
        lens.save()
        second = Lens(self.dir.name, max_fields=2)
        second.load()
        # c starts exact in this delta, but the disk side folded c into
        # overflow; the merged provenance must still remember c.
        second.infer([{"c": 9}])
        second.save()
        reloaded = Lens(self.dir.name, max_fields=2)
        reloaded.load()
        self.assertEqual(reloaded.schema()["overflow"]["names"], ["c"])
        self.assertEqual(reloaded.check({"a": 1, "b": 1, "c": 3}), [])
        self.assertEqual(
            reloaded.check({"a": 1, "b": 1, "zzz": 3}),
            ["$.zzz: unexpected field"],
        )

    def test_legacy_overflow_without_names_keeps_permissive(self):
        from schema_lens.core import _checksum

        lens = Lens(self.dir.name, max_fields=2)
        lens.infer([{"a": 1, "b": 1, "c": 3}])
        lens.save()
        path = revision_path(self.dir.name, 1)
        payload = json.loads(path.read_text())
        del payload["root"]["overflow"]["names"]
        payload["checksum"] = _checksum(payload["root"])
        path.write_text(json.dumps(payload), encoding="utf-8")
        reloaded = Lens(self.dir.name, max_fields=2)
        reloaded.load()
        self.assertIsNone(reloaded.schema()["overflow"]["names"])
        # Unknown provenance widens for any name.
        self.assertEqual(reloaded.check({"a": 1, "b": 1, "zzz": 3}), [])
        # Merging more batches in keeps the unknown provenance permissive.
        reloaded.infer([{"d": 4}])
        reloaded.save()
        again = Lens(self.dir.name, max_fields=2)
        again.load()
        self.assertIsNone(again.schema()["overflow"]["names"])
        self.assertEqual(again.check({"a": 1, "b": 1, "qqq": 3}), [])

    def test_dotted_overflow_name_uses_quoted_path_in_check(self):
        lens = Lens(self.dir.name, max_fields=1)
        lens.infer([{"a": 1, "x.y": 2}])
        reports = lens.check({"a": 1, "x.y": "bad"})
        self.assertEqual(
            reports, ['$["x.y"]: type string not in field types [number]']
        )

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
        self._write([{"a": 2, "b": 3}])
        code, _ = self._run("--path", self.lens_dir, "check", self.records)
        self.assertEqual(code, 0)
        # A never-seen name is unexpected even though an overflow entry
        # exists (only names that really folded into it widen).
        self._write([{"a": 2, "b": 3, "c": 4}])
        code, out = self._run("--path", self.lens_dir, "check", self.records)
        self.assertEqual(code, 1)
        self.assertIn("line 1: $.c: unexpected field", out)
        self._write([{"a": "bad"}])
        code, out = self._run("--path", self.lens_dir, "check", self.records)
        self.assertEqual(code, 1)
        self.assertIn("$.a: type string not in field types [number]", out)


def baseline_path(directory: str, anchor: int) -> Path:
    return Path(directory, VERSIONS_DIRNAME, f"baseline-{anchor:010d}.json")


def _build_history(directory: str, count: int, **lens_kwargs) -> Lens:
    lens = Lens(directory, **lens_kwargs)
    for index in range(count):
        lens.infer([{f"f{index}": index}, {"shared": "s"}])
        lens.save()
    return lens


class CompactionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = _build_history(self.dir.name, 8, compact_keep=3)
        self.before = {v: self.lens.schema(version=v) for v in self.lens.versions()}

    def test_compact_returns_the_merged_range(self):
        result = self.lens.compact()
        self.assertEqual(result, {"anchor": 5, "merged": [1, 2, 3, 4, 5]})

    def test_versions_after_compaction_start_at_anchor(self):
        self.lens.compact()
        self.assertEqual(self.lens.versions(), [5, 6, 7, 8])

    def test_baseline_schema_equals_anchor(self):
        self.lens.compact()
        self.assertEqual(self.lens.schema(version=5), self.before[5])

    def test_surviving_revisions_stay_equivalent(self):
        self.lens.compact()
        for version in (6, 7, 8):
            self.assertEqual(self.lens.schema(version=version), self.before[version])

    def test_merged_revisions_raise_schema_conflict(self):
        result = self.lens.compact()
        for version in result["merged"][:-1]:
            with self.assertRaises(SchemaConflict):
                self.lens.schema(version=version)
            with self.assertRaises(SchemaConflict):
                self.lens.check({"f0": 1}, version=version)
            with self.assertRaises(SchemaConflict):
                self.lens.load(version=version)

    def test_merged_files_are_gone_baseline_is_the_floor(self):
        self.lens.compact()
        names = set(os.listdir(Path(self.dir.name, VERSIONS_DIRNAME)))
        self.assertEqual(
            names,
            {
                "manifest.json",
                "baseline-0000000005.json",
                "revision-0000000006.json",
                "revision-0000000007.json",
                "revision-0000000008.json",
            },
        )

    def test_no_compaction_within_keep_bound(self):
        result = self.lens.compact(keep=10)
        self.assertIsNone(result)
        self.assertEqual(self.lens.versions(), [1, 2, 3, 4, 5, 6, 7, 8])

    def test_compaction_keeps_exact_statistics(self):
        self.lens.compact()
        reloaded = Lens(self.dir.name)
        self.assertEqual(reloaded.schema(version=5), self.before[5])
        reloaded.load()
        self.assertEqual(reloaded.stats()["records"], 16)

    def test_check_against_baseline_uses_its_schema(self):
        self.lens.compact()
        # f0..f4 were folded before the anchor; f5..f7 never were.
        self.assertEqual(
            self.lens.check({"f4": 4, "shared": "s"}, version=5), []
        )
        reports = self.lens.check({"f5": 5}, version=5)
        self.assertIn("$.f5: unexpected field", reports)
        # Surviving revisions keep their own wider schema.
        self.assertEqual(
            self.lens.check({"f7": 7, "shared": "s"}, version=8), []
        )

    def test_rollback_after_compaction_targets_only_live_versions(self):
        self.lens.compact()
        with self.assertRaises(SchemaConflict):
            self.lens.rollback(1)
        new_revision = self.lens.rollback(5)
        self.assertEqual(new_revision, 9)
        self.assertEqual(self.lens.schema(version=9), self.before[5])
        self.assertEqual(self.lens.versions(), [5, 6, 7, 8, 9])

    def test_commits_after_compaction_build_on_head(self):
        self.lens.compact()
        self.lens.load()
        self.lens.infer([{"later": True}])
        self.lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        fields = set(reloaded.schema()["fields"])
        self.assertEqual(
            fields,
            {f"f{i}" for i in range(8)} | {"shared", "later"},
        )
        self.assertEqual(reloaded.versions(), [5, 6, 7, 8, 9])

    def test_status_reports_baseline_and_merged(self):
        status = self.lens.compaction_status()
        self.assertFalse(status["running"])
        self.assertIsNone(status["baseline"])
        self.assertEqual(status["merged"], [])
        self.assertEqual(status["versions"], [1, 2, 3, 4, 5, 6, 7, 8])
        self.lens.compact()
        status = self.lens.compaction_status()
        self.assertEqual(status["baseline"], 5)
        self.assertEqual(status["merged"], [1, 2, 3, 4, 5])
        self.assertEqual(status["versions"], [5, 6, 7, 8])
        self.assertFalse(status["running"])

    def test_repeated_compaction_replaces_the_baseline(self):
        first = self.lens.compact()
        self.assertEqual(first["anchor"], 5)
        for index in range(8, 12):
            self.lens.infer([{f"f{index}": index}])
            self.lens.save()
        second = self.lens.compact()
        self.assertEqual(second["merged"], [5, 6, 7, 8, 9])
        self.assertEqual(second["anchor"], 9)
        names = os.listdir(Path(self.dir.name, VERSIONS_DIRNAME))
        baselines = [n for n in names if n.startswith("baseline-")]
        self.assertEqual(baselines, ["baseline-0000000009.json"])
        with self.assertRaises(SchemaConflict):
            self.lens.schema(version=5)
        self.assertEqual(self.lens.versions(), [9, 10, 11, 12])

    def test_compaction_survives_reopen(self):
        self.lens.compact()
        fresh = Lens(self.dir.name)
        self.assertEqual(fresh.versions(), [5, 6, 7, 8])
        fresh.load()
        self.assertEqual(fresh.schema(), self.lens.schema())
        self.assertEqual(fresh.schema(version=5), self.before[5])

    def test_compaction_with_capped_fields_stays_exact(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(
            lambda: __import__("shutil").rmtree(directory, ignore_errors=True)
        )
        lens = Lens(directory, max_fields=2)
        for index in range(7):
            lens.infer([{f"f{index}": index, "x": index}])
            lens.save()
        before = {v: lens.schema(version=v) for v in lens.versions()}
        result = lens.compact(keep=2)
        for version in lens.versions():
            self.assertEqual(lens.schema(version=version), before[version])
        overflow = lens.schema(version=result["anchor"])["overflow"]
        self.assertEqual(overflow["names"], ["f1", "f2", "f3", "f4"])
        # Only names truly folded into overflow widen.
        self.assertEqual(
            lens.check(
                {"f0": 1, "x": 1, "brand_new": 1}, version=result["anchor"]
            ),
            ["$.brand_new: unexpected field"],
        )
        self.assertEqual(
            lens.check(
                {"f0": 1, "x": 1, "f3": "bad"}, version=result["anchor"]
            ),
            ["$.f3: type string not in field types [number]"],
        )

    def test_compaction_fold_includes_rollback_revisions(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(
            lambda: __import__("shutil").rmtree(directory, ignore_errors=True)
        )
        lens = Lens(directory)
        for index in range(3):
            lens.infer([{f"f{index}": index}])
            lens.save()
        lens.rollback(1)  # revision 4 is equivalent to revision 1
        for index in range(3, 6):
            lens.infer([{f"g{index}": index}])
            lens.save()
        before = {v: lens.schema(version=v) for v in lens.versions()}
        result = lens.compact(keep=2)
        for version in lens.versions():
            self.assertEqual(lens.schema(version=version), before[version])
        self.assertEqual(result["anchor"], 5)

    def test_baseline_is_never_pruned_by_retention(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(
            lambda: __import__("shutil").rmtree(directory, ignore_errors=True)
        )
        lens = Lens(directory, max_versions=2, compact_keep=1)
        for index in range(5):
            lens.infer([{f"f{index}": index}])
            lens.save()
        result = lens.compact()
        for index in range(5, 9):
            lens.infer([{f"f{index}": index}])
            lens.save()
        self.assertTrue(baseline_path(directory, result["anchor"]).exists())
        self.assertEqual(lens.versions()[0], result["anchor"])

    def test_version_lookup_is_one_file_not_a_scan(self):
        from schema_lens import core as core_module

        decoded = []
        original = core_module._decode_revision
        try:
            core_module._decode_revision = lambda path: (
                decoded.append(Path(path).name) or original(path)
            )
            # One explicit revision decodes exactly that one file.
            self.lens.schema(version=7)
            self.assertEqual(decoded, ["revision-0000000007.json"])
            self.lens.compact()
            decoded.clear()
            self.lens.schema(version=6)
            self.assertEqual(decoded, ["revision-0000000006.json"])
            decoded.clear()
            self.lens.schema(version=5)  # anchor -> one baseline file
            self.assertEqual(decoded, [])  # baseline uses a separate decoder
        finally:
            core_module._decode_revision = original

    def test_versions_listing_parses_no_revision_files(self):
        from schema_lens import core as core_module

        calls = []
        original = core_module._decode_revision
        try:
            core_module._decode_revision = lambda path: (
                calls.append(path) or original(path)
            )
            self.assertEqual(self.lens.versions(), [1, 2, 3, 4, 5, 6, 7, 8])
            self.lens.compact()
            calls.clear()
            self.assertEqual(self.lens.versions(), [5, 6, 7, 8])
            self.assertEqual(calls, [])
        finally:
            core_module._decode_revision = original

    def test_one_corrupt_surviving_revision_isolates(self):
        self.lens.compact()
        path = revision_path(self.dir.name, 7)
        path.write_text("{broken", encoding="utf-8")
        # Baseline and other revisions stay readable.
        self.assertEqual(self.lens.schema(version=5), self.before[5])
        self.assertEqual(self.lens.schema(version=6), self.before[6])
        with self.assertRaises(SchemaConflict):
            self.lens.schema(version=7)
        self.assertEqual(self.lens.schema(version=8), self.before[8])

    def test_corrupt_baseline_does_not_expose_merged_numbers(self):
        result = self.lens.compact()
        path = baseline_path(self.dir.name, result["anchor"])
        path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            self.lens.schema(version=result["anchor"])
        # A merged-away revision stays merged-away (no file resurrection).
        with self.assertRaises(SchemaConflict):
            self.lens.schema(version=1)
        self.assertEqual(self.lens.schema(version=6), self.before[6])

    def test_compaction_does_not_read_the_surviving_tail(self):        # Compaction must cost less than re-reading every revision: the
        # surviving newest revisions are never decoded to build the
        # baseline.
        from schema_lens import core as core_module

        decoded = []
        original = core_module._decode_revision

        def spy(path):
            decoded.append(Path(path).name)
            return original(path)

        fresh = Lens(self.dir.name, compact_keep=1)
        for index in range(8, 11):
            fresh.infer([{f"f{index}": index}])
            fresh.save()
        core_module._decode_revision = spy
        try:
            fresh.compact(keep=2)
        finally:
            core_module._decode_revision = original
        self.assertNotIn("revision-0000000010.json", decoded)


class CompactionCrashTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _history(self, count=5):
        return _build_history(self.dir.name, count)

    def test_stray_baseline_without_manifest_is_rolled_back(self):
        self._history(4)
        versions_dir = Path(self.dir.name, VERSIONS_DIRNAME)
        # A staged baseline that never had its manifest switched.
        (versions_dir / "baseline-0000000002.json").write_text(
            (versions_dir / "revision-0000000002.json").read_text(),
            encoding="utf-8",
        )
        (versions_dir / "revision-0000000009.json.tmp").write_text(
            "{half", encoding="utf-8"
        )
        other = Lens(self.dir.name)
        other.infer([{"z": 1}])
        other.save()  # taking the lock reconciles the leftovers
        names = {p.name for p in versions_dir.iterdir()}
        self.assertNotIn("baseline-0000000002.json", names)
        self.assertNotIn("revision-0000000009.json.tmp", names)
        self.assertEqual(Lens(self.dir.name).versions(), [1, 2, 3, 4, 5])

    def test_manifest_without_baseline_file_is_corrupt(self):
        lens = self._history(4)
        result = lens.compact(keep=2)
        os.unlink(baseline_path(self.dir.name, result["anchor"]))
        # The pointer names a baseline that is gone; that is corruption,
        # never a silent fallback to an older revision.
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).versions()
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).load()

    def test_extra_baseline_after_switch_is_cleaned_up(self):
        lens = self._history(5)
        result = lens.compact(keep=2)
        versions_dir = Path(self.dir.name, VERSIONS_DIRNAME)
        stale = versions_dir / "baseline-0000000002.json"
        stale.write_text(
            (versions_dir / f"baseline-{result['anchor']:010d}.json").read_text(),
            encoding="utf-8",
        )
        other = Lens(self.dir.name)
        other.infer([{"z": 1}])
        other.save()
        self.assertFalse(stale.exists())
        names = [n for n in os.listdir(versions_dir) if n.startswith("baseline-")]
        self.assertEqual(names, [f"baseline-{result['anchor']:010d}.json"])

    def test_merged_revision_left_after_switch_is_removed_on_reconcile(self):
        lens = self._history(5)
        lens.compact(keep=2)
        versions_dir = Path(self.dir.name, VERSIONS_DIRNAME)
        # Resurrect one merged file as if its deletion was interrupted.
        ghost = versions_dir / "revision-0000000001.json"
        ghost.write_text("{}", encoding="utf-8")
        other = Lens(self.dir.name)
        other.infer([{"z": 1}])
        other.save()
        self.assertFalse(ghost.exists())
        # A directory listing cannot resurrect a compacted-away number.
        self.assertEqual(other.versions()[0], 3)
        with self.assertRaises(SchemaConflict):
            other.schema(version=1)

    def test_no_two_baselines_ever_coexist(self):
        lens = self._history(6)
        lens.compact(keep=2)
        for index in range(6, 9):
            lens.infer([{f"f{index}": index}])
            lens.save()
        lens.compact(keep=2)
        versions_dir = Path(self.dir.name, VERSIONS_DIRNAME)
        baselines = [
            n for n in os.listdir(versions_dir) if n.startswith("baseline-")
        ]
        self.assertEqual(len(baselines), 1)
        fresh = Lens(self.dir.name)
        fresh.load()
        self.assertEqual(len(fresh.versions()), 3)


@unittest.skipIf(fcntl is None, "flock is only available on POSIX")
class BackgroundCompactionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = _build_history(self.dir.name, 6, compact_keep=2)

    def _wait(self, lens, timeout=10.0):
        import time as _time

        remaining = timeout
        while remaining > 0 and lens.compaction_status()["running"]:
            _time.sleep(0.02)
            remaining -= 0.02
        self.assertFalse(lens.compaction_status()["running"])

    def test_background_compaction_eventually_switches(self):
        from schema_lens import core as core_module

        # Slow the staging decode so the run is observably in flight.
        original = core_module._decode_revision

        def slow(path):
            import time as _time

            _time.sleep(0.05)
            return original(path)

        core_module._decode_revision = slow
        try:
            self.lens.compact(background=True)
            self.assertTrue(self.lens.compaction_status()["running"])
        finally:
            core_module._decode_revision = original
        self._wait(self.lens)
        status = self.lens.compaction_status()
        self.assertIsNone(status["error"])
        self.assertIsNotNone(status["baseline"])
        self.assertEqual(status["versions"][0], status["baseline"])

    def test_concurrent_commits_and_reads_never_see_a_broken_read(self):
        from schema_lens import core as core_module

        original = core_module._decode_revision

        def slow(path):
            import time as _time

            _time.sleep(0.02)
            return original(path)

        core_module._decode_revision = slow
        self.addCleanup(setattr, core_module, "_decode_revision", original)
        self.lens.compact(background=True)

        writer = Lens(self.dir.name)
        writer.load()
        bad_reads = []
        for index in range(6, 16):
            writer.infer([{f"f{index}": index}])
            try:
                writer.save()
            except SchemaConflict:
                pass
            reader = Lens(self.dir.name)
            try:
                reader.load()
                root = reader.schema()
                if not isinstance(root.get("count"), int):
                    bad_reads.append("no count")
            except SchemaConflict as exc:
                bad_reads.append(str(exc))
        self._wait(self.lens)
        self.assertEqual(bad_reads, [])
        final = Lens(self.dir.name)
        final.load()
        self.assertIn("f15", final.schema()["fields"])
        baselines = [
            n
            for n in os.listdir(Path(self.dir.name, VERSIONS_DIRNAME))
            if n.startswith("baseline-")
        ]
        self.assertEqual(len(baselines), 1)

    def test_second_background_start_while_running_conflicts(self):
        self.lens.compact(background=True)
        try:
            with self.assertRaises(SchemaConflict):
                self.lens.compact(background=True)
        finally:
            self._wait(self.lens)

    def test_background_failure_is_reported_in_status(self):
        from unittest import mock

        with mock.patch(
            "schema_lens.core._decode_revision",
            side_effect=SchemaConflict("simulated corruption"),
        ):
            self.lens.compact(background=True)
            self._wait(self.lens)
        status = self.lens.compaction_status()
        self.assertFalse(status["running"])
        self.assertIn("simulated corruption", status["error"])
        # Nothing switched: the history is intact and still compactable.
        self.assertEqual(status["versions"], [1, 2, 3, 4, 5, 6])
        self.assertTrue(self.lens.compact())


class CompactionCliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens_dir = str(Path(self.dir.name, "lens"))
        self.records = str(Path(self.dir.name, "records.jsonl"))

    def _commit(self, record):
        with open(self.records, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _run(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli_main(list(argv))
        return code, out.getvalue()

    def _history(self, count):
        for index in range(count):
            self._commit({f"f{index}": index})
            code, _ = self._run(
                "--path", self.lens_dir, "infer", self.records
            )
            self.assertEqual(code, 0)

    def test_compact_command_outputs_anchor(self):
        self._history(5)
        code, out = self._run(
            "--path", self.lens_dir, "--keep", "2", "compact"
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(out),
            {"anchor": 3, "compacted": True, "merged": [1, 2, 3]},
        )

    def test_compact_within_bound_reports_not_compacted(self):
        self._history(2)
        code, out = self._run(
            "--path", self.lens_dir, "--keep", "10", "compact"
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), {"compacted": False})

    def test_status_command_reports_state(self):
        self._history(4)
        code, out = self._run("--path", self.lens_dir, "status")
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(out),
            {
                "baseline": None,
                "error": None,
                "merged": [],
                "range": None,
                "running": False,
                "versions": [1, 2, 3, 4],
            },
        )
        self._run("--path", self.lens_dir, "--keep", "1", "compact")
        code, out = self._run("--path", self.lens_dir, "status")
        status = json.loads(out)
        self.assertEqual(status["baseline"], 3)
        self.assertEqual(status["versions"], [3, 4])

    def test_keep_only_valid_with_compact(self):
        code, _ = self._run(
            "--path", self.lens_dir, "--keep", "2", "versions"
        )
        self.assertEqual(code, 2)

    def test_compact_rejects_positional_target(self):
        code, _ = self._run("--path", self.lens_dir, "compact", "3")
        self.assertEqual(code, 2)

    def test_background_compact_detaches_and_finishes(self):
        import time as _time

        self._history(4)
        code, out = self._run(
            "--path", self.lens_dir, "--keep", "1",
            "--background", "compact",
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), {"started": True})
        status = {}
        for _ in range(100):
            code, out = self._run("--path", self.lens_dir, "status")
            status = json.loads(out)
            if status["baseline"] is not None:
                break
            _time.sleep(0.05)
        self.assertEqual(status["baseline"], 3)
        self.assertFalse(status["running"])


def _publish(directory, records, revision, **limits):
    """Fold ``records`` in isolation and publish the exact root as a revision.

    Each revision is therefore a standalone accepted set (no folding onto a
    predecessor), which gives compat tests precise control over required
    fields and type sets on either side of a comparison.
    """
    from schema_lens.core import _encode_revision, _revision_path

    folder = Lens(tempfile.mkdtemp(), **limits)
    folder.infer(records)
    root = folder.schema()
    versions_dir = Path(directory, VERSIONS_DIRNAME)
    versions_dir.mkdir(parents=True, exist_ok=True)
    _revision_path(versions_dir, revision).write_text(
        _encode_revision(revision, root), encoding="utf-8"
    )
    return root


class CompatTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)

    def _publish_pair(self, old_records, new_records, **limits):
        _publish(self.dir.name, old_records, 1, **limits)
        _publish(self.dir.name, new_records, 2, **limits)
        return self.lens.compat(1, 2)

    def test_report_shape_and_key_set(self):
        _publish(self.dir.name, [{"a": 1}], 1)
        _publish(self.dir.name, [{"a": 2}], 2)
        report = self.lens.compat(1, 2)
        self.assertEqual(
            set(report),
            {"from", "to", "backward", "forward", "changes", "unknown_reasons"},
        )
        self.assertEqual(report["from"], 1)
        self.assertEqual(report["to"], 2)

    def test_self_comparison_is_compatible_and_empty(self):
        _publish(self.dir.name, [{"a": 1}, {"a": 2}], 1)
        report = self.lens.compat(1, 1)
        self.assertEqual(report["backward"], "compatible")
        self.assertEqual(report["forward"], "compatible")
        self.assertEqual(report["changes"], [])
        self.assertEqual(report["unknown_reasons"], [])

    def test_repeated_analysis_is_stable(self):
        _publish(self.dir.name, [{"a": 1, "b": "x"}], 1)
        _publish(self.dir.name, [{"a": "s", "c": True}], 2)
        first = self.lens.compat(1, 2)
        for _ in range(3):
            self.assertEqual(self.lens.compat(1, 2), first)
        # Reversing the pair swaps the directional verdicts and endpoints
        # and inverts each change, but is itself stable.
        reversed_report = self.lens.compat(2, 1)
        self.assertEqual(reversed_report["from"], 2)
        self.assertEqual(reversed_report["to"], 1)
        self.assertEqual(reversed_report["backward"], first["forward"])
        self.assertEqual(reversed_report["forward"], first["backward"])
        self.assertEqual(
            len(reversed_report["changes"]), len(first["changes"])
        )
        self.assertEqual(self.lens.compat(2, 1), reversed_report)

    def test_identical_schemas_distinct_revisions_are_compatible(self):
        _publish(self.dir.name, [{"a": 1, "b": "x"}], 1)
        _publish(self.dir.name, [{"a": 2, "b": "y"}], 2)
        report = self.lens.compat(1, 2)
        self.assertEqual(
            (report["backward"], report["forward"]),
            ("compatible", "compatible"),
        )
        self.assertEqual(report["changes"], [])

    def test_add_optional_field_is_backward_compatible_forward_breaking(self):
        report = self._publish_pair(
            [{"a": 1}, {"a": 2}], [{"a": 3, "b": "x"}, {"a": 4}]
        )
        self.assertEqual(report["backward"], "compatible")
        self.assertEqual(report["forward"], "breaking")
        self.assertEqual(
            report["changes"],
            [{"path": "$.b", "change": "field_added", "breaking": False}],
        )

    def test_add_required_field_is_backward_breaking(self):
        report = self._publish_pair(
            [{"a": 1}, {"a": 2}], [{"a": 3, "b": "x"}, {"a": 4, "b": "y"}]
        )
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "breaking")
        self.assertEqual(
            report["changes"],
            [{"path": "$.b", "change": "field_added", "breaking": True}],
        )

    def test_remove_field_is_backward_breaking_forward_depends_on_optionality(self):
        # b required on the old side.
        report = self._publish_pair(
            [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], [{"a": 3}, {"a": 4}]
        )
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "breaking")
        self.assertIn(
            {"path": "$.b", "change": "field_removed", "breaking": True},
            report["changes"],
        )
        # b optional on the old side: new records omit it, so the forward
        # direction still accepts them.
        report = self._publish_pair(
            [{"a": 1, "b": "x"}, {"a": 2}], [{"a": 3}, {"a": 4}]
        )
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "compatible")

    def test_required_becomes_optional(self):
        report = self._publish_pair(
            [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
            [{"a": 3, "b": "x"}, {"a": 4}],
        )
        self.assertEqual(report["backward"], "compatible")
        self.assertEqual(report["forward"], "breaking")
        self.assertEqual(
            report["changes"],
            [{"path": "$.b", "change": "required_to_optional",
              "breaking": False}],
        )

    def test_optional_becomes_required(self):
        report = self._publish_pair(
            [{"a": 1, "b": "x"}, {"a": 2}],
            [{"a": 3, "b": "x"}, {"a": 4, "b": "y"}],
        )
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "compatible")
        self.assertEqual(
            report["changes"],
            [{"path": "$.b", "change": "optional_to_required",
              "breaking": True}],
        )

    def test_type_set_widens(self):
        report = self._publish_pair(
            [{"a": 1}, {"a": 2}], [{"a": 3}, {"a": "s"}]
        )
        self.assertEqual(report["backward"], "compatible")
        self.assertEqual(report["forward"], "breaking")
        self.assertEqual(
            report["changes"],
            [{"path": "$.a", "change": "type_string_added", "breaking": False}],
        )

    def test_type_set_narrows(self):
        report = self._publish_pair(
            [{"a": 3}, {"a": "s"}], [{"a": 1}, {"a": 2}]
        )
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "compatible")
        self.assertEqual(
            report["changes"],
            [{"path": "$.a", "change": "type_string_removed", "breaking": True}],
        )

    def test_nested_object_branch_narrows(self):
        report = self._publish_pair(
            [{"user": {"x": 1}}, {"user": {"x": "s"}}],
            [{"user": {"x": 2}}, {"user": {"x": 3}}],
        )
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "compatible")
        self.assertEqual(
            report["changes"],
            [{"path": "$.user.x", "change": "type_string_removed",
              "breaking": True}],
        )

    def test_nested_required_field_added_is_breaking(self):
        report = self._publish_pair(
            [{"user": {"x": 1}}],
            [{"user": {"x": 2, "y": 9}}, {"user": {"x": 3, "y": 8}}],
        )
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(
            report["changes"],
            [{"path": "$.user.y", "change": "field_added", "breaking": True}],
        )

    def test_array_element_kind_narrows(self):
        report = self._publish_pair(
            [{"tags": ["a", 1]}], [{"tags": ["b", "c"]}]
        )
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "compatible")
        self.assertEqual(
            report["changes"],
            [{"path": "$.tags[]", "change": "element_type_number_removed",
              "breaking": True}],
        )

    def test_array_element_kind_widens(self):
        report = self._publish_pair(
            [{"tags": ["a"]}], [{"tags": ["b", 1, None]}]
        )
        self.assertEqual(report["backward"], "compatible")
        self.assertEqual(report["forward"], "breaking")
        categories = {c["change"] for c in report["changes"]}
        self.assertEqual(
            categories, {"element_type_null_added", "element_type_number_added"}
        )
        self.assertTrue(all(not c["breaking"] for c in report["changes"]))

    def test_array_element_object_field_narrows(self):
        report = self._publish_pair(
            [{"items": [{"x": 1}, {"x": "s"}]}],
            [{"items": [{"x": 2}, {"x": 3}]}],
        )
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(
            report["changes"],
            [{"path": "$.items[].x", "change": "type_string_removed",
              "breaking": True}],
        )

    def test_changes_are_sorted_by_path_then_category(self):
        report = self._publish_pair(
            [{"z": 1}, {"a": 2}], [{"z": 3, "a": 4, "m": 5}]
        )
        paths = [c["path"] for c in report["changes"]]
        self.assertEqual(paths, sorted(paths))
        self.assertEqual(paths, ["$.a", "$.m", "$.z"])

    def test_quoted_dotted_key_path(self):
        report = self._publish_pair(
            [{"a.b": 1}, {"a.b": "s"}], [{"a.b": 2}, {"a.b": 3}]
        )
        self.assertEqual(
            report["changes"],
            [{"path": '$["a.b"]', "change": "type_string_removed",
              "breaking": True}],
        )

    def test_quoted_backslash_key_path(self):
        report = self._publish_pair(
            [{"a\\b": 1}, {"a\\b": "s"}], [{"a\\b": 2}]
        )
        self.assertEqual(report["changes"][0]["path"], '$["a\\\\b"]')

    def test_literal_star_key_is_quoted(self):
        report = self._publish_pair(
            [{"*": 1}, {"*": "s"}], [{"*": 2}]
        )
        self.assertEqual(
            report["changes"],
            [{"path": '$["*"]', "change": "type_string_removed",
              "breaking": True}],
        )
        self.assertNotIn("$.*", report["unknown_reasons"])

    def test_overflow_on_either_side_is_unknown_both_ways(self):
        _publish(self.dir.name, [{"a": 1, "b": 2}], 1, max_fields=1)
        _publish(self.dir.name, [{"a": 3, "c": 4}], 2, max_fields=1)
        report = self.lens.compat(1, 2)
        self.assertEqual(report["backward"], "unknown")
        self.assertEqual(report["forward"], "unknown")
        self.assertEqual(report["unknown_reasons"], ["$.*"])

    def test_overflow_marker_distinct_from_literal_star_field(self):
        _publish(self.dir.name, [{"*": 1, "b": 2}], 1, max_fields=1)
        _publish(self.dir.name, [{"*": 3, "c": 4}], 2, max_fields=1)
        report = self.lens.compat(1, 2)
        self.assertIn("$.*", report["unknown_reasons"])
        # The real "*" field is never rendered as the bare marker.
        for change in report["changes"]:
            self.assertNotEqual(change["path"], "$.*")

    def test_depth_capped_branch_is_unknown_both_ways(self):
        _publish(self.dir.name, [{"u": {"k": {"deep": 1}}}], 1, max_depth=1)
        _publish(self.dir.name, [{"u": {"k": {"other": 2}}}], 2, max_depth=1)
        report = self.lens.compat(1, 2)
        self.assertEqual(report["backward"], "unknown")
        self.assertEqual(report["forward"], "unknown")
        self.assertIn("$.u.k", report["unknown_reasons"])

    def test_depth_capped_self_comparison_is_clean(self):
        _publish(self.dir.name, [{"u": {"k": {"deep": 1}}}], 1, max_depth=1)
        report = self.lens.compat(1, 1)
        self.assertEqual(
            (report["backward"], report["forward"]),
            ("compatible", "compatible"),
        )
        self.assertEqual(report["unknown_reasons"], [])

    def test_unrelated_approximate_branch_still_allows_precise_verdict(self):
        # An approximated deep branch coexisting with an exact, breaking
        # change at another path makes the report unknown but still lists
        # the precise change.
        _publish(
            self.dir.name,
            [{"u": {"k": {"deep": 1}}, "x": 1}],
            1,
            max_depth=1,
        )
        _publish(
            self.dir.name,
            [{"u": {"k": {"deep": 2}}, "x": "s"}],
            2,
            max_depth=1,
        )
        report = self.lens.compat(1, 2)
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "breaking")
        self.assertIn("$.u.k", report["unknown_reasons"])
        self.assertIn(
            {"path": "$.x", "change": "type_string_added", "breaking": False},
            report["changes"],
        )

    def test_approximate_array_element_branch_is_unknown(self):
        _publish(self.dir.name, [{"items": [{"k": {"d": 1}}]}], 1,
                 max_depth=1)
        _publish(self.dir.name, [{"items": [{"k": {"d": 2, "e": 3}}]}], 2,
                 max_depth=1)
        report = self.lens.compat(1, 2)
        self.assertEqual(report["backward"], "unknown")
        self.assertEqual(report["forward"], "unknown")
        self.assertTrue(report["unknown_reasons"])

    def test_compat_does_not_change_open_snapshot_or_state(self):
        _publish(self.dir.name, [{"a": 1}], 1)
        _publish(self.dir.name, [{"a": 2, "b": "x"}], 2)
        reader = Lens(self.dir.name)
        reader.load(version=1)
        before = reader.schema()
        before_versions = reader.versions()
        self.lens.compat(1, 2)
        reader.compat(1, 2)
        self.assertEqual(reader.schema(), before)
        self.assertEqual(reader.versions(), before_versions)
        # compat must not have created a new revision.
        self.assertEqual(reader.versions(), [1, 2])

    def test_missing_revision_raises_schema_conflict(self):
        _publish(self.dir.name, [{"a": 1}], 1)
        with self.assertRaises(SchemaConflict):
            self.lens.compat(1, 2)
        with self.assertRaises(SchemaConflict):
            self.lens.compat(2, 1)
        with self.assertRaises(SchemaConflict):
            self.lens.compat(9, 10)

    def test_compacted_away_revision_raises(self):
        lens = Lens(self.dir.name, compact_keep=1)
        for index in range(4):
            lens.infer([{f"f{index}": index}])
            lens.save()
        lens.compact()
        with self.assertRaises(SchemaConflict):
            lens.compat(1, 4)
        # Baseline anchor to surviving tail is still comparable.
        report = lens.compat(3, 4)
        self.assertIn(report["backward"], ("compatible", "breaking", "unknown"))

    def test_corrupt_revision_raises_schema_conflict(self):
        _publish(self.dir.name, [{"a": 1}], 1)
        _publish(self.dir.name, [{"a": 2}], 2)
        revision_path(self.dir.name, 2).write_text("{broken", encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            self.lens.compat(1, 2)
        with self.assertRaises(SchemaConflict):
            self.lens.compat(2, 1)

    def test_truncated_revision_raises(self):
        _publish(self.dir.name, [{"a": 1}], 1)
        _publish(self.dir.name, [{"a": 2}], 2)
        path = revision_path(self.dir.name, 1)
        raw = path.read_text(encoding="utf-8")
        path.write_text(raw[: len(raw) // 3], encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            self.lens.compat(1, 2)

    def test_invalid_revision_arguments_raise(self):
        for bad_old, bad_new in ((0, 1), (1, 0), (-1, 2), (True, 1),
                                 (1, False)):
            with self.assertRaises(SchemaConflict):
                self.lens.compat(bad_old, bad_new)

    def test_verdicts_never_use_other_strings(self):
        _publish(self.dir.name, [{"a": 1}], 1)
        _publish(self.dir.name, [{"a": 2}], 2)
        for report in (self.lens.compat(1, 2), self.lens.compat(2, 1),
                       self.lens.compat(1, 1)):
            self.assertIn(report["backward"],
                          ("compatible", "breaking", "unknown"))
            self.assertIn(report["forward"],
                          ("compatible", "breaking", "unknown"))

    def test_real_inference_and_rollback_round_trip(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        lens.infer([{"a": "s", "b": "x"}])
        lens.save()
        lens.rollback(1)  # revision 3 narrows a and drops b
        report = lens.compat(2, 3)
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "compatible")
        self.assertIn(
            {"path": "$.a", "change": "type_string_removed", "breaking": True},
            report["changes"],
        )
        self.assertIn(
            {"path": "$.b", "change": "field_removed", "breaking": True},
            report["changes"],
        )


class CompatCliTests(unittest.TestCase):
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

    def _commit(self, records):
        self._write(records)
        code, _ = self._run("--path", self.lens_dir, "infer", self.records)
        self.assertEqual(code, 0)

    def test_compat_outputs_json_report(self):
        self._commit([{"a": 1}])
        self._commit([{"a": 2, "b": "x"}])
        code, out = self._run("--path", self.lens_dir, "compat", "1", "2")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["from"], 1)
        self.assertEqual(report["to"], 2)
        self.assertEqual(report["backward"], "compatible")
        self.assertEqual(report["forward"], "breaking")
        self.assertEqual(
            report["changes"],
            [{"path": "$.b", "change": "field_added", "breaking": False}],
        )
        self.assertEqual(report["unknown_reasons"], [])

    def test_compat_self_is_zero_with_empty_changes(self):
        self._commit([{"a": 1}])
        code, out = self._run("--path", self.lens_dir, "compat", "1", "1")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["backward"], "compatible")
        self.assertEqual(report["forward"], "compatible")
        self.assertEqual(report["changes"], [])

    def test_compat_breaking_still_exits_zero(self):
        self._commit([{"a": 1}])
        self._commit([{"a": 2, "b": "x"}])
        code, _ = self._run(
            "--path", self.lens_dir, "rollback", "1"
        )
        self.assertEqual(code, 0)
        code, out = self._run("--path", self.lens_dir, "compat", "2", "3")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["backward"], "breaking")

    def test_compat_unknown_still_exits_zero(self):
        self._write([{"a": 1, "b": 2}])
        code, _ = self._run(
            "--path", self.lens_dir, "--max-fields", "1",
            "infer", self.records,
        )
        self.assertEqual(code, 0)
        self._write([{"a": 3, "c": 4}])
        code, _ = self._run(
            "--path", self.lens_dir, "--max-fields", "1",
            "infer", self.records,
        )
        self.assertEqual(code, 0)
        code, out = self._run("--path", self.lens_dir, "compat", "1", "2")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["backward"], "unknown")
        self.assertEqual(report["forward"], "unknown")

    def test_compat_missing_revision_exits_two(self):
        self._commit([{"a": 1}])
        code, out = self._run("--path", self.lens_dir, "compat", "1", "9")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_compat_corrupt_revision_exits_two(self):
        self._commit([{"a": 1}])
        self._commit([{"a": 2}])
        (Path(self.lens_dir, VERSIONS_DIRNAME,
              "revision-0000000002.json")).write_text("{broken", encoding="utf-8")
        code, out = self._run("--path", self.lens_dir, "compat", "1", "2")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_compat_wrong_arg_count_exits_two_and_prints_no_report(self):
        self._commit([{"a": 1}])
        for argv in (("compat",), ("compat", "1"), ("compat", "1", "2", "3")):
            code, out = self._run("--path", self.lens_dir, *argv)
            self.assertEqual(code, 2)
            self.assertEqual(out, "")

    def test_compat_non_integer_revision_exits_two(self):
        self._commit([{"a": 1}])
        code, out = self._run(
            "--path", self.lens_dir, "compat", "1", "abc"
        )
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_compat_does_not_change_versions(self):
        self._commit([{"a": 1}])
        self._commit([{"a": 2}])
        self._run("--path", self.lens_dir, "compat", "1", "2")
        code, out = self._run("--path", self.lens_dir, "versions")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), [1, 2])

    def test_compat_version_flag_rejected(self):
        self._commit([{"a": 1}])
        code, _ = self._run(
            "--path", self.lens_dir, "--version", "1", "compat", "1", "1"
        )
        self.assertEqual(code, 2)


class CompatMatrixTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)

    def _publish(self, records, revision, **limits):
        _publish(self.dir.name, records, revision, **limits)

    def test_default_uses_all_revisions_in_ascending_order(self):
        for number in range(1, 4):
            self._publish([{"a": number, f"f{number}": 1}], number)
        report = self.lens.compat_matrix()
        self.assertEqual(set(report), {"revisions", "pairs"})
        self.assertEqual(report["revisions"], [1, 2, 3])

    def test_pairs_cover_every_ordered_pair_including_self(self):
        for number in range(1, 4):
            self._publish([{"a": number}], number)
        report = self.lens.compat_matrix()
        endpoints = [(pair["from"], pair["to"]) for pair in report["pairs"]]
        self.assertEqual(
            endpoints,
            [(1, 1), (1, 2), (1, 3),
             (2, 1), (2, 2), (2, 3),
             (3, 1), (3, 2), (3, 3)],
        )
        for pair in report["pairs"]:
            self.assertEqual(
                set(pair),
                {"from", "to", "backward", "forward", "changes",
                 "unknown_reasons"},
            )

    def test_each_pair_matches_single_compat_report(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2, "b": "x"}, {"a": 3}], 2)
        report = self.lens.compat_matrix()
        for pair in report["pairs"]:
            self.assertEqual(
                pair, self.lens.compat(pair["from"], pair["to"])
            )

    def test_directions_stay_separate(self):
        # Revision 2 adds an optional field: 1 -> 2 is backward
        # compatible / forward breaking; 2 -> 1 inverts the directions.
        self._publish([{"a": 1}, {"a": 2}], 1)
        self._publish([{"a": 3, "b": "x"}, {"a": 4}], 2)
        pairs = {(p["from"], p["to"]): p for p in self.lens.compat_matrix()["pairs"]}
        forward_evolution = pairs[(1, 2)]
        self.assertEqual(forward_evolution["backward"], "compatible")
        self.assertEqual(forward_evolution["forward"], "breaking")
        reverse = pairs[(2, 1)]
        self.assertEqual(reverse["backward"], "breaking")
        self.assertEqual(reverse["forward"], "compatible")

    def test_self_pairs_are_compatible_and_empty_even_when_approximate(self):
        self._publish([{"a": 1, "b": 2, "c": 3}], 1, max_fields=1)
        self._publish([{"u": {"k": {"deep": 1}}}], 2, max_depth=1)
        report = self.lens.compat_matrix()
        for pair in report["pairs"]:
            if pair["from"] == pair["to"]:
                self.assertEqual(pair["backward"], "compatible")
                self.assertEqual(pair["forward"], "compatible")
                self.assertEqual(pair["changes"], [])
                self.assertEqual(pair["unknown_reasons"], [])

    def test_approximate_pairs_still_report_fully(self):
        self._publish([{"a": 1, "b": 2}], 1, max_fields=1)
        self._publish([{"a": 3, "c": 4}], 2, max_fields=1)
        self._publish([{"a": 5}], 3)
        pairs = {(p["from"], p["to"]): p for p in self.lens.compat_matrix()["pairs"]}
        tainted = pairs[(1, 2)]
        self.assertEqual(tainted["backward"], "unknown")
        self.assertEqual(tainted["forward"], "unknown")
        self.assertEqual(tainted["unknown_reasons"], ["$.*"])
        # Pairs not touching the approximated revision are unaffected.
        for endpoints in ((3, 3),):
            pair = pairs[endpoints]
            self.assertEqual(pair["backward"], "compatible")
            self.assertEqual(pair["unknown_reasons"], [])

    def test_explicit_selection_and_subset_order(self):
        for number in range(1, 5):
            self._publish([{"a": number}], number)
        report = self.lens.compat_matrix([1, 3, 4])
        self.assertEqual(report["revisions"], [1, 3, 4])
        endpoints = [(p["from"], p["to"]) for p in report["pairs"]]
        self.assertEqual(
            endpoints,
            [(1, 1), (1, 3), (1, 4),
             (3, 1), (3, 3), (3, 4),
             (4, 1), (4, 3), (4, 4)],
        )
        # A tuple selection works the same as a list.
        self.assertEqual(
            self.lens.compat_matrix((1, 3, 4)), report
        )

    def test_repeated_calls_are_identical_in_content_and_order(self):
        self._publish([{"a": 1, "z": 1}], 1)
        self._publish([{"a": "s", "m": 2}], 2)
        first = self.lens.compat_matrix()
        for _ in range(3):
            self.assertEqual(self.lens.compat_matrix(), first)
        subset = self.lens.compat_matrix([1, 2])
        for _ in range(2):
            self.assertEqual(self.lens.compat_matrix([1, 2]), subset)

    def test_default_on_empty_history_raises_schema_conflict(self):
        with self.assertRaises(SchemaConflict):
            self.lens.compat_matrix()

    def test_invalid_selections_raise_value_error(self):
        self._publish([{"a": 1}], 1)
        for bad in (
            [],
            (),
            [True],
            [1, True],
            [False],
            [0],
            [-1],
            [1, 0],
            [1, 2.0],
            ["1", "2"],
            [1, 1],
            [2, 1],
            [3, 2, 1],
            [1, 3, 3],
        ):
            with self.assertRaises(ValueError):
                self.lens.compat_matrix(bad)
        with self.assertRaises(ValueError):
            self.lens.compat_matrix("1,2")
        with self.assertRaises(ValueError):
            self.lens.compat_matrix(12)

    def test_missing_revision_raises_schema_conflict(self):
        self._publish([{"a": 1}], 1)
        with self.assertRaises(SchemaConflict):
            self.lens.compat_matrix([1, 2])
        with self.assertRaises(SchemaConflict):
            self.lens.compat_matrix([9])

    def test_compacted_away_revision_raises(self):
        lens = Lens(self.dir.name, compact_keep=1)
        for index in range(4):
            lens.infer([{f"f{index}": index}])
            lens.save()
        lens.compact()
        with self.assertRaises(SchemaConflict):
            lens.compat_matrix([1, 4])
        # The surviving logical history still produces a full matrix.
        report = lens.compat_matrix()
        self.assertEqual(report["revisions"], [3, 4])
        self.assertEqual(len(report["pairs"]), 4)

    def test_corrupt_revision_raises_schema_conflict(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2}], 2)
        revision_path(self.dir.name, 2).write_text("{broken", encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            self.lens.compat_matrix()
        with self.assertRaises(SchemaConflict):
            self.lens.compat_matrix([1, 2])

    def test_matrix_does_not_change_open_snapshot_or_state(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2, "b": "x"}], 2)
        reader = Lens(self.dir.name)
        reader.load(version=1)
        before = reader.schema()
        before_versions = reader.versions()
        self.lens.compat_matrix()
        reader.compat_matrix([1, 2])
        self.assertEqual(reader.schema(), before)
        self.assertEqual(reader.versions(), before_versions)
        self.assertEqual(reader.versions(), [1, 2])

    def test_verdicts_never_use_other_strings(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": "s"}], 2)
        for pair in self.lens.compat_matrix()["pairs"]:
            self.assertIn(pair["backward"],
                          ("compatible", "breaking", "unknown"))
            self.assertIn(pair["forward"],
                          ("compatible", "breaking", "unknown"))


class CompatMatrixCliTests(unittest.TestCase):
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

    def _commit(self, records):
        self._write(records)
        code, _ = self._run("--path", self.lens_dir, "infer", self.records)
        self.assertEqual(code, 0)

    def test_matrix_outputs_json_matrix(self):
        self._commit([{"a": 1}])
        self._commit([{"a": 2, "b": "x"}])
        code, out = self._run("--path", self.lens_dir, "matrix")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["revisions"], [1, 2])
        self.assertEqual(
            [(p["from"], p["to"]) for p in report["pairs"]],
            [(1, 1), (1, 2), (2, 1), (2, 2)],
        )
        evolving = report["pairs"][1]
        self.assertEqual(evolving["backward"], "compatible")
        self.assertEqual(evolving["forward"], "breaking")
        self.assertTrue(out.endswith("\n"))

    def test_matrix_explicit_revisions(self):
        for number in range(1, 4):
            self._commit([{"a": number}])
        code, out = self._run(
            "--path", self.lens_dir, "matrix", "--revisions", "1,3"
        )
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["revisions"], [1, 3])
        self.assertEqual(
            [(p["from"], p["to"]) for p in report["pairs"]],
            [(1, 1), (1, 3), (3, 1), (3, 3)],
        )

    def test_matrix_empty_history_exits_two_without_report(self):
        code, out = self._run("--path", self.lens_dir, "matrix")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_matrix_invalid_revision_lists_exit_two(self):
        self._commit([{"a": 1}])
        for value in ("", "1,1", "2,1", "0", "-1", "true", "abc", "1,,2",
                      "1.5"):
            code, out = self._run(
                "--path", self.lens_dir, "matrix", "--revisions", value
            )
            self.assertEqual(code, 2, value)
            self.assertEqual(out, "", value)

    def test_matrix_missing_revision_exits_two(self):
        self._commit([{"a": 1}])
        code, out = self._run(
            "--path", self.lens_dir, "matrix", "--revisions", "1,9"
        )
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_matrix_corrupt_revision_exits_two(self):
        self._commit([{"a": 1}])
        self._commit([{"a": 2}])
        (Path(self.lens_dir, VERSIONS_DIRNAME,
              "revision-0000000002.json")).write_text(
            "{broken", encoding="utf-8"
        )
        code, out = self._run("--path", self.lens_dir, "matrix")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_matrix_takes_no_positional_arguments(self):
        self._commit([{"a": 1}])
        code, out = self._run("--path", self.lens_dir, "matrix", "1")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_revisions_flag_rejected_for_other_commands(self):
        self._commit([{"a": 1}])
        code, out = self._run(
            "--path", self.lens_dir, "versions", "--revisions", "1"
        )
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_matrix_approximate_self_pair_is_clean(self):
        self._write([{"a": 1, "b": 2}])
        code, _ = self._run(
            "--path", self.lens_dir, "--max-fields", "1",
            "infer", self.records,
        )
        self.assertEqual(code, 0)
        code, out = self._run("--path", self.lens_dir, "matrix")
        self.assertEqual(code, 0)
        report = json.loads(out)
        (self_pair,) = report["pairs"]
        self.assertEqual(self_pair["backward"], "compatible")
        self.assertEqual(self_pair["forward"], "compatible")
        self.assertEqual(self_pair["changes"], [])
        self.assertEqual(self_pair["unknown_reasons"], [])

    def test_matrix_output_is_deterministic(self):
        self._commit([{"a": 1, "z": 1}])
        self._commit([{"a": "s", "m": 2}])
        _, first = self._run("--path", self.lens_dir, "matrix")
        for _ in range(2):
            _, again = self._run("--path", self.lens_dir, "matrix")
            self.assertEqual(again, first)

    def test_matrix_does_not_change_versions(self):
        self._commit([{"a": 1}])
        self._commit([{"a": 2}])
        self._run("--path", self.lens_dir, "matrix")
        self._run(
            "--path", self.lens_dir, "matrix", "--revisions", "1,2"
        )
        code, out = self._run("--path", self.lens_dir, "versions")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), [1, 2])

    def test_matrix_version_flag_rejected(self):
        self._commit([{"a": 1}])
        code, _ = self._run(
            "--path", self.lens_dir, "--version", "1", "matrix"
        )
        self.assertEqual(code, 2)


class WitnessTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)

    def _publish(self, records, revision, **limits):
        _publish(self.dir.name, records, revision, **limits)

    def _publish_pair(self, old_records, new_records, **limits):
        self._publish(old_records, 1, **limits)
        self._publish(new_records, 2, **limits)
        return self.lens.compat_witness(1, 2)

    def _verify(self, report, accepting, rejecting):
        """Check every witness through the real Lens.check pipeline.

        ``accepting``/``rejecting`` are the revisions for the backward
        direction (old/new); forward swaps them. Only non-null witnesses
        are checked.
        """
        checked = {}
        for direction, acc, rej in (
            ("backward", accepting, rejecting),
            ("forward", rejecting, accepting),
        ):
            witness = report["witnesses"][direction]
            if witness is None:
                continue
            self.assertEqual(set(witness), {"record", "reports"})
            record = witness["record"]
            self.assertIsInstance(record, dict)
            self.assertEqual(self.lens.check(record, version=acc), [])
            reports = self.lens.check(record, version=rej)
            self.assertTrue(reports)
            self.assertEqual(witness["reports"], reports)
            checked[direction] = witness
        return checked

    def test_report_keeps_all_compat_fields_and_appends_witnesses(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2}], 2)
        report = self.lens.compat_witness(1, 2)
        compat = self.lens.compat(1, 2)
        self.assertEqual(
            set(report),
            set(compat) | {"witnesses"},
        )
        for key in compat:
            self.assertEqual(report[key], compat[key])
        self.assertEqual(report["witnesses"],
                         {"backward": None, "forward": None})

    def test_witness_order_follows_direction_roles(self):
        # 1 -> 2: old has required b, new dropped it.
        report = self._publish_pair(
            [{"a": 1, "b": "x"}], [{"a": 2}]
        )
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "breaking")
        witnesses = self._verify(report, 1, 2)
        # Backward: an old-legal record carrying b is unexpected under new.
        self.assertEqual(
            witnesses["backward"]["reports"], ["$.b: unexpected field"]
        )
        # Forward: a new-legal record omits b and misses it under old.
        self.assertEqual(
            witnesses["forward"]["reports"], ["$.b: missing required field"]
        )

    def test_add_required_field_witnesses(self):
        report = self._publish_pair(
            [{"a": 1}], [{"a": 2, "b": "x"}, {"a": 3, "b": "y"}]
        )
        witnesses = self._verify(report, 1, 2)
        # The old-legal witness simply omits the new required field.
        self.assertEqual(witnesses["backward"]["record"], {"a": 1})
        self.assertEqual(
            witnesses["backward"]["reports"], ["$.b: missing required field"]
        )
        # A new-legal record carrying b is unexpected under old.
        self.assertIn("b", witnesses["forward"]["record"])
        self.assertEqual(
            witnesses["forward"]["reports"], ["$.b: unexpected field"]
        )

    def test_add_optional_field_only_forward_has_witness(self):
        report = self._publish_pair(
            [{"a": 1}, {"a": 2}], [{"a": 3, "b": "x"}, {"a": 4}]
        )
        self.assertEqual(report["backward"], "compatible")
        self.assertEqual(report["forward"], "breaking")
        self.assertIsNone(report["witnesses"]["backward"])
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["forward"]["reports"], ["$.b: unexpected field"]
        )

    def test_remove_optional_field_only_backward_has_witness(self):
        report = self._publish_pair(
            [{"a": 1, "b": "x"}, {"a": 2}], [{"a": 3}, {"a": 4}]
        )
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "compatible")
        self.assertIsNone(report["witnesses"]["forward"])
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["backward"]["reports"], ["$.b: unexpected field"]
        )

    def test_optionality_changes(self):
        report = self._publish_pair(
            [{"a": 1, "b": "x"}, {"a": 2}],
            [{"a": 3, "b": "x"}, {"a": 4, "b": "y"}],
        )
        self.assertEqual(report["backward"], "breaking")
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["backward"]["reports"], ["$.b: missing required field"]
        )
        # The record really omits b rather than carrying null.
        self.assertNotIn("b", witnesses["backward"]["record"])

        report = self._publish_pair(
            [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
            [{"a": 3, "b": "x"}, {"a": 4}],
        )
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["forward"]["reports"], ["$.b: missing required field"]
        )
        self.assertNotIn("b", witnesses["forward"]["record"])

    def test_type_set_narrows_and_widens(self):
        # Narrowing old -> new: only backward is breaking; new-legal
        # numbers already passed the wider old check.
        report = self._publish_pair(
            [{"a": 3}, {"a": "s"}], [{"a": 1}, {"a": 2}]
        )
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["backward"]["reports"],
            ["$.a: type string not in field types [number]"],
        )
        self.assertIsNone(report["witnesses"]["forward"])

        report = self._publish_pair(
            [{"a": 1}, {"a": 2}], [{"a": 3}, {"a": "s"}]
        )
        witnesses = self._verify(report, 1, 2)
        self.assertIsNone(report["witnesses"]["backward"])
        self.assertEqual(
            witnesses["forward"]["reports"],
            ["$.a: type string not in field types [number]"],
        )

    def test_boolean_is_not_used_as_a_number(self):
        report = self._publish_pair([{"a": True}], [{"a": 1}])
        witnesses = self._verify(report, 1, 2)
        self.assertIs(witnesses["backward"]["record"]["a"], True)
        self.assertEqual(
            witnesses["backward"]["reports"],
            ["$.a: type boolean not in field types [number]"],
        )
        self.assertIs(witnesses["forward"]["record"]["a"], 1)
        self.assertEqual(
            witnesses["forward"]["reports"],
            ["$.a: type number not in field types [boolean]"],
        )

    def test_null_is_distinct_from_missing(self):
        # A null type branch removed: the witness carries null, it does
        # not merely omit the field.
        report = self._publish_pair([{"a": None}], [{"a": 1}])
        witnesses = self._verify(report, 1, 2)
        self.assertIn("a", witnesses["backward"]["record"])
        self.assertIsNone(witnesses["backward"]["record"]["a"])
        self.assertEqual(
            witnesses["backward"]["reports"],
            ["$.a: type null not in field types [number]"],
        )

    def test_null_type_branch_added_forward_witness_carries_null(self):
        report = self._publish_pair([{"a": 1}], [{"a": None}])
        witnesses = self._verify(report, 1, 2)
        self.assertIsNone(witnesses["forward"]["record"]["a"])
        self.assertEqual(
            witnesses["forward"]["reports"],
            ["$.a: type null not in field types [number]"],
        )

    def test_whole_array_type_branch_even_when_only_empty_arrays_seen(self):
        # The old side never saw the field as an array; the new side only
        # ever saw an empty array. An empty array is still kind array, so
        # old check rejects it at the field's type set.
        report = self._publish_pair([{"a": 1}], [{"a": []}])
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(witnesses["forward"]["record"], {"a": []})
        self.assertEqual(
            witnesses["forward"]["reports"],
            ["$.a: type array not in field types [number]"],
        )

    def test_nested_object_witness_fills_required_siblings(self):
        report = self._publish_pair(
            [{"user": {"x": 1, "name": "n"}}, {"user": {"x": 2, "name": "m"}}],
            [{"user": {"x": 3, "name": "o", "y": 9}},
             {"user": {"x": 4, "name": "p", "y": 8}}],
        )
        witnesses = self._verify(report, 1, 2)
        # The old-legal record keeps the required sibling name but omits y.
        self.assertEqual(
            witnesses["backward"]["record"],
            {"user": {"name": "x", "x": 1}},
        )
        self.assertEqual(
            witnesses["backward"]["reports"],
            ["$.user.y: missing required field"],
        )

    def test_nested_object_field_removed(self):
        report = self._publish_pair(
            [{"user": {"x": 1, "y": 2}}], [{"user": {"x": 3}}]
        )
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["backward"]["reports"], ["$.user.y: unexpected field"]
        )
        self.assertEqual(
            witnesses["forward"]["reports"], ["$.user.y: missing required field"]
        )

    def test_array_element_kind_witnesses(self):
        report = self._publish_pair(
            [{"tags": ["a", 1]}], [{"tags": ["b", "c"]}]
        )
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["backward"]["reports"],
            ["$.tags[0]: type number not in field types [string]"],
        )

    def test_array_element_object_witness(self):
        report = self._publish_pair(
            [{"items": [{"x": 1}, {"x": "s"}]}],
            [{"items": [{"x": 2}, {"x": 3}]}],
        )
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["backward"]["reports"],
            ["$.items[0].x: type string not in field types [number]"],
        )

    def test_multi_level_arrays(self):
        report = self._publish_pair(
            [{"grid": [[[{"k": 1}]], [[{"k": "s"}]]]}],
            [{"grid": [[[{"k": 2}]]]}],
        )
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["backward"]["reports"],
            ["$.grid[0][0][0].k: type string not in field types [number]"],
        )
        self.assertIsNone(report["witnesses"]["forward"])

        report = self._publish_pair(
            [{"grid": [[[{"k": 2}]]]}],
            [{"grid": [[[{"k": 1}]], [[{"k": "s"}]]]}],
        )
        witnesses = self._verify(report, 1, 2)
        self.assertIsNone(report["witnesses"]["backward"])
        self.assertEqual(
            witnesses["forward"]["reports"],
            ["$.grid[0][0][0].k: type string not in field types [number]"],
        )

    def test_empty_array_uses_existing_check_semantics(self):
        # New only ever saw empty arrays, so an element is rejected against
        # an empty element type set; the old-legal record carries one.
        report = self._publish_pair([{"a": ["x"]}], [{"a": []}])
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["backward"]["reports"],
            ["$.a[0]: type string not in field types []"],
        )
        # Reverse evolution: old only saw [], new widened with elements.
        self._publish([{"a": []}], 1)
        self._publish([{"a": ["x"]}], 2)
        reverse = self.lens.compat_witness(1, 2)
        witnesses = self._verify(reverse, 1, 2)
        self.assertIsNone(reverse["witnesses"]["backward"])
        self.assertEqual(
            witnesses["forward"]["reports"],
            ["$.a[0]: type string not in field types []"],
        )

    def test_quoted_special_keys_keep_real_names(self):
        report = self._publish_pair(
            [{"a.b": 1, "x\\y": 2, "*": 3}],
            [{"a.b": 4}],
        )
        witnesses = self._verify(report, 1, 2)
        record = witnesses["backward"]["record"]
        # The record keeps the literal key names.
        self.assertIn("a.b", record)
        self.assertIn("x\\y", record)
        self.assertIn("*", record)
        self.assertEqual(
            witnesses["backward"]["reports"],
            ['$["*"]: unexpected field',
             '$["x\\\\y"]: unexpected field'],
        )
        # The omission witness keeps the surviving real key.
        forward_record = witnesses["forward"]["record"]
        self.assertEqual(forward_record, {"a.b": 1})
        self.assertEqual(
            witnesses["forward"]["reports"],
            ['$["*"]: missing required field',
             '$["x\\\\y"]: missing required field'],
        )

    def test_reports_are_the_full_check_list(self):
        # Two removed required fields: the rejection list carries both
        # reports, exactly as check returns them.
        report = self._publish_pair(
            [{"a": 1, "b": 2, "c": 3}], [{"a": 1}]
        )
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["forward"]["reports"],
            ["$.b: missing required field",
             "$.c: missing required field"],
        )

    def test_unknown_pair_has_no_witnesses(self):
        self._publish([{"a": 1, "b": 2}], 1, max_fields=1)
        self._publish([{"a": 3, "c": 4}], 2, max_fields=1)
        report = self.lens.compat_witness(1, 2)
        self.assertEqual(report["backward"], "unknown")
        self.assertEqual(report["forward"], "unknown")
        self.assertIsNone(report["witnesses"]["backward"])
        self.assertIsNone(report["witnesses"]["forward"])

    def test_breaking_elsewhere_still_witnessed_under_unknown_branch(self):
        self._publish(
            [{"u": {"k": {"deep": 1}}, "x": 1}], 1, max_depth=1
        )
        self._publish(
            [{"u": {"k": {"deep": 2}}, "x": "s"}], 2, max_depth=1
        )
        report = self.lens.compat_witness(1, 2)
        self.assertEqual(report["backward"], "breaking")
        self.assertEqual(report["forward"], "breaking")
        self.assertIn("$.u.k", report["unknown_reasons"])
        witnesses = self._verify(report, 1, 2)
        self.assertEqual(
            witnesses["backward"]["reports"],
            ["$.x: type number not in field types [string]"],
        )
        self.assertEqual(
            witnesses["forward"]["reports"],
            ["$.x: type string not in field types [number]"],
        )

    def test_self_comparison_witnesses_are_null_even_when_approximate(self):
        self._publish([{"a": 1, "b": 2, "c": 3}], 1, max_fields=1)
        report = self.lens.compat_witness(1, 1)
        self.assertEqual(report["backward"], "compatible")
        self.assertEqual(report["forward"], "compatible")
        self.assertEqual(report["changes"], [])
        self.assertEqual(report["unknown_reasons"], [])
        self.assertEqual(report["witnesses"],
                         {"backward": None, "forward": None})

    def test_compatible_pair_witnesses_are_null(self):
        self._publish([{"a": 1, "b": "x"}], 1)
        self._publish([{"a": 2, "b": "y"}], 2)
        report = self.lens.compat_witness(1, 2)
        self.assertEqual(report["backward"], "compatible")
        self.assertEqual(report["forward"], "compatible")
        self.assertEqual(report["witnesses"],
                         {"backward": None, "forward": None})

    def test_repeated_analysis_is_content_stable(self):
        self._publish([{"a": 1, "z": 1, "b": "s"}], 1)
        self._publish([{"a": "s", "m": 2}], 2)
        first = self.lens.compat_witness(1, 2)
        for _ in range(3):
            self.assertEqual(self.lens.compat_witness(1, 2), first)
        # Serialized key order and bytes are stable too.
        encoded = json.dumps(first, sort_keys=True)
        for _ in range(3):
            self.assertEqual(
                json.dumps(self.lens.compat_witness(1, 2), sort_keys=True),
                encoded,
            )

    def test_reversed_pair_swaps_the_witness_roles(self):
        self._publish([{"a": 1, "b": "x"}], 1)
        self._publish([{"a": 2}], 2)
        forward = self.lens.compat_witness(1, 2)
        reverse = self.lens.compat_witness(2, 1)
        self.assertEqual(
            reverse["witnesses"]["backward"]["record"],
            forward["witnesses"]["forward"]["record"],
        )
        self.assertEqual(
            reverse["witnesses"]["forward"]["record"],
            forward["witnesses"]["backward"]["record"],
        )

    def test_works_before_any_load(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2, "b": "x"}], 2)
        fresh = Lens(self.dir.name)
        report = fresh.compat_witness(1, 2)
        self._verify(report, 1, 2)
        # Nothing was loaded into memory.
        with self.assertRaises(SchemaConflict):
            fresh.stats()

    def test_does_not_change_open_snapshot_pending_batch_or_history(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2, "b": "x"}], 2)
        reader = Lens(self.dir.name)
        reader.load(version=1)
        reader.infer([{"pending": 1}])
        before = reader.schema()
        before_stats = reader.stats()
        before_versions = reader.versions()
        self.lens.compat_witness(1, 2)
        reader.compat_witness(1, 2)
        reader.compat_witness(2, 1)
        self.assertEqual(reader.schema(), before)
        self.assertEqual(reader.stats(), before_stats)
        self.assertEqual(reader.versions(), before_versions)
        self.assertEqual(reader.versions(), [1, 2])
        # The pending batch is still committable on top.
        reader.save()
        self.assertEqual(reader.versions(), [1, 2, 3])

    def test_baseline_anchor_is_usable_as_input(self):
        lens = Lens(self.dir.name, compact_keep=1)
        for index in range(4):
            lens.infer([{f"f{index}": index}])
            lens.save()
        lens.compact()
        versions = lens.versions()
        report = lens.compat_witness(versions[0], versions[-1])
        self.assertIn(report["backward"],
                      ("compatible", "breaking", "unknown"))
        self.assertIn(report["forward"],
                      ("compatible", "breaking", "unknown"))

    def test_compacted_away_revision_raises(self):
        lens = Lens(self.dir.name, compact_keep=1)
        for index in range(4):
            lens.infer([{f"f{index}": index}])
            lens.save()
        lens.compact()
        with self.assertRaises(SchemaConflict):
            lens.compat_witness(1, 4)

    def test_missing_revision_raises(self):
        self._publish([{"a": 1}], 1)
        with self.assertRaises(SchemaConflict):
            self.lens.compat_witness(1, 2)
        with self.assertRaises(SchemaConflict):
            self.lens.compat_witness(9, 10)

    def test_corrupt_revision_raises(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2}], 2)
        revision_path(self.dir.name, 2).write_text("{broken", encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            self.lens.compat_witness(1, 2)

    def test_invalid_revision_arguments_raise(self):
        for bad_old, bad_new in (
            (0, 1), (1, 0), (-1, 2), (2, -1),
            (True, 1), (1, False), (1.5, 2), (1, 2.0),
            ("1", 2), (1, "2"),
        ):
            with self.assertRaises(SchemaConflict):
                self.lens.compat_witness(bad_old, bad_new)

    def test_legacy_single_file_migrates_then_witnesses(self):
        from schema_lens.core import _checksum

        root = Lens(tempfile.mkdtemp())
        root.infer([{"a": 1}])
        snapshot = root.schema()
        Path(self.dir.name, SCHEMA_FILENAME).write_text(
            json.dumps(
                {"version": 2, "checksum": _checksum(snapshot),
                 "root": snapshot},
                indent=2, sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        lens = Lens(self.dir.name)
        report = lens.compat_witness(1, 1)
        self.assertEqual(report["witnesses"],
                         {"backward": None, "forward": None})


class WitnessCliTests(unittest.TestCase):
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
        err = io.StringIO()
        with redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def _commit(self, records):
        self._write(records)
        code, _, _ = self._run("--path", self.lens_dir, "infer", self.records)
        self.assertEqual(code, 0)

    def test_witness_outputs_json_report(self):
        # rev 2 drops b; a rollback republishes rev 1's exact schema as
        # rev 3, so 2 -> 3 is a clean required-field addition.
        self._commit([{"a": 1, "b": "x"}])
        self._commit([{"a": 2}])
        code, _, _ = self._run("--path", self.lens_dir, "rollback", "1")
        self.assertEqual(code, 0)
        code, out, err = self._run(
            "--path", self.lens_dir, "witness", "2", "3"
        )
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        report = json.loads(out)
        self.assertEqual(
            set(report),
            {"from", "to", "backward", "forward", "changes",
             "unknown_reasons", "witnesses"},
        )
        self.assertTrue(out.endswith("\n"))
        backward = report["witnesses"]["backward"]
        self.assertEqual(backward["record"], {"a": 1})
        self.assertEqual(
            backward["reports"], ["$.b: missing required field"]
        )
        # The printed record itself verifies through the check command's
        # underlying API against both revisions.
        lens = Lens(self.lens_dir)
        self.assertEqual(lens.check(backward["record"], version=2), [])
        self.assertEqual(
            lens.check(backward["record"], version=3),
            backward["reports"],
        )

    def test_compat_fields_byte_match_compat_command(self):
        self._commit([{"a": 1, "z": 1}])
        self._commit([{"a": "s", "m": 2}])
        _, compat_out, _ = self._run(
            "--path", self.lens_dir, "compat", "1", "2"
        )
        _, witness_out, _ = self._run(
            "--path", self.lens_dir, "witness", "1", "2"
        )
        compat_report = json.loads(compat_out)
        witness_report = json.loads(witness_out)
        for key, value in compat_report.items():
            self.assertEqual(witness_report[key], value)

    def test_unknown_and_breaking_still_exit_zero(self):
        self._write([{"a": 1, "b": 2}])
        code, _, _ = self._run(
            "--path", self.lens_dir, "--max-fields", "1",
            "infer", self.records,
        )
        self.assertEqual(code, 0)
        self._write([{"a": 3, "c": 4}])
        code, _, _ = self._run(
            "--path", self.lens_dir, "--max-fields", "1",
            "infer", self.records,
        )
        self.assertEqual(code, 0)
        code, out, _ = self._run(
            "--path", self.lens_dir, "witness", "1", "2"
        )
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["backward"], "unknown")
        self.assertIsNone(report["witnesses"]["backward"])

    def test_self_comparison_exits_zero_with_null_witnesses(self):
        self._commit([{"a": 1}])
        code, out, _ = self._run(
            "--path", self.lens_dir, "witness", "1", "1"
        )
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["witnesses"],
                         {"backward": None, "forward": None})

    def test_output_is_byte_stable(self):
        self._commit([{"a": 1, "b": "x"}, {"a": 2, "c": 1}])
        self._commit([{"a": 3}, {"a": 4, "d": "y"}])
        _, first, _ = self._run("--path", self.lens_dir, "witness", "1", "2")
        for _ in range(3):
            _, again, _ = self._run(
                "--path", self.lens_dir, "witness", "1", "2"
            )
            self.assertEqual(again, first)

    def test_missing_revision_exits_two_with_empty_stdout(self):
        self._commit([{"a": 1}])
        code, out, err = self._run(
            "--path", self.lens_dir, "witness", "1", "9"
        )
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertNotEqual(err, "")

    def test_corrupt_revision_exits_two_with_empty_stdout(self):
        self._commit([{"a": 1}])
        self._commit([{"a": 2}])
        (Path(self.lens_dir, VERSIONS_DIRNAME,
              "revision-0000000002.json")).write_text(
            "{broken", encoding="utf-8"
        )
        code, out, _ = self._run(
            "--path", self.lens_dir, "witness", "1", "2"
        )
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_wrong_arg_count_exits_two_with_empty_stdout(self):
        self._commit([{"a": 1}])
        for argv in (
            ("witness",),
            ("witness", "1"),
            ("witness", "1", "2", "3"),
        ):
            code, out, _ = self._run("--path", self.lens_dir, *argv)
            self.assertEqual(code, 2)
            self.assertEqual(out, "")

    def test_non_integer_revision_exits_two(self):
        self._commit([{"a": 1}])
        for value in ("abc", "1.5", "true"):
            code, out, _ = self._run(
                "--path", self.lens_dir, "witness", "1", value
            )
            self.assertEqual(code, 2)
            self.assertEqual(out, "")

    def test_non_positive_revision_exits_two(self):
        self._commit([{"a": 1}])
        for value in ("0", "-1"):
            code, out, _ = self._run(
                "--path", self.lens_dir, "witness", value, "1"
            )
            self.assertEqual(code, 2)
            self.assertEqual(out, "")

    def test_witness_does_not_change_versions(self):
        self._commit([{"a": 1}])
        self._commit([{"a": 2}])
        self._run("--path", self.lens_dir, "witness", "1", "2")
        self._run("--path", self.lens_dir, "witness", "2", "1")
        code, out, _ = self._run("--path", self.lens_dir, "versions")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), [1, 2])

    def test_witness_version_flag_rejected(self):
        self._commit([{"a": 1}])
        code, out, _ = self._run(
            "--path", self.lens_dir, "--version", "1", "witness", "1", "1"
        )
        self.assertEqual(code, 2)
        self.assertEqual(out, "")

    def test_witness_works_on_compacted_anchor(self):
        for number in range(1, 5):
            self._commit([{"a": number, f"f{number}": 1}])
        code, _, _ = self._run(
            "--path", self.lens_dir, "--keep", "1", "compact"
        )
        self.assertEqual(code, 0)
        code, out, _ = self._run(
            "--path", self.lens_dir, "versions"
        )
        self.assertEqual(code, 0)
        versions = json.loads(out)
        code, out, _ = self._run(
            "--path", self.lens_dir, "witness",
            str(versions[0]), str(versions[-1]),
        )
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertIn(report["backward"],
                      ("compatible", "breaking", "unknown"))


class MigrateTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)

    def _publish(self, records, revision, **limits):
        return _publish(self.dir.name, records, revision, **limits)

    def test_report_shape_is_compat_plus_migration(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2}], 2)
        report = self.lens.migrate({"a": 1}, 1, 2, [])
        self.assertEqual(
            set(report),
            {"from", "to", "backward", "forward", "changes",
             "unknown_reasons", "migration"},
        )
        self.assertEqual(
            {key: report[key] for key in
             ("from", "to", "backward", "forward", "changes",
              "unknown_reasons")},
            self.lens.compat(1, 2),
        )
        self.assertEqual(
            report["migration"],
            {"stage": "done", "record": {"a": 1}, "reports": []},
        )

    def test_rename_between_revisions(self):
        self._publish([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], 1)
        self._publish([{"a": 1, "c": "x"}, {"a": 2, "c": "y"}], 2)
        report = self.lens.migrate(
            {"a": 1, "b": "x"}, 1, 2,
            [{"op": "rename", "path": ["b"], "to": "c"}],
        )
        self.assertEqual(
            report["migration"],
            {"stage": "done", "record": {"a": 1, "c": "x"}, "reports": []},
        )

    def test_empty_rules_validates_without_transforming(self):
        self._publish([{"a": 1}, {"a": 2}], 1)
        report = self.lens.migrate({"a": 1}, 1, 1, [])
        self.assertEqual(
            report["migration"],
            {"stage": "done", "record": {"a": 1}, "reports": []},
        )

    def test_source_failure_reports_source_stage(self):
        self._publish([{"a": 1}, {"a": 2}], 1)
        self._publish([{"a": 3}, {"a": 4}], 2)
        record = {"a": "bad", "extra": 1}
        report = self.lens.migrate(record, 1, 2, [{"op": "drop", "path": ["a"]}])
        self.assertEqual(report["migration"]["stage"], "source")
        self.assertIsNone(report["migration"]["record"])
        self.assertEqual(
            report["migration"]["reports"],
            self.lens.check(record, version=1),
        )
        self.assertTrue(report["migration"]["reports"])

    def test_target_failure_reports_target_stage(self):
        self._publish([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], 1)
        self._publish([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], 2)
        record = {"a": 1, "b": "x"}
        rules = [{"op": "drop", "path": ["a"]}]
        report = self.lens.migrate(record, 1, 2, rules)
        self.assertEqual(report["migration"]["stage"], "target")
        self.assertIsNone(report["migration"]["record"])
        self.assertEqual(
            report["migration"]["reports"],
            self.lens.check({"b": "x"}, version=2),
        )
        self.assertTrue(report["migration"]["reports"])

    def test_rules_apply_in_order(self):
        self._publish([{"a": 1, "b": 2}], 1)
        self._publish([{"a": 1, "c": 2, "d": 0}], 2)
        report = self.lens.migrate(
            {"a": 1, "b": 2}, 1, 2,
            [
                {"op": "rename", "path": ["b"], "to": "c"},
                {"op": "default", "path": ["d"], "value": 0},
            ],
        )
        self.assertEqual(
            report["migration"]["record"], {"a": 1, "c": 2, "d": 0}
        )

    def test_rename_keeps_field_position(self):
        self._publish([{"a": 1, "b": 2, "c": 3}], 1)
        self._publish([{"a": 1, "B": 2, "c": 3}], 2)
        report = self.lens.migrate(
            {"a": 1, "b": 2, "c": 3}, 1, 2,
            [{"op": "rename", "path": ["b"], "to": "B"}],
        )
        self.assertEqual(
            list(report["migration"]["record"]), ["a", "B", "c"]
        )

    def test_rename_missing_source_skips(self):
        self._publish([{"a": 1, "b": 2}, {"a": 3}], 1)
        self._publish([{"a": 1, "c": 2}, {"a": 3}], 2)
        report = self.lens.migrate(
            {"a": 1}, 1, 2, [{"op": "rename", "path": ["b"], "to": "c"}]
        )
        self.assertEqual(report["migration"]["record"], {"a": 1})

    def test_rename_onto_existing_field_raises(self):
        self._publish([{"a": 1, "b": 2}], 1)
        with self.assertRaises(ValueError):
            self.lens.migrate(
                {"a": 1, "b": 2}, 1, 1,
                [{"op": "rename", "path": ["a"], "to": "b"}],
            )

    def test_rename_to_same_name_raises(self):
        self._publish([{"a": 1}], 1)
        with self.assertRaises(ValueError):
            self.lens.migrate(
                {"a": 1}, 1, 1, [{"op": "rename", "path": ["a"], "to": "a"}]
            )

    def test_drop_missing_field_skips(self):
        self._publish([{"a": 1}], 1)
        report = self.lens.migrate({"a": 1}, 1, 1, [{"op": "drop", "path": ["b"]}])
        self.assertEqual(report["migration"]["record"], {"a": 1})

    def test_default_fills_missing_but_keeps_null(self):
        self._publish([{"a": 1, "b": "x"}, {"a": 2, "b": None}], 1)
        report = self.lens.migrate(
            {"a": 1, "b": None}, 1, 1,
            [{"op": "default", "path": ["b"], "value": "z"}],
        )
        self.assertEqual(report["migration"]["record"], {"a": 1, "b": None})

    def test_null_segment_walks_every_array_element(self):
        self._publish([{"items": [{"x": 1}, {"x": 2}]}], 1)
        self._publish([{"items": [{"y": 1}, {"y": 2}]}], 2)
        report = self.lens.migrate(
            {"items": [{"x": 1}, {"x": 2}]}, 1, 2,
            [{"op": "rename", "path": ["items", None, "x"], "to": "y"}],
        )
        self.assertEqual(
            report["migration"]["record"], {"items": [{"y": 1}, {"y": 2}]}
        )

    def test_dotted_name_is_one_literal_field(self):
        self._publish([{"items": [{"a.b": 1}]}], 1)
        self._publish([{"items": [{"c": 1}]}], 2)
        report = self.lens.migrate(
            {"items": [{"a.b": 1}]}, 1, 2,
            [{"op": "rename", "path": ["items", None, "a.b"], "to": "c"}],
        )
        self.assertEqual(report["migration"]["record"], {"items": [{"c": 1}]})

    def test_missing_parent_skips_without_creating(self):
        self._publish([{"a": 1}], 1)
        report = self.lens.migrate(
            {"a": 1}, 1, 1,
            [
                {"op": "default", "path": ["gone", "b"], "value": 1},
                {"op": "drop", "path": ["gone", "b"]},
            ],
        )
        self.assertEqual(report["migration"]["record"], {"a": 1})

    def test_string_segment_on_non_object_raises(self):
        self._publish([{"a": 1}, {"a": {"b": 2}}], 1)
        with self.assertRaises(ValueError):
            self.lens.migrate({"a": 1}, 1, 1, [{"op": "drop", "path": ["a", "b"]}])

    def test_null_segment_on_non_array_raises(self):
        self._publish([{"a": [1]}, {"a": {"b": 2}}], 1)
        with self.assertRaises(ValueError):
            self.lens.migrate(
                {"a": {"b": 2}}, 1, 1, [{"op": "drop", "path": ["a", None, "b"]}]
            )

    def test_null_segment_over_non_object_element_raises(self):
        self._publish([{"a": [1, 2]}], 1)
        with self.assertRaises(ValueError):
            self.lens.migrate({"a": [1]}, 1, 1, [{"op": "drop", "path": ["a", None, "b"]}])

    def test_empty_array_selects_nothing(self):
        self._publish([{"a": [{"b": 1}]}], 1)
        report = self.lens.migrate(
            {"a": []}, 1, 1, [{"op": "drop", "path": ["a", None, "b"]}]
        )
        self.assertEqual(report["migration"]["record"], {"a": []})

    def test_invalid_rule_structures_raise(self):
        self._publish([{"a": 1}], 1)
        bad_rules = [
            "not-a-rule",
            {"op": "move", "path": ["a"]},
            {"op": "drop"},
            {"op": "drop", "path": "a"},
            {"op": "drop", "path": []},
            {"op": "drop", "path": [None]},
            {"op": "drop", "path": [1]},
            {"op": "drop", "path": ["a"], "to": "b"},
            {"op": "rename", "path": ["a"]},
            {"op": "rename", "path": ["a"], "to": 1},
            {"op": "default", "path": ["a"]},
            {"op": "default", "path": ["a"], "value": object()},
        ]
        for rules in bad_rules:
            with self.subTest(rules=rules):
                with self.assertRaises(ValueError):
                    self.lens.migrate({"a": 1}, 1, 1, [rules])
        with self.assertRaises(ValueError):
            self.lens.migrate({"a": 1}, 1, 1, {"op": "drop"})

    def test_non_object_record_and_non_json_values_raise(self):
        self._publish([{"a": 1}], 1)
        for record in ([1], "x", 1, None, {"a": {1, 2}}, {"a": float("nan")}):
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    self.lens.migrate(record, 1, 1, [])

    def test_invalid_and_missing_revisions_raise_conflict(self):
        self._publish([{"a": 1}], 1)
        for old, new in ((0, 1), (1, -1), (True, 1), ("1", 1), (1, 2), (2, 1)):
            with self.subTest(old=old, new=new):
                with self.assertRaises(SchemaConflict):
                    self.lens.migrate({"a": 1}, old, new, [])

    def test_corrupt_revision_raises_conflict(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2}], 2)
        revision_path(self.dir.name, 2).write_text("not json", encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            self.lens.migrate({"a": 1}, 1, 2, [])

    def test_compacted_revision_raises_but_anchor_reads(self):
        lens = _build_history(self.dir.name, 5, compact_keep=2)
        lens.compact()
        self.assertEqual(lens.versions(), [3, 4, 5])
        report = lens.migrate({"f2": 2}, 3, 4, [])
        self.assertEqual(report["migration"]["stage"], "done")
        with self.assertRaises(SchemaConflict):
            lens.migrate({}, 1, 4, [])

    def test_same_revision_applies_rules_and_validates(self):
        self._publish([{"a": 1, "b": 2}], 1)
        report = self.lens.migrate(
            {"a": 1, "b": 2}, 1, 1, [{"op": "rename", "path": ["b"], "to": "c"}]
        )
        self.assertEqual(report["backward"], "compatible")
        self.assertEqual(report["forward"], "compatible")
        self.assertEqual(report["changes"], [])
        self.assertEqual(report["migration"]["stage"], "target")
        self.assertEqual(
            report["migration"]["reports"],
            self.lens.check({"a": 1, "c": 2}, version=1),
        )

    def test_repeated_migration_is_stable(self):
        self._publish([{"a": 1, "b": "x"}], 1)
        self._publish([{"a": 1, "c": "x"}], 2)
        rules = [{"op": "rename", "path": ["b"], "to": "c"}]
        first = self.lens.migrate({"a": 1, "b": "x"}, 1, 2, rules)
        for _ in range(3):
            again = self.lens.migrate({"a": 1, "b": "x"}, 1, 2, rules)
            self.assertEqual(
                json.dumps(first), json.dumps(again)
            )

    def test_inputs_and_state_are_untouched(self):
        self._publish([{"a": 1, "b": "x"}], 1)
        self._publish([{"a": 1, "c": "x"}], 2)
        self.lens.infer([{"pending": 1}])
        snapshot = self.lens.schema()
        record = {"a": 1, "b": "x"}
        rules = [
            {"op": "rename", "path": ["b"], "to": "c"},
            {"op": "default", "path": ["d"], "value": ["e"]},
        ]
        self.lens.migrate(record, 1, 2, rules)
        self.assertEqual(record, {"a": 1, "b": "x"})
        self.assertEqual(
            rules,
            [
                {"op": "rename", "path": ["b"], "to": "c"},
                {"op": "default", "path": ["d"], "value": ["e"]},
            ],
        )
        self.assertEqual(self.lens.schema(), snapshot)
        self.assertEqual(self.lens.versions(), [1, 2])
        self.assertEqual(self.lens.check({"pending": 1}), [])

    def test_default_value_is_copied_per_element(self):
        self._publish(
            [{"items": [{"x": 1, "tags": ["t"]}, {"x": 2}]}], 1
        )
        rules = [{"op": "default", "path": ["items", None, "tags"], "value": ["t"]}]
        report = self.lens.migrate({"items": [{"x": 1}, {"x": 2}]}, 1, 1, rules)
        record = report["migration"]["record"]
        self.assertEqual(
            record, {"items": [{"x": 1, "tags": ["t"]}, {"x": 2, "tags": ["t"]}]}
        )
        record["items"][0]["tags"].append("u")
        self.assertEqual(rules[0]["value"], ["t"])
        self.assertEqual(record["items"][1]["tags"], ["t"])


class MigrateRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)

    def _publish(self, records, revision, **limits):
        return _publish(self.dir.name, records, revision, **limits)

    def test_report_is_full_migrate_report_plus_rollback(self):
        self._publish([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], 1)
        self._publish([{"a": 1, "c": "x"}, {"a": 2, "c": "y"}], 2)
        rules = [{"op": "rename", "path": ["b"], "to": "c"}]
        back = [{"op": "rename", "path": ["c"], "to": "b"}]
        report = self.lens.migrate_roundtrip(
            {"a": 1, "b": "x"}, 1, 2, rules, back
        )
        self.assertEqual(
            set(report),
            {"from", "to", "backward", "forward", "changes",
             "unknown_reasons", "migration", "rollback"},
        )
        forward = self.lens.migrate({"a": 1, "b": "x"}, 1, 2, rules)
        for key in ("from", "to", "backward", "forward", "changes",
                    "unknown_reasons", "migration"):
            self.assertEqual(report[key], forward[key])
        self.assertEqual(
            report["rollback"],
            {"stage": "done", "record": {"a": 1, "b": "x"},
             "reports": [], "differences": []},
        )

    def test_successful_roundtrip_restores_record(self):
        self._publish([{"items": [{"x": 1}, {"x": 2}]}], 1)
        self._publish([{"items": [{"y": 1}, {"y": 2}]}], 2)
        report = self.lens.migrate_roundtrip(
            {"items": [{"x": 1}, {"x": 2}]}, 1, 2,
            [{"op": "rename", "path": ["items", None, "x"], "to": "y"}],
            [{"op": "rename", "path": ["items", None, "y"], "to": "x"}],
        )
        self.assertEqual(report["migration"]["stage"], "done")
        self.assertEqual(
            report["rollback"]["record"], {"items": [{"x": 1}, {"x": 2}]}
        )
        self.assertEqual(report["rollback"]["stage"], "done")
        self.assertEqual(report["rollback"]["differences"], [])

    def test_key_order_ignored_in_recovery_comparison(self):
        # rev1 accepts both keys; the roundtrip renames move a key to a new
        # position, so the recovered object differs only in key order.
        self._publish([{"a": 1, "b": 2}], 1)
        self._publish([{"a": 1, "c": 2}], 2)
        report = self.lens.migrate_roundtrip(
            {"b": 2, "a": 1}, 1, 2,
            [{"op": "rename", "path": ["b"], "to": "c"}],
            [{"op": "rename", "path": ["c"], "to": "b"}],
        )
        # The recovered object keeps the rename's position rather than
        # being reordered, but comparison ignores key order.
        self.assertEqual(list(report["rollback"]["record"]), ["b", "a"])
        self.assertEqual(report["rollback"]["stage"], "done")
        self.assertEqual(report["rollback"]["differences"], [])

    def test_empty_rules_and_same_revision_roundtrip(self):
        self._publish([{"a": 1}, {"a": 2}], 1)
        report = self.lens.migrate_roundtrip({"a": 1}, 1, 1, [], [])
        self.assertEqual(
            report["migration"],
            {"stage": "done", "record": {"a": 1}, "reports": []},
        )
        self.assertEqual(
            report["rollback"],
            {"stage": "done", "record": {"a": 1},
             "reports": [], "differences": []},
        )

    def test_same_revision_applies_both_rule_groups(self):
        # Revision 1 requires b and rejects an unexpected c: renaming b to c
        # fails the forward target check, so no rollback rules ever run.
        self._publish([{"a": 1, "b": 2}], 1)
        report = self.lens.migrate_roundtrip(
            {"a": 1, "b": 2}, 1, 1,
            [{"op": "rename", "path": ["b"], "to": "c"}],
            [{"op": "rename", "path": ["c"], "to": "b"}],
        )
        self.assertEqual(report["migration"]["stage"], "target")
        self.assertEqual(
            report["migration"]["reports"],
            self.lens.check({"a": 1, "c": 2}, version=1),
        )
        self.assertIsNone(report["rollback"])

    def test_forward_source_failure_keeps_source_result_and_null_rollback(self):
        self._publish([{"a": 1}, {"a": 2}], 1)
        self._publish([{"a": 3}, {"a": 4}], 2)
        record = {"a": "bad", "extra": 1}
        report = self.lens.migrate_roundtrip(
            record, 1, 2, [{"op": "drop", "path": ["a"]}], []
        )
        self.assertEqual(
            report["migration"],
            self.lens.migrate(record, 1, 2, [{"op": "drop", "path": ["a"]}])[
                "migration"
            ],
        )
        self.assertEqual(report["migration"]["stage"], "source")
        self.assertIsNone(report["rollback"])

    def test_forward_target_failure_keeps_target_result_and_null_rollback(self):
        self._publish([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], 1)
        record = {"a": 1, "b": "x"}
        rules = [{"op": "drop", "path": ["a"]}]
        report = self.lens.migrate_roundtrip(record, 1, 1, rules, [])
        self.assertEqual(report["migration"]["stage"], "target")
        self.assertIsNone(report["migration"]["record"])
        self.assertIsNone(report["rollback"])

    def test_invalid_rollback_rule_raises_even_when_source_check_fails(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2}], 2)
        with self.assertRaises(ValueError):
            self.lens.migrate_roundtrip(
                {"a": "bad"}, 1, 2,
                [], [{"op": "drop", "path": "a"}],
            )

    def test_invalid_rollback_rule_raises_even_when_target_check_fails(self):
        self._publish([{"a": 1, "b": "x"}], 1)
        with self.assertRaises(ValueError):
            self.lens.migrate_roundtrip(
                {"a": 1, "b": "x"}, 1, 1,
                [{"op": "drop", "path": ["a"]}],
                [{"op": "move", "path": ["a"]}],
            )

    def test_invalid_rollback_rule_structures_raise(self):
        self._publish([{"a": 1}], 1)
        bad_groups = [
            [{"op": "drop"}],
            [{"op": "drop", "path": []}],
            [{"op": "drop", "path": [None]}],
            "not-a-list",
            [{"op": "rename", "path": ["a"]}],
            [{"op": "rename", "path": ["a"], "to": "a"}],
            [{"op": "default", "path": ["a"]}],
            [{"op": "default", "path": ["a"], "value": float("inf")}],
        ]
        for rollback_rules in bad_groups:
            with self.subTest(rollback_rules=rollback_rules):
                with self.assertRaises(ValueError):
                    self.lens.migrate_roundtrip(
                        {"a": 1}, 1, 1, [], rollback_rules
                    )

    def test_rollback_runtime_path_error_raises_after_forward_done(self):
        self._publish([{"a": [1]}, {"a": {"b": 2}}], 1)
        with self.assertRaises(ValueError):
            self.lens.migrate_roundtrip(
                {"a": {"b": 2}}, 1, 1,
                [],
                [{"op": "drop", "path": ["a", None, "b"]}],
            )

    def test_rollback_rename_conflict_raises(self):
        self._publish([{"a": 1, "b": 2}], 1)
        with self.assertRaises(ValueError):
            self.lens.migrate_roundtrip(
                {"a": 1, "b": 2}, 1, 1,
                [],
                [{"op": "rename", "path": ["a"], "to": "b"}],
            )

    def test_rollback_target_failure_reports_target_stage(self):
        self._publish([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], 1)
        self._publish([{"a": 1, "c": "x"}, {"a": 2, "c": "y"}], 2)
        report = self.lens.migrate_roundtrip(
            {"a": 1, "b": "x"}, 1, 2,
            [{"op": "rename", "path": ["b"], "to": "c"}],
            [
                {"op": "drop", "path": ["c"]},
                {"op": "default", "path": ["b"], "value": 1},
            ],
        )
        rollback = report["rollback"]
        self.assertEqual(rollback["stage"], "target")
        self.assertIsNone(rollback["record"])
        self.assertEqual(
            rollback["reports"],
            self.lens.check({"a": 1, "b": 1}, version=1),
        )
        self.assertTrue(rollback["reports"])
        self.assertEqual(rollback["differences"], [])

    def _difference_roundtrip(self, original, recovered_rules):
        """Forward no-op self-migration; rollback rules shape the recovery."""
        return self.lens.migrate_roundtrip(
            original, 1, 1, [], recovered_rules
        )["rollback"]

    def test_different_missing_key_names_field_path(self):
        self._publish([{"a": 1, "b": "x"}, {"a": 2}], 1)
        rollback = self._difference_roundtrip(
            {"a": 1, "b": "x"}, [{"op": "drop", "path": ["b"]}]
        )
        self.assertEqual(rollback["stage"], "different")
        self.assertEqual(rollback["record"], {"a": 1})
        self.assertEqual(rollback["reports"], [])
        self.assertEqual(rollback["differences"], ["$.b"])

    def test_missing_distinguished_from_null_in_both_directions(self):
        self._publish([{"b": None}, {}], 1)
        present_null = self._difference_roundtrip(
            {"b": None}, [{"op": "drop", "path": ["b"]}]
        )
        self.assertEqual(present_null["differences"], ["$.b"])
        absent = self._difference_roundtrip(
            {}, [{"op": "default", "path": ["b"], "value": None}]
        )
        self.assertEqual(absent["differences"], ["$.b"])

    def test_boolean_distinguished_from_number(self):
        self._publish([{"a": True}, {"a": False}, {"a": 1}, {}], 1)
        rollback = self._difference_roundtrip(
            {"a": True},
            [{"op": "drop", "path": ["a"]},
             {"op": "default", "path": ["a"], "value": 1}],
        )
        self.assertEqual(rollback["differences"], ["$.a"])

    def test_ints_and_floats_compare_numerically(self):
        self._publish([{"a": 1}, {"a": 2.5}, {"a": 1.5}, {}], 1)
        equal = self._difference_roundtrip(
            {"a": 1},
            [{"op": "drop", "path": ["a"]},
             {"op": "default", "path": ["a"], "value": 1.0}],
        )
        self.assertEqual(equal["stage"], "done")
        self.assertEqual(equal["differences"], [])
        unequal = self._difference_roundtrip(
            {"a": 1},
            [{"op": "drop", "path": ["a"]},
             {"op": "default", "path": ["a"], "value": 1.5}],
        )
        self.assertEqual(unequal["stage"], "different")
        self.assertEqual(unequal["differences"], ["$.a"])

    def test_equal_length_arrays_compare_by_index(self):
        self._publish([{"a": [1, 2]}, {}], 1)
        rollback = self._difference_roundtrip(
            {"a": [1, 2]},
            [{"op": "drop", "path": ["a"]},
             {"op": "default", "path": ["a"], "value": [1, 3]}],
        )
        self.assertEqual(rollback["differences"], ["$.a[1]"])

    def test_unequal_array_length_names_only_array_path(self):
        self._publish([{"a": [1]}, {"a": [1, 2]}, {}], 1)
        rollback = self._difference_roundtrip(
            {"a": [1, 2]},
            [{"op": "drop", "path": ["a"]},
             {"op": "default", "path": ["a"], "value": [1]}],
        )
        self.assertEqual(rollback["differences"], ["$.a"])

    def test_type_mismatch_does_not_expand(self):
        self._publish([{"a": {"b": 1}}, {"a": 2}, {"a": {}}, {}], 1)
        rollback = self._difference_roundtrip(
            {"a": {"b": 1}},
            [{"op": "drop", "path": ["a"]},
             {"op": "default", "path": ["a"], "value": 2}],
        )
        self.assertEqual(rollback["differences"], ["$.a"])

    def test_nested_object_differences_recurse_key_by_key(self):
        self._publish([{"a": {"b": 1, "c": 2}}, {"a": {}}], 1)
        rollback = self._difference_roundtrip(
            {"a": {"b": 1, "c": 2}},
            [{"op": "drop", "path": ["a", "c"]}],
        )
        self.assertEqual(rollback["differences"], ["$.a.c"])

    def test_dotted_key_uses_quoted_path_and_differences_are_sorted(self):
        records = [
            {"a.b": 1, "c": 2, "d": 3},
            {},
            {"a.b": 1},
            {"c": 2},
            {"d": 3},
        ]
        self._publish(records, 1)
        rollback = self._difference_roundtrip(
            {"a.b": 1, "c": 2, "d": 3},
            [
                {"op": "drop", "path": ["d"]},
                {"op": "drop", "path": ["c"]},
                {"op": "drop", "path": ["a.b"]},
            ],
        )
        self.assertEqual(
            rollback["differences"], ["$.c", "$.d", '$["a.b"]']
        )

    def test_approximate_check_passes_but_comparison_covers_real_data(self):
        # A depth-capped old revision cannot inspect inside ``a``, so the
        # recovered object passes check even though a nested value changed;
        # the recovery comparison must still report the real difference.
        self._publish([{"a": {"b": 1}}], 1, max_depth=0)
        self._publish([{"a": {"c": 1}}], 2, max_depth=0)
        report = self.lens.migrate_roundtrip(
            {"a": {"b": 1}}, 1, 2,
            [{"op": "rename", "path": ["a", "b"], "to": "c"}],
            [
                {"op": "drop", "path": ["a", "c"]},
                {"op": "default", "path": ["a", "b"], "value": 2},
            ],
        )
        self.assertEqual(report["migration"]["stage"], "done")
        rollback = report["rollback"]
        self.assertEqual(rollback["stage"], "different")
        self.assertEqual(rollback["reports"], [])
        self.assertEqual(rollback["record"], {"a": {"b": 2}})
        self.assertEqual(rollback["differences"], ["$.a.b"])

    def test_non_object_record_and_non_json_values_raise(self):
        self._publish([{"a": 1}], 1)
        for record in ([1], "x", 1, None, {"a": {1, 2}}, {"a": float("nan")}):
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    self.lens.migrate_roundtrip(record, 1, 1, [], [])

    def test_invalid_and_missing_revisions_raise_conflict(self):
        self._publish([{"a": 1}], 1)
        for old, new in ((0, 1), (1, -1), (True, 1), ("1", 1), (1, 2)):
            with self.subTest(old=old, new=new):
                with self.assertRaises(SchemaConflict):
                    self.lens.migrate_roundtrip({"a": 1}, old, new, [], [])

    def test_corrupt_and_compacted_revisions_raise_conflict(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2}], 2)
        revision_path(self.dir.name, 2).write_text("not json", encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            self.lens.migrate_roundtrip({"a": 1}, 1, 2, [], [])
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        lens = _build_history(other.name, 5, compact_keep=2)
        lens.compact()
        with self.assertRaises(SchemaConflict):
            lens.migrate_roundtrip({}, 1, 4, [], [])
        report = lens.migrate_roundtrip({"f2": 2, "shared": "s"}, 3, 4, [], [])
        self.assertEqual(report["migration"]["stage"], "done")
        self.assertEqual(report["rollback"]["stage"], "done")

    def test_repeated_roundtrip_is_stable(self):
        self._publish([{"a": 1, "b": "x"}], 1)
        self._publish([{"a": 1, "c": "x"}], 2)
        rules = [{"op": "rename", "path": ["b"], "to": "c"}]
        back = [{"op": "drop", "path": ["c"]}]
        first = self.lens.migrate_roundtrip(
            {"a": 1, "b": "x"}, 1, 2, rules, back
        )
        encoded = json.dumps(first)
        for _ in range(3):
            self.assertEqual(
                encoded,
                json.dumps(
                    self.lens.migrate_roundtrip(
                        {"a": 1, "b": "x"}, 1, 2, rules, back
                    )
                ),
            )

    def test_inputs_and_state_are_untouched(self):
        self._publish([{"a": 1, "b": "x"}], 1)
        self._publish([{"a": 1, "c": "x"}], 2)
        self.lens.infer([{"pending": 1}])
        snapshot = self.lens.schema()
        versions_dir = Path(self.dir.name, VERSIONS_DIRNAME)
        before = sorted(os.listdir(versions_dir))
        record = {"a": 1, "b": "x"}
        rules = [{"op": "rename", "path": ["b"], "to": "c"}]
        back = [
            {"op": "rename", "path": ["c"], "to": "b"},
            {"op": "default", "path": ["d"], "value": ["e"]},
        ]
        self.lens.migrate_roundtrip(record, 1, 2, rules, back)
        self.assertEqual(record, {"a": 1, "b": "x"})
        self.assertEqual(rules, [{"op": "rename", "path": ["b"], "to": "c"}])
        self.assertEqual(
            back,
            [
                {"op": "rename", "path": ["c"], "to": "b"},
                {"op": "default", "path": ["d"], "value": ["e"]},
            ],
        )
        self.assertEqual(self.lens.schema(), snapshot)
        self.assertEqual(self.lens.versions(), [1, 2])
        self.assertEqual(self.lens.check({"pending": 1}), [])
        self.assertEqual(sorted(os.listdir(versions_dir)), before)

    def test_returned_records_are_independent_copies(self):
        self._publish(
            [{"items": [{"x": 1, "tags": ["t"]}, {"x": 2}]}], 1
        )
        self._publish(
            [{"items": [{"y": 1, "tags": ["t"]}, {"y": 2}]}], 2
        )
        rules = [{"op": "rename", "path": ["items", None, "x"], "to": "y"}]
        back = [
            {"op": "rename", "path": ["items", None, "y"], "to": "x"},
            {"op": "default", "path": ["items", None, "tags"], "value": ["t"]},
        ]
        first = self.lens.migrate_roundtrip(
            {"items": [{"x": 1}, {"x": 2}]}, 1, 2, rules, back
        )
        record = first["rollback"]["record"]
        self.assertEqual(
            record,
            {"items": [{"x": 1, "tags": ["t"]}, {"x": 2, "tags": ["t"]}]},
        )
        record["items"][0]["tags"].append("u")
        again = self.lens.migrate_roundtrip(
            {"items": [{"x": 1}, {"x": 2}]}, 1, 2, rules, back
        )
        self.assertEqual(
            again["rollback"]["record"]["items"][0]["tags"], ["t"]
        )
        self.assertEqual(back[1]["value"], ["t"])


class MigrateStreamTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)

    def _publish(self, records, revision, **limits):
        return _publish(self.dir.name, records, revision, **limits)

    def _rename_pair(self):
        self._publish([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], 1)
        self._publish([{"a": 1, "c": "x"}, {"a": 2, "c": "y"}], 2)
        return (
            [{"op": "rename", "path": ["b"], "to": "c"}],
            [{"op": "rename", "path": ["c"], "to": "b"}],
        )

    # -- basic shape and equivalence with the single-record entries ------

    def test_results_have_exactly_index_report_error(self):
        rules, _back = self._rename_pair()
        results = list(
            self.lens.migrate_stream(
                [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], 1, 2, rules
            )
        )
        self.assertEqual([r["index"] for r in results], [1, 2])
        for result in results:
            self.assertEqual(set(result), {"index", "report", "error"})
            self.assertIsNone(result["error"])

    def test_indexes_run_continuously_from_one_through_errors(self):
        self._publish([{"a": 1}], 1)
        records = [{"a": 1}, "not-an-object", {"a": 2}, [1], {"a": 3}]
        results = list(self.lens.migrate_stream(records, 1, 1, []))
        self.assertEqual([r["index"] for r in results], [1, 2, 3, 4, 5])
        self.assertIsNone(results[0]["error"])
        self.assertEqual(
            results[1], {"index": 2, "report": None,
                         "error": "record must be a JSON object"}
        )
        self.assertIsNone(results[2]["error"])
        self.assertIsNone(results[3]["report"])
        self.assertIsNotNone(results[3]["error"])
        self.assertIsNone(results[4]["error"])

    def test_without_rollback_rules_report_is_the_migrate_report(self):
        rules, _back = self._rename_pair()
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        results = list(self.lens.migrate_stream(records, 1, 2, rules))
        for index, record in enumerate(records, 1):
            self.assertEqual(
                results[index - 1]["report"],
                self.lens.migrate(record, 1, 2, rules),
            )
            self.assertNotIn("rollback", results[index - 1]["report"])

    def test_rollback_list_gives_roundtrip_report_empty_list_enabled(self):
        rules, back = self._rename_pair()
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        with_rules = list(
            self.lens.migrate_stream(records, 1, 2, rules, back)
        )
        empty = list(self.lens.migrate_stream(records, 1, 2, rules, []))
        for index, record in enumerate(records, 1):
            self.assertEqual(
                with_rules[index - 1]["report"],
                self.lens.migrate_roundtrip(record, 1, 2, rules, back),
            )
            self.assertIn("rollback", with_rules[index - 1]["report"])
            # An empty rollback array still enables the rollback object.
            self.assertEqual(
                empty[index - 1]["report"],
                self.lens.migrate_roundtrip(record, 1, 2, rules, []),
            )
            self.assertIn("rollback", empty[index - 1]["report"])

    def test_source_and_target_failures_keep_stage_and_full_reports(self):
        self._publish([{"a": 1}, {"a": 2}], 1)
        self._publish([{"a": 3}, {"a": 4}], 2)
        records = [{"a": "bad", "extra": 1}, {"a": 1}]
        results = list(
            self.lens.migrate_stream(
                records, 1, 2, [{"op": "drop", "path": ["a"]}]
            )
        )
        self.assertEqual(results[0]["report"]["migration"]["stage"], "source")
        self.assertEqual(
            results[0]["report"]["migration"]["reports"],
            self.lens.check({"a": "bad", "extra": 1}, version=1),
        )
        self.assertIsNone(results[0]["report"]["migration"]["record"])
        # Dropping the required a fails the target check on revision 2.
        self.assertEqual(results[1]["report"]["migration"]["stage"], "target")
        self.assertTrue(results[1]["report"]["migration"]["reports"])

    def test_rules_keep_rename_drop_default_and_wildcard_semantics(self):
        self._publish([{"items": [{"x": 1}, {"x": 2}]}], 1)
        self._publish([{"items": [{"y": 1, "t": "z"}, {"y": 2, "t": "z"}]}], 2)
        rules = [
            {"op": "rename", "path": ["items", None, "x"], "to": "y"},
            {"op": "default", "path": ["items", None, "t"], "value": "z"},
        ]
        results = list(
            self.lens.migrate_stream([{"items": [{"x": 1}, {"x": 2}]}], 1, 2, rules)
        )
        self.assertEqual(
            results[0]["report"]["migration"]["record"],
            {"items": [{"y": 1, "t": "z"}, {"y": 2, "t": "z"}]},
        )

    def test_same_revision_still_applies_rules(self):
        self._publish([{"a": 1, "b": 2}], 1)
        results = list(
            self.lens.migrate_stream(
                [{"a": 1, "b": 2}], 1, 1,
                [{"op": "rename", "path": ["b"], "to": "c"}],
            )
        )
        report = results[0]["report"]
        self.assertEqual(report["migration"]["stage"], "target")
        self.assertEqual(
            report["migration"]["reports"],
            self.lens.check({"a": 1, "c": 2}, version=1),
        )

    def test_approximate_branch_and_missing_vs_null_survive(self):
        self._publish([{"a": {"b": 1}}], 1, max_depth=0)
        self._publish([{"a": {"c": 1}}], 2, max_depth=0)
        rules = [{"op": "rename", "path": ["a", "b"], "to": "c"}]
        back = [
            {"op": "drop", "path": ["a", "c"]},
            {"op": "default", "path": ["a", "b"], "value": 2},
        ]
        results = list(
            self.lens.migrate_stream(
                [{"a": {"b": 1}}], 1, 2, rules, back
            )
        )
        rollback = results[0]["report"]["rollback"]
        self.assertEqual(rollback["stage"], "different")
        self.assertEqual(rollback["differences"], ["$.a.b"])

    # -- per-record error isolation --------------------------------------

    def test_non_json_records_become_errors_and_stream_continues(self):
        self._publish([{"a": 1}], 1)
        records = [
            {"a": 1},
            {"a": {1, 2}},
            {"a": float("nan")},
            {"a": float("inf")},
            None,
            1,
            "x",
            {"a": 2},
        ]
        results = list(self.lens.migrate_stream(records, 1, 1, []))
        statuses = [(r["report"] is None, r["error"] is None) for r in results]
        self.assertEqual(
            statuses,
            [(False, True), (True, False), (True, False), (True, False),
             (True, False), (True, False), (True, False), (False, True)],
        )

    def test_forward_rule_value_error_becomes_per_record_error(self):
        self._publish([{"a": [1]}, {"a": {"b": 2}}], 1)
        # Every record passes the source check (a is array or object); the
        # drop of a.b only errors when a is reached as a non-object.
        records = [{"a": [1]}, {"a": {"b": 2}}, {"a": [1]}]
        results = list(
            self.lens.migrate_stream(
                records, 1, 1, [{"op": "drop", "path": ["a", "b"]}]
            )
        )
        self.assertIsNone(results[0]["report"])
        self.assertEqual(
            results[0]["error"], "string path segment requires an object"
        )
        self.assertIsNone(results[1]["error"])
        self.assertEqual(results[1]["report"]["migration"]["stage"], "target")
        self.assertIsNone(results[2]["report"])
        self.assertIsNotNone(results[2]["error"])

    def test_rollback_rule_value_error_becomes_per_record_error(self):
        self._publish([{"a": [1]}, {"a": {"b": 2}}], 1)
        results = list(
            self.lens.migrate_stream(
                [{"a": {"b": 2}}, {"a": {"b": 2}}], 1, 1, [],
                [{"op": "drop", "path": ["a", None, "b"]}],
            )
        )
        for result in results:
            self.assertIsNone(result["report"])
            self.assertEqual(
                result["error"], "null path segment requires an array"
            )

    def test_rollback_rename_conflict_is_a_per_record_error(self):
        self._publish([{"a": 1, "b": 2}], 1)
        results = list(
            self.lens.migrate_stream(
                [{"a": 1, "b": 2}, {"a": 1, "b": 2}], 1, 1, [],
                [{"op": "rename", "path": ["a"], "to": "b"}],
            )
        )
        self.assertTrue(all(r["report"] is None for r in results))
        self.assertIn("already exists", results[0]["error"])

    # -- eager validation, no consumption --------------------------------

    def _tracking_source(self, records):
        source = {"started": False, "records": records}

        def generate():
            source["started"] = True
            yield from records

        source["iterator_factory"] = generate
        return source

    def test_call_does_not_consume_records(self):
        rules, _back = self._rename_pair()
        source = self._tracking_source([{"a": 1, "b": "x"}])
        results = self.lens.migrate_stream(source["iterator_factory"](), 1, 2, rules)
        self.assertFalse(source["started"])
        first = next(results)
        self.assertTrue(source["started"])
        self.assertIsNone(first["error"])

    def test_one_record_consumed_per_pull_without_prefetch(self):
        self._publish([{"a": 1}], 1)
        pulled = []

        def generate():
            for record in [{"a": 1}, {"a": 2}, {"a": 3}]:
                pulled.append(record)
                yield record

        results = self.lens.migrate_stream(generate(), 1, 1, [])
        self.assertEqual(pulled, [])
        next(results)
        self.assertEqual(len(pulled), 1)
        next(results)
        self.assertEqual(len(pulled), 2)
        next(results)
        self.assertEqual(len(pulled), 3)
        with self.assertRaises(StopIteration):
            next(results)
        self.assertEqual(len(pulled), 3)

    def test_one_shot_iterator_is_accepted(self):
        self._publish([{"a": 1}], 1)
        results = self.lens.migrate_stream(iter([{"a": 1}, {"a": 2}]), 1, 1, [])
        self.assertEqual([next(results)["index"] for _ in range(2)], [1, 2])

    def test_invalid_rules_raise_value_error_without_consuming(self):
        self._publish([{"a": 1}], 1)
        source = self._tracking_source([{"a": 1}])
        with self.assertRaises(ValueError):
            self.lens.migrate_stream(
                source["iterator_factory"](), 1, 1, [{"op": "move"}]
            )
        self.assertFalse(source["started"])

    def test_invalid_rollback_rules_raise_value_error_without_consuming(self):
        self._publish([{"a": 1}], 1)
        source = self._tracking_source([{"a": 1}])
        with self.assertRaises(ValueError):
            self.lens.migrate_stream(
                source["iterator_factory"](), 1, 1, [],
                [{"op": "drop", "path": "a"}],
            )
        self.assertFalse(source["started"])

    def test_non_iterable_records_raise_value_error(self):
        self._publish([{"a": 1}], 1)
        for records in (7, 3.5, object()):
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    self.lens.migrate_stream(records, 1, 1, [])

    def test_rules_are_validated_before_revisions(self):
        # Illegal rules win over the missing revision: ValueError, and no
        # record is consumed.
        source = self._tracking_source([{"a": 1}])
        with self.assertRaises(ValueError):
            self.lens.migrate_stream(
                source["iterator_factory"](), 1, 2, [{"op": "nope"}]
            )
        self.assertFalse(source["started"])

    def test_bad_revisions_raise_conflict_without_consuming(self):
        self._publish([{"a": 1}], 1)
        for old, new in ((0, 1), (1, -1), (True, 1), ("1", 1), (1, 2), (2, 1)):
            source = self._tracking_source([{"a": 1}])
            with self.subTest(old=old, new=new):
                with self.assertRaises(SchemaConflict):
                    self.lens.migrate_stream(
                        source["iterator_factory"](), old, new, []
                    )
                self.assertFalse(source["started"])

    def test_corrupt_revision_raises_conflict_without_consuming(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 2}], 2)
        revision_path(self.dir.name, 2).write_text("not json", encoding="utf-8")
        source = self._tracking_source([{"a": 1}])
        with self.assertRaises(SchemaConflict):
            self.lens.migrate_stream(source["iterator_factory"](), 1, 2, [])
        self.assertFalse(source["started"])

    def test_empty_input_runs_all_checks(self):
        self._publish([{"a": 1}], 1)
        self.assertEqual(list(self.lens.migrate_stream([], 1, 1, [])), [])
        self.assertEqual(list(self.lens.migrate_stream(iter([]), 1, 1, [], [])), [])
        with self.assertRaises(ValueError):
            self.lens.migrate_stream([], 1, 1, [{"op": "bad"}])
        with self.assertRaises(SchemaConflict):
            self.lens.migrate_stream([], 9, 9, [])
        with self.assertRaises(ValueError):
            self.lens.migrate_stream(42, 1, 1, [])

    # -- iterator failures -----------------------------------------------

    def test_iterator_non_stopiteration_exception_propagates_verbatim(self):
        self._publish([{"a": 1}], 1)

        def generate():
            yield {"a": 1}
            raise RuntimeError("stream broke")
            yield {"a": 2}  # pragma: no cover - unreachable

        results = self.lens.migrate_stream(generate(), 1, 1, [])
        first = next(results)
        self.assertEqual(first["index"], 1)
        with self.assertRaisesRegex(RuntimeError, "stream broke"):
            next(results)

    def test_iterator_value_error_is_not_turned_into_a_result(self):
        self._publish([{"a": 1}], 1)

        def generate():
            yield {"a": 1}
            raise ValueError("source problem")

        results = self.lens.migrate_stream(generate(), 1, 1, [])
        first = next(results)
        self.assertIsNone(first["error"])
        with self.assertRaisesRegex(ValueError, "source problem"):
            next(results)

    # -- freezing, independence, streaming memory ------------------------

    def test_snapshots_survive_compaction_that_removes_their_revisions(self):
        lens = _build_history(self.dir.name, 5, compact_keep=2)
        other = Lens(self.dir.name)
        records = [{"f0": 0, "shared": "s"}, {"f2": 2, "shared": "s"}]
        expected = [other.migrate(record, 1, 4, []) for record in records]
        stream = lens.migrate_stream(iter(records), 1, 4, [])
        first = next(stream)
        self.assertEqual(first["report"], expected[0])
        # Revision 1 is merged into the baseline while the stream is open.
        self.assertEqual(lens.compact(), {"anchor": 3, "merged": [1, 2, 3]})
        with self.assertRaises(SchemaConflict):
            lens.migrate(records[0], 1, 4, [])
        rest = list(stream)
        self.assertEqual(len(rest), 1)
        self.assertEqual(rest[0]["report"], expected[1])
        # The anchor revision itself keeps answering the frozen pair too.
        anchor_stream = list(lens.migrate_stream([{"f2": 2}], 3, 4, []))
        self.assertEqual(anchor_stream[0]["report"]["migration"]["stage"], "done")

    def test_caller_mutating_rules_after_call_does_not_change_batch(self):
        rules, _back = self._rename_pair()
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        expected = [
            self.lens.migrate(record, 1, 2, rules) for record in records
        ]
        stream = self.lens.migrate_stream(iter(records), 1, 2, rules)
        first = next(stream)
        rules.append({"op": "drop", "path": ["a"]})
        rules[0]["to"] = "zzz"
        self.assertEqual(first["report"], expected[0])
        self.assertEqual(next(stream)["report"], expected[1])

    def test_input_records_are_not_modified(self):
        rules, back = self._rename_pair()
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        originals = copy.deepcopy(records)
        list(self.lens.migrate_stream(records, 1, 2, rules, back))
        self.assertEqual(records, originals)

    def test_returned_records_are_independent_of_inputs_and_each_other(self):
        self._publish(
            [{"items": [{"x": 1, "tags": ["t"]}, {"x": 2}]}], 1
        )
        self._publish(
            [{"items": [{"x": 1, "tags": ["t"]}, {"x": 2}]}], 2
        )
        rules = [{"op": "default", "path": ["items", None, "tags"],
                  "value": ["t"]}]
        inputs = [{"items": [{"x": 1}, {"x": 2}]},
                  {"items": [{"x": 9}, {"x": 8}]}]
        stream = self.lens.migrate_stream(iter(inputs), 1, 2, rules)
        first = next(stream)["report"]["migration"]["record"]
        first["items"][0]["tags"].append("u")
        second = next(stream)["report"]["migration"]["record"]
        self.assertEqual(second["items"][0]["tags"], ["t"])
        self.assertEqual(rules[0]["value"], ["t"])
        self.assertEqual(inputs[0], {"items": [{"x": 1}, {"x": 2}]})

    def test_mutating_one_result_never_affects_later_results(self):
        self._publish([{"a": 1, "b": "x"}], 1)
        stream = self.lens.migrate_stream(
            iter([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]), 1, 1, []
        )
        first = next(stream)
        first["report"]["migration"]["record"]["a"] = 999
        first["report"]["migration"]["reports"].append("tampered")
        second = next(stream)
        self.assertEqual(
            second["report"]["migration"]["record"], {"a": 2, "b": "y"}
        )
        self.assertEqual(second["report"]["migration"]["reports"], [])

    def test_processed_records_are_not_retained(self):
        import gc
        import weakref

        self._publish([{"a": 1}], 1)

        class Record(dict):
            """A dict subclass so weak references to records are possible."""

        refs: list = []

        def generate():
            # Each record is freshly created, so its only strong
            # references are the ones the pipeline transiently holds.
            for index in range(50):
                record = Record(a=index)
                refs.append(weakref.ref(record))
                yield record

        stream = self.lens.migrate_stream(generate(), 1, 1, [])
        last = next(stream)
        for _ in range(49):
            # Replacing the previous result drops the record it carried.
            last = next(stream)
        gc.collect()
        # Earlier input records are collectable; only the one still in the
        # latest result stays alive.
        self.assertTrue(all(ref() is None for ref in refs[:-1]))
        self.assertIsNotNone(refs[-1]())

    def test_large_stream_keeps_continuous_indices(self):
        self._publish([{"a": 1}], 1)
        count = 20000

        def generate():
            for index in range(count):
                yield {"a": index}

        stream = self.lens.migrate_stream(generate(), 1, 1, [])
        last = None
        for consumed, result in enumerate(stream, 1):
            last = result
            self.assertEqual(result["index"], consumed)
        self.assertEqual(last["index"], count)

    def test_repeated_streams_are_stable(self):
        rules, back = self._rename_pair()
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        first = [
            json.dumps(r, sort_keys=True)
            for r in self.lens.migrate_stream(records, 1, 2, rules, back)
        ]
        for _ in range(3):
            again = [
                json.dumps(r, sort_keys=True)
                for r in self.lens.migrate_stream(records, 1, 2, rules, back)
            ]
            self.assertEqual(again, first)

    # -- state and persistence guarantees --------------------------------

    def test_legacy_single_file_migrates_on_first_read(self):
        folder = Lens(tempfile.mkdtemp())
        folder.infer([{"a": 1, "b": "x"}, {"a": 2}])
        root = folder.schema()
        from schema_lens.core import _checksum

        legacy = Path(self.dir.name, SCHEMA_FILENAME)
        legacy.write_text(
            json.dumps({"version": 2, "checksum": _checksum(root), "root": root})
        )
        lens = Lens(self.dir.name)
        results = lens.migrate_stream([{"a": 1, "b": "x"}], 1, 1, [])
        # The eager snapshot read performed the legacy migration already.
        self.assertEqual(lens.versions(), [1])
        self.assertFalse(legacy.exists())
        self.assertEqual(next(results)["report"]["migration"]["stage"], "done")

    def test_open_snapshot_batches_history_and_files_are_untouched(self):
        rules, _back = self._rename_pair()
        self.lens.load()
        self.lens.infer([{"pending": 1}])
        snapshot = self.lens.schema()
        versions_dir = Path(self.dir.name, VERSIONS_DIRNAME)
        before = sorted(os.listdir(versions_dir))
        list(
            self.lens.migrate_stream(
                [{"a": 1, "b": "x"}, "bad"], 1, 2, rules
            )
        )
        self.assertEqual(self.lens.schema(), snapshot)
        self.assertEqual(self.lens.check({"pending": 1}), [])
        self.assertEqual(self.lens.versions(), [1, 2])
        self.assertEqual(sorted(os.listdir(versions_dir)), before)

    def test_commit_after_return_does_not_change_open_batch_results(self):
        rules, _back = self._rename_pair()
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        expected = [self.lens.migrate(r, 1, 2, rules) for r in records]
        stream = self.lens.migrate_stream(iter(records), 1, 2, rules)
        first = next(stream)
        # A third revision lands while the batch is only half consumed.
        self._publish([{"a": 1, "d": "z"}], 3)
        rest = list(stream)
        self.assertEqual(first["report"], expected[0])
        self.assertEqual(rest[0]["report"], expected[1])


class MigratePathStreamTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)

    def _publish(self, records, revision, **limits):
        return _publish(self.dir.name, records, revision, **limits)

    def _rename_chain(self):
        # A field renamed a -> b -> c over three revisions.
        self._publish([{"a": 1}], 1)
        self._publish([{"b": 1}], 2)
        self._publish([{"c": 1}], 3)
        return [
            [{"op": "rename", "path": ["a"], "to": "b"}],
            [{"op": "rename", "path": ["b"], "to": "c"}],
        ]

    # -- basic shape -----------------------------------------------------

    def test_results_have_exactly_index_report_error(self):
        groups = self._rename_chain()
        results = list(
            self.lens.migrate_path_stream([{"a": 1}, {"a": 2}], [1, 2, 3], groups)
        )
        self.assertEqual([r["index"] for r in results], [1, 2])
        for result in results:
            self.assertEqual(set(result), {"index", "report", "error"})
            self.assertIsNone(result["error"])

    def test_field_renamed_twice_reaches_the_endpoint_record(self):
        groups = self._rename_chain()
        results = list(
            self.lens.migrate_path_stream([{"a": 7}], [1, 2, 3], groups)
        )
        report = results[0]["report"]
        self.assertEqual(
            report["migration"],
            {"stage": "done", "record": {"c": 7}, "reports": []},
        )
        self.assertEqual(
            [s["migration"]["record"] for s in report["steps"]],
            [{"b": 7}, {"c": 7}],
        )

    def test_report_keeps_endpoint_compat_fields_and_original_verdicts(self):
        groups = self._rename_chain()
        results = list(
            self.lens.migrate_path_stream([{"a": 1}], [1, 2, 3], groups)
        )
        report = results[0]["report"]
        compat = self.lens.compat(1, 3)
        self.assertEqual(
            set(report),
            {"from", "to", "backward", "forward", "changes",
             "unknown_reasons", "steps", "migration"},
        )
        for key in ("from", "to", "backward", "forward", "changes",
                    "unknown_reasons"):
            self.assertEqual(report[key], compat[key])

    def test_steps_are_complete_migrate_reports_in_path_order(self):
        groups = self._rename_chain()
        results = list(
            self.lens.migrate_path_stream([{"a": 1}], [1, 2, 3], groups)
        )
        report = results[0]["report"]
        self.assertEqual(len(report["steps"]), 2)
        self.assertEqual(
            [(s["from"], s["to"]) for s in report["steps"]], [(1, 2), (2, 3)]
        )
        for step in report["steps"]:
            self.assertEqual(
                set(step),
                {"from", "to", "backward", "forward", "changes",
                 "unknown_reasons", "migration"},
            )
        self.assertEqual(
            report["steps"][0],
            self.lens.migrate({"a": 1}, 1, 2, groups[0]),
        )
        self.assertEqual(
            report["steps"][1],
            self.lens.migrate({"b": 1}, 2, 3, groups[1]),
        )

    def test_migration_equals_last_attempted_segments_migration(self):
        groups = self._rename_chain()
        results = list(
            self.lens.migrate_path_stream([{"a": 1}], [1, 2, 3], groups)
        )
        report = results[0]["report"]
        self.assertEqual(report["migration"], report["steps"][-1]["migration"])

    def test_same_version_segments_still_run_rules(self):
        self._publish([{"a": 1, "b": 2}], 1)
        groups = [[{"op": "rename", "path": ["a"], "to": "c"}]]
        results = list(
            self.lens.migrate_path_stream([{"a": 1, "b": 2}], [1, 1], groups)
        )
        report = results[0]["report"]
        self.assertEqual(report["from"], 1)
        self.assertEqual(report["to"], 1)
        self.assertEqual(report["steps"][0]["from"], 1)
        self.assertEqual(report["steps"][0]["to"], 1)
        # The rename makes c unexpected under revision 1's schema.
        self.assertEqual(report["migration"]["stage"], "target")
        self.assertTrue(report["migration"]["reports"])

    def test_descending_order_repeats_and_adjacent_identical_versions(self):
        groups = self._rename_chain()
        back_groups = [
            [],
            [{"op": "rename", "path": ["c"], "to": "b"}],
            [{"op": "rename", "path": ["b"], "to": "a"}],
        ]
        results = list(
            self.lens.migrate_path_stream([{"c": 1}], [3, 3, 2, 1], back_groups)
        )
        report = results[0]["report"]
        self.assertEqual(
            [(s["from"], s["to"]) for s in report["steps"]],
            [(3, 3), (3, 2), (2, 1)],
        )
        self.assertEqual(report["from"], 3)
        self.assertEqual(report["to"], 1)
        self.assertEqual(
            report["migration"],
            {"stage": "done", "record": {"a": 1}, "reports": []},
        )

    def test_each_segment_reuses_existing_migrate_semantics_with_wildcards(self):
        self._publish([{"items": [{"x": 1}]}], 1)
        self._publish([{"items": [{"y": 1, "t": "z"}]}], 2)
        self._publish([{"items": [{"z": 1, "t": "z"}]}], 3)
        groups = [
            [{"op": "rename", "path": ["items", None, "x"], "to": "y"},
             {"op": "default", "path": ["items", None, "t"], "value": "z"}],
            [{"op": "rename", "path": ["items", None, "y"], "to": "z"}],
        ]
        results = list(
            self.lens.migrate_path_stream(
                [{"items": [{"x": 1}]}], [1, 2, 3], groups
            )
        )
        self.assertEqual(
            results[0]["report"]["migration"]["record"],
            {"items": [{"z": 1, "t": "z"}]},
        )

    # -- stopping mid-path ----------------------------------------------

    def test_source_stop_terminates_the_path_and_keeps_failing_step(self):
        # Revisions 1 and 2 require a; a record without a fails step 1's
        # source check, so step 2 must never run.
        self._publish([{"a": 1}], 1)
        self._publish([{"a": 1}], 2)
        self._publish([{"b": 1}], 3)
        step2_rules = [{"op": "rename", "path": ["a"], "to": "b"}]
        results = list(
            self.lens.migrate_path_stream(
                [{"b": 1}], [1, 2, 3], [[], step2_rules]
            )
        )
        report = results[0]["report"]
        # Step 1 stops at the source check: only it is retained, with a
        # null record, and segment 2 never runs.
        self.assertEqual(report["migration"]["stage"], "source")
        self.assertEqual(len(report["steps"]), 1)
        self.assertEqual(report["steps"][0]["from"], 1)
        self.assertIsNone(report["steps"][0]["migration"]["record"])

    def test_target_stop_terminates_the_path_and_keeps_failing_step(self):
        # A valid rev-1 record with no rules fails rev 2's required b.
        self._publish([{"a": 1}], 1)
        self._publish([{"b": 1}], 2)
        self._publish([{"c": 1}], 3)
        groups = [[], [{"op": "rename", "path": ["b"], "to": "c"}]]
        results = list(
            self.lens.migrate_path_stream([{"a": 1}], [1, 2, 3], groups)
        )
        report = results[0]["report"]
        self.assertEqual(report["migration"]["stage"], "target")
        self.assertEqual(len(report["steps"]), 1)
        self.assertEqual(
            report["migration"]["reports"],
            self.lens.check({"a": 1}, version=2),
        )
        self.assertEqual(report["migration"], report["steps"][-1]["migration"])

    def test_later_step_receives_the_previous_steps_migrated_record(self):
        groups = self._rename_chain()
        results = list(
            self.lens.migrate_path_stream([{"a": 5}], [1, 2, 3], groups)
        )
        report = results[0]["report"]
        # Step 2's source check (against revision 2, which requires b)
        # passes precisely because it sees step 1's {"b": 5}.
        self.assertEqual(report["steps"][1]["migration"]["stage"], "done")
        self.assertEqual(report["steps"][1]["migration"]["record"], {"c": 5})

    # -- per-record error isolation --------------------------------------

    def test_non_json_records_become_errors_and_stream_continues(self):
        groups = self._rename_chain()
        records = [
            {"a": 1}, {"a": {1, 2}}, {"a": float("nan")}, None, 1, "x",
            {"a": 2},
        ]
        results = list(
            self.lens.migrate_path_stream(records, [1, 2, 3], groups)
        )
        self.assertEqual([r["index"] for r in results], [1, 2, 3, 4, 5, 6, 7])
        self.assertIsNone(results[0]["error"])
        for bad in results[1:6]:
            self.assertIsNone(bad["report"])
            self.assertIsNotNone(bad["error"])
        self.assertIsNone(results[6]["error"])

    def test_step_rule_value_error_is_prefixed_and_isolated(self):
        self._publish([{"a": [1]}, {"a": {"x": 1}}], 1)
        groups = [
            [{"op": "drop", "path": ["a", "x"]}],
            [{"op": "drop", "path": ["a", "x"]}],
        ]
        # [1] passes the rev-1 source check; the null-free path "a.x" on
        # an array makes step 1 raise. Reusing revision 1 keeps both
        # segments on the same (frozen) snapshot.
        results = list(
            self.lens.migrate_path_stream(
                [{"a": [1]}, {"a": {"x": 1}}, {"a": [1]}], [1, 1], groups[:1]
            )
        )
        self.assertEqual(
            results[0],
            {"index": 1, "report": None,
             "error": "step 1: string path segment requires an object"},
        )
        self.assertIsNone(results[1]["error"])
        self.assertEqual(
            results[2]["error"],
            "step 1: string path segment requires an object",
        )

    def test_step_prefix_names_the_failing_segment(self):
        self._publish([{"a": [1]}], 1)
        self._publish([{"a": [1]}], 2)
        # Step 1 carries no rules and the record passes both revs 1 and
        # 2; step 2's null path then meets the non-array number inside a.
        self._publish([{"a": [{"x": 1}]}], 3)
        groups = [
            [],
            [{"op": "drop", "path": ["a", None, "x"]}],
        ]
        results = list(
            self.lens.migrate_path_stream(
                [{"a": [1]}], [1, 2, 2], groups
            )
        )
        self.assertIsNone(results[0]["report"])
        self.assertTrue(results[0]["error"].startswith("step 2: "))

    def test_iterator_non_stopiteration_exception_propagates_verbatim(self):
        groups = self._rename_chain()

        def generate():
            yield {"a": 1}
            raise RuntimeError("stream broke")

        stream = self.lens.migrate_path_stream(generate(), [1, 2, 3], groups)
        self.assertEqual(next(stream)["index"], 1)
        with self.assertRaisesRegex(RuntimeError, "stream broke"):
            next(stream)

    def test_iterator_value_error_is_not_turned_into_a_result(self):
        groups = [[]]
        self._publish([{"a": 1}], 1)

        def generate():
            yield {"a": 1}
            raise ValueError("source problem")

        stream = self.lens.migrate_path_stream(generate(), [1, 1], groups)
        self.assertIsNone(next(stream)["error"])
        with self.assertRaisesRegex(ValueError, "source problem"):
            next(stream)

    # -- eager validation, no consumption --------------------------------

    def _tracking_source(self, records):
        source = {"started": False, "records": records}

        def generate():
            source["started"] = True
            yield from records

        source["iterator_factory"] = generate
        return source

    def test_call_does_not_consume_records(self):
        groups = self._rename_chain()
        source = self._tracking_source([{"a": 1}])
        results = self.lens.migrate_path_stream(
            source["iterator_factory"](), [1, 2, 3], groups
        )
        self.assertFalse(source["started"])
        first = next(results)
        self.assertTrue(source["started"])
        self.assertIsNone(first["error"])

    def test_one_record_consumed_per_pull_without_prefetch(self):
        self._publish([{"a": 1}], 1)
        pulled = []

        def generate():
            for value in (1, 2, 3):
                record = {"a": value}
                pulled.append(record)
                yield record

        stream = self.lens.migrate_path_stream(generate(), [1, 1], [[]])
        self.assertEqual(pulled, [])
        next(stream)
        self.assertEqual(len(pulled), 1)
        next(stream)
        self.assertEqual(len(pulled), 2)
        next(stream)
        self.assertEqual(len(pulled), 3)
        with self.assertRaises(StopIteration):
            next(stream)
        self.assertEqual(len(pulled), 3)

    def test_path_container_validation(self):
        groups = self._rename_chain()
        for revisions, rule_groups in (
            ([1], []),
            ([1, 2], [[]]),
            ("x", []),
            ((1, 2), [[]]),
            ([1, 2], groups),
            ([1, 2, 3], groups[:1]),
            ([1, 2], "x"),
            ([1, 2], ["x"]),
            ([1, 2], [{"op": "drop", "path": ["a"]}]),
        ):
            with self.subTest(revisions=revisions, rule_groups=rule_groups):
                with self.assertRaises(ValueError):
                    self.lens.migrate_path_stream(None, revisions, rule_groups)

    def test_illegal_rule_in_any_group_raises_without_consuming(self):
        self._rename_chain()
        source = self._tracking_source([{"a": 1}])
        with self.assertRaises(ValueError):
            self.lens.migrate_path_stream(
                source["iterator_factory"](),
                [1, 2, 3],
                [[{"op": "rename", "path": ["a"], "to": "b"}],
                 [{"op": "nope"}]],
            )
        self.assertFalse(source["started"])

    def test_containers_validated_before_revisions_and_records(self):
        source = self._tracking_source([{"a": 1}])
        with self.assertRaises(ValueError):
            self.lens.migrate_path_stream(
                source["iterator_factory"](), [1], []
            )
        self.assertFalse(source["started"])

    def test_bad_revisions_raise_conflict_without_consuming(self):
        self._rename_chain()
        for revisions in ([0, 1], [1, -1], [True, 1], [1, "2"],
                          [1, 2, 9], [1, 2.0]):
            source = self._tracking_source([{"a": 1}])
            with self.subTest(revisions=revisions):
                with self.assertRaises(SchemaConflict):
                    self.lens.migrate_path_stream(
                        source["iterator_factory"](),
                        revisions,
                        [[] for _ in range(len(revisions) - 1)],
                    )
                self.assertFalse(source["started"])

    def test_corrupt_revision_raises_conflict_without_consuming(self):
        self._rename_chain()
        revision_path(self.dir.name, 2).write_text("not json", encoding="utf-8")
        source = self._tracking_source([{"a": 1}])
        with self.assertRaises(SchemaConflict):
            self.lens.migrate_path_stream(
                source["iterator_factory"](), [1, 2, 3], [[], []]
            )
        self.assertFalse(source["started"])

    def test_each_distinct_snapshot_is_read_once_and_repeats_share_it(self):
        groups = self._rename_chain()
        seen = []
        original = Lens._read_revision

        def reading(lens, revision):
            seen.append(revision)
            return original(lens, revision)

        Lens._read_revision = reading
        try:
            list(
                self.lens.migrate_path_stream(
                    [{"a": 1}], [2, 1, 2, 3, 3], [[], [], [], []]
                )
            )
        finally:
            Lens._read_revision = original
        self.assertEqual(sorted(seen), [1, 2, 3])

    def test_non_iterable_records_raise_value_error(self):
        self._rename_chain()
        for records in (7, 3.5, object()):
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    self.lens.migrate_path_stream(records, [1, 2], [[]])

    def test_empty_input_runs_all_checks_and_yields_nothing(self):
        groups = self._rename_chain()
        self.assertEqual(
            list(self.lens.migrate_path_stream([], [1, 2, 3], groups)), []
        )
        self.assertEqual(
            list(self.lens.migrate_path_stream(iter([]), [1, 1], [[]])), []
        )
        with self.assertRaises(ValueError):
            self.lens.migrate_path_stream([], [1, 2, 3], [[{"op": "bad"}]])
        with self.assertRaises(SchemaConflict):
            self.lens.migrate_path_stream([], [9, 9], [[]])
        with self.assertRaises(ValueError):
            self.lens.migrate_path_stream([], [1], [])
        with self.assertRaises(ValueError):
            self.lens.migrate_path_stream(42, [1, 1], [[]])

    # -- freezing, independence, streaming memory ------------------------

    def test_snapshots_survive_compaction_between_pulls(self):
        lens = _build_history(self.dir.name, 5, compact_keep=2)
        other = Lens(self.dir.name)
        records = [{"f0": 0, "shared": "s"}, {"f2": 2, "shared": "s"}]
        # A path that revises 1 -> 3 (baseline after compaction) -> 4
        # with no rules; the published histories fold f{i} as required,
        # so each step lands at target for these records - we only need
        # the frozen snapshots to keep answering.
        expected_first = (
            list(other.migrate_path_stream([records[0]], [1, 3, 4], [[], []]))
            [0]["report"]
        )
        stream = lens.migrate_path_stream(iter(records), [1, 3, 4], [[], []])
        first = next(stream)["report"]
        self.assertEqual(first, expected_first)
        self.assertEqual(lens.compact(), {"anchor": 3, "merged": [1, 2, 3]})
        with self.assertRaises(SchemaConflict):
            lens.migrate_path_stream([records[0]], [1, 3], [[]])
        rest = list(stream)
        self.assertEqual(len(rest), 1)
        anchor = list(
            lens.migrate_path_stream(
                [{"f2": 2, "shared": "s"}], [3, 4], [[]]
            )
        )
        self.assertEqual(anchor[0]["report"]["steps"][-1]["migration"]["stage"],
                         "done")

    def test_caller_mutating_path_inputs_after_call_does_not_change_batch(self):
        groups = self._rename_chain()
        records = [{"a": 1}, {"a": 2}]
        expected = [
            list(self.lens.migrate_path_stream([r], [1, 2, 3], groups))[0]["report"]
            for r in records
        ]
        revisions = [1, 2, 3]
        stream = self.lens.migrate_path_stream(
            iter(records), revisions, copy.deepcopy(groups)
        )
        first = next(stream)
        revisions.append(999)
        revisions[0] = 9
        groups[0][0]["to"] = "zzz"
        groups.append([{"op": "drop", "path": ["a"]}])
        self.assertEqual(first["report"], expected[0])
        self.assertEqual(next(stream)["report"], expected[1])

    def test_commit_after_return_does_not_change_results(self):
        groups = self._rename_chain()
        records = [{"a": 1}, {"a": 2}]
        expected = [
            list(self.lens.migrate_path_stream([r], [1, 2, 3], groups))[0]["report"]
            for r in records
        ]
        stream = self.lens.migrate_path_stream(iter(records), [1, 2, 3], groups)
        first = next(stream)
        self._publish([{"q": 1}], 4)
        rest = list(stream)
        self.assertEqual(first["report"], expected[0])
        self.assertEqual(rest[0]["report"], expected[1])

    def test_input_records_are_not_modified(self):
        groups = self._rename_chain()
        records = [{"a": 1}, {"a": 2}]
        originals = copy.deepcopy(records)
        list(self.lens.migrate_path_stream(records, [1, 2, 3], groups))
        self.assertEqual(records, originals)

    def test_returned_records_are_independent_of_inputs_defaults_and_each_other(self):
        self._publish(
            [{"items": [{"x": 1, "tags": ["t"]}, {"x": 2}]}], 1
        )
        # Revision 2 always carries the tags array; step 1 defaults it
        # and step 2 is a same-version segment that just re-checks.
        self._publish(
            [{"items": [{"x": 1, "tags": ["t"]}, {"x": 2, "tags": ["t"]}]}], 2
        )
        default_value = ["t"]
        groups = [
            [{"op": "default", "path": ["items", None, "tags"],
              "value": default_value}],
            [],
        ]
        inputs = [{"items": [{"x": 1}]}, {"items": [{"x": 2}]}]
        stream = self.lens.migrate_path_stream(iter(inputs), [1, 2, 2], groups)
        first = next(stream)["report"]["migration"]["record"]
        first["items"][0]["tags"].append("u")
        second = next(stream)["report"]["migration"]["record"]
        self.assertEqual(second["items"][0]["tags"], ["t"])
        self.assertEqual(default_value, ["t"])
        self.assertEqual(inputs[0], {"items": [{"x": 1}]})

    def test_mutating_one_result_never_affects_later_results(self):
        self._publish([{"a": 1}], 1)
        stream = self.lens.migrate_path_stream(
            iter([{"a": 1}, {"a": 2}]), [1, 1], [[]]
        )
        first = next(stream)
        first["report"]["migration"]["record"]["a"] = 999
        first["report"]["steps"][0]["migration"]["reports"].append("tampered")
        second = next(stream)
        self.assertEqual(
            second["report"]["migration"]["record"], {"a": 2}
        )
        self.assertEqual(second["report"]["steps"][0]["migration"]["reports"], [])

    def test_processed_records_are_not_retained(self):
        import gc
        import weakref

        self._publish([{"a": 1}], 1)

        class Record(dict):
            """A dict subclass so weak references to records are possible."""

        refs: list = []

        def generate():
            for index in range(50):
                record = Record(a=index)
                refs.append(weakref.ref(record))
                yield record

        stream = self.lens.migrate_path_stream(generate(), [1, 1], [[]])
        last = next(stream)
        for _ in range(49):
            last = next(stream)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs[:-1]))
        self.assertIsNotNone(refs[-1]())

    def test_large_stream_keeps_continuous_indices(self):
        self._publish([{"a": 1}], 1)
        count = 20000

        def generate():
            for index in range(count):
                yield {"a": index}

        stream = self.lens.migrate_path_stream(generate(), [1, 1, 1], [[], []])
        last = None
        for consumed, result in enumerate(stream, 1):
            last = result
            self.assertEqual(result["index"], consumed)
        self.assertEqual(last["index"], count)

    def test_repeated_streams_are_stable(self):
        groups = self._rename_chain()
        records = [{"a": 1}, {"a": 2}]
        first = [
            json.dumps(r, sort_keys=True)
            for r in self.lens.migrate_path_stream(records, [1, 2, 3], groups)
        ]
        for _ in range(3):
            again = [
                json.dumps(r, sort_keys=True)
                for r in self.lens.migrate_path_stream(records, [1, 2, 3], groups)
            ]
            self.assertEqual(again, first)

    # -- state and persistence guarantees --------------------------------

    def test_legacy_single_file_migrates_on_first_read(self):
        folder = Lens(tempfile.mkdtemp())
        folder.infer([{"a": 1}])
        root = folder.schema()
        from schema_lens.core import _checksum

        legacy = Path(self.dir.name, SCHEMA_FILENAME)
        legacy.write_text(
            json.dumps({"version": 2, "checksum": _checksum(root), "root": root})
        )
        lens = Lens(self.dir.name)
        results = lens.migrate_path_stream([{"a": 1}], [1, 1], [[]])
        self.assertEqual(lens.versions(), [1])
        self.assertFalse(legacy.exists())
        self.assertEqual(
            next(results)["report"]["migration"]["stage"], "done"
        )

    def test_open_snapshot_batches_and_files_are_untouched(self):
        groups = self._rename_chain()
        self.lens.load()
        self.lens.infer([{"pending": 1}])
        snapshot = self.lens.schema()
        versions_dir = Path(self.dir.name, VERSIONS_DIRNAME)
        before = sorted(os.listdir(versions_dir))
        list(
            self.lens.migrate_path_stream(
                [{"a": 1}, "bad"], [1, 2, 3], groups
            )
        )
        self.assertEqual(self.lens.schema(), snapshot)
        self.assertEqual(self.lens.check({"pending": 1}), [])
        self.assertEqual(self.lens.versions(), [1, 2, 3])
        self.assertEqual(sorted(os.listdir(versions_dir)), before)


class MigratePathStreamRollbackTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)

    def _publish(self, records, revision, **limits):
        return _publish(self.dir.name, records, revision, **limits)

    def _rename_chain(self):
        # A field renamed a -> b -> c over three revisions; the rollback
        # groups rename it back, group i belonging to forward segment i+1.
        self._publish([{"a": 1}], 1)
        self._publish([{"b": 1}], 2)
        self._publish([{"c": 1}], 3)
        forward = [
            [{"op": "rename", "path": ["a"], "to": "b"}],
            [{"op": "rename", "path": ["b"], "to": "c"}],
        ]
        backward = [
            [{"op": "rename", "path": ["b"], "to": "a"}],
            [{"op": "rename", "path": ["c"], "to": "b"}],
        ]
        return forward, backward

    # -- output shape ----------------------------------------------------

    def test_omitted_and_none_leave_the_report_unchanged(self):
        groups, back = self._rename_chain()
        records = [{"a": 1}, {"a": 2}]
        plain = list(self.lens.migrate_path_stream(records, [1, 2, 3], groups))
        none = list(
            self.lens.migrate_path_stream(
                records, [1, 2, 3], groups, rollback_rule_groups=None
            )
        )
        self.assertEqual(plain, none)
        for result in plain:
            self.assertNotIn("rollback_steps", result["report"])
            self.assertNotIn("rollback", result["report"])
        enabled = list(
            self.lens.migrate_path_stream(records, [1, 2, 3], groups, back)
        )
        self.assertEqual(len(enabled), 2)
        for result in enabled:
            self.assertEqual(set(result), {"index", "report", "error"})
            self.assertIsNone(result["error"])
            self.assertEqual(
                set(result["report"]),
                {"from", "to", "backward", "forward", "changes",
                 "unknown_reasons", "steps", "migration",
                 "rollback_steps", "rollback"},
            )
        # The forward part of the report is exactly the plain report.
        for with_back, without in zip(enabled, plain):
            for key in without["report"]:
                self.assertEqual(with_back["report"][key],
                                 without["report"][key])

    def test_full_roundtrip_recovers_the_original_record(self):
        groups, back = self._rename_chain()
        results = list(
            self.lens.migrate_path_stream([{"a": 7}], [1, 2, 3], groups, back)
        )
        report = results[0]["report"]
        self.assertEqual(
            report["rollback"],
            {"stage": "done", "record": {"a": 7}, "reports": [],
             "differences": []},
        )
        # Reverse segments execute last-forward-segment first.
        self.assertEqual(
            [(s["from"], s["to"]) for s in report["rollback_steps"]],
            [(3, 2), (2, 1)],
        )
        self.assertEqual(
            [s["migration"]["record"] for s in report["rollback_steps"]],
            [{"b": 7}, {"a": 7}],
        )

    def test_rollback_steps_are_complete_migrate_reports_in_execution_order(self):
        groups, back = self._rename_chain()
        results = list(
            self.lens.migrate_path_stream([{"a": 1}], [1, 2, 3], groups, back)
        )
        report = results[0]["report"]
        self.assertEqual(len(report["rollback_steps"]), 2)
        for step in report["rollback_steps"]:
            self.assertEqual(
                set(step),
                {"from", "to", "backward", "forward", "changes",
                 "unknown_reasons", "migration"},
            )
        self.assertEqual(
            report["rollback_steps"][0],
            self.lens.migrate({"c": 1}, 3, 2, back[1]),
        )
        self.assertEqual(
            report["rollback_steps"][1],
            self.lens.migrate({"b": 1}, 2, 1, back[0]),
        )

    def test_rollback_group_applies_its_rules_in_their_own_order(self):
        self._publish([{"a": 1}], 1)
        self._publish([{"b": 1}], 2)
        groups = [[{"op": "rename", "path": ["a"], "to": "b"}]]
        # drop then re-default inside one group, applied in group order.
        back = [[
            {"op": "rename", "path": ["b"], "to": "a"},
            {"op": "drop", "path": ["a"]},
            {"op": "default", "path": ["a"], "value": 3},
        ]]
        results = list(
            self.lens.migrate_path_stream([{"a": 9}], [1, 2], groups, back)
        )
        rollback = results[0]["report"]["rollback"]
        self.assertEqual(rollback["stage"], "different")
        self.assertEqual(rollback["record"], {"a": 3})
        self.assertEqual(rollback["differences"], ["$.a"])

    def test_same_version_segment_runs_its_rollback_rules(self):
        self._publish([{"a": 1}], 1)
        groups = [[
            {"op": "drop", "path": ["a"]},
            {"op": "default", "path": ["a"], "value": 5},
        ]]
        back = [[
            {"op": "drop", "path": ["a"]},
            {"op": "default", "path": ["a"], "value": 7},
        ]]
        results = list(
            self.lens.migrate_path_stream([{"a": 9}], [1, 1], groups, back)
        )
        report = results[0]["report"]
        self.assertEqual(report["migration"]["record"], {"a": 5})
        self.assertEqual(
            [(s["from"], s["to"]) for s in report["rollback_steps"]],
            [(1, 1)],
        )
        self.assertEqual(report["rollback"]["stage"], "different")
        self.assertEqual(report["rollback"]["record"], {"a": 7})
        self.assertEqual(report["rollback"]["differences"], ["$.a"])

    # -- forward not done -------------------------------------------------

    def test_forward_source_stop_gives_empty_steps_and_null_rollback(self):
        groups, back = self._rename_chain()
        results = list(
            self.lens.migrate_path_stream([{"x": 1}], [1, 2, 3], groups, back)
        )
        report = results[0]["report"]
        self.assertEqual(report["migration"]["stage"], "source")
        self.assertEqual(report["rollback_steps"], [])
        self.assertIsNone(report["rollback"])

    def test_forward_mid_path_stop_gives_empty_steps_and_null_rollback(self):
        groups, back = self._rename_chain()
        # Step 1 carries no rules, so {"a": 1} fails revision 2's
        # required b at the target check and step 2 never runs.
        results = list(
            self.lens.migrate_path_stream(
                [{"a": 1}], [1, 2, 3], [[], groups[1]], back
            )
        )
        report = results[0]["report"]
        self.assertEqual(report["migration"]["stage"], "target")
        self.assertEqual(len(report["steps"]), 1)
        self.assertEqual(report["rollback_steps"], [])
        self.assertIsNone(report["rollback"])

    # -- rollback check failures -----------------------------------------

    def test_rollback_target_stop_keeps_failing_segment_and_stops(self):
        groups, _back = self._rename_chain()
        # No rollback rules: the recovered {"c": 7} fails revision 2.
        results = list(
            self.lens.migrate_path_stream(
                [{"a": 7}], [1, 2, 3], groups, [[], []]
            )
        )
        report = results[0]["report"]
        self.assertEqual(report["migration"]["stage"], "done")
        # Only the first reverse segment (3 -> 2) ran; (2 -> 1) never did.
        self.assertEqual(len(report["rollback_steps"]), 1)
        self.assertEqual(
            (report["rollback_steps"][0]["from"],
             report["rollback_steps"][0]["to"]),
            (3, 2),
        )
        self.assertEqual(
            report["rollback"],
            {"stage": "target", "record": None,
             "reports": self.lens.check({"c": 7}, version=2),
             "differences": []},
        )
        self.assertTrue(report["rollback"]["reports"])

    def test_rollback_stop_in_a_later_reverse_segment(self):
        groups, back = self._rename_chain()
        # The first reverse segment recovers b; the second has no rules,
        # so {"b": 7} fails revision 1's required a.
        results = list(
            self.lens.migrate_path_stream(
                [{"a": 7}], [1, 2, 3], groups, [[], back[1]]
            )
        )
        report = results[0]["report"]
        self.assertEqual(len(report["rollback_steps"]), 2)
        self.assertEqual(
            report["rollback"],
            {"stage": "target", "record": None,
             "reports": self.lens.check({"b": 7}, version=1),
             "differences": []},
        )

    def test_rollback_object_has_exactly_the_roundtrip_keys(self):
        groups, back = self._rename_chain()
        results = list(
            self.lens.migrate_path_stream([{"a": 1}], [1, 2, 3], groups, back)
        )
        self.assertEqual(
            set(results[0]["report"]["rollback"]),
            {"stage", "record", "reports", "differences"},
        )

    def test_rollback_different_lists_every_differing_path(self):
        self._publish([{"a": 1, "b": "x"}], 1)
        self._publish([{"c": 1, "b": "x"}], 2)
        groups = [[{"op": "rename", "path": ["a"], "to": "c"}]]
        back = [[{"op": "drop", "path": ["c"]},
                 {"op": "default", "path": ["a"], "value": 0}]]
        results = list(
            self.lens.migrate_path_stream(
                [{"a": 7, "b": "y"}], [1, 2], groups, back
            )
        )
        rollback = results[0]["report"]["rollback"]
        self.assertEqual(rollback["stage"], "different")
        self.assertEqual(rollback["record"], {"a": 0, "b": "y"})
        self.assertEqual(rollback["reports"], [])
        self.assertEqual(rollback["differences"], ["$.a"])

    # -- per-record errors -------------------------------------------------

    def test_rollback_rule_error_is_prefixed_with_forward_segment_number(self):
        # Every revision tolerates an optional "extra" field, so one
        # record can carry a value the rollback rule collides with.
        self._publish([{"a": 1}, {"a": 2, "extra": 1}], 1)
        self._publish([{"b": 1}, {"b": 2, "extra": 1}], 2)
        self._publish([{"c": 1}, {"c": 2, "extra": 1}], 3)
        groups = [
            [{"op": "rename", "path": ["a"], "to": "b"}],
            [{"op": "rename", "path": ["b"], "to": "c"}],
        ]
        back = [
            [{"op": "rename", "path": ["b"], "to": "a"},
             {"op": "rename", "path": ["extra"], "to": "a"}],
            [{"op": "rename", "path": ["c"], "to": "b"}],
        ]
        results = list(
            self.lens.migrate_path_stream(
                [{"a": 1, "extra": 5}, {"a": 2}], [1, 2, 3], groups, back
            )
        )
        # The failing group belongs to forward segment 1 (it runs last
        # in the reverse order, but keeps the forward segment's number).
        self.assertEqual(
            results[0],
            {"index": 1, "report": None,
             "error": "rollback step 1: rename target 'a' already exists"},
        )
        # The stream continues; the next record rolls back cleanly.
        self.assertIsNone(results[1]["error"])
        self.assertEqual(
            results[1]["report"]["rollback"],
            {"stage": "done", "record": {"a": 2}, "reports": [],
             "differences": []},
        )

    def test_rollback_error_numbers_the_forward_segment_not_execution_order(self):
        groups, back = self._rename_chain()
        # The failing group belongs to forward segment 1, so it executes
        # last in the rollback yet reports "rollback step 1".
        back[0] = [{"op": "drop", "path": ["b", "x"]}]
        results = list(
            self.lens.migrate_path_stream([{"a": 1}], [1, 2, 3], groups, back)
        )
        self.assertIsNone(results[0]["report"])
        self.assertEqual(
            results[0]["error"],
            "rollback step 1: string path segment requires an object",
        )

    def test_forward_step_error_still_uses_the_plain_prefix(self):
        groups, back = self._rename_chain()
        groups[0] = [{"op": "drop", "path": ["a", "x"]}]
        results = list(
            self.lens.migrate_path_stream([{"a": 1}], [1, 2, 3], groups, back)
        )
        self.assertEqual(
            results[0]["error"],
            "step 1: string path segment requires an object",
        )

    def test_non_json_record_error_carries_no_rollback_keys(self):
        groups, back = self._rename_chain()
        results = list(
            self.lens.migrate_path_stream(
                [{"a": 1}, None, {"a": 2}], [1, 2, 3], groups, back
            )
        )
        self.assertIsNone(results[1]["report"])
        self.assertIsNotNone(results[1]["error"])
        self.assertEqual(results[2]["report"]["rollback"]["stage"], "done")

    # -- eager validation --------------------------------------------------

    def _tracking_source(self, records):
        source = {"started": False, "records": records}

        def generate():
            source["started"] = True
            yield from records

        source["iterator_factory"] = generate
        return source

    def test_rollback_container_validation(self):
        groups, back = self._rename_chain()
        for rollback_rule_groups in (
            "x",
            7,
            {"a": 1},
            back[:1],
            back + [[]],
            ["x", []],
            [[{"op": "nope"}], []],
            [[], [{"op": "rename", "path": ["c"], "to": "c"}]],
        ):
            with self.subTest(rollback_rule_groups=rollback_rule_groups):
                with self.assertRaises(ValueError):
                    self.lens.migrate_path_stream(
                        None, [1, 2, 3], groups, rollback_rule_groups
                    )

    def test_rollback_rules_validated_before_revisions_and_records(self):
        groups, _back = self._rename_chain()
        source = self._tracking_source([{"a": 1}])
        # An illegal rollback rule reports as ValueError even though the
        # revisions are bad too (rules are validated first) ...
        with self.assertRaises(ValueError):
            self.lens.migrate_path_stream(
                source["iterator_factory"](),
                [1, 2, 99],
                groups,
                [[{"op": "nope"}], []],
            )
        self.assertFalse(source["started"])
        # ... and a valid rollback configuration still hits the bad
        # revision as SchemaConflict, before any record is consumed.
        with self.assertRaises(SchemaConflict):
            self.lens.migrate_path_stream(
                source["iterator_factory"](),
                [1, 2, 99],
                groups,
                [[], []],
            )
        self.assertFalse(source["started"])

    def test_empty_input_validates_rollback_and_yields_nothing(self):
        groups, back = self._rename_chain()
        self.assertEqual(
            list(self.lens.migrate_path_stream([], [1, 2, 3], groups, back)),
            [],
        )
        with self.assertRaises(ValueError):
            self.lens.migrate_path_stream([], [1, 2, 3], groups, [[{"op": "bad"}], []])
        with self.assertRaises(ValueError):
            self.lens.migrate_path_stream([], [1, 2, 3], groups, [[]])
        with self.assertRaises(SchemaConflict):
            self.lens.migrate_path_stream([], [1, 2, 9], groups, back)
        with self.assertRaises(ValueError):
            self.lens.migrate_path_stream(42, [1, 2, 3], groups, back)

    def test_non_iterable_records_raise_value_error_with_rollback(self):
        groups, back = self._rename_chain()
        with self.assertRaises(ValueError):
            self.lens.migrate_path_stream(7, [1, 2, 3], groups, back)

    # -- streaming, freezing, independence ---------------------------------

    def test_one_record_consumed_per_pull_with_rollback(self):
        groups, back = self._rename_chain()
        pulled = []

        def generate():
            for value in (1, 2, 3):
                record = {"a": value}
                pulled.append(record)
                yield record

        stream = self.lens.migrate_path_stream(
            generate(), [1, 2, 3], groups, back
        )
        self.assertEqual(pulled, [])
        next(stream)
        self.assertEqual(len(pulled), 1)
        next(stream)
        self.assertEqual(len(pulled), 2)
        list(stream)
        self.assertEqual(len(pulled), 3)

    def test_caller_mutating_rollback_groups_after_call_does_not_change_batch(self):
        groups, back = self._rename_chain()
        records = [{"a": 1}, {"a": 2}]
        expected = [
            list(self.lens.migrate_path_stream([r], [1, 2, 3], groups, back))[0]
            for r in records
        ]
        mutable_back = copy.deepcopy(back)
        stream = self.lens.migrate_path_stream(
            iter(records), [1, 2, 3], groups, mutable_back
        )
        first = next(stream)
        mutable_back[0][0]["to"] = "zzz"
        mutable_back.append([{"op": "drop", "path": ["a"]}])
        self.assertEqual(first["report"], expected[0]["report"])
        self.assertEqual(next(stream)["report"], expected[1]["report"])

    def test_commit_after_return_does_not_change_rollback_results(self):
        groups, back = self._rename_chain()
        records = [{"a": 1}, {"a": 2}]
        expected = [
            list(self.lens.migrate_path_stream([r], [1, 2, 3], groups, back))[0]
            for r in records
        ]
        stream = self.lens.migrate_path_stream(
            iter(records), [1, 2, 3], groups, back
        )
        first = next(stream)
        self._publish([{"q": 1}], 4)
        rest = list(stream)
        self.assertEqual(first["report"], expected[0]["report"])
        self.assertEqual(rest[0]["report"], expected[1]["report"])

    def test_input_records_and_rules_are_not_modified(self):
        groups, back = self._rename_chain()
        records = [{"a": 1}, {"a": 2}]
        originals = copy.deepcopy(records)
        frozen_groups = copy.deepcopy(groups)
        frozen_back = copy.deepcopy(back)
        list(self.lens.migrate_path_stream(records, [1, 2, 3], groups, back))
        self.assertEqual(records, originals)
        self.assertEqual(groups, frozen_groups)
        self.assertEqual(back, frozen_back)

    def test_rollback_results_are_independent_of_each_other(self):
        groups, back = self._rename_chain()
        stream = self.lens.migrate_path_stream(
            iter([{"a": 1}, {"a": 2}]), [1, 2, 3], groups, back
        )
        first = next(stream)["report"]
        first["rollback"]["record"]["a"] = 999
        first["rollback_steps"][0]["migration"]["record"]["b"] = 999
        first["rollback"]["differences"].append("tampered")
        second = next(stream)["report"]
        self.assertEqual(second["rollback"]["record"], {"a": 2})
        self.assertEqual(
            second["rollback_steps"][0]["migration"]["record"], {"b": 2}
        )
        self.assertEqual(second["rollback"]["differences"], [])

    def test_rollback_record_is_independent_of_the_step_records(self):
        groups, back = self._rename_chain()
        report = list(
            self.lens.migrate_path_stream([{"a": 1}], [1, 2, 3], groups, back)
        )[0]["report"]
        report["rollback"]["record"]["a"] = 999
        self.assertEqual(
            report["rollback_steps"][-1]["migration"]["record"], {"a": 1}
        )

    def test_processed_records_are_not_retained_with_rollback(self):
        import gc
        import weakref

        groups, back = self._rename_chain()

        class Record(dict):
            """A dict subclass so weak references to records are possible."""

        refs: list = []

        def generate():
            for index in range(50):
                record = Record(a=index)
                refs.append(weakref.ref(record))
                yield record

        stream = self.lens.migrate_path_stream(
            generate(), [1, 2, 3], groups, back
        )
        last = next(stream)
        for _ in range(49):
            last = next(stream)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs[:-1]))
        self.assertIsNotNone(refs[-1]())

    def test_repeated_streams_with_rollback_are_stable(self):
        groups, back = self._rename_chain()
        records = [{"a": 1}, {"a": 2}]
        first = [
            json.dumps(r, sort_keys=True)
            for r in self.lens.migrate_path_stream(records, [1, 2, 3], groups, back)
        ]
        for _ in range(3):
            again = [
                json.dumps(r, sort_keys=True)
                for r in self.lens.migrate_path_stream(
                    records, [1, 2, 3], groups, back
                )
            ]
            self.assertEqual(again, first)

    # -- persistence guarantees ---------------------------------------------

    def test_legacy_single_file_migrates_on_first_read_with_rollback(self):
        folder = Lens(tempfile.mkdtemp())
        folder.infer([{"a": 1}])
        root = folder.schema()
        from schema_lens.core import _checksum

        legacy = Path(self.dir.name, SCHEMA_FILENAME)
        legacy.write_text(
            json.dumps({"version": 2, "checksum": _checksum(root), "root": root})
        )
        lens = Lens(self.dir.name)
        results = lens.migrate_path_stream(
            [{"a": 1}], [1, 1], [[]], [[]]
        )
        self.assertEqual(lens.versions(), [1])
        self.assertFalse(legacy.exists())
        report = next(results)["report"]
        self.assertEqual(report["migration"]["stage"], "done")
        self.assertEqual(report["rollback"]["stage"], "done")

    def test_open_snapshot_batches_and_files_are_untouched_with_rollback(self):
        groups, back = self._rename_chain()
        self.lens.load()
        self.lens.infer([{"pending": 1}])
        snapshot = self.lens.schema()
        versions_dir = Path(self.dir.name, VERSIONS_DIRNAME)
        before = sorted(os.listdir(versions_dir))
        list(
            self.lens.migrate_path_stream(
                [{"a": 1}, "bad"], [1, 2, 3], groups, back
            )
        )
        self.assertEqual(self.lens.schema(), snapshot)
        self.assertEqual(self.lens.check({"pending": 1}), [])
        self.assertEqual(self.lens.versions(), [1, 2, 3])
        self.assertEqual(sorted(os.listdir(versions_dir)), before)


if __name__ == "__main__":
    unittest.main()

