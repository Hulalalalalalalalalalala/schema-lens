import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from schema_lens import Lens, SchemaConflict
from schema_lens.__main__ import _read_records, main as cli_main
from schema_lens.core import LOCK_FILENAME, SCHEMA_FILENAME, SCHEMA_VERSION

try:
    import fcntl
except ImportError:
    fcntl = None


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


class PersistenceTests(unittest.TestCase):
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

    def test_load_replaces_memory_wholesale(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()
        lens.infer([{"b": 2}])
        lens.load()
        self.assertEqual(sorted(lens.schema()["fields"]), ["a"])

    def test_load_missing_file_raises(self):
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).load()

    def test_load_corrupt_file_raises_and_preserves_memory(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        before = lens.schema()
        Path(self.dir.name, "schema.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            lens.load()
        self.assertEqual(lens.schema(), before)

    def test_load_malformed_schema_raises_and_preserves_memory(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        before = lens.schema()
        Path(self.dir.name, "schema.json").write_text(
            json.dumps({"root": {"kind": "object", "fields": "oops"}}),
            encoding="utf-8",
        )
        with self.assertRaises(SchemaConflict):
            lens.load()
        self.assertEqual(lens.schema(), before)

    def test_save_without_schema_raises(self):
        with self.assertRaises(SchemaConflict):
            Lens(self.dir.name).save()


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
        # A record without the sparsely observed fields still matches, but a
        # wrong value for the always-present field a is still reported.
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


class ChecksummedFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)
        self.lens.infer([{"a": 1, "b": "x"}, {"a": "s"}])
        self.lens.save()

    def test_file_carries_version_and_checksum_of_schema(self):
        payload = json.loads(Path(self.dir.name, SCHEMA_FILENAME).read_text())
        self.assertEqual(payload["version"], SCHEMA_VERSION)
        self.assertEqual(len(payload["checksum"]), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in payload["checksum"]))

    def test_checksum_mismatch_raises_and_preserves_memory(self):
        path = Path(self.dir.name, SCHEMA_FILENAME)
        payload = json.loads(path.read_text())
        payload["root"]["count"] += 100
        path.write_text(json.dumps(payload), encoding="utf-8")
        before = self.lens.schema()
        with self.assertRaises(SchemaConflict):
            self.lens.load()
        self.assertEqual(self.lens.schema(), before)

    def test_rewritten_checksum_raises_and_preserves_memory(self):
        path = Path(self.dir.name, SCHEMA_FILENAME)
        payload = json.loads(path.read_text())
        payload["checksum"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        before = self.lens.schema()
        with self.assertRaises(SchemaConflict):
            self.lens.load()
        self.assertEqual(self.lens.schema(), before)

    def test_unknown_version_raises_and_preserves_memory(self):
        path = Path(self.dir.name, SCHEMA_FILENAME)
        payload = json.loads(path.read_text())
        payload["version"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")
        before = self.lens.schema()
        with self.assertRaises(SchemaConflict):
            self.lens.load()
        self.assertEqual(self.lens.schema(), before)

    def test_truncated_half_file_raises_and_preserves_memory(self):
        path = Path(self.dir.name, SCHEMA_FILENAME)
        raw = path.read_text(encoding="utf-8")
        path.write_text(raw[: len(raw) // 2], encoding="utf-8")
        before = self.lens.schema()
        with self.assertRaises(SchemaConflict):
            self.lens.load()
        self.assertEqual(self.lens.schema(), before)

    def test_successful_commit_leaves_no_half_file(self):
        names = set(os.listdir(self.dir.name))
        self.assertIn(SCHEMA_FILENAME, names)
        self.assertNotIn(SCHEMA_FILENAME + ".tmp", names)
        # A stale temp file from a crashed writer never masks the full schema.
        Path(self.dir.name, SCHEMA_FILENAME + ".tmp").write_text("{half", encoding="utf-8")
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.schema(), self.lens.schema())

    def test_save_against_corrupt_disk_raises_and_preserves_memory(self):
        from schema_lens.core import SCHEMA_VERSION, _checksum

        with tempfile.TemporaryDirectory() as clean:
            lens = Lens(clean)
            lens.infer([{"a": 1}])
            lens.save()
            first_batch = lens.schema()
            lens.infer([{"b": 2}])
            memory = lens.schema()
            Path(clean, SCHEMA_FILENAME).write_text("{broken", encoding="utf-8")
            with self.assertRaises(SchemaConflict):
                lens.save()
            self.assertEqual(lens.schema(), memory)
            # Once disk is healthy again, retrying commits the held delta.
            Path(clean, SCHEMA_FILENAME).write_text(
                json.dumps(
                    {
                        "version": SCHEMA_VERSION,
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


@unittest.skipIf(fcntl is None, "flock is only available on POSIX")
class LockTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.holder = Path(self.dir.name, "hold_lock.py")
        self.holder.write_text(_LOCK_HOLDER, encoding="utf-8")

    def test_contended_lock_raises_conflict_without_touching_commit(self):
        lens_dir = self.dir.name
        lens = Lens(lens_dir)
        lens.infer([{"a": 1}])
        lens.save()
        before = Path(lens_dir, SCHEMA_FILENAME).read_bytes()
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
            # The failed commit neither touched the stored file nor rolled
            # back (nor published) the in-memory schema.
            self.assertEqual(
                Path(lens_dir, SCHEMA_FILENAME).read_bytes(), before
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

    def test_two_threads_in_one_process_are_serialized(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.save()

        import threading

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
        out = io.StringIO()
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
        # b seen twice, c once; types only widen across batches.
        self.assertEqual(overflow["observed"], 3)
        self.assertEqual(sorted(overflow["types"]), ["boolean", "number", "string"])
        # The exact field is untouched by the approximation.
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
        # Depth 1 stays exact; the deeper addr subtree is collapsed.
        self.assertNotIn("approximate", user)
        self.assertEqual(user["count"], 2)
        self.assertEqual(user["fields"]["name"]["observed"], 2)
        addr = user["fields"]["addr"]["types"]["object"]
        self.assertTrue(addr["approximate"])
        self.assertNotIn("fields", addr)
        self.assertEqual(addr["count"], 1)
        self.assertEqual(lens.stats()["approximate"], ["$.user.addr"])
        # Checking does not descend into the collapsed subtree.
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

    def test_stats_lists_overflow_location(self):
        lens = Lens(self.dir.name, max_fields=1)
        lens.infer([{"a": 1, "b": 2}])
        self.assertEqual(lens.stats()["approximate"], ['$["*"]'])

    def test_approximation_survives_save_and_load(self):
        lens = Lens(self.dir.name, max_fields=1, max_depth=1)
        lens.infer([{"a": 1, "b": {"c": 2}}])
        lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.schema(), lens.schema())
        self.assertEqual(reloaded.stats()["approximate"], lens.stats()["approximate"])
        # Appending after reload keeps the overflow monotone.
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

    def test_record_stream_is_not_retained(self):
        # Folding a large stream keeps only the bounded statistics tree.
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


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        lens = Lens(tempfile.mkdtemp())
        lens.infer([{"a": 1, "b": "x"}, {"a": "s"}])
        self.root = lens.schema()

    def _write_version_one(self):
        from schema_lens.core import _checksum

        path = Path(self.dir.name, SCHEMA_FILENAME)
        payload = {"version": 1, "checksum": _checksum(self.root), "root": self.root}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    def test_version_one_file_migrates_on_load(self):
        from schema_lens.core import _checksum

        path = self._write_version_one()
        lens = Lens(self.dir.name)
        lens.load()
        self.assertEqual(lens.schema(), self.root)
        migrated = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(migrated["version"], SCHEMA_VERSION)
        self.assertEqual(migrated["checksum"], _checksum(migrated["root"]))
        self.assertEqual(migrated["root"], self.root)

    def test_version_one_file_migrates_on_save_and_keeps_batches(self):
        self._write_version_one()
        lens = Lens(self.dir.name)
        lens.load()
        lens.infer([{"c": True}])
        lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.stats()["records"], 3)
        self.assertEqual(sorted(reloaded.schema()["fields"]), ["a", "b", "c"])
        payload = json.loads(Path(self.dir.name, SCHEMA_FILENAME).read_text())
        self.assertEqual(payload["version"], SCHEMA_VERSION)

    def test_failed_migration_preserves_file_and_memory(self):
        from unittest import mock

        path = self._write_version_one()
        before = path.read_bytes()
        lens = Lens(self.dir.name)
        with mock.patch(
            "schema_lens.core._write_payload", side_effect=OSError("disk full")
        ):
            with self.assertRaises(SchemaConflict):
                lens.load()
        # The original file is untouched and memory holds no schema.
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(SchemaConflict):
            lens.schema()
        # The read can be retried later and migrates cleanly.
        lens.load()
        self.assertEqual(lens.schema(), self.root)
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8"))["version"], SCHEMA_VERSION
        )

    def test_failed_migration_during_save_preserves_memory(self):
        from unittest import mock

        self._write_version_one()
        lens = Lens(self.dir.name)
        lens.infer([{"c": 1}])
        memory = lens.schema()
        with mock.patch(
            "schema_lens.core._write_payload", side_effect=OSError("read-only")
        ):
            with self.assertRaises(SchemaConflict):
                lens.save()
        self.assertEqual(lens.schema(), memory)
        # The version 1 file is still there, so the retry migrates and
        # commits the held delta on top of it.
        lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.stats()["records"], 3)
        self.assertEqual(sorted(reloaded.schema()["fields"]), ["a", "b", "c"])

    def test_migrated_schema_equivalent_to_fresh_fold(self):
        self._write_version_one()
        migrated = Lens(self.dir.name)
        migrated.load()
        fresh = Lens(tempfile.mkdtemp())
        fresh.infer([{"a": 1, "b": "x"}, {"a": "s"}])
        self.assertEqual(migrated.schema(), fresh.schema())


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
        self.assertEqual(stats["approximate"], ['$["*"]'])
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


if __name__ == "__main__":
    unittest.main()
