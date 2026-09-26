import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from schema_lens import Lens, SchemaConflict
from schema_lens.__main__ import main as cli_main


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


class IncrementalCommitTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _one_shot(self, batches):
        lens = Lens(self.dir.name)
        lens.infer([record for batch in batches for record in batch])
        return lens.schema()

    def test_repeated_infer_save_equals_one_shot(self):
        batches = [
            [{"a": 1, "b": "x"}, {"a": 2}],
            [{"b": "y", "c": [1, "s"]}],
            [{"a": "str", "c": []}],
        ]
        lens = Lens(self.dir.name)
        for batch in batches:
            lens.infer(batch)
            lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.schema(), self._one_shot(batches))

    def test_separate_instances_accumulate_batches(self):
        first = Lens(self.dir.name)
        first.infer([{"a": 1}])
        first.save()
        second = Lens(self.dir.name)
        second.infer([{"b": 2}, {"a": 3}])
        second.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        schema = reloaded.schema()
        self.assertEqual(schema["count"], 3)
        self.assertEqual(schema["fields"]["a"]["observed"], 2)
        self.assertTrue(schema["fields"]["a"]["optional"])
        self.assertTrue(schema["fields"]["b"]["optional"])

    def test_concurrent_saves_lose_no_batches(self):
        import threading

        workers = 6
        batches_per_worker = 4
        barrier = threading.Barrier(workers)
        failures = []

        def work(tag):
            lens = Lens(self.dir.name)
            try:
                for i in range(batches_per_worker):
                    lens.infer([{"worker": tag, "i": i}])
                    while True:
                        try:
                            lens.save()
                            break
                        except SchemaConflict:
                            continue
            except Exception as exc:  # pragma: no cover - failure path
                failures.append(exc)
            finally:
                barrier.wait()

        threads = [threading.Thread(target=work, args=(t,)) for t in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(failures, [])
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.schema()["count"], workers * batches_per_worker)
        self.assertEqual(
            reloaded.schema()["fields"]["i"]["observed"], workers * batches_per_worker
        )

    def test_lock_contention_raises_and_preserves_memory(self):
        fcntl = _import_fcntl()
        if fcntl is None:
            self.skipTest("fcntl unavailable")
        import os

        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        before = lens.schema()
        lock_path = Path(self.dir.name, "schema.json.lock")
        lens.path.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with self.assertRaises(SchemaConflict):
                lens.save()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        self.assertEqual(lens.schema(), before)
        lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.schema(), before)

    def test_failed_save_can_be_retried(self):
        import os

        fcntl = _import_fcntl()
        if fcntl is None:
            self.skipTest("fcntl unavailable")
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1}])
        lens.path.mkdir(parents=True, exist_ok=True)
        lock_path = Path(self.dir.name, "schema.json.lock")
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with self.assertRaises(SchemaConflict):
                lens.save()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        lens.save()
        lens.infer([{"a": 2}])
        lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.schema()["count"], 2)
        self.assertEqual(reloaded.schema()["fields"]["a"]["observed"], 2)


def _import_fcntl():
    try:
        import fcntl

        return fcntl
    except ImportError:
        return None


class ChecksumTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.file = Path(self.dir.name, "schema.json")

    def _saved_lens(self):
        lens = Lens(self.dir.name)
        lens.infer([{"a": 1, "b": "x"}, {"a": 2}])
        lens.save()
        return lens

    def test_saved_file_carries_version_and_checksum(self):
        self._saved_lens()
        payload = json.loads(self.file.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], 1)
        self.assertIsInstance(payload["checksum"], str)
        self.assertEqual(len(payload["checksum"]), 64)

    def test_tampered_content_fails_checksum_and_preserves_memory(self):
        lens = self._saved_lens()
        before = lens.schema()
        payload = json.loads(self.file.read_text(encoding="utf-8"))
        payload["root"]["count"] = 99
        self.file.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            lens.load()
        self.assertEqual(lens.schema(), before)

    def test_tampered_checksum_fails_and_preserves_memory(self):
        lens = self._saved_lens()
        before = lens.schema()
        payload = json.loads(self.file.read_text(encoding="utf-8"))
        payload["checksum"] = "0" * 64
        self.file.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            lens.load()
        self.assertEqual(lens.schema(), before)

    def test_wrong_version_fails_and_preserves_memory(self):
        lens = self._saved_lens()
        before = lens.schema()
        payload = json.loads(self.file.read_text(encoding="utf-8"))
        payload["version"] = 999
        self.file.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            lens.load()
        self.assertEqual(lens.schema(), before)

    def test_truncated_file_fails_and_preserves_memory(self):
        lens = self._saved_lens()
        before = lens.schema()
        raw = self.file.read_text(encoding="utf-8")
        self.file.write_text(raw[: len(raw) // 2], encoding="utf-8")
        with self.assertRaises(SchemaConflict):
            lens.load()
        self.assertEqual(lens.schema(), before)

    def test_no_tmp_file_left_after_save(self):
        self._saved_lens()
        self.assertFalse(Path(self.dir.name, "schema.json.tmp").exists())
        self.assertTrue(self.file.exists())


class DottedKeyTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens = Lens(self.dir.name)
        self.lens.infer([{"a.b": 1, "a": {"b": 2}, "back\\slash": "q"}])

    def test_dotted_key_is_not_split_into_path_segments(self):
        reports = self.lens.check({"a": {"b": 2}, "back\\slash": "q"})
        self.assertIn('$["a.b"]: missing required field', reports)

    def test_unexpected_dotted_key_uses_bracket_form(self):
        reports = self.lens.check(
            {"a.b": 1, "a": {"b": 2}, "back\\slash": "q", "x.y": 1}
        )
        self.assertIn('$["x.y"]: unexpected field', reports)

    def test_backslash_key_is_uniquely_locatable(self):
        reports = self.lens.check({"a.b": 1, "a": {"b": 2}, "back\\slash": 5})
        self.assertIn(
            '$["back\\\\slash"]: type number not in field types [string]', reports
        )

    def test_dotted_keys_roundtrip_through_disk(self):
        self.lens.save()
        reloaded = Lens(self.dir.name)
        reloaded.load()
        self.assertEqual(reloaded.schema(), self.lens.schema())


class RecordsFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.lens_dir = str(Path(self.dir.name, "lens"))
        self.records = str(Path(self.dir.name, "records.jsonl"))

    def _run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_missing_records_file_fails_with_value_error(self):
        code, _, err = self._run(
            "--path", self.lens_dir, "infer", self.records
        )
        self.assertEqual(code, 2)
        self.assertIn(self.records, err)

    def test_non_object_line_fails_with_line_number(self):
        with open(self.records, "w", encoding="utf-8") as handle:
            handle.write('{"a": 1}\n[1, 2]\n')
        code, _, err = self._run("--path", self.lens_dir, "infer", self.records)
        self.assertEqual(code, 2)
        self.assertIn(f"{self.records}:2", err)

    def test_non_object_line_fails_check_with_line_number(self):
        with open(self.records, "w", encoding="utf-8") as handle:
            handle.write('{"a": 1}\n')
        self._run("--path", self.lens_dir, "infer", self.records)
        with open(self.records, "w", encoding="utf-8") as handle:
            handle.write('{"a": 1}\n"oops"\n')
        code, _, err = self._run("--path", self.lens_dir, "check", self.records)
        self.assertEqual(code, 2)
        self.assertIn(f"{self.records}:2", err)

    def test_cli_infer_appends_across_invocations(self):
        with open(self.records, "w", encoding="utf-8") as handle:
            handle.write('{"a": 1}\n{"a": 2, "b": "x"}\n')
        code, out, _ = self._run("--path", self.lens_dir, "infer", self.records)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["records"], 2)
        with open(self.records, "w", encoding="utf-8") as handle:
            handle.write('{"a": 3, "c": true}\n')
        code, out, _ = self._run("--path", self.lens_dir, "infer", self.records)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["records"], 3)
        code, out, _ = self._run("--path", self.lens_dir, "show")
        self.assertEqual(code, 0)
        schema = json.loads(out)
        self.assertEqual(schema["count"], 3)
        self.assertEqual(schema["fields"]["a"]["observed"], 3)
        self.assertTrue(schema["fields"]["b"]["optional"])
        self.assertTrue(schema["fields"]["c"]["optional"])


if __name__ == "__main__":
    unittest.main()