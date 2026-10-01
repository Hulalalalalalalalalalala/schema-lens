"""Unit tests for schema_lens."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from schema_lens import Lens, SchemaConflict
from schema_lens.cli import main


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lens = Lens(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_fold_marks_missing_optional(self):
        schema = self.lens.infer([
            {"id": 1, "tags": ["a"]},
            {"id": 2},
        ])
        id_field = schema["fields"]["id"]
        tags_field = schema["fields"]["tags"]
        self.assertTrue(id_field["required"])
        self.assertEqual(id_field["schema"], {"kind": "integer"})
        self.assertFalse(tags_field["required"])
        self.assertEqual(
            tags_field["schema"],
            {"kind": "array", "items": {"kind": "string"}},
        )

    def test_later_field_is_optional(self):
        schema = self.lens.infer([{"a": 1}, {"a": 2, "b": 3}])
        self.assertTrue(schema["fields"]["a"]["required"])
        self.assertFalse(schema["fields"]["b"]["required"])

    def test_empty_sequence_saves_nothing(self):
        self.assertIsNone(self.lens.infer([]))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "schema.json")))

    def test_empty_array_has_null_items(self):
        schema = self.lens.infer([{"xs": []}])
        self.assertEqual(schema["fields"]["xs"]["schema"],
                         {"kind": "array", "items": None})

    def test_empty_array_does_not_change_items_later(self):
        # Empty first, populated later: items learned from later elements.
        schema = self.lens.infer([{"xs": []}, {"xs": [1]}])
        self.assertEqual(schema["fields"]["xs"]["schema"]["items"],
                         {"kind": "integer"})

    def test_boolean_is_not_integer(self):
        with self.assertRaises(SchemaConflict):
            self.lens.infer([{"v": 1}, {"v": True}])

    def test_integer_widens_to_number(self):
        schema = self.lens.infer([{"v": 1}, {"v": 2.5}])
        self.assertEqual(schema["fields"]["v"]["schema"]["kind"], "number")
        schema = self.lens.infer([{"w": 2.5}, {"w": 1}])
        self.assertEqual(schema["fields"]["w"]["schema"]["kind"], "number")

    def test_shape_morph_conflicts(self):
        with self.assertRaises(SchemaConflict):
            self.lens.infer([{"v": [1]}, {"v": {"x": 1}}])
        with self.assertRaises(SchemaConflict):
            self.lens.infer([{"v": {"x": 1}}, {"v": [1]}])

    def test_leaf_type_conflict(self):
        with self.assertRaises(SchemaConflict):
            self.lens.infer([{"v": "a"}, {"v": 1}])

    def test_nested_object_merge(self):
        schema = self.lens.infer([
            {"a": {"x": 1}},
            {"a": {"x": 2, "y": 3}},
        ])
        a = schema["fields"]["a"]["schema"]
        self.assertTrue(a["fields"]["x"]["required"])
        self.assertFalse(a["fields"]["y"]["required"])

    def test_array_element_merge(self):
        schema = self.lens.infer([
            {"xs": [1, 2]},
            {"xs": [3, 4.5]},
        ])
        self.assertEqual(schema["fields"]["xs"]["schema"]["items"]["kind"],
                         "number")

    def test_stats(self):
        self.lens.infer([
            {"id": 1, "tags": ["a"]},
            {"id": 2},
        ])
        stats = self.lens.stats()
        self.assertEqual(stats["records"], 2)
        self.assertEqual(stats["fields"], ["id", "tags"])
        self.assertEqual(stats["optional_fields"], ["tags"])
        self.assertEqual(stats["observed_types"],
                         {"id": ["integer"], "tags": ["array"]})
        self.assertEqual(stats["field_observations"], {"id": 2, "tags": 1})


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lens = Lens(self.tmp.name)
        self.lens.infer([
            {"id": 1, "tags": ["a"], "meta": {"name": "x"}},
            {"id": 2},
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_record(self):
        self.assertEqual(self.lens.check({"id": 3}), [])

    def test_type_mismatch_and_unexpected(self):
        reports = self.lens.check({"id": "3", "extra": 1})
        self.assertEqual(reports, [
            "type_mismatch:$.id:integer:string",
            "unexpected_field:$.extra",
        ])

    def test_missing_required_path(self):
        reports = self.lens.check({"tags": []})
        self.assertEqual(reports, ["missing_required:$.id"])

    def test_nested_paths_and_array_indices(self):
        reports = self.lens.check({"id": 1, "meta": {"name": 2},
                                   "tags": ["ok", 3]})
        self.assertIn("type_mismatch:$.meta.name:string:integer", reports)
        self.assertIn("type_mismatch:$.tags[1]:string:integer", reports)

    def test_integer_accepted_for_number(self):
        lens = Lens(self.tmp.name)
        lens.infer([{"v": 1.5}])
        self.assertEqual(lens.check({"v": 1}), [])

    def test_boolean_rejected_for_integer(self):
        lens = Lens(self.tmp.name)
        lens.infer([{"v": 1}])
        self.assertEqual(lens.check({"v": False}),
                         ["type_mismatch:$.v:integer:boolean"])

    def test_shape_mismatch_reports_actual_kind(self):
        reports = self.lens.check({"id": 1, "tags": {"x": 1}})
        self.assertEqual(reports, ["type_mismatch:$.tags:array:object"])


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_load_roundtrip(self):
        lens = Lens(self.path)
        before = lens.infer([{"id": 1, "tags": ["a"]}, {"id": 2}])
        Lens(self.path).load()
        reloaded = Lens(self.path)
        reloaded.load()
        self.assertEqual(reloaded.schema(), before)
        with open(os.path.join(self.path, "schema.json"), encoding="utf-8") as f:
            text = f.read()
        self.assertEqual(json.loads(text), before)  # valid JSON file
        self.assertIn("\n", text)

    def test_load_rejects_invalid_schema(self):
        with open(os.path.join(self.path, "schema.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"kind": "weird"}, f)
        lens = Lens(self.path)
        with self.assertRaises(SchemaConflict):
            lens.load()

    def test_load_rejects_bad_json(self):
        with open(os.path.join(self.path, "schema.json"), "w",
                  encoding="utf-8") as f:
            f.write("{not json")
        with self.assertRaises(SchemaConflict):
            Lens(self.path).load()

    def test_missing_file_is_os_error(self):
        with self.assertRaises(FileNotFoundError):
            Lens(self.path).load()

    def test_invalid_field_entry_rejected(self):
        with open(os.path.join(self.path, "schema.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"kind": "object",
                       "fields": {"a": {"required": "yes"}}}, f)
        with self.assertRaises(SchemaConflict):
            Lens(self.path).load()

    def test_second_infer_extends_stored_schema(self):
        Lens(self.path).infer([{"id": 1}])
        Lens(self.path).infer([{"id": 2, "name": "n"}])
        lens = Lens(self.path)
        lens.load()
        self.assertFalse(lens.schema()["fields"]["name"]["required"])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, args, stdin_text):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            old_stdin = sys.stdin
            sys.stdin = io.StringIO(stdin_text)
            try:
                code = main(args)
            finally:
                sys.stdin = old_stdin
        return code, out.getvalue(), err.getvalue()

    def test_end_to_end_exit_codes(self):
        code, _, err = self.run_cli(
            ["--path", self.path, "infer"],
            '{"id": 1, "tags": ["a"]}\n\n{"id": 2}\n',
        )
        self.assertEqual(code, 0, err)

        code, out, _ = self.run_cli(
            ["--path", self.path, "show"], "")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["fields"]["id"]["schema"],
                         {"kind": "integer"})

        code, out, _ = self.run_cli(
            ["--path", self.path, "check"], '{"id": 3}\n')
        self.assertEqual(code, 0)
        self.assertEqual(out, "")

        code, out, _ = self.run_cli(
            ["--path", self.path, "check"], '{"id": "3", "extra": 1}\n')
        self.assertEqual(code, 1)
        lines = out.strip().splitlines()
        self.assertEqual(lines, [
            "type_mismatch:$.id:integer:string",
            "unexpected_field:$.extra",
        ])

    def test_input_errors_exit_2(self):
        self.run_cli(["--path", self.path, "infer"], '{"id": 1}\n')
        for bad, fragment in [
            ('{"id": 1}\n{broken\n', "input_error:2"),  # syntax error (line 2)
            ('{"id": 1, "id": 2}\n', "duplicate"),    # duplicate field
            ('[1, 2]\n', "not an object"),             # non-object
        ]:
            code, out, err = self.run_cli(
                ["--path", self.path, "check"], bad)
            self.assertEqual(code, 2, bad)
            self.assertIn("input_error:", err)
            self.assertIn(fragment, err, (bad, err))

    def test_blank_lines_ignored(self):
        code, _, err = self.run_cli(
            ["--path", self.path, "infer"],
            "\n  \n\t\n",
        )
        self.assertEqual(code, 0, err)

    def test_check_without_schema_is_storage_error(self):
        code, _, err = self.run_cli(
            ["--path", os.path.join(self.path, "missing"), "check"], "{}")
        self.assertEqual(code, 2)
        self.assertIn("storage_error:", err)

    def test_infer_conflict_exit_2(self):
        code, _, err = self.run_cli(
            ["--path", self.path, "infer"], '{"v": 1}\n{"v": "x"}\n')
        self.assertEqual(code, 2)
        self.assertIn("input_error:2", err)

    def test_deviation_and_input_error_precedence(self):
        self.run_cli(["--path", self.path, "infer"], '{"id": 1}\n')
        code, out, err = self.run_cli(
            ["--path", self.path, "check"],
            '{"id": "x"}\nnot-json\n')
        self.assertEqual(code, 2)  # input error wins over deviations
        self.assertIn("type_mismatch:$.id", out)
        self.assertIn("input_error:2", err)


if __name__ == "__main__":
    unittest.main()
