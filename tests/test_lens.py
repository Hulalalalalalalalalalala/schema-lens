"""Tests for schema inference, checking, optionality and persistence."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from schema_lens import Lens, SchemaConflict


class LensTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def lens(self) -> Lens:
        return Lens(os.path.join(self.dir, "lens"))

    # -- inference ---------------------------------------------------------

    def test_merges_types_seen_for_the_same_field(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1}, {"a": "x"}, {"a": True}, {"a": None}])
        field = lens.schema()["fields"]["a"]
        self.assertEqual(field["type"], ["boolean", "null", "number", "string"])
        self.assertEqual(field["count"], 4)
        self.assertFalse(field["optional"])

    def test_bool_is_not_number(self) -> None:
        lens = self.lens()
        lens.infer([{"a": True}])
        self.assertEqual(lens.schema()["fields"]["a"]["type"], ["boolean"])

    def test_field_present_in_every_record_is_required(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1, "b": 2}, {"a": 3}])
        fields = lens.schema()["fields"]
        self.assertFalse(fields["a"]["optional"])
        self.assertTrue(fields["b"]["optional"])
        self.assertEqual(fields["b"]["count"], 1)

    def test_batches_accumulate_for_optionality(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1}])
        lens.infer([{"a": 2}, {"b": 3}])
        fields = lens.schema()["fields"]
        # Seen in 2 of 3 records -> optional across the full folded stream.
        self.assertTrue(fields["a"]["optional"])
        self.assertEqual(fields["a"]["count"], 2)
        self.assertTrue(fields["b"]["optional"])

    def test_nested_objects_merge_by_field_name(self) -> None:
        lens = self.lens()
        lens.infer(
            [
                {"o": {"x": 1, "y": "s"}},
                {"o": {"x": 2}},
            ]
        )
        nested = lens.schema()["fields"]["o"]["fields"]
        self.assertFalse(nested["x"]["optional"])
        self.assertTrue(nested["y"]["optional"])

    def test_array_element_types_are_merged(self) -> None:
        lens = self.lens()
        lens.infer([{"tags": [1, 2]}, {"tags": ["a", None, True]}])
        items = lens.schema()["fields"]["tags"]["items"]
        self.assertEqual(items["type"], ["boolean", "null", "number", "string"])

    def test_empty_array_has_empty_items(self) -> None:
        lens = self.lens()
        lens.infer([{"tags": []}, {"tags": ["a"]}])
        items = lens.schema()["fields"]["tags"]["items"]
        self.assertEqual(items["type"], ["string"])

    # -- checking ----------------------------------------------------------

    def test_check_reports_unknown_field_with_path(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1}])
        issues = lens.check({"a": 1, "b": 2})
        self.assertEqual(issues, ["b: field is not in schema"])

    def test_check_reports_missing_required_field(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        self.assertEqual(lens.check({}), ["a: required field is missing",
                                          "b: required field is missing"])

    def test_check_allows_missing_optional_field(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1}, {"a": 2, "b": 3}])
        self.assertEqual(lens.check({"a": 1}), [])

    def test_check_reports_wrong_type(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1}])
        issues = lens.check({"a": "x"})
        self.assertEqual(len(issues), 1)
        self.assertTrue(issues[0].startswith("a: type 'string'"))

    def test_check_nested_paths_are_unique(self) -> None:
        lens = self.lens()
        lens.infer([{"o": {"x": 1}}, {"tags": [{"y": 1}]}])
        issues = lens.check({"o": {"x": 1, "z": 2}, "tags": [{"y": "nope"}]})
        self.assertIn("o.z: field is not in schema", issues)
        type_issues = [i for i in issues if i.startswith("tags[0].y:")]
        self.assertEqual(len(type_issues), 1)

    def test_check_does_not_mutate_schema(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1}])
        before = lens.schema()
        lens.check({"a": "new", "b": 2})
        self.assertEqual(before, lens.schema())

    def test_check_reports_type_mismatch_inside_array_element(self) -> None:
        lens = self.lens()
        lens.infer([{"tags": [1, 2]}])
        issues = lens.check({"tags": [1, "bad", 3]})
        self.assertEqual(len(issues), 1)
        self.assertTrue(issues[0].startswith("tags[1]: type 'string'"))

    # -- errors before a schema exists ------------------------------------

    def test_check_without_schema_raises(self) -> None:
        with self.assertRaises(SchemaConflict):
            self.lens().check({"a": 1})

    def test_schema_without_schema_raises(self) -> None:
        with self.assertRaises(SchemaConflict):
            self.lens().schema()

    def test_stats_without_schema_raises(self) -> None:
        with self.assertRaises(SchemaConflict):
            self.lens().stats()

    def test_save_without_schema_raises(self) -> None:
        with self.assertRaises(SchemaConflict):
            self.lens().save()

    # -- persistence -------------------------------------------------------

    def test_save_then_load_round_trips(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1, "o": {"x": [1, "s"]}}, {"a": "s"}])
        lens.save()

        reloaded = self.lens()
        reloaded.load()
        self.assertEqual(reloaded.schema(), lens.schema())

        issues = reloaded.check({"a": 1, "o": {"x": [2, "t"]}})
        self.assertEqual(issues, [])
        bad = reloaded.check({"a": []})
        self.assertEqual(len(bad), 1)

    def test_load_missing_file_raises_and_keeps_memory(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1}])
        before = lens.schema()
        with self.assertRaises(SchemaConflict):
            lens.load()
        self.assertEqual(lens.schema(), before)

    def test_load_corrupt_file_raises_and_keeps_memory(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1}])
        lens.save()
        with open(lens.schema_path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        before = lens.schema()
        with self.assertRaises(SchemaConflict):
            lens.load()
        self.assertEqual(lens.schema(), before)

    def test_load_structurally_invalid_schema_raises(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1}])
        lens.save()
        with open(lens.schema_path, encoding="utf-8") as handle:
            good = json.loads(handle.read())
        good["fields"]["a"]["type"] = ["number", "wibble"]
        with open(lens.schema_path, "w", encoding="utf-8") as handle:
            json.dump(good, handle)
        before = lens.schema()
        with self.assertRaises(SchemaConflict):
            lens.load()
        self.assertEqual(lens.schema(), before)

    def test_load_replaces_memory_wholesale(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1}])
        lens.save()

        # Overwrite the persisted file with a different schema entirely.
        other = Lens(os.path.join(self.dir, "other"))
        other.infer([{"zzz": "q"}])
        other.save()
        import shutil

        shutil.copy(other.schema_path, lens.schema_path)

        lens.load()
        self.assertEqual(set(lens.schema()["fields"]), {"zzz"})
        # The old in-memory field must be gone, not merged with the new one.
        self.assertNotIn("a", lens.schema()["fields"])

    def test_non_json_value_raises_type_error(self) -> None:
        lens = self.lens()
        with self.assertRaises(TypeError):
            lens.infer([{"a": object()}])

    # -- stats -------------------------------------------------------------

    def test_stats_reports_fields_optionality_types_counts(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1, "b": 2}, {"a": "x"}])
        stats = lens.stats()
        self.assertEqual(stats["fields"], ["a", "b"])
        self.assertEqual(stats["optional_fields"], ["b"])
        self.assertEqual(stats["types"]["a"], ["number", "string"])
        self.assertEqual(stats["counts"], {"a": 2, "b": 1})


if __name__ == "__main__":
    unittest.main()
