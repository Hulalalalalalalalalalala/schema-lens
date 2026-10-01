"""Unit tests for the schema_lens package."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from schema_lens import Lens, SchemaConflict


class LensTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def lens(self) -> Lens:
        return Lens(os.path.join(self.dir, "lens"))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def test_worked_example_schema(self) -> None:
        lens = self.lens()
        lens.infer([{"id": 1, "tags": ["a"]}, {"id": 2}])
        schema = lens.schema()
        self.assertEqual(schema["kind"], "object")
        id_field = schema["fields"]["id"]
        self.assertTrue(id_field["required"])
        self.assertEqual(id_field["schema"], {"kind": "integer"})
        tags_field = schema["fields"]["tags"]
        self.assertFalse(tags_field["required"])
        self.assertEqual(
            tags_field["schema"], {"kind": "array", "items": {"kind": "string"}}
        )

    def test_missing_field_becomes_optional_then_merges(self) -> None:
        lens = self.lens()
        lens.infer([{"id": 1}])
        self.assertTrue(lens.schema()["fields"]["id"]["required"])
        lens.infer([{"id": 2, "name": "x"}])
        fields = lens.schema()["fields"]
        self.assertTrue(fields["id"]["required"])
        self.assertFalse(fields["name"]["required"])
        self.assertEqual(fields["name"]["schema"], {"kind": "string"})
        # A field optional once stays optional even when seen again.
        lens.infer([{"id": 3, "name": "y"}])
        self.assertFalse(lens.schema()["fields"]["name"]["required"])

    def test_empty_sequence_leaves_schema_unchanged(self) -> None:
        lens = self.lens()
        before = lens.schema()
        self.assertEqual(lens.infer([]), before)
        lens.infer([{"id": 1}])
        snapshot = json.loads(json.dumps(lens.schema()))
        lens.infer([])
        self.assertEqual(lens.schema(), snapshot)
        self.assertEqual(lens.stats()["records"], 1)

    def test_empty_array_items_is_null_then_merges_elements(self) -> None:
        lens = self.lens()
        lens.infer([{"tags": []}])
        self.assertIsNone(lens.schema()["fields"]["tags"]["schema"]["items"])
        lens.infer([{"tags": []}])
        self.assertIsNone(lens.schema()["fields"]["tags"]["schema"]["items"])
        lens.infer([{"tags": ["a"]}])
        self.assertEqual(
            lens.schema()["fields"]["tags"]["schema"]["items"], {"kind": "string"}
        )

    def test_array_element_types_merge(self) -> None:
        lens = self.lens()
        lens.infer([{"vals": [1, 2]}])
        lens.infer([{"vals": [3]}])
        self.assertEqual(
            lens.schema()["fields"]["vals"]["schema"],
            {"kind": "array", "items": {"kind": "integer"}},
        )

    def test_nested_object_fields_merge(self) -> None:
        lens = self.lens()
        lens.infer([{"user": {"id": 1}}])
        lens.infer([{"user": {"id": 2, "name": "a"}}])
        user = lens.schema()["fields"]["user"]["schema"]
        self.assertEqual(user["kind"], "object")
        self.assertTrue(user["fields"]["id"]["required"])
        self.assertFalse(user["fields"]["name"]["required"])

    def test_integer_widens_to_number_in_either_order(self) -> None:
        lens = self.lens()
        lens.infer([{"v": 1}, {"v": 2.5}])
        self.assertEqual(lens.schema()["fields"]["v"]["schema"]["kind"], "number")
        lens2 = self.lens()
        lens2.infer([{"v": 2.5}, {"v": 1}])
        self.assertEqual(lens2.schema()["fields"]["v"]["schema"]["kind"], "number")

    def test_boolean_is_not_integer(self) -> None:
        lens = self.lens()
        with self.assertRaises(SchemaConflict):
            lens.infer([{"v": True}, {"v": 1}])
        lens2 = self.lens()
        with self.assertRaises(SchemaConflict):
            lens2.infer([{"v": 1}, {"v": False}])

    def test_scalar_type_conflict(self) -> None:
        lens = self.lens()
        with self.assertRaises(SchemaConflict):
            lens.infer([{"v": "a"}, {"v": 1}])

    def test_shape_morph_conflicts(self) -> None:
        lens = self.lens()
        with self.assertRaises(SchemaConflict):
            lens.infer([{"v": {"a": 1}}, {"v": 1}])
        lens2 = self.lens()
        with self.assertRaises(SchemaConflict):
            lens2.infer([{"v": [1]}, {"v": 1}])
        lens3 = self.lens()
        with self.assertRaises(SchemaConflict):
            lens3.infer([{"v": 1}, {"v": [1]}])

    def test_array_element_conflict(self) -> None:
        lens = self.lens()
        lens.infer([{"vals": [1]}])
        with self.assertRaises(SchemaConflict):
            lens.infer([{"vals": ["a"]}])

    def test_conflict_does_not_mutate_schema_or_stats(self) -> None:
        lens = self.lens()
        lens.infer([{"a": 1}])
        snapshot = json.loads(json.dumps(lens.schema()))
        with self.assertRaises(SchemaConflict):
            lens.infer([{"a": "x"}])
        self.assertEqual(lens.schema(), snapshot)
        self.assertEqual(lens.stats()["records"], 1)

    def test_non_object_record_rejected(self) -> None:
        lens = self.lens()
        with self.assertRaises(SchemaConflict):
            lens.infer([[1, 2]])
        with self.assertRaises(SchemaConflict):
            lens.infer([None])

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------
    def test_check_clean_record(self) -> None:
        lens = self.lens()
        lens.infer([{"id": 1, "tags": ["a"]}, {"id": 2}])
        self.assertEqual(lens.check({"id": 3}), [])
        self.assertEqual(lens.check({"id": 3, "tags": []}), [])

    def test_check_type_mismatch_and_unexpected_field(self) -> None:
        lens = self.lens()
        lens.infer([{"id": 1, "tags": ["a"]}, {"id": 2}])
        self.assertEqual(
            lens.check({"id": "3", "extra": 1}),
            ["type_mismatch:$.id:integer:string", "unexpected_field:$.extra"],
        )

    def test_check_missing_required_path(self) -> None:
        lens = self.lens()
        lens.infer([{"id": 1, "tags": ["a"]}])
        self.assertEqual(lens.check({}), ["missing_required:$.id",
                                         "missing_required:$.tags"])

    def test_check_optional_field_may_be_absent(self) -> None:
        lens = self.lens()
        lens.infer([{"id": 1, "tags": ["a"]}, {"id": 2}])
        self.assertEqual(lens.check({"id": 9}), [])

    def test_check_array_index_paths(self) -> None:
        lens = self.lens()
        lens.infer([{"tags": ["a", "b"]}])
        self.assertEqual(
            lens.check({"tags": ["a", 2]}),
            ["type_mismatch:$.tags[1]:string:integer"],
        )

    def test_check_nested_object_path(self) -> None:
        lens = self.lens()
        lens.infer([{"user": {"id": 1}}])
        self.assertEqual(
            lens.check({"user": {"id": "x"}}),
            ["type_mismatch:$.user.id:integer:string"],
        )
        self.assertEqual(lens.check({"user": {}}), ["missing_required:$.user.id"])

    def test_integer_satisfies_number_schema(self) -> None:
        lens = self.lens()
        lens.infer([{"v": 1.5}, {"v": 2}])
        self.assertEqual(lens.check({"v": 10}), [])
        self.assertEqual(
            lens.check({"v": "x"}), ["type_mismatch:$.v:number:string"]
        )

    def test_check_null_and_boolean_kinds(self) -> None:
        lens = self.lens()
        lens.infer([{"a": None, "b": True}])
        self.assertEqual(lens.check({"a": None, "b": False}), [])
        self.assertEqual(
            lens.check({"a": 1, "b": 1}),
            ["type_mismatch:$.a:null:integer",
             "type_mismatch:$.b:boolean:integer"],
        )

    def test_empty_array_schema_accepts_any_elements(self) -> None:
        lens = self.lens()
        lens.infer([{"tags": []}])
        self.assertEqual(lens.check({"tags": [1, "x", None, {}]}), [])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def test_stats_worked_example(self) -> None:
        lens = self.lens()
        lens.infer([{"id": 1, "tags": ["a"]}, {"id": 2}])
        stats = lens.stats()
        self.assertEqual(stats["records"], 2)
        self.assertEqual(stats["fields"], ["$.id", "$.tags"])
        self.assertEqual(stats["optional_fields"], ["$.tags"])
        self.assertEqual(
            stats["observed_types"],
            {
                "$": ["object"],
                "$.id": ["integer"],
                "$.tags": ["array"],
                "$.tags[]": ["string"],
            },
        )
        self.assertEqual(
            stats["field_observations"],
            {"$": 2, "$.id": 2, "$.tags": 1, "$.tags[]": 1},
        )

    def test_stats_counts_every_array_element(self) -> None:
        lens = self.lens()
        lens.infer([{"tags": ["a", "b"]}, {"tags": ["c"]}, {"tags": []}])
        stats = lens.stats()
        self.assertEqual(stats["records"], 3)
        self.assertEqual(stats["field_observations"]["$.tags"], 3)
        self.assertEqual(stats["field_observations"]["$.tags[]"], 3)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def test_save_writes_sorted_utf8_json(self) -> None:
        lens = self.lens()
        # Field names are part of the schema, so a non-ASCII name must be
        # written as readable UTF-8 rather than an \uXXXX escape.
        lens.infer([{"z": 1, "café": ["x"]}])
        lens.save()
        target = os.path.join(self.dir, "lens", "schema.json")
        with open(target, "rb") as handle:
            raw = handle.read()
        self.assertTrue(raw.startswith(b"{"))
        text = raw.decode("utf-8")
        self.assertIn("café", text)
        # Keys sorted: "records" before "schema" before "stats".
        self.assertLess(text.index("records"), text.index("schema"))
        self.assertLess(text.index('"schema"'), text.index("stats"))

    def test_save_load_round_trip(self) -> None:
        lens = self.lens()
        lens.infer([{"id": 1, "tags": ["a"]}, {"id": 2}])
        lens.save()
        reloaded = self.lens()
        reloaded.load()
        self.assertEqual(reloaded.schema(), lens.schema())
        self.assertEqual(reloaded.stats(), lens.stats())
        self.assertEqual(reloaded.check({"id": "x"}),
                         ["type_mismatch:$.id:integer:string"])

    def test_load_missing_file_raises_oserror(self) -> None:
        lens = self.lens()
        with self.assertRaises(OSError):
            lens.load()

    def test_load_invalid_schema_raises_conflict_and_keeps_memory(self) -> None:
        lens = self.lens()
        lens.infer([{"id": 1}])
        good = json.loads(json.dumps(lens.schema()))
        target = os.path.join(self.dir, "lens", "schema.json")

        def corrupt(document: object) -> None:
            lens.save()
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(document, handle)

        bad_documents = [
            {"records": 0, "schema": {"kind": "wat"}},
            {"records": 0, "schema": {"kind": "object"}},  # missing fields
            {"records": 0, "schema": {"kind": "array"}},  # missing items
            {
                "records": 0,
                "schema": {
                    "kind": "object",
                    "fields": {"id": {"required": "yes",
                                      "schema": {"kind": "integer"}}},
                },
            },
            {"records": 0, "schema": {"kind": "integer", "fields": {}}},
            ["not", "an", "object"],
        ]
        for document in bad_documents:
            corrupt(document)
            with self.subTest(document=document):
                with self.assertRaises(SchemaConflict):
                    lens.load()
                self.assertEqual(lens.schema(), good)
                self.assertEqual(lens.stats()["records"], 1)

    def test_load_bad_json_raises_conflict(self) -> None:
        lens = self.lens()
        lens.save()
        target = os.path.join(self.dir, "lens", "schema.json")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaises(SchemaConflict):
            lens.load()

    def test_load_accepts_bare_schema_node(self) -> None:
        lens = self.lens()
        lens.save()
        target = os.path.join(self.dir, "lens", "schema.json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump({"kind": "object", "fields": {}}, handle)
        lens.load()
        self.assertEqual(lens.schema(), {"kind": "object", "fields": {}})

    def test_save_creates_missing_directory(self) -> None:
        lens = Lens(os.path.join(self.dir, "nested", "deeper", "lens"))
        lens.infer([{"id": 1}])
        lens.save()
        self.assertTrue(
            os.path.exists(
                os.path.join(self.dir, "nested", "deeper", "lens", "schema.json")
            )
        )


if __name__ == "__main__":
    unittest.main()
