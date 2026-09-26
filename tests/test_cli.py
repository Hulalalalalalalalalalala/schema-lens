"""End-to-end tests for the ``python -m schema_lens`` command line."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

from schema_lens.__main__ import main


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.lens_dir = os.path.join(self.dir, "lens")
        self.records = os.path.join(self.dir, "records.jsonl")

    def write_records(self, rows: list[dict]) -> None:
        with open(self.records, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_infer_show_check_cycle(self) -> None:
        self.write_records([{"a": 1, "b": "x"}, {"a": 2}])

        code, out, err = self.run_cli("--path", self.lens_dir,
                                      "infer", self.records)
        self.assertEqual(code, 0)
        self.assertIn('"b"', out)

        code, shown, err = self.run_cli("--path", self.lens_dir, "show")
        self.assertEqual(code, 0)
        schema = json.loads(shown)
        self.assertTrue(schema["fields"]["b"]["optional"])
        self.assertFalse(schema["fields"]["a"]["optional"])

        bad_records = os.path.join(self.dir, "bad.jsonl")
        with open(bad_records, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"a": "string!"}) + "\n")
            handle.write(json.dumps({"a": 3}) + "\n")
        code, out, err = self.run_cli("--path", self.lens_dir,
                                      "check", bad_records)
        self.assertEqual(code, 1)
        self.assertIn("record 1", out)
        self.assertIn("a: type 'string'", out)
        self.assertNotIn("record 2", out)

    def test_check_clean_records_exits_zero(self) -> None:
        self.write_records([{"a": 1}])
        self.run_cli("--path", self.lens_dir, "infer", self.records)
        code, out, err = self.run_cli("--path", self.lens_dir,
                                      "check", self.records)
        self.assertEqual(code, 0)
        self.assertEqual(out, "")

    def test_show_without_schema_is_conflict(self) -> None:
        code, out, err = self.run_cli("--path", self.lens_dir, "show")
        self.assertEqual(code, 2)
        self.assertIn("schema conflict", err)


if __name__ == "__main__":
    unittest.main()
