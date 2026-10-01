"""End-to-end tests for the ``python -m schema_lens`` CLI."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest

from schema_lens.__main__ import main  # noqa: F401  (ensures importability)


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "lens")

    def _run(self, command: str, stdin: str = "") -> tuple[int, str]:
        import contextlib

        from schema_lens import __main__ as cli

        reader = io.TextIOWrapper(io.BytesIO(stdin.encode("utf-8")), encoding="utf-8")
        out = io.StringIO()
        old_stdin = os.sys.stdin
        os.sys.stdin = reader
        try:
            with contextlib.redirect_stdout(out):
                try:
                    code = cli.main(["--path", self.path, command])
                except SystemExit as exc:
                    code = exc.code
        finally:
            os.sys.stdin = old_stdin
        return code, out.getvalue()

    def test_worked_example(self) -> None:
        records = '{"id": 1, "tags": ["a"]}\n{"id": 2}\n'
        code, output = self._run("infer", records)
        self.assertEqual(code, 0)
        self.assertEqual(output, "")

        code, output = self._run("show")
        self.assertEqual(code, 0)
        schema = json.loads(output)
        self.assertTrue(schema["fields"]["id"]["required"])
        self.assertEqual(schema["fields"]["id"]["schema"], {"kind": "integer"})
        self.assertFalse(schema["fields"]["tags"]["required"])
        self.assertEqual(
            schema["fields"]["tags"]["schema"],
            {"kind": "array", "items": {"kind": "string"}},
        )

        code, output = self._run("check", '{"id": 3}\n')
        self.assertEqual(code, 0)
        self.assertEqual(output, "")

        code, output = self._run("check", '{"id": "3", "extra": 1}\n')
        self.assertEqual(code, 1)
        self.assertEqual(
            output.splitlines(),
            ["type_mismatch:$.id:integer:string", "unexpected_field:$.extra"],
        )

    def test_blank_lines_ignored(self) -> None:
        code, output = self._run("infer", "\n  \n" + '{"id": 1}\n' + "\n")
        self.assertEqual((code, output), (0, ""))

    def test_input_errors_exit_2_with_line_numbers(self) -> None:
        self._run("infer", '{"id": 1}\n')
        stream = (
            '{"id": 1}\n'
            "not json\n"
            "[1, 2]\n"
            '{"a": 1, "a": 2}\n'
            '{"id": NaN}\n'
        )
        code, output = self._run("check", stream)
        self.assertEqual(code, 2)
        lines = output.splitlines()
        self.assertTrue(lines[0].startswith("input_error:2:"))
        self.assertEqual(lines[1], "input_error:3:record is not a JSON object")
        self.assertTrue(lines[2].startswith("input_error:4:duplicate field: a"))
        self.assertTrue(lines[3].startswith("input_error:5:"))

    def test_check_deviations_exit_1(self) -> None:
        self._run("infer", '{"id": 1}\n')
        code, output = self._run("check", '{"id": "x"}\n{"id": 2}\n')
        self.assertEqual(code, 1)
        self.assertEqual(output, "type_mismatch:$.id:integer:string\n")

    def test_infer_conflict_exit_2_and_schema_saved_from_good_records(self) -> None:
        code, output = self._run("infer", '{"v": 1}\n{"v": "x"}\n{"v": 2}\n')
        self.assertEqual(code, 2)
        self.assertIn("input_error:2:type conflict", output)
        # Records before and after the conflicting line were folded in.
        _, show = self._run("show")
        self.assertEqual(json.loads(show)["fields"]["v"]["schema"],
                         {"kind": "integer"})

    def test_check_without_schema_is_storage_error(self) -> None:
        code, output = self._run("check", '{"id": 1}\n')
        self.assertEqual(code, 2)
        self.assertTrue(output.startswith("storage_error:"))

    def test_show_without_schema_is_storage_error(self) -> None:
        code, _ = self._run("show")
        self.assertEqual(code, 2)

    def test_infer_empty_input_still_writes_empty_schema(self) -> None:
        code, _ = self._run("infer", "")
        self.assertEqual(code, 0)
        code, output = self._run("show")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output), {"kind": "object", "fields": {}})


if __name__ == "__main__":
    unittest.main()
