"""
Unit and Integration Tests for UniqToken CLI.
"""

from __future__ import annotations

import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import uniqtoken.cli as cli


class CLITests(unittest.TestCase):
    def test_cli_train_encode_decode_roundtrip(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corpus_file = tmp / "corpus.txt"
            model_dir = tmp / "model"
            ids_file = tmp / "ids.json"
            decoded_file = tmp / "decoded.txt"

            corpus_content = "\n".join(
                [
                    "the quick brown fox jumps over the lazy dog",
                    "machine learning and natural language processing",
                    "custom tokenizer training verification",
                ]
            )
            corpus_file.write_text(corpus_content, encoding="utf-8")

            # 1. Train
            ret = cli.main(
                [
                    "train",
                    "--corpus",
                    str(corpus_file),
                    "--vocab-size",
                    "320",
                    "--ranking-strategy",
                    "pmi",
                    "--out",
                    str(model_dir),
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTrue((model_dir / "tokenizer.json").exists())

            # 2. Encode to IDs (JSON)
            ret = cli.main(
                [
                    "encode",
                    "--model",
                    str(model_dir),
                    "--input",
                    "the quick brown fox",
                    "--to-ids",
                    "--json",
                    "--out",
                    str(ids_file),
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTrue(ids_file.exists())
            ids_data = json.loads(ids_file.read_text(encoding="utf-8"))
            self.assertIsInstance(ids_data, list)
            self.assertGreater(len(ids_data), 0)

            # 3. Decode
            ret = cli.main(
                [
                    "decode",
                    "--model",
                    str(model_dir),
                    "--input",
                    str(ids_file),
                    "--out",
                    str(decoded_file),
                ]
            )
            self.assertEqual(ret, 0)
            decoded_text = decoded_file.read_text(encoding="utf-8")
            self.assertEqual(decoded_text, "the quick brown fox")

    def test_cli_decode_empty_ids_writes_empty_file(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corpus_file = tmp / "corpus.txt"
            model_dir = tmp / "model"
            decoded_file = tmp / "decoded.txt"
            corpus_file.write_text("test corpus", encoding="utf-8")
            self.assertEqual(
                cli.main(
                    [
                        "train",
                        "--corpus",
                        str(corpus_file),
                        "--vocab-size",
                        "320",
                        "--out",
                        str(model_dir),
                    ]
                ),
                0,
            )
            self.assertEqual(
                cli.main(
                    [
                        "decode",
                        "--model",
                        str(model_dir),
                        "--input",
                        "[]",
                        "--out",
                        str(decoded_file),
                    ]
                ),
                0,
            )
            self.assertEqual(decoded_file.read_text(encoding="utf-8"), "")

    def test_cli_encode_with_metrics_and_offsets(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corpus_file = tmp / "corpus.txt"
            model_dir = tmp / "model"
            metrics_file = tmp / "metrics.json"
            offsets_file = tmp / "offsets.json"

            corpus_file.write_text("the quick brown fox jumps over the lazy dog\n", encoding="utf-8")

            cli.main(
                [
                    "train",
                    "--corpus",
                    str(corpus_file),
                    "--vocab-size",
                    "320",
                    "--out",
                    str(model_dir),
                ]
            )

            # Encode with metrics
            ret = cli.main(
                [
                    "encode",
                    "--model",
                    str(model_dir),
                    "--input",
                    "the quick brown fox",
                    "--with-metrics",
                    "--out",
                    str(metrics_file),
                ]
            )
            self.assertEqual(ret, 0)
            report = json.loads(metrics_file.read_text(encoding="utf-8"))
            self.assertIn("num_tokens", report)
            self.assertIn("bytes_per_token", report)
            self.assertIn("byte_fallback_rate", report)

            # Encode with offsets
            ret = cli.main(
                [
                    "encode",
                    "--model",
                    str(model_dir),
                    "--input",
                    "the quick brown fox",
                    "--with-offsets",
                    "--out",
                    str(offsets_file),
                ]
            )
            self.assertEqual(ret, 0)
            offsets = json.loads(offsets_file.read_text(encoding="utf-8"))
            self.assertIsInstance(offsets, list)
            self.assertIn("start", offsets[0])
            self.assertIn("end", offsets[0])

    def test_cli_superbpe_train(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corpus_file = tmp / "corpus.txt"
            model_dir = tmp / "sbp_model"

            corpus_file.write_text("hello world from uniqtoken superbpe compression\n", encoding="utf-8")

            ret = cli.main(
                [
                    "train",
                    "--corpus",
                    str(corpus_file),
                    "--vocab-size",
                    "320",
                    "--superbpe-merges",
                    "5",
                    "--out",
                    str(model_dir),
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTrue((model_dir / "tokenizer.json").exists())

    def test_cli_train_defaults_ranking_strategy_to_char_savings(self):
        parser = cli.build_parser()
        args = parser.parse_args(["train", "--corpus", "dummy.txt", "--out", "dummy_dir"])
        self.assertEqual(args.ranking_strategy, "char_savings")

    def test_cli_train_with_preset_and_digit_chunking(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corpus_file = tmp / "code_corpus.txt"
            model_dir = tmp / "code_model"

            corpus_file.write_text("x = 0xDEADBEEF + 123456\ny = 0b1010\n", encoding="utf-8")

            ret = cli.main(
                [
                    "train",
                    "--corpus",
                    str(corpus_file),
                    "--vocab-size",
                    "320",
                    "--preset",
                    "code",
                    "--digit-chunk-size",
                    "3",
                    "--out",
                    str(model_dir),
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTrue((model_dir / "tokenizer.json").exists())

    def test_cli_downstream_eval_smoke(self):
        ret = cli.main(["eval-downstream", "--smoke-test"])
        self.assertEqual(ret, 0)

    def test_cli_train_sample_corpus_with_progress_and_no_progress(self):
        sample_path = Path(__file__).parent / "sample.txt"
        self.assertTrue(sample_path.exists(), "tests/sample.txt fixture must exist")

        with TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "sample_model"
            # 1. Train with progress on sample corpus
            ret = cli.main(
                [
                    "train",
                    "--corpus",
                    str(sample_path),
                    "--vocab-size",
                    "500",
                    "--out",
                    str(model_dir),
                ]
            )
            self.assertEqual(ret, 0)
            self.assertTrue((model_dir / "tokenizer.json").exists())

            # 2. Train with --no-progress flag
            no_prog_dir = Path(tmp_dir) / "no_prog_model"
            ret_no_prog = cli.main(
                [
                    "train",
                    "--corpus",
                    str(sample_path),
                    "--vocab-size",
                    "500",
                    "--no-progress",
                    "--out",
                    str(no_prog_dir),
                ]
            )
            self.assertEqual(ret_no_prog, 0)
            self.assertTrue((no_prog_dir / "tokenizer.json").exists())


class CLICompareTests(unittest.TestCase):
    model_dir: Path
    _tmpdir: TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory()
        cls.addClassCleanup(cls._tmpdir.cleanup)
        tmp = Path(cls._tmpdir.name)
        corpus_file = tmp / "compare_corpus.txt"
        cls.model_dir = tmp / "compare_model"
        corpus_file.write_text("the quick brown fox jumps over the lazy dog\n", encoding="utf-8")
        ret = cli.main(["train", "--corpus", str(corpus_file), "--vocab-size", "320", "--out", str(cls.model_dir)])
        assert ret == 0

    def _run_compare(self, argv: list[str]) -> tuple[int, str]:
        buffer = StringIO()
        with redirect_stdout(buffer):
            ret = cli.main(argv)
        return ret, buffer.getvalue()

    def test_compare_parsing_defaults(self):
        parser = cli.build_parser()
        args = parser.parse_args(["compare", "--input", "hello"])
        self.assertEqual(args.models, "uniqtoken,tiktoken")
        self.assertIsNone(args.model)
        self.assertFalse(args.no_color)
        self.assertIs(args.func, cli.compare_command)

    def test_compare_uniqtoken_no_color(self):
        ret, output = self._run_compare(
            [
                "compare",
                "--model",
                str(self.model_dir),
                "--input",
                "the quick brown fox",
                "--models",
                "uniqtoken",
                "--no-color",
            ]
        )
        self.assertEqual(ret, 0)
        self.assertIn("UniqToken:", output)
        self.assertRegex(output, r"\(\d+ tokens\)")
        self.assertNotIn("\033[", output)

    def test_compare_colored_output_has_ansi(self):
        ret, output = self._run_compare(
            ["compare", "--model", str(self.model_dir), "--input", "hello world", "--models", "uniqtoken"]
        )
        self.assertEqual(ret, 0)
        self.assertIn("\033[", output)

    def test_compare_unknown_engine_rejected(self):
        ret, _ = self._run_compare(["compare", "--model", str(self.model_dir), "--input", "hi", "--models", "wat"])
        self.assertEqual(ret, 1)

    def test_compare_missing_tiktoken_skips_gracefully(self):
        with patch.object(cli, "_tiktoken_tokens", return_value=None):
            ret, output = self._run_compare(
                [
                    "compare",
                    "--model",
                    str(self.model_dir),
                    "--input",
                    "hello world",
                    "--models",
                    "uniqtoken,tiktoken",
                    "--no-color",
                ]
            )
        self.assertEqual(ret, 0)
        self.assertIn("UniqToken:", output)
        self.assertIn("skipping tiktoken", output)

    def test_compare_tiktoken_only_unavailable_exits_nonzero(self):
        with patch.object(cli, "_tiktoken_tokens", return_value=None):
            ret, _ = self._run_compare(["compare", "--input", "hello", "--models", "tiktoken", "--no-color"])
        self.assertEqual(ret, 1)

    def test_compare_two_engines_prints_savings(self):
        with patch.object(cli, "_tiktoken_tokens", return_value=["Hello", " world"]):
            ret, output = self._run_compare(
                [
                    "compare",
                    "--model",
                    str(self.model_dir),
                    "--input",
                    "hello world",
                    "--models",
                    "uniqtoken,tiktoken",
                    "--no-color",
                ]
            )
        self.assertEqual(ret, 0)
        self.assertIn("Tiktoken", output)
        self.assertIn("Token Savings:", output)

    def test_compare_tiktoken_uses_bytes_and_treats_specials_literally(self) -> None:
        captured: dict[str, Any] = {}

        class FakeEncoding:
            def encode(self, text: str, disallowed_special: Any = None) -> list[int]:
                captured["text"] = text
                captured["disallowed_special"] = disallowed_special
                return [0, 1]

            def decode_tokens_bytes(self, token_ids: list[int]) -> list[bytes]:
                captured["token_ids"] = list(token_ids)
                return [b"hel", b"lo\xff"]

        fake_tiktoken: Any = SimpleNamespace(get_encoding=lambda name: FakeEncoding())
        with patch.dict(sys.modules, {"tiktoken": fake_tiktoken}):
            pieces = cli._tiktoken_tokens("hello <|endoftext|>")
        self.assertEqual(captured["text"], "hello <|endoftext|>")
        self.assertEqual(captured["disallowed_special"], ())
        self.assertEqual(captured["token_ids"], [0, 1])
        # Lone 0xff byte is escaped, never replaced with U+FFFD.
        self.assertEqual(pieces, ["hel", "lo\\xff"])
        assert pieces is not None
        self.assertNotIn("\ufffd", "".join(pieces))

    def test_compare_without_model_trains_demo_tokenizer(self):
        ret, output = self._run_compare(["compare", "--input", "hello world", "--models", "uniqtoken", "--no-color"])
        self.assertEqual(ret, 0)
        self.assertIn("UniqToken:", output)

    def test_colorize_tokens_unit(self):
        self.assertEqual(cli._colorize_tokens(["a", "b"], False), "[a][b]")
        colored = cli._colorize_tokens(["a", "b"], True)
        self.assertIn("\033[", colored)
        self.assertIn("[a]", colored)
        self.assertIn("[b]", colored)

    def test_colorize_tokens_escapes_terminal_controls(self):
        hostile = cli._colorize_tokens(["\x1b[31mok", "a\rb"], False)
        self.assertEqual(hostile, "[\\x1b[31mok][a\\rb]")
        self.assertNotIn("\x1b", hostile)
        self.assertNotIn("\r", hostile)


class CLIStreamingTrainTests(unittest.TestCase):
    """Streaming train mode must not materialize the corpus and must agree with standard training."""

    def _write_corpus(self, tmp: Path) -> Path:
        corpus_file = tmp / "corpus.txt"
        corpus_file.write_text(
            "the quick brown fox jumps over the lazy dog\nmachine learning and natural language processing\n",
            encoding="utf-8",
        )
        return corpus_file

    def test_streaming_train_matches_non_streaming_vocab(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corpus_file = self._write_corpus(tmp)
            std_dir = tmp / "model_std"
            stream_dir = tmp / "model_stream"
            base_args = ["train", "--corpus", str(corpus_file), "--vocab-size", "320", "--no-progress"]
            self.assertEqual(cli.main([*base_args, "--out", str(std_dir)]), 0)
            self.assertEqual(cli.main([*base_args, "--streaming", "--out", str(stream_dir)]), 0)
            std_ids = json.loads((std_dir / "tokenizer.json").read_text(encoding="utf-8"))["token_to_id"]
            stream_ids = json.loads((stream_dir / "tokenizer.json").read_text(encoding="utf-8"))["token_to_id"]
            self.assertEqual(std_ids, stream_ids)

    def test_streaming_rejects_superbpe_combo(self):
        with TemporaryDirectory() as tmp_dir:
            corpus_file = self._write_corpus(Path(tmp_dir))
            ret = cli.main(
                [
                    "train",
                    "--corpus",
                    str(corpus_file),
                    "--out",
                    str(Path(tmp_dir) / "m"),
                    "--streaming",
                    "--superbpe-merges",
                    "5",
                    "--no-progress",
                ]
            )
            self.assertEqual(ret, 1)

    def test_streaming_rejects_bad_chunk_size(self):
        with TemporaryDirectory() as tmp_dir:
            corpus_file = self._write_corpus(Path(tmp_dir))
            for bad in ("0", "-5"):
                ret = cli.main(
                    [
                        "train",
                        "--corpus",
                        str(corpus_file),
                        "--out",
                        str(Path(tmp_dir) / "m"),
                        "--streaming",
                        "--chunk-size-mb",
                        bad,
                        "--no-progress",
                    ]
                )
                self.assertEqual(ret, 1)


if __name__ == "__main__":
    unittest.main()
