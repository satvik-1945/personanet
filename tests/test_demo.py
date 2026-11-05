"""Presentation regressions and a true base-model bypass, without model downloads."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

from personanet.traits import one_hot
from personanet.generate import clean_answer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts/11_demo.py")
demo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = demo
spec.loader.exec_module(demo)
ESCAPES = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class DemoTests(unittest.TestCase):
    def test_layout_fills_terminal_and_follows_resize(self):
        with patch.object(demo.shutil, "get_terminal_size") as size:
            size.return_value = os.terminal_size((160, 50))
            ui = demo.DemoUI()
            self.assertEqual(ui.width, 160)
            size.return_value = os.terminal_size((64, 24))
            self.assertEqual(ui.width, 64)
            self.assertTrue(ui.compact)
            self.assertEqual(demo.DemoUI(width=40).width, 40)

    def test_frames_fit_narrow_and_wide_terminals(self):
        for width in (20, 40, 64, 88, 160):
            with self.subTest(width=width), patch.object(
                demo.shutil, "get_terminal_size", return_value=os.terminal_size((width, 40))
            ), patch.object(demo.time, "sleep"), contextlib.redirect_stdout(io.StringIO()) as out:
                ui = demo.DemoUI(auto=True)
                ui.banner()
                ui.box("A long title to fit", demo.GREEN + "verylongword" * 20 + demo.RESET)
                demo.show_frozen_proof(ui, demo.load_frozen_check(ROOT / "experiments/compare/frozen_check.json"))
                demo.show_scoreboard(ui)
                lines = ESCAPES.sub("", out.getvalue()).splitlines()
                self.assertTrue(all(len(line) <= width for line in lines))
                self.assertIn("─" * (width - 2), "\n".join(lines))

    def test_styled_wrap_preserves_text_and_long_words(self):
        text = "Hello " + demo.BOLD + "personality" * 8 + demo.RESET + " world"
        lines = demo.wrap_styled(text, 20)
        self.assertEqual("".join(demo.ANSI_RE.sub("", line) for line in lines), demo.ANSI_RE.sub("", text))
        self.assertTrue(all(len(demo.ANSI_RE.sub("", line)) <= 20 for line in lines))

    def test_missing_proof_cannot_claim_identical_weights(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            demo.show_frozen_proof(demo.DemoUI(), {})
        self.assertIn("NOT VERIFIED", out.getvalue())
        self.assertNotIn("BIT-IDENTICAL: TRUE", out.getvalue())

    def test_bold_preserves_trained_trait_index(self):
        self.assertEqual(one_hot("bold"), one_hot("stud"))
        self.assertNotEqual(one_hot("bold"), one_hot("polite"))

    def test_generation_limit_is_distinct_from_eos_stop(self):
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 99
        tokenizer.decode.return_value = "unfinished"
        limited = demo.decode_result(tokenizer, torch.tensor([1, 2, 3]), 3)
        self.assertTrue(limited.hit_limit)
        finished = demo.decode_result(tokenizer, torch.tensor([1, 2, 99]), 3)
        self.assertFalse(finished.hit_limit)
        shorter = demo.decode_result(tokenizer, torch.tensor([1, 99]), 3)
        self.assertFalse(shorter.hit_limit)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            demo.show_answer(demo.DemoUI(), "vanilla", limited, 0.5)
        self.assertIn("TOKEN BUDGET REACHED", out.getvalue())

    def test_reasoning_cleanup_handles_open_closed_and_prompt_supplied_tags(self):
        cases = [
            ("<think>private trace</think>Final answer", False, "Final answer"),
            ("<think>unfinished private trace", False, ""),
            ("unfinished private trace", True, ""),
            ("private trace</think>Final answer", True, "Final answer"),
            ("Final answer", False, "Final answer"),
            ("<THINK>private trace</THINK>Final answer", False, "Final answer"),
        ]
        for text, starts_in_think, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(clean_answer(text, starts_in_think=starts_in_think), expected)

    def test_decode_hides_prompt_supplied_thinking_and_removes_special_tokens(self):
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 99
        tokenizer.all_special_tokens = ["<eos>", "<think>", "</think>"]
        tokenizer.decode.return_value = "private trace</think>Final answer<eos>"
        result = demo.decode_result(tokenizer, torch.tensor([1, 2, 99]), 8, starts_in_think=True)
        self.assertEqual(result.text, "Final answer")
        self.assertFalse(tokenizer.decode.call_args.kwargs["skip_special_tokens"])
        tokenizer.decode.return_value = "unfinished private trace"
        result = demo.decode_result(tokenizer, torch.tensor([1, 2]), 2, starts_in_think=True)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            demo.show_answer(demo.DemoUI(), "vanilla", result, 1)
        self.assertNotIn("private trace", out.getvalue())
        self.assertIn("No final answer within budget", out.getvalue())

    def test_processing_indicator_cleans_up_on_failure(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with demo.DemoUI().processing("vanilla"):
                    raise RuntimeError("failed")
        self.assertIn("Processing", out.getvalue())
        self.assertTrue(out.getvalue().endswith("\r\033[2K"))

    def test_library_diagnostics_are_replayed_on_failure(self):
        import sys

        with contextlib.redirect_stderr(io.StringIO()) as err:
            with demo.library_diagnostics():
                print("loading chatter", file=sys.stderr)
            self.assertEqual(err.getvalue(), "")
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with demo.library_diagnostics():
                    print("failure details", file=sys.stderr)
                    raise RuntimeError("failed")
            self.assertIn("failure details", err.getvalue())

    def test_vanilla_bypasses_head_and_decodes_only_continuation(self):
        bundle = MagicMock()
        bundle.device = "cpu"
        bundle.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        bundle.model.base.generate.return_value = torch.tensor([[1, 2, 3]])
        bundle.tokenizer.decode.return_value = "A real continuation"
        bundle.tokenizer.apply_chat_template.return_value = "A normal chat prompt"
        answer = demo.generate_one(bundle, "Question?", "vanilla", 80)
        self.assertEqual(answer, "A real continuation")
        bundle.model.generate.assert_not_called()
        call = bundle.model.base.generate.call_args.kwargs
        self.assertNotIn("trait_vec", call)
        self.assertFalse(call["do_sample"])
        self.assertEqual(bundle.tokenizer.decode.call_args.args[0].tolist(), [3])
        messages = bundle.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]["content"], demo.DEFAULT_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
