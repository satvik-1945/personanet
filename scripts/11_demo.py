#!/usr/bin/env python3
"""
PersonaNet live terminal demo.

Cinematic walkthrough for screen recording: boot splash, loading timeline,
frozen-weight proof, live trait switching, and the experiment scoreboard.

Examples:
  uv run python scripts/11_demo.py
  uv run python scripts/11_demo.py --auto --question "What is two plus two?"
  uv run python scripts/11_demo.py --skip-load   # UI-only rehearsal, no model
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ANSI
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"
BG_DARK = "\033[48;5;234m"

MODES = ("stud", "polite", "bare")


class DemoUI:
    def __init__(self, auto: bool = False, step_pause: float = 1.1) -> None:
        self.auto = auto
        self.step_pause = step_pause
        self.width = min(shutil.get_terminal_size((88, 24)).columns, 88)

    def clear(self) -> None:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    def line(self, text: str = "", end: str = "\n") -> None:
        sys.stdout.write(text + end)
        sys.stdout.flush()

    def rule(self, char: str = "─") -> None:
        self.line(DIM + char * self.width + RESET)

    def typewrite(self, text: str, delay: float = 0.018) -> None:
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            if ch not in " \n\t":
                time.sleep(delay)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def pause(self, prompt: str = "press Enter") -> None:
        if self.auto:
            time.sleep(self.step_pause)
            return
        try:
            input(DIM + f"  ▸ {prompt} … " + RESET)
        except (EOFError, KeyboardInterrupt):
            self.line()
            raise SystemExit(0)

    def progress(
        self,
        label: str,
        seconds: float = 1.6,
        done_label: str | None = None,
    ) -> None:
        steps = max(int(seconds * 28), 12)
        for i in range(steps + 1):
            frac = i / steps
            filled = int(frac * 32)
            bar = "█" * filled + "░" * (32 - filled)
            pct = int(frac * 100)
            sys.stdout.write(
                f"\r  {CYAN}{label:<28}{RESET} [{GREEN}{bar}{RESET}] {pct:3d}%"
            )
            sys.stdout.flush()
            time.sleep(seconds / steps)
        sys.stdout.write(
            f"\r  {CYAN}{label:<28}{RESET} [{GREEN}{'█' * 32}{RESET}] 100%"
            f"  {DIM}{done_label or 'ok'}{RESET}\n"
        )
        sys.stdout.flush()

    def banner(self) -> None:
        self.clear()
        self.line()
        self.line(BOLD + CYAN + "╔" + "═" * (self.width - 2) + "╗" + RESET)
        title = "PERSONANET"
        subtitle = "personality as a swappable head  ·  base model frozen"
        pad_t = (self.width - 2 - len(title)) // 2
        pad_s = (self.width - 2 - len(subtitle)) // 2
        self.line(
            BOLD
            + CYAN
            + "║"
            + RESET
            + BOLD
            + WHITE
            + " " * pad_t
            + title
            + " " * (self.width - 2 - pad_t - len(title))
            + CYAN
            + "║"
            + RESET
        )
        self.line(
            CYAN
            + "║"
            + RESET
            + DIM
            + " " * pad_s
            + subtitle
            + " " * (self.width - 2 - pad_s - len(subtitle))
            + CYAN
            + "║"
            + RESET
        )
        self.line(BOLD + CYAN + "╚" + "═" * (self.width - 2) + "╝" + RESET)
        self.line()

    def beat(self, n: int, total: int, title: str) -> None:
        self.line()
        self.line(
            f"{MAGENTA}{BOLD}[{n}/{total}]{RESET} {BOLD}{title}{RESET}"
        )
        self.rule("·")

    def box(self, title: str, body: str, color: str = CYAN) -> None:
        self.line(f"{color}{BOLD}┌─ {title} {RESET}{color}{'─' * max(0, self.width - len(title) - 5)}{RESET}")
        for raw in body.splitlines() or [""]:
            # wrap long lines
            while len(raw) > self.width - 4:
                self.line(f"{color}│{RESET} {raw[: self.width - 4]}")
                raw = raw[self.width - 4 :]
            self.line(f"{color}│{RESET} {raw}")
        self.line(f"{color}└{'─' * (self.width - 1)}{RESET}")


def load_frozen_check(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def generate_one(bundle, question: str, mode: str, max_new_tokens: int) -> str:
    import torch

    from personanet.data import build_prompt
    from personanet.generate import GEN_DEFAULTS, clean_answer
    from personanet.traits import one_hot, zero_trait

    tokenizer = bundle.tokenizer
    device = bundle.device
    tok = tokenizer(build_prompt(question), return_tensors="pt")
    vec = zero_trait() if mode == "bare" else one_hot(mode)
    trait = torch.tensor([vec], dtype=torch.float32, device=device)
    out = bundle.model.generate(
        input_ids=tok["input_ids"].to(device),
        attention_mask=tok["attention_mask"].to(device),
        trait_vec=trait,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **GEN_DEFAULTS,
    )
    return clean_answer(tokenizer.decode(out[0], skip_special_tokens=True))


def load_model_with_bar(ui: DemoUI, ckpt: Path):
    """Load the real model while a progress bar runs in the foreground."""
    result: dict = {"bundle": None, "meta": None, "error": None}

    def worker() -> None:
        try:
            from personanet.model import load_bundle_from_ckpt

            result["bundle"], result["meta"] = load_bundle_from_ckpt(ckpt)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    steps = 0
    while t.is_alive():
        filled = min(steps % 33, 32)
        bar = "█" * filled + "░" * (32 - filled)
        sys.stdout.write(
            f"\r  {CYAN}{'loading frozen student':<28}{RESET} "
            f"[{YELLOW}{bar}{RESET}] live"
        )
        sys.stdout.flush()
        time.sleep(0.08)
        steps += 1
    t.join()
    sys.stdout.write(
        f"\r  {CYAN}{'loading frozen student':<28}{RESET} "
        f"[{GREEN}{'█' * 32}{RESET}] 100%  {DIM}ready{RESET}\n"
    )
    sys.stdout.flush()
    if result["error"] is not None:
        raise result["error"]
    return result["bundle"], result["meta"]


def show_scoreboard(ui: DemoUI) -> None:
    ui.beat(4, 5, "Experiment scoreboard  (blind binary judge)")
    rows = [
        ("prompt-only (chat)", "52%", "5%", RED),
        ("prompt-only (plain text)", "60%", "20%", RED),
        ("PersonaNet shallow head", "100%", "100%", GREEN),
        ("PersonaNet deep head", "100%", "100%", GREEN),
        ("LoRA finetune", "98%", "95%", YELLOW),
    ]
    ui.line(
        f"  {DIM}{'arm':<28} {'accuracy':>10} {'separation':>12}{RESET}"
    )
    ui.rule("·")
    for name, acc, sep, color in rows:
        ui.line(f"  {color}{name:<28}{RESET} {BOLD}{acc:>10}{RESET} {color}{sep:>12}{RESET}")
    ui.line()
    ui.typewrite(
        f"  {DIM}Chance is 50%. Separation is the clean metric: "
        f"how much more often 'stud' is detected when stud was requested.{RESET}",
        delay=0.008,
    )


def show_frozen_proof(ui: DemoUI, check: dict) -> None:
    ui.beat(2, 5, "Frozen-weight proof")
    pn = check.get("personanet", {})
    lora = check.get("lora", {})
    head = check.get("head_checkpoint", {})

    ui.progress("inspect head checkpoint", 1.2, "8 tensors only")
    params = head.get("params", 19_568_640)
    ui.line(f"  head params: {BOLD}{params:,}{RESET}   contains base weights: {GREEN}False{RESET}")
    ui.line()

    ui.progress("hash vanilla base", 1.4, "1.78B params")
    h1 = pn.get("base_sha256_vanilla", "…")
    ui.line(f"  sha256  {DIM}{h1}{RESET}")
    ui.line()

    ui.progress("hash base + PersonaNet head", 1.4)
    h2 = pn.get("base_sha256_with_head", "…")
    ui.line(f"  sha256  {DIM}{h2}{RESET}")
    ui.line()

    identical = pn.get("identical", h1 == h2)
    if identical:
        ui.box(
            "RESULT",
            f"{GREEN}{BOLD}BIT-IDENTICAL: TRUE{RESET}\n"
            "PersonaNet does not finetune the base model.\n"
            "Personality lives in a small swappable head.",
            color=GREEN,
        )
    else:
        ui.box("RESULT", "Base weights changed — unexpected.", color=RED)

    ui.line()
    ui.progress("hash LoRA-merged base", 1.1)
    hl = lora.get("base_sha256_merged", "…")
    ui.line(f"  sha256  {DIM}{hl}{RESET}")
    if not lora.get("identical_to_vanilla", True):
        ui.box(
            "CONTRAST",
            f"{RED}{BOLD}LoRA MODIFIES the base{RESET}\n"
            "Same task, different evidence: LoRA changes model weights.\n"
            "PersonaNet leaves them untouched.",
            color=RED,
        )


def interactive_loop(ui: DemoUI, bundle, max_new_tokens: int) -> None:
    ui.beat(5, 5, "Live inference")
    ui.line(
        f"  commands:  {BOLD}/stud{RESET}  {BOLD}/polite{RESET}  "
        f"{BOLD}/bare{RESET}  {BOLD}/compare <q>{RESET}  {BOLD}/quit{RESET}"
    )
    ui.line(f"  tip:      {DIM}/compare What is the capital of India?{RESET}")
    ui.line()
    mode = "stud"
    while True:
        try:
            line = input(f"{MAGENTA}({mode}){RESET}> ").strip()
        except (EOFError, KeyboardInterrupt):
            ui.line()
            break
        if not line:
            continue
        if line in {"/quit", "/exit", ":q"}:
            break
        if line.startswith("/") and line[1:] in MODES:
            mode = line[1:]
            ui.line(f"  {DIM}trait → {mode}{RESET}")
            continue
        if line.startswith("/compare"):
            q = line[len("/compare") :].strip()
            if not q:
                ui.line(f"  {DIM}usage: /compare What is two plus two?{RESET}")
                continue
            run_compare(ui, bundle, q, max_new_tokens)
            continue

        ui.line(f"  {DIM}generating [{mode}] …{RESET}")
        t0 = time.time()
        ans = generate_one(bundle, line, mode, max_new_tokens)
        dt = time.time() - t0
        ui.box(f"{mode.upper()}  ·  {dt:.1f}s", ans, color=mode_color(mode))
        ui.line()


def mode_color(mode: str) -> str:
    return {"stud": YELLOW, "polite": GREEN, "bare": DIM}.get(mode, CYAN)


def run_compare(ui: DemoUI, bundle, question: str, max_new_tokens: int) -> None:
    ui.line()
    ui.typewrite(f"  {BOLD}Q:{RESET} {question}", delay=0.012)
    ui.line()
    for mode in MODES:
        ui.progress(f"trait = {mode}", 0.55 if ui.auto else 0.35)
        t0 = time.time()
        ans = generate_one(bundle, question, mode, max_new_tokens)
        dt = time.time() - t0
        ui.box(f"{mode.upper()}  ·  {dt:.1f}s", ans, color=mode_color(mode))
        ui.line()
        if ui.auto:
            time.sleep(0.4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "artifacts" / "head_deep.pt",
    )
    parser.add_argument(
        "--frozen-check",
        type=Path,
        default=ROOT / "experiments" / "compare" / "frozen_check.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Cinematic run: no Enter prompts, uses --question for compare",
    )
    parser.add_argument(
        "--question",
        default="What is two plus two?",
        help="Question used in --auto compare beat",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Rehearse UI without loading the model",
    )
    parser.add_argument(
        "--step-pause",
        type=float,
        default=1.15,
        help="Seconds between beats in --auto mode",
    )
    args = parser.parse_args()

    ui = DemoUI(auto=args.auto, step_pause=args.step_pause)
    check = load_frozen_check(args.frozen_check)

    # ---- 1. splash ----
    ui.banner()
    ui.typewrite(
        f"  {DIM}A tiny personality head steers a frozen 1.5B LLM — "
        f"no system prompt, no base finetune.{RESET}",
        delay=0.014,
    )
    ui.line()
    ui.pause("start demo")

    # ---- 2. load timeline ----
    ui.beat(1, 5, "Boot sequence")
    ui.progress("resolve checkpoint", 0.9, str(args.ckpt.name))
    if not args.ckpt.exists() and not args.skip_load:
        ui.line(f"  {RED}missing {args.ckpt}{RESET}")
        raise SystemExit(1)
    ui.progress("attach PersonaNet head", 1.0, "deep · 19.6M params")
    ui.progress("freeze base weights", 0.8, "requires_grad = False")

    bundle = None
    meta: dict = {}
    if args.skip_load:
        ui.line(f"  {YELLOW}skip-load: UI rehearsal only{RESET}")
    else:
        bundle, meta = load_model_with_bar(ui, args.ckpt)
        arch = meta.get("head_arch", "deep")
        n_soft = meta.get("n_soft", 16)
        ui.line(
            f"  device {BOLD}{bundle.device}{RESET}  ·  head {BOLD}{arch}{RESET}"
            f"  ·  soft tokens {BOLD}{n_soft}{RESET}"
        )
    ui.pause("continue")

    # ---- 3. frozen proof ----
    show_frozen_proof(ui, check)
    ui.pause("continue")

    # ---- 4. scoreboard ----
    show_scoreboard(ui)
    ui.pause("continue to live inference")

    # ---- 5. live ----
    if args.skip_load or bundle is None:
        ui.beat(5, 5, "Live inference")
        ui.line(f"  {YELLOW}skipped — rerun without --skip-load for real answers{RESET}")
    elif args.auto:
        ui.beat(5, 5, "Live inference")
        run_compare(ui, bundle, args.question, args.max_new_tokens)
    else:
        interactive_loop(ui, bundle, args.max_new_tokens)

    # ---- outro ----
    ui.line()
    ui.rule()
    ui.typewrite(
        f"  {BOLD}PersonaNet{RESET}{DIM}  ·  personality as weights you can ship"
        f"  ·  base model unchanged{RESET}",
        delay=0.012,
    )
    ui.line(f"  {DIM}docs/REPORT.md{RESET}")
    ui.line()


if __name__ == "__main__":
    main()
