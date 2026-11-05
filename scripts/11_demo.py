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
import contextlib
import io
import json
import re
import shutil
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
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

MODES = ("vanilla", "bold", "polite")
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
SAMPLE_QUESTIONS = (
    "Why do droplets appear on the outside of a cold glass?",
    "Why does a metal spoon feel colder than a wooden spoon in the same room?",
    "What does airplane mode on a phone actually turn off?",
    "Why is using a password manager better than reusing one password?",
    "Why does bread go stale faster in the fridge?",
    "Why does leftover pizza turn soggy in the microwave?",
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@contextlib.contextmanager
def library_diagnostics(verbose: bool = False):
    """Keep library chatter off recording frames; replay it if an operation fails."""
    if verbose:
        yield
        return
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(buffer):
            yield
    except BaseException:
        sys.stderr.write(buffer.getvalue())
        raise


def wrap_styled(text: str, width: int) -> list[str]:
    """Wrap visible text, keeping ANSI styles out of the column budget."""
    tokens = re.findall(r"\x1b\[[0-9;]*m|[^\x1b]", text)
    plain = ANSI_RE.sub("", text)
    chunks = textwrap.wrap(
        plain, width=max(1, width), expand_tabs=False,
        replace_whitespace=False, drop_whitespace=False,
    ) or [""]
    result = []
    index = 0
    style = ""
    for chunk in chunks:
        rendered = style
        count = 0
        while index < len(tokens):
            token = tokens[index]
            if ANSI_RE.fullmatch(token):
                style = "" if token == RESET else style + token
            elif count == len(chunk):
                break
            else:
                count += 1
            rendered += token
            index += 1
        result.append(rendered + RESET)
    return result


class DemoUI:
    def __init__(
        self,
        auto: bool = False,
        step_pause: float = 1.1,
        width: int | None = None,
        compact: bool = False,
        verbose: bool = False,
    ) -> None:
        self.auto = auto
        self.step_pause = step_pause
        self.max_width = width
        self.force_compact = compact
        self.verbose = verbose

    @property
    def width(self) -> int:
        columns = shutil.get_terminal_size((88, 24)).columns
        return max(4, min(self.max_width or columns, columns))

    @property
    def compact(self) -> bool:
        return (
            self.force_compact
            or shutil.get_terminal_size((88, 24)).lines < 34
            or self.width < 76
        )

    def clear(self) -> None:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    def line(self, text: str = "", end: str = "\n") -> None:
        lines = [line for raw in text.split("\n") for line in wrap_styled(raw, self.width)]
        sys.stdout.write("\n".join(lines) + end)
        sys.stdout.flush()

    def rule(self, char: str = "─") -> None:
        self.line(DIM + char * self.width + RESET)

    def typewrite(self, text: str, delay: float = 0.018) -> None:
        for raw in text.split("\n"):
            for line in wrap_styled(raw, self.width):
                for token in re.findall(r"\x1b\[[0-9;]*m|[^\x1b]", line):
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    if len(token) == 1 and not token.isspace():
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

    @contextlib.contextmanager
    def processing(self, mode: str):
        """Animate a single status line; never stream model reasoning."""
        finished = threading.Event()
        started = time.monotonic()
        frame = 0

        def render() -> None:
            nonlocal frame
            elapsed = time.monotonic() - started
            status = f"  {'|/-'[frame % 3]} {mode.upper()} · Processing… {elapsed:.1f}s"
            sys.stdout.write(f"\r\033[2K{CYAN}{status[:self.width]}{RESET}")
            sys.stdout.flush()
            frame += 1

        def animate() -> None:
            while not finished.wait(0.1):
                render()

        render()
        animation = threading.Thread(target=animate, daemon=True)
        animation.start()
        try:
            yield
        finally:
            finished.set()
            animation.join()
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()

    def progress(
        self,
        label: str,
        seconds: float = 1.6,
        done_label: str | None = None,
    ) -> None:
        steps = max(int(seconds * 28), 12)
        for i in range(steps + 1):
            frac = i / steps
            pct = int(frac * 100)
            sys.stdout.write(self.progress_text(label, pct, GREEN))
            sys.stdout.flush()
            time.sleep(seconds / steps)
        sys.stdout.write(self.progress_text(label, 100, GREEN, done_label or "ok") + "\n")
        sys.stdout.flush()

    def progress_text(
        self,
        label: str,
        pct: int,
        color: str,
        suffix: str = "",
    ) -> str:
        width = self.width
        suffix = suffix[:max(0, width // 3)]
        if width < 64:
            room = max(0, width - 10 - len(suffix))
            text = f"  {label[:room]} {pct:3d}% {suffix}".strip()
            return f"\r\033[2K{CYAN}{text[:width]}{RESET}"

        label_width = min(28, width // 3)
        bar_width = max(1, width - label_width - len(suffix) - 13)
        filled = int(pct / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        return (
            f"\r\033[2K  {CYAN}{label[:label_width]:<{label_width}}{RESET} "
            f"[{color}{bar}{RESET}] {pct:3d}%  {DIM}{suffix}{RESET}"
        )

    def digest(self, value: str) -> str:
        if not self.compact or len(value) <= 32:
            return value
        return f"{value[:16]}…{value[-8:]}"

    def banner(self) -> None:
        self.clear()
        self.line()
        self.line(BOLD + CYAN + "╔" + "═" * (self.width - 2) + "╗" + RESET)
        title = "PERSONANET"[:self.width - 2]
        subtitle = (
            "swappable personality · frozen base"
            if self.compact
            else "personality as a swappable head  ·  base model frozen"
        )
        subtitle = subtitle[:self.width - 2]
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
        self.clear()
        self.line()
        self.line(
            f"{MAGENTA}{BOLD}[{n}/{total}]{RESET} {BOLD}{title}{RESET}"
        )
        self.rule("·")

    def box(self, title: str, body: str, color: str = CYAN) -> None:
        title = title[:max(0, self.width - 5)]
        self.line(f"{color}{BOLD}┌─ {title} {RESET}{color}{'─' * max(0, self.width - len(title) - 5)}┐{RESET}")
        for raw in body.splitlines() or [""]:
            wrapped = wrap_styled(raw, self.width - 4)
            for line in wrapped:
                padding = self.width - 4 - len(ANSI_RE.sub("", line))
                self.line(f"{color}│{RESET} {line}{' ' * padding} {color}│{RESET}")
        self.line(f"{color}└{'─' * (self.width - 2)}┘{RESET}")


def load_frozen_check(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


@dataclass
class GenerationResult:
    text: str
    tokens: int
    budget: int
    hit_limit: bool


def decode_result(
    tokenizer, token_ids, budget: int, *, starts_in_think: bool = False,
) -> GenerationResult:
    from personanet.generate import clean_answer

    tokens = token_ids.tolist()
    eos = tokenizer.eos_token_id
    stop_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
    hit_limit = len(tokens) >= budget and (not tokens or tokens[-1] not in stop_ids)
    # Preserve reasoning delimiters until after channel cleanup. Some tokenizers
    # classify them as special tokens and would strip them too early.
    answer = clean_answer(
        tokenizer.decode(token_ids, skip_special_tokens=False),
        starts_in_think=starts_in_think,
    )
    for special in tokenizer.all_special_tokens:
        answer = answer.replace(special, "")
    return GenerationResult(
        text=answer.strip(),
        tokens=len(tokens), budget=budget, hit_limit=hit_limit,
    )


def generate_result(
    bundle, question: str, mode: str, max_new_tokens: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> GenerationResult:
    import torch

    from personanet.data import build_prompt
    from personanet.generate import GEN_DEFAULTS
    from personanet.traits import one_hot, zero_trait

    tokenizer = bundle.tokenizer
    device = bundle.device
    if mode == "vanilla":
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
        tok = tokenizer(text, return_tensors="pt")
        input_ids = tok["input_ids"].to(device)
        with torch.inference_mode():
            out = bundle.model.base.generate(
                input_ids=input_ids,
                attention_mask=tok["attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **GEN_DEFAULTS,
            )
        starts_in_think = text.lower().rfind("<think>") > text.lower().rfind("</think>")
        return decode_result(
            tokenizer, out[0][input_ids.shape[1]:], max_new_tokens,
            starts_in_think=starts_in_think,
        )
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
    return decode_result(tokenizer, out[0], max_new_tokens)


def generate_one(
    bundle, question: str, mode: str, max_new_tokens: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    return generate_result(bundle, question, mode, max_new_tokens, system_prompt).text


def show_answer(ui: DemoUI, mode: str, result: GenerationResult, seconds: float) -> None:
    ui.box(
        f"{mode.upper()} · {seconds:.1f}s · {result.tokens} tokens",
        result.text or "[No final answer within budget · reasoning hidden]", color=mode_color(mode),
    )
    if mode == "vanilla":
        ui.line(f"  {DIM}Final answer only · reasoning hidden · token count includes processing.{RESET}")
    if result.hit_limit:
        ui.line(f"  {YELLOW}TOKEN BUDGET REACHED ({result.tokens}/{result.budget}) · output may be incomplete.{RESET}")
        ui.line("  Rerun with a larger --max-new-tokens value for more room.")
        if mode == "vanilla":
            ui.line("  R1 may use the budget for reasoning before reaching its final answer.")


def load_model_with_bar(ui: DemoUI, ckpt: Path):
    """Load the real model while a progress bar runs in the foreground."""
    result: dict = {"bundle": None, "meta": None, "error": None}

    def worker() -> None:
        try:
            with library_diagnostics(ui.verbose):
                from personanet.model import load_bundle_from_ckpt

                result["bundle"], result["meta"] = load_bundle_from_ckpt(ckpt)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    steps = 0
    while t.is_alive():
        status = f"  {'|/-'[steps % 3]} Loading frozen base + head · {steps * 0.08:.1f}s"
        sys.stdout.write(f"\r\033[2K{YELLOW}{status[:ui.width]}{RESET}")
        sys.stdout.flush()
        time.sleep(0.08)
        steps += 1
    t.join()
    sys.stdout.write("\r\033[2K")
    if result["error"] is not None:
        raise result["error"]
    ui.line(f"  {GREEN}Frozen base + personality head loaded{RESET}")
    return result["bundle"], result["meta"]


def show_scoreboard(ui: DemoUI) -> None:
    ui.beat(3, 4, "Recorded evaluation · blind binary judge")
    ui.line(f"  {DIM}20 held-out questions / arm · Qwen2.5:7b judge · docs/REPORT.md{RESET}")
    rows = [
        ("prompt-only (chat)", "52%", "5%", RED),
        ("prompt-only (plain text)", "60%", "20%", RED),
        ("PersonaNet shallow head", "100%", "100%", GREEN),
        ("PersonaNet deep head", "100%", "100%", GREEN),
        ("LoRA finetune", "98%", "95%", YELLOW),
    ]
    name_width = max(4, min(28, ui.width - 16))
    ui.line(f"  {DIM}{'arm':<{name_width}} {'acc':>5} {'sep':>5}{RESET}")
    ui.rule("·")
    for name, acc, sep, color in rows:
        short_name = name.replace("PersonaNet ", "PN ") if ui.compact else name
        ui.line(
            f"  {color}{short_name[:name_width]:<{name_width}}{RESET} "
            f"{BOLD}{acc:>5}{RESET} {color}{sep:>5}{RESET}"
        )
    ui.line()
    explanation = (
        "  Chance: 50%. Separation measures whether changing the trait changes the answer."
        if ui.compact
        else "  Chance is 50%. Separation measures how much more often "
        "Bold is detected when Bold was requested rather than Polite."
    )
    ui.typewrite(f"{DIM}{explanation}{RESET}", delay=0.008)


def show_frozen_proof(ui: DemoUI, check: dict) -> None:
    ui.beat(2, 4, "Frozen-weight proof · saved verification")
    pn = check.get("personanet", {})
    lora = check.get("lora", {})
    head = check.get("head_checkpoint", {})
    h1 = pn.get("base_sha256_vanilla")
    h2 = pn.get("base_sha256_with_head")
    ui.line(f"  {DIM}Saved artifact: {head.get('path', 'unavailable')}{RESET}")
    ui.line("  Hashes below are recorded evidence, not a live hashing operation.")
    if not h1 or not h2 or "identical" not in pn:
        ui.box("NOT VERIFIED", "No complete saved proof. Run scripts/10_verify_frozen.py.", YELLOW)
        return
    ui.line(f"  head params: {head.get('params', 'unknown')} · base weights stored: {head.get('contains_base_weights', 'unknown')}")
    ui.line(f"  vanilla SHA256:   {DIM}{ui.digest(h1)}{RESET}")
    ui.line(f"  with head SHA256: {DIM}{ui.digest(h2)}{RESET}")
    if pn["identical"] is True and h1 == h2:
        ui.box(
            "RESULT",
            f"{GREEN}{BOLD}BIT-IDENTICAL: TRUE{RESET}\n"
            "PersonaNet does not finetune the base model.\n"
            "Personality lives in a small swappable head.",
            color=GREEN,
        )
    else:
        ui.box("RESULT", "Base weights changed — unexpected.", color=RED)

    if lora.get("identical_to_vanilla") is False:
        ui.box(
            "LoRA comparison",
            "Merged LoRA changes the effective base weights.\n"
            "LoRA also freezes original weights during adapter training.\n"
            "PersonaNet instead learns a separate soft-prefix head.",
            color=YELLOW,
        )


def interactive_loop(ui: DemoUI, bundle, max_new_tokens: int, system_prompt: str) -> None:
    ui.beat(4, 4, "Live inference")
    ui.line(
        f"  commands:  {BOLD}/bold{RESET}  {BOLD}/polite{RESET}  "
        f"{BOLD}/vanilla{RESET}  {BOLD}/compare <q>{RESET}  "
        f"{BOLD}/examples{RESET}  {BOLD}/quit{RESET}"
    )
    ui.line(f"  tip:      {DIM}/compare What is the capital of India?{RESET}")
    ui.line(f"  {DIM}/examples lists fresh questions for rehearsal.{RESET}")
    ui.line()
    mode = "bold"
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
        if line == "/examples":
            ui.line("  Sample questions (absent from the local data snapshot checked on 2026-11-05):")
            for question in SAMPLE_QUESTIONS:
                ui.line(f"  /compare {question}")
            continue
        if line.startswith("/") and line[1:] in (*MODES, "stud", "bare"):
            mode = "bold" if line == "/stud" else line[1:]
            ui.line(f"  {DIM}trait → {mode}{RESET}")
            continue
        if line.startswith("/compare"):
            q = line[len("/compare") :].strip()
            if not q:
                ui.line(f"  {DIM}usage: /compare What is two plus two?{RESET}")
                continue
            run_compare(ui, bundle, q, max_new_tokens, system_prompt)
            continue

        t0 = time.time()
        with ui.processing(mode), library_diagnostics(ui.verbose):
            result = generate_result(bundle, line, mode, max_new_tokens, system_prompt)
        dt = time.time() - t0
        show_answer(ui, mode, result, dt)
        ui.line()


def mode_color(mode: str) -> str:
    return {"bold": YELLOW, "polite": GREEN, "vanilla": CYAN, "bare": DIM}.get(mode, CYAN)


def run_compare(
    ui: DemoUI, bundle, question: str, max_new_tokens: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> None:
    for index, mode in enumerate(MODES, 1):
        ui.beat(4, 4, f"Same question · {index}/3 · {mode.upper()}")
        ui.typewrite(f"  {BOLD}Q:{RESET} {question}", delay=0.012)
        if mode == "vanilla":
            ui.line(f"  System: {system_prompt}")
            ui.line("  Head bypassed · original base model · chat template")
        else:
            ui.line(f"  PersonaNet {mode.title()} · no style system prompt · plain Q/A")
        ui.line(f"  {DIM}Greedy decoding · same {max_new_tokens}-token budget for all three{RESET}")
        t0 = time.time()
        with ui.processing(mode), library_diagnostics(ui.verbose):
            result = generate_result(bundle, question, mode, max_new_tokens, system_prompt)
        dt = time.time() - t0
        show_answer(ui, mode, result, dt)
        ui.line()
        ui.pause("next comparison" if index < 3 else "finish comparison")


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
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Generation budget shared by all live panels (default: 512; R1 needs reasoning room)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show library diagnostics during loading and generation")
    parser.add_argument(
        "--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
        help="Explicit system prompt for the vanilla baseline only",
    )
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
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Clear between beats and shorten output for small recording windows",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Optional maximum width; default fills terminal and follows resizing",
    )
    args = parser.parse_args()
    if args.width is not None and args.width < 20:
        parser.error("--width must be at least 20 columns")
    if args.max_new_tokens < 1 or args.step_pause < 0:
        parser.error("token budget must be positive and step pause non-negative")

    ui = DemoUI(
        auto=args.auto,
        step_pause=args.step_pause,
        width=args.width,
        compact=args.compact,
        verbose=args.verbose,
    )
    check = load_frozen_check(args.frozen_check)

    # ---- 1. splash ----
    ui.banner()
    if args.skip_load:
        ui.line(f"  {YELLOW}UI REHEARSAL · no model loaded · no generated answers{RESET}")
    intro = (
        "  Tiny personality head. Frozen 1.5B LLM.\n  No style prompt or base finetune."
        if ui.compact
        else "  A tiny personality head steers a frozen 1.5B LLM — "
        "no system prompt, no base finetune."
    )
    ui.typewrite(f"{DIM}{intro}{RESET}", delay=0.014)
    ui.line()
    ui.pause("start demo")

    # ---- 2. load timeline ----
    ui.beat(1, 4, "Boot sequence")
    ui.line(f"  Checkpoint: {args.ckpt.name}")
    if not args.ckpt.exists() and not args.skip_load:
        ui.line(f"  {RED}missing {args.ckpt}{RESET}")
        raise SystemExit(1)

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
        base_params = sum(p.numel() for p in bundle.model.base.parameters())
        head_params = sum(p.numel() for p in bundle.model.head.parameters())
        ui.line(f"  Base: {base_params:,} params · head: {head_params:,} ({100 * head_params / base_params:.2f}%)")
        frozen = all(not p.requires_grad for p in bundle.model.base.parameters())
        ui.line(f"  Base parameters frozen: {frozen}")
    ui.pause("continue")

    # ---- 3. frozen proof ----
    show_frozen_proof(ui, check)
    ui.pause("continue")

    # ---- 4. scoreboard ----
    show_scoreboard(ui)
    ui.pause("continue to live inference")

    # ---- 5. live ----
    if args.skip_load or bundle is None:
        for index, mode in enumerate(MODES, 1):
            ui.beat(4, 4, f"UI rehearsal · {index}/3 · {mode.upper()}")
            ui.line(f"  Q: {args.question}")
            ui.box(mode.upper(), "[UI placeholder — no model loaded, no generated answer]")
            ui.pause("next rehearsal panel")
    elif args.auto:
        run_compare(ui, bundle, args.question, args.max_new_tokens, args.system_prompt)
    else:
        interactive_loop(ui, bundle, args.max_new_tokens, args.system_prompt)

    # ---- outro ----
    ui.line()
    ui.rule()
    outro = (
        f"  {BOLD}PersonaNet{RESET}{DIM} · shippable personality weights\n"
        "  base model unchanged"
        if ui.compact
        else f"  {BOLD}PersonaNet{RESET}{DIM}  ·  personality as weights you can ship"
        "  ·  base model unchanged"
    )
    ui.typewrite(f"{outro}{RESET}", delay=0.012)
    ui.line(f"  {DIM}docs/REPORT.md{RESET}")
    ui.line()


if __name__ == "__main__":
    main()
