#!/usr/bin/env python3
"""
Evidence that PersonaNet does not finetune the base model.

Checks three things:
  1. A head checkpoint contains only head tensors, no base-model weights.
  2. Base weights hash identically before and after the head is attached.
  3. The LoRA arm, by contrast, does change base weights once merged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from personanet.model import DEFAULT_STUDENT, load_base, load_bundle_from_ckpt

ROOT = Path(__file__).resolve().parents[1]


def hash_params(module) -> tuple[str, int]:
    """Order-stable SHA256 over every parameter of a module."""
    digest = hashlib.sha256()
    count = 0
    for name, p in sorted(module.state_dict().items()):
        digest.update(name.encode())
        t = p.detach().to("cpu")
        digest.update(t.numpy().tobytes())
        count += t.numel()
    return digest.hexdigest(), count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_STUDENT)
    parser.add_argument(
        "--ckpt", type=Path, default=ROOT / "artifacts" / "head_deep.pt"
    )
    parser.add_argument(
        "--lora-dir", type=Path, default=ROOT / "artifacts" / "lora_stud_polite"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "compare" / "frozen_check.json",
    )
    args = parser.parse_args()

    report: dict[str, object] = {}

    # 1. what is actually inside the head checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    head_keys = list(ckpt["head_state_dict"].keys())
    head_numel = sum(t.numel() for t in ckpt["head_state_dict"].values())
    looks_like_base = [k for k in head_keys if k.startswith(("model.", "base."))]
    report["head_checkpoint"] = {
        "path": str(args.ckpt),
        "tensors": head_keys,
        "params": head_numel,
        "base_weight_tensors_found": looks_like_base,
        "contains_base_weights": bool(looks_like_base),
    }
    print(f"head ckpt tensors: {head_keys}")
    print(f"head ckpt params: {head_numel:,}  contains base weights: {bool(looks_like_base)}")

    # 2. base weights before vs after attaching the head
    base, _, _ = load_base(args.model_name, device=torch.device("cpu"))
    before, n_before = hash_params(base)
    del base
    print(f"vanilla base sha256: {before}  ({n_before:,} params)")

    bundle, _ = load_bundle_from_ckpt(args.ckpt, device=torch.device("cpu"))
    after, n_after = hash_params(bundle.model.base)
    del bundle
    print(f"base with head attached: {after}  ({n_after:,} params)")

    report["personanet"] = {
        "base_sha256_vanilla": before,
        "base_sha256_with_head": after,
        "base_params": n_before,
        "identical": before == after,
    }
    print(f"PersonaNet leaves base bit-identical: {before == after}")

    # 3. the LoRA arm for contrast
    if args.lora_dir.exists():
        from peft import PeftModel

        base2, _, _ = load_base(args.model_name, device=torch.device("cpu"))
        merged = PeftModel.from_pretrained(base2, str(args.lora_dir))
        merged = merged.merge_and_unload()
        lora_hash, _ = hash_params(merged)
        del merged, base2
        report["lora"] = {
            "base_sha256_merged": lora_hash,
            "identical_to_vanilla": lora_hash == before,
        }
        print(f"LoRA merged base sha256: {lora_hash}")
        print(f"LoRA leaves base bit-identical: {lora_hash == before}")
    else:
        report["lora"] = {"skipped": f"missing {args.lora_dir}"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
