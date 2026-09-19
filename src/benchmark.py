from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from data import load_examples, make_loaders, SimpleTokenizer, HashNgramTokenizer, HashWordNgramTokenizer
from models import build_model, count_parameters


def closest_width(kind: str, vocab: int, classes: int, target: int, depth: int, max_len: int) -> tuple[int, int]:
    candidates = list(range(48, 161, 4))
    scored = []
    for width in candidates:
        model = build_model(kind, vocab, classes, width, depth, max_len)
        total = count_parameters(model, trainable_only=False)
        scored.append((abs(total - target), width, total))
    _, width, params = min(scored)
    return width, params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ag_news")
    p.add_argument("--shots", type=int, default=16)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--output-dir", default="E:\\nlp_arch_lab\\runs")
    p.add_argument("--tokenizer", choices=["word", "hashchar3", "hashword2"], default="word")
    p.add_argument("--classifier-head", choices=["dense", "support"], default="dense")
    p.add_argument("--evidence-routing", choices=["global", "positional", "edge", "adaptive"], default="global")
    p.add_argument("--sifter-residual-scale", type=float, default=1.0)
    args = p.parse_args()
    # Match against the actual few-shot vocabulary, not a synthetic vocabulary size.
    # This keeps the embedding contribution in the parameter budget honest.
    project_root = Path(__file__).resolve().parents[1]
    probe_train, probe_test, probe_classes, _ = load_examples(args.dataset, args.seeds[0], args.shots, str(project_root / "data"))
    if args.tokenizer == "hashchar3":
        probe_tokenizer = HashNgramTokenizer(4096, 3)
    elif args.tokenizer == "hashword2":
        probe_tokenizer = HashWordNgramTokenizer(8192)
    else:
        probe_tokenizer = SimpleTokenizer.build((x.text for x in probe_train + probe_test))
    probe_vocab = len(probe_tokenizer.vocab)
    target_model = build_model("transformer", probe_vocab, probe_classes, 96, args.depth, args.max_len)
    target_params = count_parameters(target_model, trainable_only=False)
    widths = {}
    for kind in ["transformer", "mamba", "sifter"]:
        widths[kind] = closest_width(kind, probe_vocab, probe_classes, target_params, args.depth, args.max_len)
    print(json.dumps({"probe_vocab": probe_vocab, "probe_classes": probe_classes, "target_params": target_params, "matched_widths": widths}, indent=2))
    if args.smoke_only:
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_results = []
    for seed in args.seeds:
        for kind in ["transformer", "mamba", "sifter"]:
            width = widths[kind][0]
            cmd = [sys.executable, str(Path(__file__).with_name("train.py")), "--model", kind,
                   "--dataset", args.dataset, "--shots", str(args.shots), "--seed", str(seed),
                   "--epochs", str(args.epochs), "--width", str(width), "--depth", str(args.depth),
                   "--max-len", str(args.max_len), "--vocab-scope", "all", "--tokenizer", args.tokenizer,
                   "--classifier-head", args.classifier_head, "--evidence-routing", args.evidence_routing,
                   "--sifter-residual-scale", str(args.sifter_residual_scale),
                   "--output-dir", str(out)]
            print("running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            result_path = out / f"{kind}_seed{seed}.json"
            all_results.append(json.loads(result_path.read_text(encoding="utf-8")))
    summary = out / "summary.json"
    summary.write_text(json.dumps({"probe_vocab": probe_vocab, "probe_classes": probe_classes, "target_params": target_params, "matched_widths": widths, "classifier_head": args.classifier_head, "evidence_routing": args.evidence_routing, "sifter_residual_scale": args.sifter_residual_scale, "results": all_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
