from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data import load_examples, make_loaders
from models import build_model, count_parameters


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    conf = None
    with torch.no_grad():
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(ids, mask)
            pred = logits.argmax(-1)
            total += labels.numel()
            correct += (pred == labels).sum().item()
            if conf is None:
                n = logits.shape[-1]
                conf = torch.zeros(n, n, dtype=torch.long)
            for truth, guess in zip(labels.cpu(), pred.cpu()):
                conf[truth, guess] += 1
    f1s = []
    if conf is not None:
        for i in range(conf.shape[0]):
            tp = conf[i, i].item()
            fp = conf[:, i].sum().item() - tp
            fn = conf[i, :].sum().item() - tp
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1s.append(2 * precision * recall / max(1e-9, precision + recall))
    return {"accuracy": correct / max(1, total), "macro_f1": float(np.mean(f1s)) if f1s else 0.0}


def initialize_sifter_prototypes(model, loader, device, num_classes: int, routing: str = "global"):
    """Select evidence channels and initialize the support prototype head.

    The channel selector uses leave-one-out support accuracy only. It chooses
    whether this task is better served by global lexical evidence, relative
    position evidence, or a boundary-emphasized mixture, without looking at
    the evaluation labels.
    """
    channel_values, labels = [], []
    model.eval()
    with torch.no_grad():
        for ids, mask, batch_labels in loader:
            channel_values.append(model.sparse.channel_features(ids.to(device), mask.to(device)).cpu())
            labels.append(batch_labels.cpu())
    channel_values = torch.cat(channel_values, dim=0)
    labels = torch.cat(labels, dim=0)
    candidates = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # global
            [0.0, 1.0, 1.0, 1.0, 1.0],  # relative position
            [1.0, 1.0, 1.0, 1.0, 1.0],  # global + position
            [0.0, 2.0, 1.0, 1.0, 2.0],  # boundary emphasis
            [0.5, 1.0, 0.5, 0.5, 1.0],  # soft boundary emphasis
        ],
        dtype=channel_values.dtype,
    )
    if routing == "global":
        selected_scale = candidates[0]
    elif routing == "positional":
        selected_scale = candidates[1]
    elif routing == "edge":
        selected_scale = candidates[3]
    elif routing == "adaptive":
        selected_scale = candidates[0]
    else:
        raise ValueError(f"unknown evidence routing: {routing}")
    best_score, best_scale = -1.0, selected_scale
    search_candidates = candidates if routing == "adaptive" else [selected_scale]
    for scale in search_candidates:
        features = torch.nn.functional.normalize(
            model.sparse.pack_channels(channel_values * scale.view(1, -1, 1)), dim=-1
        )
        sums = torch.zeros(num_classes, features.shape[-1])
        counts = torch.zeros(num_classes)
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
        predictions = []
        for index in range(len(features)):
            class_count = counts[labels[index]].item()
            leave_proto = (sums - torch.nn.functional.one_hot(labels[index], num_classes).float().unsqueeze(1) * features[index])
            leave_proto = leave_proto / (counts.clamp_min(1.0).unsqueeze(1) - torch.nn.functional.one_hot(labels[index], num_classes).float().unsqueeze(1)).clamp_min(1.0)
            leave_proto = torch.nn.functional.normalize(leave_proto, dim=-1)
            predictions.append((features[index] @ leave_proto.T).argmax().item())
        score = float(np.mean(np.asarray(predictions) == labels.numpy()))
        if score > best_score:
            best_score, best_scale = score, scale
    with torch.no_grad():
        model.sparse.channel_scale.copy_(best_scale.to(device))
    features = torch.nn.functional.normalize(
        model.sparse.pack_channels(channel_values * best_scale.view(1, -1, 1)), dim=-1
    )
    prototypes = torch.zeros(num_classes, features.shape[-1])
    for cls in range(num_classes):
        rows = features[labels == cls]
        if len(rows):
            prototypes[cls] = rows.mean(dim=0)
    prototypes = torch.nn.functional.normalize(prototypes, dim=-1).to(device)
    with torch.no_grad():
        model.sparse_head.weight.copy_(prototypes * 8.0)
        model.sparse_head.bias.zero_()
        model.head.weight.zero_()
        model.head.bias.zero_()
    for parameter in model.sparse_head.parameters():
        parameter.requires_grad_(False)
    model.sifter_channel_scale = tuple(float(x) for x in best_scale.tolist())
    model.sifter_channel_support_score = best_score
    model.sifter_evidence_routing = routing


def initialize_dense_prototypes(model, loader, device, num_classes: int):
    """Give dense baselines the same support-prototype evaluation privilege."""
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for ids, mask, batch_labels in loader:
            features.append(model.encode(ids.to(device), mask.to(device)).cpu())
            labels.append(batch_labels.cpu())
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    prototypes = torch.zeros(num_classes, features.shape[-1])
    for cls in range(num_classes):
        rows = features[labels == cls]
        if len(rows):
            prototypes[cls] = rows.mean(dim=0)
    prototypes = torch.nn.functional.normalize(prototypes, dim=-1).to(device)
    with torch.no_grad():
        model.head.weight.copy_(prototypes * 8.0)
        model.head.bias.zero_()
    for parameter in model.head.parameters():
        parameter.requires_grad_(False)


def train_one(args) -> dict:
    seed_everything(args.seed)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    cache_root = project_root / "cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_rows, test_rows, num_classes, source = load_examples(args.dataset, args.seed, args.shots, str(data_root))
    tokenizer, train_loader, test_loader = make_loaders(train_rows, test_rows, args.max_len, args.batch_size, args.seed, args.vocab_scope, args.tokenizer)
    model = build_model(
        args.model,
        len(tokenizer.vocab),
        num_classes,
        args.width,
        args.depth,
        args.max_len,
        args.slots,
        args.ablation,
        getattr(tokenizer, "idf", None),
        args.sifter_residual_scale,
    ).to(device)
    if args.model == "sifter" and args.tokenizer in {"word", "hashchar3", "hashword2"}:
        initialize_sifter_prototypes(model, train_loader, device, num_classes, args.evidence_routing)
    elif args.classifier_head == "support" and args.model in {"transformer", "mamba"}:
        initialize_dense_prototypes(model, train_loader, device, num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    best = {"accuracy": 0.0, "macro_f1": 0.0}
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss = loss_fn(logits, labels)
            if not loss.requires_grad:
                # SIFTER's support prototype head is intentionally fixed in
                # the strict few-shot protocol; there is no gradient step to
                # apply when the neural residual is disabled.
                running += loss.item()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        metrics = evaluate(model, test_loader, device)
        if metrics["macro_f1"] > best["macro_f1"]:
            best = metrics
            torch.save({"model": model.state_dict(), "vocab": tokenizer.vocab, "args": vars(args)}, root / f"{args.model}_seed{args.seed}.pt")
        if args.verbose:
            print(f"epoch={epoch+1:02d} loss={running/max(1,len(train_loader)):.4f} acc={metrics['accuracy']:.4f} f1={metrics['macro_f1']:.4f}")
    elapsed = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    result = {
        "model": args.model, "ablation": args.ablation, "seed": args.seed, "dataset": args.dataset, "dataset_source": source,
        "shots_per_class": args.shots, "vocab_scope": args.vocab_scope, "tokenizer": args.tokenizer, "classifier_head": args.classifier_head, "evidence_routing": args.evidence_routing, "width": args.width, "depth": args.depth, "max_len": args.max_len,
        "parameters": count_parameters(model), "total_parameters": count_parameters(model, trainable_only=False), "device": str(device), "best": best,
        "sifter_residual_scale": args.sifter_residual_scale,
        "evidence_channel_scale": list(getattr(model, "sifter_channel_scale", (1.0, 0.0, 0.0, 0.0, 0.0))),
        "train_seconds": elapsed, "peak_memory_mb": peak_mb, "vocab_size": len(tokenizer.vocab),
    }
    with (root / f"{args.model}_seed{args.seed}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["transformer", "mamba", "sifter", "tessera", "dream", "harmonic"], default="sifter")
    p.add_argument("--dataset", default="ag_news")
    p.add_argument("--shots", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--ablation", choices=["none", "no_local", "no_ssm", "no_anchor"], default="none")
    p.add_argument("--vocab-scope", choices=["train", "all"], default="all")
    p.add_argument("--tokenizer", choices=["word", "hashchar3", "hashword2"], default="word")
    p.add_argument("--classifier-head", choices=["dense", "support"], default="dense")
    p.add_argument("--evidence-routing", choices=["global", "positional", "edge", "adaptive"], default="global")
    p.add_argument("--sifter-residual-scale", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="E:\\nlp_arch_lab\\runs")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    print(json.dumps(train_one(parse_args()), ensure_ascii=False, indent=2))
