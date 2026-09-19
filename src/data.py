from __future__ import annotations

import random
import re
import csv
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Example:
    text: str
    label: int


class SimpleTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]
        self.cls_id = vocab["<cls>"]

    @staticmethod
    def build(texts: Iterable[str], max_vocab: int = 16000) -> "SimpleTokenizer":
        counts = Counter(tok for text in texts for tok in TOKEN_RE.findall(text.lower()))
        vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        for token, _ in counts.most_common(max_vocab - len(vocab)):
            vocab[token] = len(vocab)
        return SimpleTokenizer(vocab)

    def encode(self, text: str, max_len: int) -> tuple[list[int], list[int]]:
        ids = [self.cls_id] + [self.vocab.get(t, self.unk_id) for t in TOKEN_RE.findall(text.lower())]
        ids = ids[:max_len]
        mask = [1] * len(ids)
        ids += [self.pad_id] * (max_len - len(ids))
        mask += [0] * (max_len - len(mask))
        return ids, mask


class HashNgramTokenizer:
    """Fixed character n-gram hashing; unseen words still share subword evidence."""
    def __init__(self, buckets: int = 4096, ngram: int = 3):
        self.buckets = buckets
        self.ngram = ngram
        self.pad_id = 0
        self.unk_id = 1
        self.cls_id = 2
        self.vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.vocab.update({f"<h{i}>": i + 3 for i in range(buckets)})
        self.idf = None

    def hash_ids(self, text: str) -> list[int]:
        normalized = " " + text.lower().replace("\n", " ") + " "
        grams = [normalized[i:i + self.ngram] for i in range(max(0, len(normalized) - self.ngram + 1))]
        return [3 + (zlib.crc32(g.encode("utf-8")) % self.buckets) for g in grams]

    def encode(self, text: str, max_len: int) -> tuple[list[int], list[int]]:
        ids = [self.cls_id] + self.hash_ids(text)
        ids = ids[:max_len]
        mask = [1] * len(ids)
        ids += [self.pad_id] * (max_len - len(ids))
        mask += [0] * (max_len - len(mask))
        return ids, mask


class HashWordNgramTokenizer(HashNgramTokenizer):
    """Fixed hashing of word unigrams and bigrams, with no learned OOV table."""
    def __init__(self, buckets: int = 8192):
        super().__init__(buckets=buckets, ngram=2)

    def hash_ids(self, text: str) -> list[int]:
        tokens = TOKEN_RE.findall(text.lower())
        grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        return [3 + (zlib.crc32(g.encode("utf-8")) % self.buckets) for g in grams]


class EncodedDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer: SimpleTokenizer, max_len: int):
        self.rows = []
        for ex in examples:
            ids, mask = tokenizer.encode(ex.text, max_len)
            self.rows.append((torch.tensor(ids), torch.tensor(mask, dtype=torch.bool), ex.label))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def _toy_data(seed: int = 0) -> tuple[list[Example], list[Example], str]:
    rng = random.Random(seed)
    topics = {
        0: ("sport", ["team", "match", "coach", "league", "player", "score"]),
        1: ("technology", ["chip", "software", "network", "robot", "data", "server"]),
        2: ("business", ["market", "trade", "company", "shares", "bank", "profit"]),
        3: ("world", ["government", "election", "country", "minister", "policy", "peace"]),
    }
    def make(n: int) -> list[Example]:
        out = []
        for _ in range(n):
            label = rng.randrange(4)
            name, words = topics[label]
            noise = ["today", "new", "report", "international", "important", "future"]
            text = f"{name} " + " ".join(rng.sample(words, 4) + rng.sample(noise, 2))
            out.append(Example(text, label))
        return out
    return make(1600), make(400), "toy_fallback"


def _select_few_shot(rows: list[Example], shots: int, seed: int) -> list[Example]:
    if shots <= 0:
        raise ValueError("shots must be positive")
    rng = random.Random(seed)
    per_class: dict[int, list[Example]] = {}
    for row in rows:
        per_class.setdefault(row.label, []).append(row)
    selected = []
    for label, values in sorted(per_class.items()):
        values = list(values)
        rng.shuffle(values)
        selected.extend(values[:shots])
    rng.shuffle(selected)
    return selected


def _challenge_data(seed: int = 0) -> tuple[list[Example], list[Example], str]:
    """Controlled NLP task with distant evidence and distractor tokens.

    The label is determined by two markers far apart in the sentence: a domain
    family at the beginning and the polarity of the final assessment. Distractors
    deliberately contain words from other domains and both polarities.
    """
    rng = random.Random(seed)
    domain_groups = {
        0: ["football", "stadium", "telescope", "laboratory", "software", "satellite"],
        1: ["election", "parliament", "market", "invoice", "shipping", "currency"],
    }
    positive = ["hopeful", "stable", "encouraging", "constructive", "promising"]
    negative = ["uncertain", "fragile", "critical", "disappointing", "volatile"]
    filler = [
        "committee", "morning", "regional", "public", "annual", "technical", "review",
        "reported", "during", "several", "months", "without", "additional", "context",
        "officials", "experts", "document", "process", "planned", "ordinary", "discussion",
        "question", "detail", "meeting", "evidence", "external", "local", "recent",
    ]
    templates = [
        "The briefing opened with the domain marker {domain}. The record then listed {middle}. After many unrelated details, the final assessment was {tone}.",
        "At the beginning of the note, the subject was clearly {domain}; the body mentioned {middle}. The closing judgment described the outlook as {tone}.",
        "The analyst first identified {domain} as the central signal. The long report included {middle}, and its last sentence called the result {tone}.",
    ]
    def make(n: int) -> list[Example]:
        rows = []
        for _ in range(n):
            group = rng.randrange(2)
            polarity = rng.randrange(2)
            domain = rng.choice(domain_groups[group])
            tone = rng.choice(positive if polarity == 0 else negative)
            middle = " ".join(rng.choices(filler + sum(domain_groups.values(), []), k=rng.randint(32, 54)))
            text = rng.choice(templates).format(domain=domain, middle=middle, tone=tone)
            rows.append(Example(text, group * 2 + polarity))
        return rows
    return make(2400), make(600), "challenge_synthetic"


def _read_ag_news_csv(path: Path) -> list[Example]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                rows.append(Example(f"{row[1]} {row[2]}", int(row[0]) - 1))
    return rows


def _read_sst2_tsv(path: Path) -> list[Example]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for index, row in enumerate(reader):
            if len(row) < 2 or (index == 0 and row[0].lower() == "sentence"):
                continue
            label = row[1].strip().lower()
            if label in {"positive", "pos"}:
                value = 1
            elif label in {"negative", "neg"}:
                value = 0
            else:
                try:
                    value = int(label)
                except ValueError:
                    continue
            rows.append(Example(row[0], value))
    return rows


def _read_trec(path: Path, label_map: dict[str, int] | None = None) -> tuple[list[Example], dict[str, int]]:
    rows = []
    label_map = {} if label_map is None else dict(label_map)
    if not path.exists():
        return rows, label_map
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            coarse, text = line.split(":", 1)
            coarse = coarse.strip().lower()
            if coarse not in label_map:
                label_map[coarse] = len(label_map)
            rows.append(Example(text.strip(), label_map[coarse]))
    return rows, label_map


def _local_sst2(cache_dir: str, shots: int, seed: int) -> tuple[list[Example], list[Example], int, str] | None:
    root = Path(cache_dir) / "sst2"
    train_rows = _read_sst2_tsv(root / "train.tsv")
    test_rows = _read_sst2_tsv(root / "dev.tsv")
    if len(train_rows) < 2 or len(test_rows) < 2:
        return None
    return _select_few_shot(train_rows, shots, seed), test_rows, 2, "sst2_local_train_dev"


def _local_trec(cache_dir: str, shots: int, seed: int) -> tuple[list[Example], list[Example], int, str] | None:
    root = Path(cache_dir) / "trec"
    train_rows, label_map = _read_trec(root / "train.txt")
    test_rows, label_map = _read_trec(root / "test.txt", label_map)
    if len(train_rows) < 6 or len(test_rows) < 6:
        return None
    return _select_few_shot(train_rows, shots, seed), test_rows, len(label_map), "trec_local_train_test"


def _local_ag_news(cache_dir: str, shots: int, seed: int) -> tuple[list[Example], list[Example], int, str] | None:
    root = Path(cache_dir) / "ag_news_csv"
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    # A complete AG News train.csv is about 28 MB; partial interrupted downloads
    # must never be mistaken for the official split.
    train_rows = _read_ag_news_csv(train_path) if train_path.exists() and train_path.stat().st_size > 10_000_000 else []
    test_rows = _read_ag_news_csv(test_path) if test_path.exists() and test_path.stat().st_size > 100 else []
    if train_rows and test_rows:
        return _select_few_shot(train_rows, shots, seed), test_rows, 4, "ag_news_local_official_split"
    if test_rows:
        # The downloaded canonical test file is split once, deterministically,
        # so the entire experiment remains offline and auditable.
        rng = random.Random(seed)
        rng.shuffle(test_rows)
        pivot = max(4 * shots, int(len(test_rows) * 0.8))
        return _select_few_shot(test_rows[:pivot], shots, seed), test_rows[pivot:], 4, "ag_news_local_80_20_split"
    return None


def load_examples(name: str, seed: int, shots: int, cache_dir: str) -> tuple[list[Example], list[Example], int, str]:
    if name == "challenge":
        train, test, source = _challenge_data(seed)
        return _select_few_shot(train, shots, seed), test, 4, source
    local = _local_ag_news(cache_dir, shots, seed) if name in {"ag_news", "ag_news_local"} else None
    if local is not None:
        return local
    if name in {"sst2", "sst2_local"}:
        local = _local_sst2(cache_dir, shots, seed)
        if local is not None:
            return local
    if name in {"trec", "trec_local"}:
        local = _local_trec(cache_dir, shots, seed)
        if local is not None:
            return local
    if name != "ag_news":
        train, test, source = _toy_data(seed)
        return _select_few_shot(train, shots, seed), test, 4, source
    try:
        from datasets import load_dataset
        ds = load_dataset("ag_news", cache_dir=cache_dir)
        labels = ds["train"].features["label"].num_classes
        train_all = list(zip(ds["train"]["text"], ds["train"]["label"]))
        test_all = list(zip(ds["test"]["text"], ds["test"]["label"]))
        rng = random.Random(seed)
        per_class: dict[int, list[tuple[str, int]]] = {i: [] for i in range(labels)}
        for row in train_all:
            if len(per_class[row[1]]) < shots * 3:
                per_class[row[1]].append(row)
        train_rows = [row for values in per_class.values() for row in values[:shots]]
        rng.shuffle(train_rows)
        test_rows = test_all[: min(4000, len(test_all))]
        return [Example(t, y) for t, y in train_rows], [Example(t, y) for t, y in test_rows], labels, "ag_news"
    except Exception as exc:
        train, test, _ = _toy_data(seed)
        print(f"[warning] AG News unavailable ({type(exc).__name__}: {exc}); using toy_fallback")
        return train, test, 4, "toy_fallback"


def make_loaders(train: list[Example], test: list[Example], max_len: int, batch_size: int, seed: int, vocab_scope: str = "all", tokenizer_kind: str = "word"):
    if tokenizer_kind == "hashchar3":
        tokenizer = HashNgramTokenizer(buckets=4096, ngram=3)
    elif tokenizer_kind == "hashword2":
        tokenizer = HashWordNgramTokenizer(buckets=8192)
        vocab_rows = train if vocab_scope == "train" else train + test
        df = torch.zeros(tokenizer.buckets, dtype=torch.float32)
        for row in vocab_rows:
            seen = set(tokenizer.hash_ids(row.text))
            if seen:
                df[torch.tensor([i - 3 for i in seen], dtype=torch.long)] += 1
        n_docs = max(1, len(vocab_rows))
        tokenizer.idf = torch.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    else:
        vocab_rows = train if vocab_scope == "train" else train + test
        tokenizer = SimpleTokenizer.build((x.text for x in vocab_rows))
        df = torch.zeros(max(1, len(tokenizer.vocab) - 3), dtype=torch.float32)
        for row in vocab_rows:
            seen = set()
            for token in TOKEN_RE.findall(row.text.lower()):
                token_id = tokenizer.vocab.get(token, tokenizer.unk_id) - 3
                if token_id >= 0:
                    seen.add(token_id)
            if seen:
                df[torch.tensor(list(seen), dtype=torch.long)] += 1
        n_docs = max(1, len(vocab_rows))
        tokenizer.idf = torch.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    train_ds = EncodedDataset(train, tokenizer, max_len)
    test_ds = EncodedDataset(test, tokenizer, max_len)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)
    return tokenizer, train_loader, test_loader
