import sys
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from data import load_examples, make_loaders
from models import build_model, count_parameters
from train import initialize_sifter_prototypes


class SifterSmokeTests(unittest.TestCase):
    def test_real_local_data_and_balanced_support(self):
        train, test, classes, source = load_examples(
            "ag_news_local", seed=42, shots=4, cache_dir=str(PROJECT / "data")
        )
        self.assertEqual(classes, 4)
        self.assertGreater(len(test), 100)
        self.assertIn("ag_news_local", source)
        counts = [sum(row.label == label for row in train) for label in range(classes)]
        self.assertEqual(counts, [4, 4, 4, 4])

    def test_sst2_local_data_is_available(self):
        train, test, classes, source = load_examples(
            "sst2_local", seed=42, shots=8, cache_dir=str(PROJECT / "data")
        )
        self.assertEqual(classes, 2)
        self.assertEqual(len(train), 16)
        self.assertGreater(len(test), 800)
        self.assertEqual(source, "sst2_local_train_dev")

    def test_trec_local_data_is_available(self):
        train, test, classes, source = load_examples(
            "trec_local", seed=42, shots=4, cache_dir=str(PROJECT / "data")
        )
        self.assertEqual(classes, 6)
        self.assertEqual(len(train), 24)
        self.assertEqual(len(test), 500)
        self.assertEqual(source, "trec_local_train_test")

    def test_sparse_channels_are_finite_and_packed(self):
        train, test, classes, _ = load_examples(
            "challenge", seed=42, shots=4, cache_dir=str(PROJECT / "data")
        )
        tokenizer, loader, _ = make_loaders(
            train, test, max_len=128, batch_size=8, seed=42, vocab_scope="all", tokenizer_kind="word"
        )
        model = build_model(
            "sifter", len(tokenizer.vocab), classes, width=48, depth=2, max_len=128, idf=tokenizer.idf
        )
        ids, mask, _ = next(iter(loader))
        channels = model.sparse.channel_features(ids, mask)
        features = model.sparse.features(ids, mask)
        self.assertEqual(channels.shape[1], 5)
        self.assertEqual(features.shape[1], model.sparse.sketch_buckets)
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.allclose(features.norm(dim=-1), torch.ones(features.shape[0]), atol=1e-4))

    def test_support_prototype_initialization_is_frozen(self):
        train, test, classes, _ = load_examples(
            "ag_news_local", seed=42, shots=4, cache_dir=str(PROJECT / "data")
        )
        tokenizer, loader, _ = make_loaders(
            train, test, max_len=128, batch_size=8, seed=42, vocab_scope="all", tokenizer_kind="word"
        )
        model = build_model(
            "sifter", len(tokenizer.vocab), classes, width=52, depth=2, max_len=128, idf=tokenizer.idf
        )
        initialize_sifter_prototypes(model, loader, torch.device("cpu"), classes, routing="global")
        self.assertFalse(model.sparse_head.weight.requires_grad)
        self.assertEqual(model.sparse_head.weight.shape[1], model.sparse.sketch_buckets)

    def test_evidence_route_is_checkpoint_persistent(self):
        model = build_model(
            "sifter", 128, 2, width=48, depth=1, max_len=32, idf=torch.ones(125)
        )
        model.sparse.channel_scale.copy_(torch.tensor([0.5, 1.0, 0.25, 0.0, 2.0]))
        self.assertIn("sparse.channel_scale", model.state_dict())

    def test_total_parameter_budget_is_close(self):
        vocab, classes = 12000, 4
        target = count_parameters(build_model("transformer", vocab, classes, 96, 4, 128), False)
        candidate = count_parameters(build_model("sifter", vocab, classes, 52, 4, 128), False)
        self.assertLess(abs(target - candidate) / target, 0.08)


if __name__ == "__main__":
    unittest.main()
