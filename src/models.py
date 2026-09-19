from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, expansion: float = 2.0):
        super().__init__()
        hidden = max(8, int(d_model * expansion))
        self.in_proj = nn.Linear(d_model, hidden * 2)
        self.out_proj = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(a) * b)


class SelectiveDiagonalSSM(nn.Module):
    """A small, transparent selective SSM implemented without custom CUDA ops."""

    def __init__(self, d_model: int):
        super().__init__()
        self.log_decay = nn.Parameter(torch.zeros(d_model))
        self.input_scale = nn.Linear(d_model, d_model)
        self.delta = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, D]. The recurrent loop is intentionally explicit for reproducibility.
        batch, length, dim = x.shape
        state = x.new_zeros(batch, dim)
        decay_base = -F.softplus(self.log_decay).view(1, dim)
        outputs = []
        for t in range(length):
            xt = x[:, t]
            delta = torch.sigmoid(self.delta(xt))
            decay = torch.exp(decay_base * (0.25 + delta))
            proposal = torch.tanh(self.input_scale(xt))
            state = decay * state + (1.0 - decay) * proposal
            if mask is not None:
                state = state * mask[:, t:t + 1].to(state.dtype)
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class BiSSM(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.forward_ssm = SelectiveDiagonalSSM(d_model)
        self.backward_ssm = SelectiveDiagonalSSM(d_model)
        self.out = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        left = self.forward_ssm(x, mask)
        rev_x = torch.flip(x, dims=[1])
        rev_mask = torch.flip(mask, dims=[1]) if mask is not None else None
        right = torch.flip(self.backward_ssm(rev_x, rev_mask), dims=[1])
        return self.out(torch.cat([left, right], dim=-1))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = GatedMLP(d_model, 2.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        attn, _ = self.attn(y, y, y, key_padding_mask=~mask.bool(), need_weights=False)
        x = x + attn
        return x + self.ffn(self.norm2(x))


class MambaLiteBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.ssm = SelectiveDiagonalSSM(d_model)
        self.backward_ssm = SelectiveDiagonalSSM(d_model)
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = GatedMLP(d_model, 2.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        y = self.local_conv(y.transpose(1, 2)).transpose(1, 2)
        left = self.ssm(y, mask)
        rev_y = torch.flip(y, dims=[1])
        rev_mask = torch.flip(mask, dims=[1])
        right = torch.flip(self.backward_ssm(rev_y, rev_mask), dims=[1])
        x = x + self.out_proj(torch.cat([left, right], dim=-1) * torch.sigmoid(gate).repeat(1, 1, 2))
        return x + self.ffn(self.ffn_norm(x))


class AnchorMixer(nn.Module):
    def __init__(self, d_model: int, slots: int = 4):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(slots, d_model) / math.sqrt(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        logits = torch.einsum("bld,kd->blk", y, self.slots)
        logits = logits / self.temperature.clamp_min(0.2)
        logits = logits.masked_fill(~mask.bool().unsqueeze(-1), -1e4)
        assign = logits.softmax(dim=-1)
        weights = mask.to(x.dtype).unsqueeze(-1) * assign
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-5)
        summaries = torch.einsum("blk,bld->bkd", weights, x) / denom.transpose(1, 2)
        return torch.einsum("blk,bkd->bld", assign, summaries)


class EventAssociativeMemory(nn.Module):
    """DREAM's core: surprise-gated competitive event memory.

    Tokens do not attend to other tokens and do not update one continuous
    channel-wise state. Each token competes for a small set of event cells;
    a write is stronger when the current token is poorly predicted by the
    event it routed to. After the scan, a low-rank relation operator binds
    the event cells before they are read back by tokens.
    """

    def __init__(self, d_model: int, slots: int = 4, relation_rank: int = 16):
        super().__init__()
        route_dim = max(8, min(32, d_model // 2))
        self.slots = slots
        self.seed = nn.Parameter(torch.randn(slots, d_model) / math.sqrt(d_model))
        self.init_proj = nn.Linear(d_model, d_model)
        self.route = nn.Linear(d_model, route_dim, bias=False)
        self.slot_route = nn.Linear(d_model, route_dim, bias=False)
        self.write = nn.Linear(d_model, d_model)
        self.predict = nn.Linear(d_model, d_model)
        self.surprise = nn.Linear(d_model, 1)
        self.update_gate = nn.Linear(d_model, 1)
        self.memory_norm = nn.LayerNorm(d_model)
        self.rel_q = nn.Linear(d_model, relation_rank, bias=False)
        self.rel_k = nn.Linear(d_model, relation_rank, bias=False)
        self.rel_out = nn.Linear(relation_rank, d_model)
        self.read_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, dim = x.shape
        valid = mask.to(x.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        context = (x * valid.unsqueeze(-1)).sum(dim=1) / denom
        memory = self.seed.unsqueeze(0).expand(batch, -1, -1) + self.init_proj(context).unsqueeze(1)
        memory = self.memory_norm(memory)
        token_reads = []
        token_routes = []
        for t in range(length):
            xt = x[:, t]
            token_key = F.normalize(self.route(xt), dim=-1)
            slot_key = F.normalize(self.slot_route(memory), dim=-1)
            logits = torch.einsum("br,bkr->bk", token_key, slot_key) * 2.5
            route = logits.softmax(dim=-1)
            is_valid = valid[:, t:t + 1]
            read = torch.einsum("bk,bkd->bd", route, memory)
            prediction = self.predict(read)
            error = torch.abs(xt - prediction)
            surprise = torch.sigmoid(self.surprise(error)) * is_valid
            write = torch.tanh(self.write(xt))
            amount = torch.sigmoid(self.update_gate(xt)) * surprise
            memory = memory + route.unsqueeze(-1) * amount.unsqueeze(-1) * (write.unsqueeze(1) - memory)
            memory = self.memory_norm(memory)
            token_reads.append(read)
            token_routes.append(route * is_valid)
        reads = torch.stack(token_reads, dim=1)
        routes = torch.stack(token_routes, dim=1)
        relation_q = self.rel_q(memory)
        relation_k = self.rel_k(memory)
        relation = torch.tanh(relation_q.unsqueeze(2) - relation_k.unsqueeze(1)).mean(dim=2)
        relation = self.rel_out(relation)
        reads = reads + torch.einsum("blk,bkd->bld", routes, relation)
        reads = self.read_out(reads)
        memory_summary = (memory + relation).mean(dim=1)
        return reads, memory_summary


class DreamBlock(nn.Module):
    def __init__(self, d_model: int, slots: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.memory = EventAssociativeMemory(d_model, slots)
        self.route_gate = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(d_model, d_model)
        self.global_proj = nn.Linear(d_model, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = GatedMLP(d_model, 1.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        reads, summary = self.memory(y, mask)
        gate = torch.sigmoid(self.route_gate(y))
        x = x + self.out_proj(reads * gate) + self.global_proj(summary).unsqueeze(1)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x + self.ffn(self.ffn_norm(x))


class TesseraEventGraph(nn.Module):
    """Chunked evidence-to-event graph operator.

    TESSERA never forms token-token attention and never scans one state for
    every token. It first compresses a sequence into a fixed number of
    evidence tiles, routes those tiles competitively into event cells, and
    applies a low-rank relation operator between event cells. This gives a
    different inductive bias from both Transformer and Mamba.
    """

    def __init__(self, d_model: int, slots: int = 4, tiles: int = 8, relation_rank: int = 16):
        super().__init__()
        self.slots = slots
        self.tiles = tiles
        self.seed = nn.Parameter(torch.randn(slots, d_model) / math.sqrt(d_model))
        self.init_proj = nn.Linear(d_model, d_model)
        self.update = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.tile_norm = nn.LayerNorm(d_model)
        self.slot_norm = nn.LayerNorm(d_model)
        self.rel_q = nn.Linear(d_model, relation_rank, bias=False)
        self.rel_k = nn.Linear(d_model, relation_rank, bias=False)
        self.rel_out = nn.Linear(relation_rank, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, dim = x.shape
        tile_size = max(1, math.ceil(length / self.tiles))
        tile_values = []
        tile_valid = []
        for start in range(0, length, tile_size):
            end = min(length, start + tile_size)
            local_mask = mask[:, start:end].to(x.dtype)
            denom = local_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            tile_values.append((x[:, start:end] * local_mask.unsqueeze(-1)).sum(dim=1) / denom)
            tile_valid.append((local_mask.sum(dim=1) > 0).to(x.dtype))
        tiles = torch.stack(tile_values, dim=1)
        tile_mask = torch.stack(tile_valid, dim=1)
        tiles = self.tile_norm(tiles)
        global_mean = (tiles * tile_mask.unsqueeze(-1)).sum(dim=1) / tile_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        slots = self.seed.unsqueeze(0).expand(batch, -1, -1) + self.init_proj(global_mean).unsqueeze(1)
        slots = self.slot_norm(slots)
        for _ in range(2):
            assign = torch.einsum("btd,bkd->btk", F.normalize(tiles, dim=-1), F.normalize(slots, dim=-1)) * 3.0
            assign = assign.softmax(dim=-1) * tile_mask.unsqueeze(-1)
            denom = assign.sum(dim=1, keepdim=False).clamp_min(1e-4).unsqueeze(-1)
            summaries = torch.einsum("btk,btd->bkd", assign, tiles) / denom
            slots = self.slot_norm(slots + self.update(torch.cat([summaries, slots], dim=-1)))
        relation_q = self.rel_q(slots)
        relation_k = self.rel_k(slots)
        relation = torch.tanh(relation_q.unsqueeze(2) - relation_k.unsqueeze(1)).mean(dim=2)
        relation = self.rel_out(relation)
        graph = slots + relation
        tile_evidence = tiles + torch.einsum("btk,bkd->btd", assign, graph)
        token_evidence = x.new_zeros(batch, length, dim)
        cursor = 0
        for index, start in enumerate(range(0, length, tile_size)):
            end = min(length, start + tile_size)
            token_evidence[:, start:end] = tile_evidence[:, index:index + 1]
            cursor = end
        summary = graph.mean(dim=1) + tile_evidence.mean(dim=1)
        return token_evidence, summary


class TesseraBlock(nn.Module):
    def __init__(self, d_model: int, slots: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.graph = TesseraEventGraph(d_model, slots)
        self.out_proj = nn.Linear(d_model, d_model)
        self.global_proj = nn.Linear(d_model, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = GatedMLP(d_model, 1.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        evidence, summary = self.graph(y, mask)
        x = x + self.out_proj(evidence) + self.global_proj(summary).unsqueeze(1)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x + self.ffn(self.ffn_norm(x))


class SparseHashEncoder(nn.Module):
    """Fixed sparse evidence memory with global and relative-position channels.

    The global channel preserves ordinary lexical evidence. The position channels
    prevent a long document from collapsing into a bag of words: an occurrence
    near the opening, middle, or closing of a document lands in a different
    evidence tile. This is fixed, differentiable, and independent of attention
    or recurrent state scanning.
    """

    def __init__(self, d_model: int, idf: Optional[torch.Tensor] = None, sketch_buckets: int = 512, position_bins: int = 4):
        super().__init__()
        self.base_buckets = sketch_buckets
        self.position_bins = max(1, position_bins)
        self.position_buckets = min(512, sketch_buckets)
        self.sketch_buckets = sketch_buckets + self.position_buckets * self.position_bins
        self.proj = nn.Linear(self.sketch_buckets, d_model, bias=False)
        if idf is None:
            idf = torch.ones(sketch_buckets, dtype=torch.float32)
        idf = idf.detach().float()
        full_ids = torch.arange(idf.numel())
        full_bucket = full_ids.remainder(self.base_buckets)
        base_idf = torch.zeros(self.base_buckets, dtype=torch.float32)
        base_count = torch.zeros(self.base_buckets, dtype=torch.float32)
        base_idf.scatter_add_(0, full_bucket, idf)
        base_count.scatter_add_(0, full_bucket, torch.ones_like(idf))
        base_idf = base_idf / base_count.clamp_min(1.0)
        position_bucket = full_ids.remainder(self.position_buckets)
        position_idf = torch.zeros(self.position_buckets, dtype=torch.float32)
        position_count = torch.zeros(self.position_buckets, dtype=torch.float32)
        position_idf.scatter_add_(0, position_bucket, idf)
        position_count.scatter_add_(0, position_bucket, torch.ones_like(idf))
        position_idf = position_idf / position_count.clamp_min(1.0)
        padded_position_idf = F.pad(position_idf, (0, self.base_buckets - self.position_buckets))
        self.register_buffer(
            "sketch_idf",
            torch.cat([base_idf, padded_position_idf.repeat(self.position_bins)]),
            persistent=False,
        )
        self.register_buffer(
            "channel_scale",
            torch.ones(1 + self.position_bins, dtype=torch.float32),
            persistent=True,
        )

    def channel_features(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ids = input_ids.to(torch.long) - 3
        valid = mask & (ids >= 0)
        base_id = ids.remainder(self.base_buckets).clamp_min(0)
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device).view(1, -1)
        valid_lengths = mask.to(torch.long).sum(dim=1, keepdim=True).clamp_min(1)
        position_bin = torch.div(positions * self.position_bins, valid_lengths, rounding_mode="floor")
        position_bin = position_bin.clamp_max(self.position_bins - 1)
        position_id = ids.remainder(self.position_buckets).clamp_min(0)
        flat_bucket = position_id + position_bin.expand_as(position_id) * self.position_buckets
        flat_counts = input_ids.new_zeros(
            input_ids.shape[0], self.position_bins * self.position_buckets, dtype=torch.float32
        )
        flat_counts.scatter_add_(1, flat_bucket, valid.to(flat_counts.dtype))
        position_counts = flat_counts.view(input_ids.shape[0], self.position_bins, self.position_buckets)
        global_counts = input_ids.new_zeros(input_ids.shape[0], self.base_buckets, dtype=torch.float32)
        global_counts.scatter_add_(1, base_id, valid.to(global_counts.dtype))
        padded_position = F.pad(position_counts, (0, self.base_buckets - self.position_buckets))
        counts = torch.cat([global_counts.unsqueeze(1), padded_position], dim=1)
        idf = self.sketch_idf.to(input_ids.device)
        global_idf = idf[:self.base_buckets].view(1, 1, self.base_buckets)
        position_idf = idf[self.base_buckets:].view(1, self.position_bins, self.base_buckets)
        channel_idf = torch.cat([global_idf, position_idf], dim=1)
        return torch.log1p(counts) * channel_idf * self.channel_scale.to(input_ids.device).view(1, -1, 1)

    def pack_channels(self, channel_values: torch.Tensor) -> torch.Tensor:
        global_values = channel_values[:, :1, :self.base_buckets]
        position_values = channel_values[:, 1:, :self.position_buckets]
        return torch.cat([global_values.flatten(1), position_values.flatten(1)], dim=1)

    def features(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.pack_channels(self.channel_features(input_ids, mask))
        return F.normalize(features, dim=-1)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(input_ids, mask))


class HarmonicBlock(nn.Module):
    def __init__(self, d_model: int, slots: int = 4, ablation: str = "none"):
        super().__init__()
        self.ablation = ablation
        self.norm = nn.LayerNorm(d_model)
        self.local = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.ssm = BiSSM(d_model)
        self.anchor = AnchorMixer(d_model, slots)
        self.branch_gate = nn.Linear(d_model, 3)
        self.mix_out = nn.Linear(d_model, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = GatedMLP(d_model, 1.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        local = self.local(y.transpose(1, 2)).transpose(1, 2) if self.ablation != "no_local" else torch.zeros_like(y)
        state = self.ssm(y, mask) if self.ablation != "no_ssm" else torch.zeros_like(y)
        global_ctx = self.anchor(y, mask) if self.ablation != "no_anchor" else torch.zeros_like(y)
        gates = self.branch_gate(y).softmax(dim=-1)
        mixed = gates[..., 0:1] * local + gates[..., 1:2] * state + gates[..., 2:3] * global_ctx
        x = x + self.mix_out(mixed)
        return x + self.ffn(self.ffn_norm(x))


class EncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int,
        depth: int,
        max_len: int,
        kind: str,
        slots: int = 4,
        ablation: str = "none",
        idf: Optional[torch.Tensor] = None,
        sifter_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.kind = kind
        self.sifter_residual_scale = float(sifter_residual_scale)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        if kind == "transformer":
            heads = max(1, min(4, d_model // 16))
            while d_model % heads:
                heads -= 1
            self.blocks = nn.ModuleList([TransformerBlock(d_model, heads) for _ in range(depth)])
        elif kind == "mamba":
            self.blocks = nn.ModuleList([MambaLiteBlock(d_model) for _ in range(depth)])
        elif kind == "harmonic":
            self.blocks = nn.ModuleList([HarmonicBlock(d_model, slots, ablation) for _ in range(depth)])
        elif kind == "dream":
            self.blocks = nn.ModuleList([DreamBlock(d_model, slots) for _ in range(depth)])
        elif kind == "tessera":
            self.blocks = nn.ModuleList([TesseraBlock(d_model, slots) for _ in range(depth)])
        elif kind == "sifter":
            self.blocks = nn.ModuleList([TesseraBlock(d_model, slots) for _ in range(depth)])
            # Keep the global lexical channel exact. Only the relative-position
            # channels use a small hash sketch, so long-vocabulary tasks retain
            # their rare-word evidence without blowing up the parameter budget.
            base_dim = int(idf.numel()) if idf is not None else max(1, vocab_size - 3)
            self.sparse = SparseHashEncoder(d_model, idf, sketch_buckets=base_dim, position_bins=4)
            self.sparse_head = nn.Linear(self.sparse.sketch_buckets, num_classes)
        else:
            raise ValueError(f"unknown model kind: {kind}")
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def _encode_pooled(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids) + self.pos[:, :input_ids.shape[1]]
        x = x * mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, mask)
            x = x * mask.unsqueeze(-1).to(x.dtype)
        # Mean pooling is shared by all three families. In the few-shot regime
        # it is less brittle than asking a randomly initialized CLS token to
        # learn the entire aggregation rule from only a handful of examples.
        denom = mask.to(x.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom
        return pooled

    def encode(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.norm(self._encode_pooled(input_ids, mask))

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self._encode_pooled(input_ids, mask)
        sparse_logits = None
        if self.kind == "sifter":
            sparse_features = self.sparse.features(input_ids, mask)
            pooled = pooled + self.sparse.proj(sparse_features)
            sparse_logits = self.sparse_head(sparse_features)
        logits = self.head(pooled)
        # The support-derived sparse classifier is the stable few-shot anchor;
        # the event-graph path contributes a deliberately small learned residual
        # so the architecture remains genuinely hybrid rather than padded.
        return logits if sparse_logits is None else sparse_logits + self.sifter_residual_scale * logits


def build_model(
    kind: str,
    vocab_size: int,
    num_classes: int,
    width: int = 96,
    depth: int = 4,
    max_len: int = 128,
    slots: int = 4,
    ablation: str = "none",
    idf: Optional[torch.Tensor] = None,
    sifter_residual_scale: float = 1.0,
) -> EncoderClassifier:
    return EncoderClassifier(
        vocab_size,
        num_classes,
        width,
        depth,
        max_len,
        kind,
        slots,
        ablation,
        idf,
        sifter_residual_scale,
    )
