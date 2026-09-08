"""
Standardized Matched-Budget (8k-128k) Subword Efficiency & Downstream LM Benchmark.
===================================================================================
Evaluates subword tokenizers under strictly matched vocabulary and analytical compute budgets:
- Vocabulary Capacities: V in {8k, 16k, 32k, 64k, 128k}
- Tokenizers: SentencePiece-Unigram, Boundary-BPE, UniqToken-SuperBPE
- Balanced Multilingual Corpus: English, Hindi, Telugu, Arabic, Chinese, Russian, Code, Finnish
- Matched Small Transformer LM Architectures: 2L-128d, 4L-256d, 8L-512d
- Hardware Acceleration: CUDA on NVIDIA GPU (with CPU fallback)
- Standard Metrics:
    * Bytes per Token (BpT) & Tokens per Byte (TpB)
    * True Information Density / Bits per Byte (TID-BPB)
    * Downstream Cross-Entropy (nats) on matched small Transformer LMs
    * Microsecond encoding latency per document
    * Peak Process RAM RSS and CUDA VRAM footprint
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import tempfile
import time
import tracemalloc
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

warnings.filterwarnings("ignore")
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[misc,assignment]
    Dataset = object  # type: ignore[misc,assignment]
    HAS_TORCH = False

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uniqtoken.bpe_trainer import BPETrainer
from uniqtoken.cem_merger import CrossEntropyMerging
from uniqtoken.tokenizer import CustomTokenizer

DEFAULT_VOCAB_BUDGETS = [8192, 16384, 32768, 65536, 131072]
DEFAULT_TRAINING_FLOPS = 1.0e12  # Analytical FLOP budget per condition (1.0 TFLOPs)


@dataclass
class LMArchConfig:
    name: str
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    batch_size: int
    lr: float


LM_CONFIGS: Dict[str, LMArchConfig] = {
    "Small (2L-128d)": LMArchConfig(
        name="Small (2L-128d)",
        num_layers=2,
        d_model=128,
        num_heads=4,
        d_ff=512,
        batch_size=16,
        lr=1e-3,
    ),
    "Medium (4L-256d)": LMArchConfig(
        name="Medium (4L-256d)",
        num_layers=4,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        batch_size=16,
        lr=8e-4,
    ),
    "Large (8L-512d)": LMArchConfig(
        name="Large (8L-512d)",
        num_layers=8,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        batch_size=8,
        lr=5e-4,
    ),
}


@dataclass
class BenchmarkRecord:
    vocab_budget: int
    actual_vocab_size: int
    lm_tier: str
    tokenizer_name: str
    seed: int
    num_layers: int
    d_model: int
    total_params: int
    non_embed_params: int
    embedding_memory_mb: float
    training_steps: int
    tokens_processed: int
    actual_flops: float
    token_ce_loss: float
    true_lm_bpb: float
    bytes_per_token: float
    tokens_per_byte: float
    fertility: float
    active_vocab_pct: float
    pct_ge_6b: float
    encode_latency_us: float
    peak_rss_mb: float
    peak_vram_mb: float
    wall_clock_sec: float


def generate_balanced_multilingual_corpus(
    num_docs_per_lang: int = 150,
    seed: int = 42,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Generates a balanced multilingual and synthetic code corpus covering 8 domains:
    English, Hindi, Telugu, Arabic, Chinese, Russian, Code, and Finnish.
    """
    rng = random.Random(seed)

    domain_specs = {
        "English": (
            "abcdefghijklmnopqrstuvwxyz",
            [
                "tion",
                "ing",
                "ness",
                "able",
                "ment",
                "ship",
                "hood",
                "ism",
                "ize",
                "ate",
                "ous",
                "ive",
                "al",
                "ity",
                "ward",
                "wise",
                "less",
                "ful",
                "ance",
                "ence",
            ],
        ),
        "Hindi": (
            "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
            [
                "कारी",
                "वादी",
                "करण",
                "शीलता",
                "पूर्वक",
                "त्मक",
                "त्व",
                "मय",
                "वान",
                "अनुसार",
                "प्रणाली",
                "योजना",
                "विज्ञान",
                "संस्थान",
            ],
        ),
        "Telugu": (
            "అఆఇఈఉఊఋఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహ",
            ["త్వము", "శీలత", "పూర్వక", "మైన", "కరమైన", "వాద", "నిర్వహణ", "వ్యవస్థ", "విధానము", "అభివృద్ధి", "పరిశోధన"],
        ),
        "Arabic": (
            "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
            ["ية", "يات", "يون", "ين", "ستان", "ات", "ان", "المعلوماتية", "الاستراتيجية", "التكنولوجية", "المؤسساتية"],
        ),
        "Chinese": (
            "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把建争性好应各想向开特立数正日月明天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈剑号巨阙珠称夜光果珍李柰菜重芥姜海咸河淡鳞潜羽翔龙师火帝鸟官人皇始制文字乃服衣裳",
            [],
        ),
        "Russian": (
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
            ["ость", "ение", "ация", "ический", "ованный", "тель", "ство", "изм", "ирование", "ование", "тельский"],
        ),
        "Finnish": (
            "abcdefghijklmnopqrstuvxyzäö",
            [
                "ssa",
                "ssä",
                "sta",
                "stä",
                "lla",
                "llä",
                "lta",
                "ltä",
                "lle",
                "ksi",
                "tta",
                "ttä",
                "ineen",
                "mme",
                "nne",
                "nsa",
                "kaan",
                "kään",
                "kin",
                "pa",
                "pä",
                "mainen",
                "llinen",
                "ton",
                "tön",
            ],
        ),
    }

    code_templates = [
        "def compute_gradients_{id}(loss, params, lr={lr}):\n    for p in params:\n        p.grad = p.grad.clamp(-1.0, 1.0)\n        p.data -= lr * p.grad\n    return params",
        "struct TensorNode_{id}<'a> {{\n    dim: usize,\n    grad: &'a mut [f32],\n    active: bool,\n}}\nimpl<'a> TensorNode_{id}<'a> {{\n    pub fn zero_grad(&mut self) {{\n        self.grad.fill(0.0);\n    }}\n}}",
        "const processRequest_{id} = async (req, res) => {{\n    const payload = JSON.parse(req.body);\n    const token = await verifyAuth(payload.auth_token);\n    return res.status(200).json({{ status: 'ok', data: payload.items }});\n}};",
        '{{\n    "model_id": "uniqtoken-transformer-{id}",\n    "vocab_size": {vocab},\n    "context_window": 4096,\n    "embedding_dim": 512,\n    "quantization": "int8",\n    "weights_sha256": "0x{hex}"\n}}',
        "for i in range({count}):\n    x = torch.randn(32, 128, device='cuda')\n    out = model(x)\n    loss = criterion(out, target)\n    loss.backward()\n    optimizer.step()",
    ]

    train_docs: List[str] = []
    val_by_domain: Dict[str, str] = {}

    # 1. Natural Language Domains
    for domain, (chars, affixes) in domain_specs.items():
        raw_words = [
            "".join(rng.choices(chars, k=rng.randint(2, 4 if domain == "Chinese" else 7))) for _ in range(6000)
        ]
        if affixes:
            extra = [w + aff for w in raw_words[:3000] for aff in rng.sample(affixes, k=min(len(affixes), 3))]
            raw_words.extend(extra)
        vocab_pool = list(dict.fromkeys(raw_words))
        n_pool = len(vocab_pool)

        docs_lang = []
        for _ in range(num_docs_per_lang):
            d_len = rng.randint(25, 45)
            sample_words = [vocab_pool[rng.randrange(n_pool)] for _ in range(d_len)]
            if rng.random() < 0.2:
                sample_words.append(f"SYS_{rng.randint(100, 99999)}")
            if rng.random() < 0.2:
                sample_words.append(f"0x{rng.randint(0, 0xFFFFFFFF):08x}")
            if rng.random() < 0.25:
                sample_words.append(str(rng.randint(10, 999999)))
            docs_lang.append("".join(sample_words) if domain == "Chinese" else " ".join(sample_words))

        split = int(num_docs_per_lang * 0.8)
        train_docs.extend(docs_lang[:split])
        val_by_domain[domain] = "\n".join(docs_lang[split:])

    # 2. Code Domain
    code_docs = []
    for _ in range(num_docs_per_lang):
        tmpl = rng.choice(code_templates)
        doc = tmpl.format(
            id=rng.randint(100, 9999),
            lr=round(rng.uniform(1e-4, 1e-2), 4),
            vocab=rng.choice([8192, 16384, 32768, 65536]),
            hex=f"{rng.randint(0, 0xFFFFFFFF):08x}",
            count=rng.randint(10, 100),
        )
        code_docs.append(doc)

    split = int(num_docs_per_lang * 0.8)
    train_docs.extend(code_docs[:split])
    val_by_domain["Code"] = "\n".join(code_docs[split:])

    return train_docs, val_by_domain


class TokenizerAdapter:
    def __init__(
        self,
        name: str,
        encode_ids_fn: Callable[[str], List[int]],
        encode_pieces_fn: Callable[[str], List[str]],
        vocab_size: int,
    ):
        self.name = name
        self.encode_to_ids = encode_ids_fn
        self.encode_pieces = encode_pieces_fn
        self.vocab_size = vocab_size


def train_sentencepiece_unigram(train_docs: List[str], target_vocab: int) -> TokenizerAdapter:
    import sentencepiece as spm

    all_text = "\n".join(train_docs)
    num_unique_chars = len(set(all_text))
    # If vocab size is smaller than unique characters + special tokens, SentencePiece requires lower coverage
    char_cov = 1.0 if target_vocab >= (num_unique_chars + 50) else 0.995

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        sp_corpus = tmp / "sp_corpus.txt"
        sp_corpus.write_text(all_text, encoding="utf-8")
        sp_prefix = tmp / "sp_model"
        spm.SentencePieceTrainer.train(
            input=str(sp_corpus),
            model_prefix=str(sp_prefix),
            model_type="unigram",
            vocab_size=target_vocab,
            character_coverage=char_cov,
            byte_fallback=True,
            hard_vocab_limit=True,
            minloglevel=2,
        )
        sp_proc = spm.SentencePieceProcessor(model_file=str(sp_prefix) + ".model")
        actual_v = sp_proc.get_piece_size()
        if actual_v != target_vocab:
            raise ValueError(
                f"SentencePiece produced {actual_v} pieces; strictly matched budget requires {target_vocab}"
            )

        def _encode_ids(t: str) -> List[int]:
            return sp_proc.encode(t, out_type=int)

        def _encode_pieces(t: str) -> List[str]:
            return list(sp_proc.encode_as_pieces(t))

        return TokenizerAdapter(
            name="SentencePiece-Unigram",
            encode_ids_fn=_encode_ids,
            encode_pieces_fn=_encode_pieces,
            vocab_size=actual_v,
        )


def train_boundary_bpe(train_docs: List[str], target_vocab: int) -> TokenizerAdapter:
    bpe = BPETrainer(target_vocab_size=target_vocab, byte_fallback=True)
    chunks = [w for doc in train_docs for w in doc.split() if w]
    model = bpe.train(chunks, verbose=False)
    actual_v = len(model.vocab)
    if actual_v != target_vocab:
        raise ValueError(f"Boundary-BPE produced {actual_v} pieces; strictly matched budget requires {target_vocab}")

    return TokenizerAdapter(
        name="Boundary-BPE",
        encode_ids_fn=lambda t: model.encode_to_ids(t),
        encode_pieces_fn=lambda t: model.encode(t),
        vocab_size=actual_v,
    )


def train_uniqtoken_superbpe(train_docs: List[str], target_vocab: int) -> TokenizerAdapter:
    sbp_merges = min(target_vocab // 10, 4000)
    base_target = max(target_vocab - sbp_merges, 1000 if target_vocab >= 2000 else target_vocab // 2)
    actual_merges = target_vocab - base_target

    base_tok = CustomTokenizer.train_from_corpus(
        corpus=train_docs,
        target_vocab_size=base_target,
        seed_multiplier=1.2,
        ranking_strategy="byte_savings",
        min_boundary_entropy=0.35,
        length_exponent=1.5,
        pruning_length_exponent=0.0,
        min_frequency=1,
        verbose=False,
    )

    pretok_chunks = [
        tok for d in train_docs for tok in base_tok.pre_tokenizer.pre_tokenize(base_tok.normalizer.normalize(d))
    ]
    cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
    sbp_model = cem.optimize(base_tok.model, chunks=pretok_chunks)
    sbp_tok = CustomTokenizer(
        normalizer=base_tok.normalizer,
        pre_tokenizer=base_tok.pre_tokenizer,
        model=sbp_model,
    )
    actual_v = len(sbp_tok.model.vocab)
    if actual_v != target_vocab:
        raise ValueError(
            f"UniqToken-SuperBPE produced {actual_v} pieces; strictly matched budget requires {target_vocab}"
        )

    return TokenizerAdapter(
        name="UniqToken-SuperBPE",
        encode_ids_fn=lambda t: sbp_tok.encode_to_ids(t),
        encode_pieces_fn=lambda t: sbp_tok.encode(t),
        vocab_size=actual_v,
    )


def calculate_analytical_flops_per_step(
    vocab_size: int,
    cfg: LMArchConfig,
    seq_len: int = 64,
) -> Tuple[int, int, float]:
    """
    Computes analytical FLOPs per training step:
    6 * P_non_embed * B * S + 12 * L * d_model * S^2 * B + 6 * (P_embed + P_head) * B * S
    """
    d_m = cfg.d_model
    l_cnt = cfg.num_layers
    d_ff = cfg.d_ff
    b_sz = cfg.batch_size

    params_per_layer = 4 * (d_m**2) + 2 * d_m * d_ff + d_ff + 9 * d_m
    p_non_embed = l_cnt * params_per_layer + seq_len * d_m + 2 * d_m
    p_embed = vocab_size * d_m
    p_head = vocab_size * d_m
    p_total = p_non_embed + p_embed + p_head

    flops_transformer = 6.0 * p_non_embed * b_sz * seq_len
    flops_attention_quad = 12.0 * l_cnt * d_m * (seq_len**2) * b_sz
    flops_embed_and_head = 6.0 * (p_embed + p_head) * b_sz * seq_len
    flops_per_step = flops_transformer + flops_attention_quad + flops_embed_and_head

    return p_total, p_non_embed, flops_per_step


if HAS_TORCH:

    class CausalMiniTransformer(nn.Module):
        def __init__(self, vocab_size: int, cfg: LMArchConfig, block_size: int = 64):
            super().__init__()
            self.block_size = block_size
            self.embed = nn.Embedding(vocab_size, cfg.d_model)
            self.pos = nn.Parameter(torch.randn(1, block_size, cfg.d_model) * 0.02)

            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.d_ff,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
            self.ln_f = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t = x.size()
            causal_mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
            h = self.embed(x) + self.pos[:, :t, :]
            h = self.encoder(h, mask=causal_mask, is_causal=True)
            h = self.ln_f(h)
            return self.head(h)

else:

    class CausalMiniTransformer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            pass


def train_and_eval_transformer(
    tok: TokenizerAdapter,
    cfg: LMArchConfig,
    train_texts: List[str],
    val_text: str,
    total_val_bytes: int,
    target_flops: float,
    block_size: int = 64,
    device: Optional[Any] = None,
    seed: int = 42,
) -> Tuple[float, float, int, int, int, int, float, float, float]:
    """
    Trains matched-compute CausalMiniTransformer on specified device and evaluates validation loss and TID-BPB.
    Returns:
    (val_ce_loss, lm_bpb, p_total, p_non_embed, steps, tokens_processed, actual_flops, peak_vram_mb, wall_clock_sec)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for Transformer LM evaluation. Install torch to execute benchmarks.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)

    p_total, p_non_embed, flops_per_step = calculate_analytical_flops_per_step(tok.vocab_size, cfg, block_size)
    if flops_per_step > target_flops * 1.5:
        raise ValueError(
            f"Configuration requires {flops_per_step:.2e} FLOPs per step, which exceeds requested target budget {target_flops:.2e} beyond 50% tolerance."
        )
    steps = max(1, int(round(target_flops / flops_per_step)))
    actual_flops = steps * flops_per_step
    tokens_processed = steps * cfg.batch_size * block_size

    class SeqDS(Dataset):
        def __init__(self, ids: List[int], blk: int):
            self.chunks = []
            for i in range(0, max(len(ids) - blk, 0), blk):
                self.chunks.append((ids[i : i + blk], ids[i + 1 : i + blk + 1]))

        def __len__(self):
            return max(len(self.chunks), 1)

        def __getitem__(self, idx):
            if not self.chunks:
                return torch.zeros(block_size, dtype=torch.long), torch.zeros(block_size, dtype=torch.long)
            x, y = self.chunks[idx % len(self.chunks)]
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    train_ids: List[int] = []
    for doc in train_texts:
        train_ids.extend(tok.encode_to_ids(doc))
    val_ids = tok.encode_to_ids(val_text)

    model = CausalMiniTransformer(tok.vocab_size, cfg, block_size).to(device)
    ds = SeqDS(train_ids, block_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()

    t_start = time.perf_counter()
    model.train()
    step_count = 0
    while step_count < steps:
        for x, y in loader:
            if x.size(0) == 0:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            step_count += 1
            if step_count >= steps:
                break

    # Validation evaluation
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    if len(val_ids) < 2:
        raise ValueError("validation text produced fewer than 2 token IDs")

    with torch.no_grad():
        for start in range(0, len(val_ids) - 1, block_size):
            chunk = torch.tensor(val_ids[start : start + block_size + 1], dtype=torch.long, device=device)
            prediction_count = len(chunk) - 1
            if prediction_count == 0:
                continue
            logits = model(chunk[:-1].unsqueeze(0))
            loss = crit(logits.view(-1, logits.size(-1)), chunk[1:])
            total_loss += float(loss.item()) * prediction_count
            n_tokens += prediction_count

    val_ce_loss = total_loss / max(n_tokens, 1)
    lm_bpb = (val_ce_loss * n_tokens) / (total_val_bytes * math.log(2))
    wall_clock = time.perf_counter() - t_start

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return (
        val_ce_loss,
        lm_bpb,
        p_total,
        p_non_embed,
        steps,
        tokens_processed,
        actual_flops,
        peak_vram_mb,
        wall_clock,
    )


def measure_encoding_latency(
    tok: TokenizerAdapter,
    sample_docs: List[str],
) -> float:
    """Measures microsecond latency per document."""
    latencies: List[float] = []
    # Warmup
    for doc in sample_docs[:5]:
        tok.encode_to_ids(doc)

    for doc in sample_docs:
        t0 = time.perf_counter_ns()
        tok.encode_to_ids(doc)
        latencies.append((time.perf_counter_ns() - t0) / 1000.0)
    return float(np.median(latencies)) if latencies else 0.0


def run_benchmark(
    vocab_budgets: List[int],
    lm_tiers: List[str],
    device_str: str = "auto",
    target_flops: float = DEFAULT_TRAINING_FLOPS,
    num_docs_per_lang: int = 150,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[BenchmarkRecord], Dict[str, Any]]:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for Transformer LM evaluation. Install torch to execute benchmarks.")

    # Device selection
    if device_str == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device_str)

    device_name = "CPU"
    if target_device.type == "cuda":
        device_name = torch.cuda.get_device_name(target_device.index or 0)

    if verbose:
        print("=" * 115)
        print("STANDARDIZED MATCHED-BUDGET TOKENIZER & DOWNSTREAM LM BENCHMARK")
        print(f"Device: {target_device} ({device_name})")
        print(f"Vocab Budgets: {vocab_budgets}")
        print(f"LM Tiers: {lm_tiers}")
        print(f"Target Analytical FLOPs: {target_flops:.3e}")
        print("=" * 115)

    tracemalloc.start()

    train_docs, val_by_domain = generate_balanced_multilingual_corpus(
        num_docs_per_lang=num_docs_per_lang,
        seed=seed,
    )
    combined_val = "\n".join(val_by_domain.values())
    total_val_bytes = len(combined_val.encode("utf-8"))
    val_words = [w for w in combined_val.split() if w]
    num_words = len(val_words)

    records: List[BenchmarkRecord] = []

    for V in vocab_budgets:
        if verbose:
            print(f"\n---> Training Tokenizer Triplet at Matched Vocab Budget V = {V:,}")

        # 1. SentencePiece-Unigram
        sp_tok = train_sentencepiece_unigram(train_docs, V)
        # 2. Boundary-BPE
        bpe_tok = train_boundary_bpe(train_docs, V)
        # 3. UniqToken-SuperBPE
        sbp_tok = train_uniqtoken_superbpe(train_docs, V)

        tok_list = [sp_tok, bpe_tok, sbp_tok]

        for tok in tok_list:
            # Tokenizer evaluation metrics
            val_pieces = tok.encode_pieces(combined_val)
            tok_counts = Counter(val_pieces)
            active_cov = len(tok_counts) / max(tok.vocab_size, 1)
            tok_bytes = [len(t.encode("utf-8")) for t in val_pieces]
            pct_ge_6b = sum(1 for b in tok_bytes if b >= 6) / max(len(tok_bytes), 1) * 100.0
            bpt = total_val_bytes / max(len(val_pieces), 1)
            tpb = 1.0 / bpt
            fert = len(val_pieces) / max(num_words, 1)

            # Stride so the latency sample spans every domain, not just the first.
            stride = max(1, len(train_docs) // 50)
            latency_us = measure_encoding_latency(tok, train_docs[::stride][:50])

            current_rss, peak_rss = tracemalloc.get_traced_memory()
            peak_rss_mb = peak_rss / (1024 * 1024)

            # Evaluate against each LM Tier
            for lm_tier_name in lm_tiers:
                cfg = LM_CONFIGS[lm_tier_name]
                if verbose:
                    print(
                        f"     [V={V:,}] {tok.name:<22} on {lm_tier_name:<18} ... ",
                        end="",
                        flush=True,
                    )

                val_ce, lm_bpb, p_tot, p_non, steps, tok_proc, flops, peak_vram, wall_sec = train_and_eval_transformer(
                    tok=tok,
                    cfg=cfg,
                    train_texts=train_docs,
                    val_text=combined_val,
                    total_val_bytes=total_val_bytes,
                    target_flops=target_flops,
                    device=target_device,
                    seed=seed,
                )

                m_embed_mb = (2 * tok.vocab_size * cfg.d_model * 4) / (1024 * 1024)

                if verbose:
                    print(
                        f"CE: {val_ce:.4f} nats | TID-BPB: {lm_bpb:.4f} | BpT: {bpt:.2f} | Latency: {latency_us:.1f}µs ({wall_sec:.1f}s)"
                    )

                rec = BenchmarkRecord(
                    vocab_budget=V,
                    actual_vocab_size=tok.vocab_size,
                    lm_tier=lm_tier_name,
                    tokenizer_name=tok.name,
                    seed=seed,
                    num_layers=cfg.num_layers,
                    d_model=cfg.d_model,
                    total_params=p_tot,
                    non_embed_params=p_non,
                    embedding_memory_mb=round(m_embed_mb, 2),
                    training_steps=steps,
                    tokens_processed=tok_proc,
                    actual_flops=flops,
                    token_ce_loss=round(val_ce, 4),
                    true_lm_bpb=round(lm_bpb, 4),
                    bytes_per_token=round(bpt, 3),
                    tokens_per_byte=round(tpb, 4),
                    fertility=round(fert, 3),
                    active_vocab_pct=round(active_cov * 100.0, 2),
                    pct_ge_6b=round(pct_ge_6b, 2),
                    encode_latency_us=round(latency_us, 2),
                    peak_rss_mb=round(peak_rss_mb, 2),
                    peak_vram_mb=round(peak_vram, 2),
                    wall_clock_sec=round(wall_sec, 3),
                )
                records.append(rec)

    tracemalloc.stop()

    metadata = {
        "benchmark": "Matched-Budget Subword Efficiency & Downstream LM Benchmark",
        "date": datetime.datetime.now().isoformat(),
        "device": str(target_device),
        "device_name": device_name,
        "cuda_available": torch.cuda.is_available() if HAS_TORCH else False,
        "pytorch_version": torch.__version__ if HAS_TORCH else "N/A",
        "deterministic": True,
        "vocab_budgets": vocab_budgets,
        "lm_tiers": lm_tiers,
        "target_flops": target_flops,
        "seed": seed,
    }

    return records, metadata


def generate_tradeoff_plots(
    records: List[BenchmarkRecord],
    output_prefix: Path,
) -> Tuple[Path, Path]:
    """
    Generates publication-quality 4-panel trade-off plots saved as PNG and SVG:
    Panel A: Compression (BpT) vs Downstream LM CE Loss
    Panel B: True Information Density (TID-BPB) across Vocabulary Scales
    Panel C: Total Model Parameters & Embedding Memory Footprint
    Panel D: Microsecond Encoding Latency vs Vocabulary Budget
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not installed; skipping plot generation.", stacklevel=2)
        return output_prefix.with_suffix(".png"), output_prefix.with_suffix(".svg")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    colors = {
        "SentencePiece-Unigram": "#1f77b4",
        "Boundary-BPE": "#ff7f0e",
        "UniqToken-SuperBPE": "#2ca02c",
    }
    markers = {
        "SentencePiece-Unigram": "o",
        "Boundary-BPE": "s",
        "UniqToken-SuperBPE": "^",
    }

    # Panel A: BpT vs CE Loss (Perplexity Trade-off)
    ax_a = axes[0, 0]
    for tok_name, color in colors.items():
        sub = [r for r in records if r.tokenizer_name == tok_name]
        if not sub:
            continue
        bpts = [r.bytes_per_token for r in sub]
        ces = [r.token_ce_loss for r in sub]
        ax_a.scatter(
            bpts,
            ces,
            color=color,
            marker=markers.get(tok_name, "o"),
            s=80,
            alpha=0.85,
            label=tok_name,
            edgecolors="black",
        )
    ax_a.set_title("Panel A: Compression (BpT) vs. Downstream CE Loss", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Bytes Per Token (BpT) [Higher = Better Compression]", fontsize=10)
    ax_a.set_ylabel("Token Cross-Entropy Loss (nats) [Lower = Better]", fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(frameon=True)

    # Panel B: TID-BPB across Vocabulary Scales
    ax_b = axes[0, 1]
    for tok_name, color in colors.items():
        sub = [r for r in records if r.tokenizer_name == tok_name]
        if not sub:
            continue
        v_dict: Dict[int, List[float]] = {}
        for r in sub:
            v_dict.setdefault(r.vocab_budget, []).append(r.true_lm_bpb)
        sorted_v = sorted(v_dict.keys())
        mean_bpb = [float(np.mean(v_dict[v])) for v in sorted_v]
        ax_b.plot(
            [str(v) for v in sorted_v],
            mean_bpb,
            color=color,
            marker=markers.get(tok_name, "o"),
            linewidth=2,
            label=tok_name,
        )
    ax_b.set_title("Panel B: True Information Density (TID-BPB) vs. Vocab Budget", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Matched Vocabulary Budget (V)", fontsize=10)
    ax_b.set_ylabel("True LM Bits-Per-Byte (TID-BPB) [Lower = Better]", fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(frameon=True)

    # Panel C: Embedding Footprint & Total Parameters
    ax_c = axes[1, 0]
    for tok_name, color in colors.items():
        sub = [r for r in records if r.tokenizer_name == tok_name]
        if not sub:
            continue
        sub_sorted = sorted(sub, key=lambda x: x.total_params)
        params_m = [r.total_params / 1e6 for r in sub_sorted]
        mem_mb = [r.embedding_memory_mb for r in sub_sorted]
        ax_c.plot(
            params_m,
            mem_mb,
            color=color,
            marker=markers.get(tok_name, "o"),
            linewidth=1.8,
            label=f"{tok_name} (M_embed)",
        )
    ax_c.set_title("Panel C: Parameter Count vs. Embedding Table Footprint", fontsize=12, fontweight="bold")
    ax_c.set_xlabel("Total Model Parameters (Millions)", fontsize=10)
    ax_c.set_ylabel("Embedding Table Footprint (MB)", fontsize=10)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(frameon=True)

    # Panel D: Microsecond Latency vs Vocab
    ax_d = axes[1, 1]
    for tok_name, color in colors.items():
        sub = [r for r in records if r.tokenizer_name == tok_name]
        if not sub:
            continue
        v_dict_lat: Dict[int, float] = {}
        for r in sub:
            v_dict_lat[r.vocab_budget] = r.encode_latency_us
        sorted_v = sorted(v_dict_lat.keys())
        lats = [v_dict_lat[v] for v in sorted_v]
        ax_d.plot(
            [str(v) for v in sorted_v],
            lats,
            color=color,
            marker=markers.get(tok_name, "o"),
            linewidth=2,
            label=tok_name,
        )
    ax_d.set_title("Panel D: Microsecond Encoding Latency per Document", fontsize=12, fontweight="bold")
    ax_d.set_xlabel("Matched Vocabulary Budget (V)", fontsize=10)
    ax_d.set_ylabel("Median Encoding Latency (µs/doc)", fontsize=10)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(frameon=True)

    plt.tight_layout()

    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    plt.savefig(png_path, format="png", bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()

    return png_path, svg_path


def main():
    parser = argparse.ArgumentParser(description="Run Matched-Budget (8k-128k) Subword Efficiency Benchmark.")
    parser.add_argument("--all", action="store_true", help="Run full factorial 5-scale evaluation")
    parser.add_argument("--smoke-test", action="store_true", help="Fast verification smoke test (< 2 minutes)")
    parser.add_argument(
        "--vocab-sizes",
        type=str,
        default=None,
        help="Comma-separated vocabulary budgets (e.g., '8192,16384,32768')",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default=None,
        help="Comma-separated LM tiers (e.g., 'Small (2L-128d),Medium (4L-256d)')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for Transformer LM evaluation (default: auto)",
    )
    parser.add_argument(
        "--target-flops",
        type=float,
        default=None,
        help="Target analytical FLOP budget per model run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for reproducibility")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks",
        help="Output directory for JSON records and figures",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        vocab_budgets = [1024, 2048]
        lm_tiers = ["Small (2L-128d)"]
        target_flops = 5.0e9  # Sub-2-minute execution
        num_docs = 40
    elif args.all:
        vocab_budgets = DEFAULT_VOCAB_BUDGETS
        lm_tiers = list(LM_CONFIGS.keys())
        target_flops = args.target_flops or DEFAULT_TRAINING_FLOPS
        num_docs = 150
    else:
        if args.vocab_sizes:
            vocab_budgets = [int(v.strip()) for v in args.vocab_sizes.split(",") if v.strip()]
        else:
            vocab_budgets = [8192, 16384]

        if args.tiers:
            requested = [t.strip() for t in args.tiers.split(",") if t.strip()]
            lm_tiers = [t for t in requested if t in LM_CONFIGS]
            if not lm_tiers:
                lm_tiers = ["Small (2L-128d)"]
        else:
            lm_tiers = ["Small (2L-128d)"]

        target_flops = args.target_flops or DEFAULT_TRAINING_FLOPS
        num_docs = 100

    records, metadata = run_benchmark(
        vocab_budgets=vocab_budgets,
        lm_tiers=lm_tiers,
        device_str=args.device,
        target_flops=target_flops,
        num_docs_per_lang=num_docs,
        seed=args.seed,
    )

    # Save JSON ledger
    records_json_path = out_dir / "matched_budget_eval_records.json"
    data_payload = {
        "metadata": metadata,
        "records": [asdict(r) for r in records],
    }
    with open(records_json_path, "w", encoding="utf-8") as f:
        json.dump(data_payload, f, indent=2)
    print(f"\n[Saved JSON Ledger]: {records_json_path}")

    # Generate plots
    plot_prefix = out_dir / "matched_budget_tradeoffs"
    png_path, svg_path = generate_tradeoff_plots(records, plot_prefix)
    print(f"[Saved Trade-off Plots]: {png_path} and {svg_path}")


if __name__ == "__main__":
    main()
