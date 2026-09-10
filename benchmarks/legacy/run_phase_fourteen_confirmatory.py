"""
Phase Fourteen B: Five-Seed Confirmatory Factorial Benchmark & Interaction Testing.
Evaluates 2 Critical Vocabulary Capacities (32K, 64K) x 3 LM Architectures (Small, Medium, Large)
across 3 Tokenizers (SentencePiece, Boundary-BPE, UniqToken Config B) across N=5 paired seeds [101, 202, 303, 404, 505].
Total: 90 matched FLOP runs (5.0e+12 analytical FLOPs on CUDA).
Executes pre-registered hypothesis testing and 2-way Repeated Measures ANOVA (V x LM Capacity interaction).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple

EXPERIMENT_VERSION = "post-tokenizer-fixes-2026-08-30"

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uniqtoken.bpe_trainer import BPETrainer
from uniqtoken.cem_merger import CrossEntropyMerging
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.seed_builder import SeedToken, SeedVocabularyBuilder
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramTrainer


def generate_high_entropy_corpus(num_docs: int = 1000, seed: int = 42) -> Tuple[List[str], Dict[str, str]]:
    rng = random.Random(seed)
    scripts = {
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
        "Tamil": (
            "அஆஇஈஉஊஎஏஐஒஓஔகஙசஞடணதநபமயரலவழளறன",
            ["மை", "வாதம்", "பூர்வ", "மான", "கரமான", "த்துவம்", "மேலாண்மை", "அமைப்பு", "வளர்ச்சி", "திட்டம்", "ஆராய்ச்சி"],
        ),
        "Bengali": (
            "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলবশষসহ",
            ["কারী", "বাদী", "করণ", "শীলতা", "মূলক", "ত্ব", "ময়", "ব্যবস্থাপনা", "পদ্ধতি", "উন্নয়ন", "গবেষণা"],
        ),
        "Arabic": (
            "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
            ["ية", "يات", "يون", "ين", "ستان", "ات", "ان", "المعلوماتية", "الاستراتيجية", "التكنولوجية", "المؤسساتية"],
        ),
        "Chinese": (
            "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把建争性好应各想向开特立数正日月明天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈剑号巨阙珠称夜光果珍李柰菜重芥姜海咸河淡鳞潜羽翔龙师火帝鸟官人皇始制文字乃服衣裳推位让国有虞陶唐吊民伐罪周发殷汤坐朝问道垂拱平章爱育黎首臣伏戎羌遐迩一体率宾归王鸣凤在竹白驹食场化被草木赖及万方盖此身发四大五常恭惟鞠养岂敢毁伤女慕贞洁男效才良知过必改得能莫忘罔谈彼短靡恃己长信使可覆器欲难量墨悲丝染诗赞羔羊景行维贤克念作圣德建名立形端表正空谷传声虚堂习听祸因恶积福缘善庆尺璧非宝寸阴是竞资父事君曰严与敬孝当竭力忠则尽命临深履薄夙兴温凊似兰斯馨如松之盛川流不息渊澄取映容止若思言辞安定笃初诚美慎终宜令荣业所基籍甚无竟学优登仕摄职从政存以甘棠去而益咏乐殊贵贱礼别尊卑上和下睦夫唱妇随",
            [],
        ),
        "Russian": (
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
            ["ость", "ение", "ация", "ический", "ованный", "тель", "ство", "изм", "ирование", "ование", "тельский"],
        ),
    }

    train_docs: List[str] = []
    val_by_lang: Dict[str, str] = {}

    for lang, (chars, affixes) in scripts.items():
        raw_words = ["".join(rng.choices(chars, k=rng.randint(2, 4 if lang == "Chinese" else 7))) for _ in range(12000)]
        if affixes:
            extra = [w + aff for w in raw_words[:6000] for aff in rng.sample(affixes, k=min(len(affixes), 3))]
            raw_words.extend(extra)
        vocab_pool = list(dict.fromkeys(raw_words))
        n_pool = len(vocab_pool)

        docs_lang = []
        for _ in range(num_docs):
            d_len = rng.randint(25, 50)
            w_sample = [vocab_pool[rng.randrange(n_pool)] for _ in range(d_len)]
            if rng.random() < 0.25:
                w_sample.append(f"SYS_{rng.randint(100, 99999)}")
            if rng.random() < 0.25:
                w_sample.append(f"0x{rng.randint(0, 0xFFFFFFFF):08x}")
            if rng.random() < 0.30:
                w_sample.append(str(rng.randint(100, 999999)))
            docs_lang.append("".join(w_sample) if lang == "Chinese" else " ".join(w_sample))

        split = int(num_docs * 0.8)
        train_docs.extend(docs_lang[:split])
        val_by_lang[lang] = "\n".join(docs_lang[split:])

    return train_docs, val_by_lang


TARGET_TRAINING_FLOPS = 5.0e12


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
    "Small (4L-128d)": LMArchConfig(
        name="Small (4L-128d)",
        num_layers=4,
        d_model=128,
        num_heads=4,
        d_ff=512,
        batch_size=16,
        lr=1e-3,
    ),
    "Medium (6L-256d)": LMArchConfig(
        name="Medium (6L-256d)",
        num_layers=6,
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
class ConfirmatoryRecord:
    vocab_size: int
    lm_tier: str
    seed: int
    model_name: str
    actual_vocab_size: int
    num_layers: int
    d_model: int
    total_params: int
    non_embed_params: int
    training_steps: int
    tokens_processed: int
    true_lm_bpb: float
    token_ce_loss: float
    bytes_per_token: float
    indic_bpt: float
    fertility: float
    active_vocab_pct: float
    pct_ge_6b: float
    actual_flops: float
    wall_clock_sec: float


def calculate_analytical_flops_per_step(
    v_sz: int,
    cfg: LMArchConfig,
    seq_len: int = 64,
) -> Tuple[int, int, float]:
    d_m = cfg.d_model
    l_cnt = cfg.num_layers
    d_ff = cfg.d_ff
    b_sz = cfg.batch_size

    params_per_layer = 4 * (d_m**2) + 2 * d_m * d_ff + 4 * d_m
    p_non_embed = l_cnt * params_per_layer + 2 * d_m
    p_embed = v_sz * d_m
    p_head = v_sz * d_m
    p_total = p_non_embed + p_embed + p_head

    flops_transformer = 6.0 * p_non_embed * b_sz * seq_len
    flops_attention_quad = 12.0 * l_cnt * d_m * (seq_len**2) * b_sz
    flops_embed_and_head = 6.0 * (p_embed + p_head) * b_sz * seq_len
    flops_per_step = flops_transformer + flops_attention_quad + flops_embed_and_head

    return p_total, p_non_embed, flops_per_step


class CausalMiniTransformer(nn.Module):
    def __init__(self, v_sz: int, cfg: LMArchConfig, block_size: int = 64):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(v_sz, cfg.d_model)
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
        self.head = nn.Linear(cfg.d_model, v_sz, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.size()
        causal_mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        h = self.embed(x) + self.pos[:, :t, :]
        h = self.encoder(h, mask=causal_mask, is_causal=True)
        h = self.ln_f(h)
        return self.head(h)


def train_and_eval_capacity_transformer(
    enc_fn: Callable[[str], List[int]],
    vocab_size: int,
    cfg: LMArchConfig,
    train_texts: List[str],
    val_text: str,
    total_val_bytes: int,
    target_flops: float = TARGET_TRAINING_FLOPS,
    block_size: int = 64,
    seed: int = 42,
) -> Tuple[float, float, int, int, int, int, float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    p_total, p_non_embed, flops_per_step = calculate_analytical_flops_per_step(vocab_size, cfg, block_size)
    steps = max(1, int(round(target_flops / flops_per_step)))
    actual_flops = steps * flops_per_step
    tokens_processed = steps * cfg.batch_size * block_size

    class SeqDS(Dataset):
        def __init__(self, ids: List[int], b_sz: int):
            self.chunks = []
            for i in range(0, max(len(ids) - b_sz + 1, 0), b_sz):
                self.chunks.append((ids[i : i + b_sz], ids[i + 1 : i + b_sz + 1]))

        def __len__(self):
            return max(len(self.chunks), 1)

        def __getitem__(self, idx):
            if not self.chunks:
                return torch.zeros(block_size, dtype=torch.long), torch.zeros(block_size, dtype=torch.long)
            x, y = self.chunks[idx % len(self.chunks)]
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids: List[int] = []
    for doc in train_texts:
        train_ids.extend(enc_fn(doc))
    val_ids = enc_fn(val_text)

    model = CausalMiniTransformer(vocab_size, cfg, block_size).to(device)
    assert model.embed.num_embeddings == vocab_size
    assert model.head.out_features == vocab_size

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

    model.eval()
    total_loss = 0.0
    n_tokens = 0
    if len(val_ids) < 2:
        raise ValueError("validation text must produce at least two token IDs")
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

    return val_ce_loss, lm_bpb, p_total, p_non_embed, steps, tokens_processed, actual_flops, wall_clock


def compute_repeated_measures_anova_2way(data_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Computes standard 2-Way Within-Subjects (Repeated Measures) ANOVA:
    data_matrix shape: (s, a, b) where s = seeds, a = levels of A (Vocab), b = levels of B (Capacity)
    """
    s, a, b = data_matrix.shape
    grand_mean = np.mean(data_matrix)

    mean_subj = np.mean(data_matrix, axis=(1, 2))  # shape (s,)
    mean_a = np.mean(data_matrix, axis=(0, 2))  # shape (a,)
    mean_b = np.mean(data_matrix, axis=(0, 1))  # shape (b,)
    mean_ab = np.mean(data_matrix, axis=0)  # shape (a, b)
    mean_as = np.mean(data_matrix, axis=2)  # shape (s, a)
    mean_bs = np.mean(data_matrix, axis=1)  # shape (s, b)

    SS_total = np.sum((data_matrix - grand_mean) ** 2)
    SS_subj = a * b * np.sum((mean_subj - grand_mean) ** 2)

    SS_A = b * s * np.sum((mean_a - grand_mean) ** 2)
    SS_B = a * s * np.sum((mean_b - grand_mean) ** 2)
    SS_AB = s * np.sum((mean_ab - mean_a[:, None] - mean_b[None, :] + grand_mean) ** 2)

    SS_As = b * np.sum((mean_as.T - mean_a[:, None] - mean_subj[None, :] + grand_mean) ** 2)
    SS_Bs = a * np.sum((mean_bs.T - mean_b[:, None] - mean_subj[None, :] + grand_mean) ** 2)

    SS_ABs = 0.0
    for k in range(s):
        for i in range(a):
            for j in range(b):
                dev = (
                    data_matrix[k, i, j]
                    - mean_ab[i, j]
                    - mean_as[k, i]
                    - mean_bs[k, j]
                    + mean_a[i]
                    + mean_b[j]
                    + mean_subj[k]
                    - grand_mean
                )
                SS_ABs += dev**2

    df_A = a - 1
    df_As = (a - 1) * (s - 1)
    df_B = b - 1
    df_Bs = (b - 1) * (s - 1)
    df_AB = (a - 1) * (b - 1)
    df_ABs = (a - 1) * (b - 1) * (s - 1)

    MS_A = SS_A / df_A
    MS_As = SS_As / df_As
    F_A = MS_A / MS_As
    p_A = float(stats.f.sf(F_A, df_A, df_As))

    MS_B = SS_B / df_B
    MS_Bs = SS_Bs / df_Bs
    F_B = MS_B / MS_Bs
    p_B = float(stats.f.sf(F_B, df_B, df_Bs))

    MS_AB = SS_AB / df_AB
    MS_ABs = SS_ABs / df_ABs
    F_AB = MS_AB / MS_ABs
    p_AB = float(stats.f.sf(F_AB, df_AB, df_ABs))

    return {
        "Factor_A (Vocab)": {"SS": float(SS_A), "df": df_A, "MS": float(MS_A), "F": float(F_A), "p": p_A},
        "Error_A": {"SS": float(SS_As), "df": df_As, "MS": float(MS_As)},
        "Factor_B (Capacity)": {"SS": float(SS_B), "df": df_B, "MS": float(MS_B), "F": float(F_B), "p": p_B},
        "Error_B": {"SS": float(SS_Bs), "df": df_Bs, "MS": float(MS_Bs)},
        "Interaction_AxB": {"SS": float(SS_AB), "df": df_AB, "MS": float(MS_AB), "F": float(F_AB), "p": p_AB},
        "Error_AB": {"SS": float(SS_ABs), "df": df_ABs, "MS": float(MS_ABs)},
    }


def run_phase_fourteen_confirmatory(
    vocab_scales: Optional[List[int]] = None,
    lm_tiers: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    num_docs: int = 1000,
) -> Dict[str, Any]:
    import sentencepiece as spm

    vocab_scales = list(vocab_scales or [32768, 65536])
    lm_tiers = list(lm_tiers or ["Small (4L-128d)", "Medium (6L-256d)", "Large (8L-512d)"])
    seeds = list(seeds or [101, 202, 303, 404, 505])

    print("=" * 175)
    print("PHASE FOURTEEN B: FIVE-SEED CONFIRMATORY FACTORIAL BENCHMARK & INTERACTION TESTING")
    print(f"Vocab Scales: {vocab_scales} | LM Tiers: {lm_tiers}")
    print(f"Seeds: {seeds} (N = {len(seeds)} paired) | Matched Training FLOPs: {TARGET_TRAINING_FLOPS:.3e} on CUDA")
    print(
        f"Factorial Design: 2 Vocab x 3 LM x 3 Tokenizers x {len(seeds)} Seeds = {len(vocab_scales) * len(lm_tiers) * 3 * len(seeds)} Total LM Runs"
    )
    print("=" * 175)

    all_records: List[ConfirmatoryRecord] = []

    for V in vocab_scales:
        print(f"\n======================================== SCALE: V = {V:,} ========================================")
        sbp_merges = min(V // 10, 4000)
        base_target = max(V - sbp_merges, 1000)
        actual_merges = V - base_target

        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\n---> [Scale V = {V:,} | Paired Seed {seed_idx}/{len(seeds)}: {seed}]")
            train_docs, val_by_lang = generate_high_entropy_corpus(num_docs=num_docs, seed=seed)
            combined_val = "\n".join(val_by_lang.values())
            total_val_bytes = len(combined_val.encode("utf-8"))
            words = [w for w in combined_val.split() if w]
            num_words = len(words)

            indic_val = "\n".join([val_by_lang[l] for l in ["Hindi", "Telugu", "Tamil", "Bengali"]])
            indic_val_bytes = len(indic_val.encode("utf-8"))

            # 1. SentencePiece
            with TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                sp_corpus = tmp / "train.txt"
                sp_corpus.write_text("\n".join(train_docs), encoding="utf-8")
                sp_prefix = tmp / "sp_model"
                spm.SentencePieceTrainer.train(
                    input=str(sp_corpus),
                    model_prefix=str(sp_prefix),
                    model_type="unigram",
                    vocab_size=V,
                    character_coverage=1.0,
                    byte_fallback=True,
                    hard_vocab_limit=False,
                    minloglevel=2,
                )
                sp_proc = spm.SentencePieceProcessor(model_file=str(sp_prefix) + ".model")
                sp_tokens = list(sp_proc.encode_as_pieces(combined_val))
                sp_enc = lambda t: sp_proc.encode(t, out_type=int)
                sp_actual_v = sp_proc.get_piece_size()

            sp_counts = Counter(sp_tokens)
            sp_active_cov = len(sp_counts) / sp_actual_v
            sp_tok_bytes = [len(t.encode("utf-8")) for t in sp_tokens]
            sp_pct_ge_6b = sum(1 for b in sp_tok_bytes if b >= 6) / max(len(sp_tok_bytes), 1) * 100.0
            sp_indic_toks = list(sp_proc.encode_as_pieces(indic_val))
            sp_indic_bpt = indic_val_bytes / max(len(sp_indic_toks), 1)
            sp_bpt = total_val_bytes / max(len(sp_tokens), 1)
            sp_fert = len(sp_tokens) / max(num_words, 1)

            # 2. Boundary-BPE
            b_bpe = BPETrainer(target_vocab_size=V, byte_fallback=True)
            bpe_chunks = [w for d in train_docs for w in d.split(" ") if w]
            bpe_model = b_bpe.train(bpe_chunks, verbose=False)
            bpe_tokens = bpe_model.encode(combined_val)
            bpe_actual_v = len(bpe_model.vocab)
            bpe_counts = Counter(bpe_tokens)
            bpe_active_cov = len(bpe_counts) / bpe_actual_v
            bpe_tok_bytes = [len(t.encode("utf-8")) for t in bpe_tokens]
            bpe_pct_ge_6b = sum(1 for b in bpe_tok_bytes if b >= 6) / max(len(bpe_tok_bytes), 1) * 100.0
            bpe_indic_toks = bpe_model.encode(indic_val)
            bpe_indic_bpt = indic_val_bytes / max(len(bpe_indic_toks), 1)
            bpe_bpt = total_val_bytes / max(len(bpe_tokens), 1)
            bpe_fert = len(bpe_tokens) / max(num_words, 1)
            bpe_enc = lambda t: bpe_model.encode_to_ids(t)

            # 3. UniqToken Config B
            tok_base = CustomTokenizer.train_from_corpus(
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
                tok for d in train_docs for tok in tok_base.pre_tokenizer.pre_tokenize(tok_base.normalizer.normalize(d))
            ]
            cem = CrossEntropyMerging(max_merges=actual_merges, cross_word=True, verbose=False)
            sbp_model = cem.optimize(tok_base.model, chunks=pretok_chunks)
            cal_tok = CustomTokenizer(
                normalizer=tok_base.normalizer, pre_tokenizer=tok_base.pre_tokenizer, model=sbp_model
            )
            cal_tokens = cal_tok.encode(combined_val)
            cal_actual_v = len(cal_tok.model.vocab)
            cal_counts = Counter(cal_tokens)
            cal_active_cov = len(cal_counts) / cal_actual_v
            cal_tok_bytes = [len(t.encode("utf-8")) for t in cal_tokens]
            cal_pct_ge_6b = sum(1 for b in cal_tok_bytes if b >= 6) / max(len(cal_tok_bytes), 1) * 100.0
            cal_indic_toks = cal_tok.encode(indic_val)
            cal_indic_bpt = indic_val_bytes / max(len(cal_indic_toks), 1)
            cal_bpt = total_val_bytes / max(len(cal_tokens), 1)
            cal_fert = len(cal_tokens) / max(num_words, 1)
            cal_enc = lambda t: cal_tok.encode_to_ids(t)

            for lm_name in lm_tiers:
                cfg = LM_CONFIGS[lm_name]

                # SP
                ce_sp, bpb_sp, p_tot_sp, p_non_sp, st_sp, tok_sp, fl_sp, wc_sp = train_and_eval_capacity_transformer(
                    enc_fn=sp_enc,
                    vocab_size=sp_actual_v,
                    cfg=cfg,
                    train_texts=train_docs,
                    val_text=combined_val,
                    total_val_bytes=total_val_bytes,
                    target_flops=TARGET_TRAINING_FLOPS,
                    seed=seed,
                )
                rec_sp = ConfirmatoryRecord(
                    vocab_size=V,
                    lm_tier=lm_name,
                    seed=seed,
                    model_name="SentencePiece-Unigram",
                    actual_vocab_size=sp_actual_v,
                    num_layers=cfg.num_layers,
                    d_model=cfg.d_model,
                    total_params=p_tot_sp,
                    non_embed_params=p_non_sp,
                    training_steps=st_sp,
                    tokens_processed=tok_sp,
                    true_lm_bpb=bpb_sp,
                    token_ce_loss=ce_sp,
                    bytes_per_token=sp_bpt,
                    indic_bpt=sp_indic_bpt,
                    fertility=sp_fert,
                    active_vocab_pct=sp_active_cov * 100.0,
                    pct_ge_6b=sp_pct_ge_6b,
                    actual_flops=fl_sp,
                    wall_clock_sec=wc_sp,
                )
                all_records.append(rec_sp)

                # Boundary-BPE
                ce_bpe, bpb_bpe, p_tot_bpe, p_non_bpe, st_bpe, tok_bpe, fl_bpe, wc_bpe = (
                    train_and_eval_capacity_transformer(
                        enc_fn=bpe_enc,
                        vocab_size=bpe_actual_v,
                        cfg=cfg,
                        train_texts=train_docs,
                        val_text=combined_val,
                        total_val_bytes=total_val_bytes,
                        target_flops=TARGET_TRAINING_FLOPS,
                        seed=seed,
                    )
                )
                rec_bpe = ConfirmatoryRecord(
                    vocab_size=V,
                    lm_tier=lm_name,
                    seed=seed,
                    model_name="Boundary-BPE",
                    actual_vocab_size=bpe_actual_v,
                    num_layers=cfg.num_layers,
                    d_model=cfg.d_model,
                    total_params=p_tot_bpe,
                    non_embed_params=p_non_bpe,
                    training_steps=st_bpe,
                    tokens_processed=tok_bpe,
                    true_lm_bpb=bpb_bpe,
                    token_ce_loss=ce_bpe,
                    bytes_per_token=bpe_bpt,
                    indic_bpt=bpe_indic_bpt,
                    fertility=bpe_fert,
                    active_vocab_pct=bpe_active_cov * 100.0,
                    pct_ge_6b=bpe_pct_ge_6b,
                    actual_flops=fl_bpe,
                    wall_clock_sec=wc_bpe,
                )
                all_records.append(rec_bpe)

                # UniqToken Config B
                ce_cal, bpb_cal, p_tot_cal, p_non_cal, st_cal, tok_cal, fl_cal, wc_cal = (
                    train_and_eval_capacity_transformer(
                        enc_fn=cal_enc,
                        vocab_size=cal_actual_v,
                        cfg=cfg,
                        train_texts=train_docs,
                        val_text=combined_val,
                        total_val_bytes=total_val_bytes,
                        target_flops=TARGET_TRAINING_FLOPS,
                        seed=seed,
                    )
                )
                rec_cal = ConfirmatoryRecord(
                    vocab_size=V,
                    lm_tier=lm_name,
                    seed=seed,
                    model_name="UniqToken-SuperBPE (Config B)",
                    actual_vocab_size=cal_actual_v,
                    num_layers=cfg.num_layers,
                    d_model=cfg.d_model,
                    total_params=p_tot_cal,
                    non_embed_params=p_non_cal,
                    training_steps=st_cal,
                    tokens_processed=tok_cal,
                    true_lm_bpb=bpb_cal,
                    token_ce_loss=ce_cal,
                    bytes_per_token=cal_bpt,
                    indic_bpt=cal_indic_bpt,
                    fertility=cal_fert,
                    active_vocab_pct=cal_active_cov * 100.0,
                    pct_ge_6b=cal_pct_ge_6b,
                    actual_flops=fl_cal,
                    wall_clock_sec=wc_cal,
                )
                all_records.append(rec_cal)

                print(
                    f"  [{lm_name:<16}] SP BPB: {bpb_sp:.3f} | BPE BPB: {bpb_bpe:.3f} | UniqToken BPB: {bpb_cal:.3f} "
                    f"(Diff UniqToken-BPE: {bpb_cal - bpb_bpe:+.3f}) | Steps: SP={st_sp} BPE={st_bpe} UniqToken={st_cal}",
                    flush=True,
                )

    # 1. Pre-Registered Paired Hypothesis Testing (N=5 paired seeds)
    print("\n" + "=" * 175)
    print("PHASE FOURTEEN B: STATISTICAL HYPOTHESIS TESTING (5 PAIRED SEEDS, HOLM-BONFERRONI ADJUSTED)")
    print("=" * 175)

    raw_tests: List[Dict[str, Any]] = []

    # Test 1: H1 (UniqToken 64K Medium < UniqToken 32K Medium)
    c64_med = [
        r.true_lm_bpb
        for r in all_records
        if r.model_name == "UniqToken-SuperBPE (Config B)" and r.vocab_size == 65536 and r.lm_tier == "Medium (6L-256d)"
    ]
    c32_med = [
        r.true_lm_bpb
        for r in all_records
        if r.model_name == "UniqToken-SuperBPE (Config B)" and r.vocab_size == 32768 and r.lm_tier == "Medium (6L-256d)"
    ]
    diff_h1 = np.array(c64_med) - np.array(c32_med)
    t_h1, p_h1 = stats.ttest_rel(c64_med, c32_med)
    raw_tests.append(
        {
            "name": "H1: BPB(UniqToken, 64K, Med) < BPB(UniqToken, 32K, Med)",
            "mean_a": float(np.mean(c64_med)),
            "mean_b": float(np.mean(c32_med)),
            "diff": float(np.mean(diff_h1)),
            "t": float(t_h1),
            "p": float(p_h1 / 2.0 if t_h1 < 0 else 1.0 - p_h1 / 2.0),
            "ci": [
                float(np.mean(diff_h1) - 2.776 * stats.sem(diff_h1)),
                float(np.mean(diff_h1) + 2.776 * stats.sem(diff_h1)),
            ],
            "cohen_dz": float(np.mean(diff_h1) / max(np.std(diff_h1, ddof=1), 1e-9)),
        }
    )

    # Test 2: H2 (CE(UniqToken, 64K, Medium) < CE(UniqToken, 64K, Small))
    ce64_med = [
        r.token_ce_loss
        for r in all_records
        if r.model_name == "UniqToken-SuperBPE (Config B)" and r.vocab_size == 65536 and r.lm_tier == "Medium (6L-256d)"
    ]
    ce64_sml = [
        r.token_ce_loss
        for r in all_records
        if r.model_name == "UniqToken-SuperBPE (Config B)" and r.vocab_size == 65536 and r.lm_tier == "Small (4L-128d)"
    ]
    diff_h2 = np.array(ce64_med) - np.array(ce64_sml)
    t_h2, p_h2 = stats.ttest_rel(ce64_med, ce64_sml)
    raw_tests.append(
        {
            "name": "H2: CE(UniqToken, 64K, Med) < CE(UniqToken, 64K, Small)",
            "mean_a": float(np.mean(ce64_med)),
            "mean_b": float(np.mean(ce64_sml)),
            "diff": float(np.mean(diff_h2)),
            "t": float(t_h2),
            "p": float(p_h2 / 2.0 if t_h2 < 0 else 1.0 - p_h2 / 2.0),
            "ci": [
                float(np.mean(diff_h2) - 2.776 * stats.sem(diff_h2)),
                float(np.mean(diff_h2) + 2.776 * stats.sem(diff_h2)),
            ],
            "cohen_dz": float(np.mean(diff_h2) / max(np.std(diff_h2, ddof=1), 1e-9)),
        }
    )

    # Test 3: H3 (BPB(UniqToken, 64K, Medium) < BPB(UniqToken, 64K, Small))
    c64_sml = [
        r.true_lm_bpb
        for r in all_records
        if r.model_name == "UniqToken-SuperBPE (Config B)" and r.vocab_size == 65536 and r.lm_tier == "Small (4L-128d)"
    ]
    diff_h3 = np.array(c64_med) - np.array(c64_sml)
    t_h3, p_h3 = stats.ttest_rel(c64_med, c64_sml)
    raw_tests.append(
        {
            "name": "H3: BPB(UniqToken, 64K, Med) < BPB(UniqToken, 64K, Small)",
            "mean_a": float(np.mean(c64_med)),
            "mean_b": float(np.mean(c64_sml)),
            "diff": float(np.mean(diff_h3)),
            "t": float(t_h3),
            "p": float(p_h3 / 2.0 if t_h3 < 0 else 1.0 - p_h3 / 2.0),
            "ci": [
                float(np.mean(diff_h3) - 2.776 * stats.sem(diff_h3)),
                float(np.mean(diff_h3) + 2.776 * stats.sem(diff_h3)),
            ],
            "cohen_dz": float(np.mean(diff_h3) / max(np.std(diff_h3, ddof=1), 1e-9)),
        }
    )

    # Test 4: H4 (BPB(UniqToken, 64K, Large) vs BPB(UniqToken, 64K, Medium) - Testing Diminishing Return)
    c64_lrg = [
        r.true_lm_bpb
        for r in all_records
        if r.model_name == "UniqToken-SuperBPE (Config B)" and r.vocab_size == 65536 and r.lm_tier == "Large (8L-512d)"
    ]
    diff_h4 = np.array(c64_lrg) - np.array(c64_med)
    t_h4, p_h4 = stats.ttest_rel(c64_lrg, c64_med)
    raw_tests.append(
        {
            "name": "H4: BPB(UniqToken, 64K, Large) < BPB(UniqToken, 64K, Med)",
            "mean_a": float(np.mean(c64_lrg)),
            "mean_b": float(np.mean(c64_med)),
            "diff": float(np.mean(diff_h4)),
            "t": float(t_h4),
            "p": float(p_h4 / 2.0 if t_h4 < 0 else 1.0 - p_h4 / 2.0),
            "ci": [
                float(np.mean(diff_h4) - 2.776 * stats.sem(diff_h4)),
                float(np.mean(diff_h4) + 2.776 * stats.sem(diff_h4)),
            ],
            "cohen_dz": float(np.mean(diff_h4) / max(np.std(diff_h4, ddof=1), 1e-9)),
        }
    )

    # Step-down Holm correction
    sorted_indices = sorted(range(len(raw_tests)), key=lambda i: raw_tests[i]["p"])
    m_hyp = len(raw_tests)
    running_adjusted = 0.0
    for rank, idx in enumerate(sorted_indices):
        multiplier = m_hyp - rank
        adjusted = min(raw_tests[idx]["p"] * multiplier, 1.0)
        running_adjusted = max(running_adjusted, adjusted)
        raw_tests[idx]["p_adj"] = running_adjusted
        raw_tests[idx]["verdict"] = "CONFIRMED (p < 0.05)" if raw_tests[idx]["p_adj"] < 0.05 else "NOT SIGNIFICANT"

    print(
        f"{'Hypothesis':<52} | {'Mean 1':<8} | {'Mean 2':<8} | {'Mean Diff':<10} | {'t(4)':<8} | {'p_adj (Holm)':<14} | {'95% CI':<24} | {'Cohen d_z':<10} | {'Verdict'}"
    )
    print("-" * 175)
    for test in raw_tests:
        ci_str = f"[{test['ci'][0]:.3f}, {test['ci'][1]:.3f}]"
        print(
            f"{test['name']:<52} | {test['mean_a']:<8.3f} | {test['mean_b']:<8.3f} | {test['diff']:<10.3f} | {test['t']:<8.2f} | {test['p_adj']:<14.4e} | {ci_str:<24} | {test['cohen_dz']:<10.2f} | {test['verdict']}"
        )
    print("=" * 175)

    # 2. Repeated Measures 2-Way ANOVA on UniqToken BPB (5 seeds x 2 Vocab x 3 Capacity)
    print("\n" + "=" * 175)
    print("REPEATED-MEASURES 2-WAY ANOVA: BPB ~ VOCAB + CAPACITY + (VOCAB x CAPACITY) + (1|SEED)")
    print("=" * 175)

    anova_matrix = np.zeros((len(seeds), len(vocab_scales), len(lm_tiers)))
    for s_idx, seed in enumerate(seeds):
        for v_idx, V in enumerate(vocab_scales):
            for l_idx, lm_name in enumerate(lm_tiers):
                rec = [
                    r
                    for r in all_records
                    if r.seed == seed
                    and r.vocab_size == V
                    and r.lm_tier == lm_name
                    and r.model_name == "UniqToken-SuperBPE (Config B)"
                ][0]
                anova_matrix[s_idx, v_idx, l_idx] = rec.true_lm_bpb

    anova_res = compute_repeated_measures_anova_2way(anova_matrix)
    print(
        f"{'Source of Variation':<32} | {'Sum of Squares (SS)':<22} | {'df':<6} | {'Mean Square (MS)':<20} | {'F-Statistic':<14} | {'p-value'}"
    )
    print("-" * 175)
    for factor in ["Factor_A (Vocab)", "Error_A", "Factor_B (Capacity)", "Error_B", "Interaction_AxB", "Error_AB"]:
        d = anova_res[factor]
        f_str = f"{d['F']:.2f}" if "F" in d else "-"
        p_str = f"{d['p']:.4e}" if "p" in d else "-"
        print(f"{factor:<32} | {d['SS']:<22.4f} | {d['df']:<6d} | {d['MS']:<20.4f} | {f_str:<14} | {p_str}")
    print("=" * 175 + "\n")

    # 3. Factorial Summary Grid
    summary_grid: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    for lm_name in lm_tiers:
        for V in vocab_scales:
            for m_name in ["SentencePiece-Unigram", "Boundary-BPE", "UniqToken-SuperBPE (Config B)"]:
                recs = [r for r in all_records if r.lm_tier == lm_name and r.vocab_size == V and r.model_name == m_name]
                summary_grid[lm_name][V][m_name] = {
                    "true_lm_bpb_mean": float(np.mean([r.true_lm_bpb for r in recs])),
                    "true_lm_bpb_std": float(np.std([r.true_lm_bpb for r in recs], ddof=1)),
                    "token_ce_loss_mean": float(np.mean([r.token_ce_loss for r in recs])),
                    "bytes_per_token_mean": float(np.mean([r.bytes_per_token for r in recs])),
                    "active_vocab_pct_mean": float(np.mean([r.active_vocab_pct for r in recs])),
                }

    print(
        f"{'LM Tier':<18} | {'Scale':<8} | {'Tokenizer':<28} | {'True BPB (Mean +- Std)':<24} | {'Token CE':<10} | {'B/Tok':<8} | {'Active %'}"
    )
    print("-" * 175)
    for lm_name in lm_tiers:
        for V in vocab_scales:
            for m_name in ["SentencePiece-Unigram", "Boundary-BPE", "UniqToken-SuperBPE (Config B)"]:
                st = summary_grid[lm_name][V][m_name]
                bpb_str = f"{st['true_lm_bpb_mean']:.3f} +- {st['true_lm_bpb_std']:.3f}"
                print(
                    f"{lm_name:<18} | V={V:<6} | {m_name:<28} | {bpb_str:<24} | {st['token_ce_loss_mean']:<10.3f} | {st['bytes_per_token_mean']:<8.2f} | {st['active_vocab_pct_mean']:<6.1f}%"
                )
            print("-" * 175)

    # 4-Panel Publication Figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=300)

    # Panel A: UniqToken True LM BPB vs V with 95% CIs
    ax_a = axes[0, 0]
    tier_colors = {"Small (4L-128d)": "#1f77b4", "Medium (6L-256d)": "#ff7f0e", "Large (8L-512d)": "#2ca02c"}
    tier_markers = {"Small (4L-128d)": "o-", "Medium (6L-256d)": "s-", "Large (8L-512d)": "^-"}

    for lm_name in lm_tiers:
        means = [summary_grid[lm_name][V]["UniqToken-SuperBPE (Config B)"]["true_lm_bpb_mean"] for V in vocab_scales]
        stds = [summary_grid[lm_name][V]["UniqToken-SuperBPE (Config B)"]["true_lm_bpb_std"] for V in vocab_scales]
        cis = [2.776 * (s / np.sqrt(len(seeds))) for s in stds]
        ax_a.errorbar(
            vocab_scales,
            means,
            yerr=cis,
            fmt=tier_markers[lm_name],
            color=tier_colors[lm_name],
            label=lm_name,
            capsize=5,
            linewidth=2.2,
            markersize=8,
        )
        for V, mean in zip(vocab_scales, means):
            ax_a.annotate(
                f"{mean:.3f}", (V, mean + 0.015), fontsize=9, color=tier_colors[lm_name], ha="center", fontweight="bold"
            )
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(vocab_scales)
    ax_a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_a.set_title(
        "Panel A: UniqToken BPB Scaling (32K -> 64K) with 95% CIs Across LM Tiers", fontsize=11, fontweight="bold"
    )
    ax_a.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_a.set_ylabel("True LM BPB (lower is better)", fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend()

    # Panel B: UniqToken Cross-Tier Progression at 64K (BPB and CE)
    ax_b = axes[0, 1]
    bpb_64_means = [summary_grid[lm][65536]["UniqToken-SuperBPE (Config B)"]["true_lm_bpb_mean"] for lm in lm_tiers]
    ce_64_means = [summary_grid[lm][65536]["UniqToken-SuperBPE (Config B)"]["token_ce_loss_mean"] for lm in lm_tiers]

    x_pos = np.arange(len(lm_tiers))
    ax_b.plot(x_pos, bpb_64_means, "ro-", linewidth=2.4, markersize=9, label="True LM BPB (Left)")
    for i, val in enumerate(bpb_64_means):
        ax_b.annotate(f"{val:.3f} BPB", (i, val + 0.01), fontsize=9.5, color="red", ha="center", fontweight="bold")
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(["Small (4L-128d)", "Medium (6L-256d)", "Large (8L-512d)"], fontsize=10)
    ax_b.set_ylabel("True LM BPB", fontsize=10, color="red")
    ax_b.set_title(
        "Panel B: UniqToken Cross-Tier Performance at V = 65,536 (Saturation Curve)", fontsize=11, fontweight="bold"
    )
    ax_b.grid(True, linestyle="--", alpha=0.5)

    ax_b_twin = ax_b.twinx()
    ax_b_twin.plot(x_pos, ce_64_means, "bs--", linewidth=2.0, markersize=8, label="Token CE Loss (Right)")
    for i, val in enumerate(ce_64_means):
        ax_b_twin.annotate(f"{val:.2f} nats", (i, val - 0.08), fontsize=9, color="blue", ha="center")
    ax_b_twin.set_ylabel("Token Cross-Entropy (nats)", fontsize=10, color="blue")

    # Panel C: Factorial Interaction Plot (Slope Differences)
    ax_c = axes[1, 0]
    for lm_name in lm_tiers:
        vals = [summary_grid[lm_name][V]["UniqToken-SuperBPE (Config B)"]["true_lm_bpb_mean"] for V in vocab_scales]
        slope = vals[1] - vals[0]
        ax_c.plot(
            [0, 1],
            vals,
            tier_markers[lm_name],
            color=tier_colors[lm_name],
            label=f"{lm_name} (Slope: {slope:+.3f})",
            linewidth=2.2,
            markersize=8,
        )
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["V = 32,768 (32K)", "V = 65,536 (64K)"], fontsize=10)
    ax_c.set_title(
        f"Panel C: 2-Way Interaction Plot (V x LM Capacity: p={anova_res['Interaction_AxB']['p']:.3g})",
        fontsize=11,
        fontweight="bold",
    )
    ax_c.set_ylabel("True LM BPB", fontsize=10)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend()

    # Panel D: 3-Way Architecture Comparison at Medium & Large Tiers
    ax_d = axes[1, 1]
    m_colors = {
        "SentencePiece-Unigram": "#1f77b4",
        "Boundary-BPE": "#2ca02c",
        "UniqToken-SuperBPE (Config B)": "#d62728",
    }
    for m_name in ["SentencePiece-Unigram", "Boundary-BPE", "UniqToken-SuperBPE (Config B)"]:
        vals_med = [summary_grid["Medium (6L-256d)"][V][m_name]["true_lm_bpb_mean"] for V in vocab_scales]
        vals_lrg = [summary_grid["Large (8L-512d)"][V][m_name]["true_lm_bpb_mean"] for V in vocab_scales]
        ax_d.plot(
            vocab_scales,
            vals_med,
            "s-",
            color=m_colors[m_name],
            label=f"{m_name} (Medium)",
            linewidth=2.2,
            markersize=8,
        )
        ax_d.plot(
            vocab_scales,
            vals_lrg,
            "^--",
            color=m_colors[m_name],
            label=f"{m_name} (Large)",
            linewidth=1.8,
            markersize=7,
            alpha=0.7,
        )
    ax_d.set_xscale("log", base=2)
    ax_d.set_xticks(vocab_scales)
    ax_d.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_d.set_title(
        "Panel D: 3-Way Comparison Across High Capacity Tiers (Medium & Large)", fontsize=11, fontweight="bold"
    )
    ax_d.set_xlabel("Vocabulary Size (V)", fontsize=10)
    ax_d.set_ylabel("True LM BPB", fontsize=10)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(fontsize=8)

    plt.tight_layout()
    plot_path = Path(__file__).resolve().parent / "phase_fourteen_confirmatory.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved Phase 14B confirmatory figure to {plot_path}")

    # Persist JSON
    output_data = {
        "experiment_version": EXPERIMENT_VERSION,
        "summary_grid": summary_grid,
        "hypothesis_tests": raw_tests,
        "repeated_measures_anova": anova_res,
        "all_records": [asdict(r) for r in all_records],
    }
    json_path = Path(__file__).resolve().parent / "phase_fourteen_confirmatory_records.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[Records] Saved Phase 14B confirmatory ledger to {json_path}")

    return output_data


if __name__ == "__main__":
    run_phase_fourteen_confirmatory(
        vocab_scales=[32768, 65536],
        lm_tiers=["Small (4L-128d)", "Medium (6L-256d)", "Large (8L-512d)"],
        seeds=[101, 202, 303, 404, 505],
        num_docs=1000,
    )
