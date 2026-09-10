from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uniqtoken.bpe_trainer import BPETrainer
from uniqtoken.cem_merger import CrossEntropyMerging
from uniqtoken.hf_exporter import HuggingFaceExporter
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramTrainer


@dataclass
class BenchmarkMetrics:
    dataset_name: str
    model_kind: str
    num_chars: int
    num_bytes: int
    num_words: int
    num_tokens: int
    bytes_per_token: float
    tokens_per_unicode_character: float  # Tokens per raw Unicode code point, including whitespace
    encode_speed_kbs: float
    encode_speed_tokens_sec: float
    decode_speed_kbs: float
    offset_overhead_ratio: float  # (time_with_offsets / time_without_offsets)
    peak_ram_mb: float = 0.0
    fallback_rate_pct: float = 0.0


class TokenizerBenchmarkSuite:
    """
    Held-out tokenizer measurement suite.

    Evaluates:
    1. Compression Ratio (Bytes / Token) across multilingual scripts.
    2. Tokens per Unicode code point.
    3. Throughput (KB/sec & Tokens/sec) on fixed evaluation corpora.
    4. Offset Span Computation Overhead.
    5. Code Indentation Context Compression Savings.
    6. Unigram vs. BPE Head-to-Head Architectural Comparison.
    7. UniqToken vs. HuggingFace, SentencePiece, and tiktoken baselines.
    """

    BENCHMARK_CORPORA = {
        "English_Prose": (
            "The architecture of transformer language models relies fundamentally on discrete "
            "tokenization subword vocabularies. Subword tokenization balances the trade-off "
            "between character-level sequence bloat and word-level vocabulary explosion. "
            "Modern systems require robust handling of diverse orthographic conventions, "
            "case normalization, and punctuation isolation.\n"
        )
        * 40,
        "Python_Code": (
            "class DistributedOptimizer:\n"
            "    def __init__(self, params, lr: float = 1e-4):\n"
            "        self.params = list(params)\n"
            "        self.lr = lr\n"
            "        self.state = {}\n\n"
            "    def step(self, closure=None):\n"
            "        loss = None\n"
            "        if closure is not None:\n"
            "            loss = closure()\n"
            "        for p in self.params:\n"
            "            if p.grad is not None:\n"
            "                d_p = p.grad.data\n"
            "                p.data.add_(d_p, alpha=-self.lr)\n"
            "        return loss\n"
        )
        * 30,
        "Indic_Hindi": (
            "प्राकृतिक भाषा प्रसंस्करण और कंप्यूटर विज्ञान में टोकनाइज़र एक अत्यंत महत्वपूर्ण घटक है। "
            "देवनागरी लिपि में मात्राओं और हलंत (विराम) का उचित संयोजन बनाए रखना आवश्यक है ताकि "
            "अक्षरों का विखंडन न हो। भाषा मॉडल की सटीकता सही टोकनीकरण पर निर्भर करती है।\n"
        )
        * 30,
        "CJK_Japanese": (
            "自然言語処理におけるトークナイザーは、テキストを一連のサブワードに分割する重要な役割を果たします。"
            "日本語のように単語間に空白が存在しない言語では、形態素解析やバイトフォールバック機構が極めて重要です。"
            "正確なアライメントとオフセット追跡が必要です。\n"
        )
        * 30,
        "Arabic_Script": (
            "تعتبر معالجة اللغات الطبيعية وتجزئة النصوص من أهم ركائز الذكاء الاصطناعي الحديث. "
            "يتطلب التعامل مع اللغة العربية دعماً دقيقاً للحركات وعلامات التشكيل لضमान عدم فقدان المعنى.\n"
        )
        * 30,
        "Arithmetic_Math": (
            "Solve the system of equations: f(x, y) = 3.14159 * x^2 + 2.71828 * y - 42.0. "
            "Given matrices A = [[12, 34], [56, 78]] and B = [[90, 11], [22, 33]], calculate det(A * B). "
            "Indices: 1048576, 2097152, 4194304, 8388608. Verify sum(x_i) for i in range(1000).\n"
        )
        * 30,
        "Agglutinative_Turkish": (
            "Doğal dil işleme modellerinde eklemeli dillerin morfolojik yapısının doğru çözümlenmesi büyük önem taşır. "
            "Türkçede çekim ve yapım eklerinin ardışık dizilimi, kelime gövdelerinin korunmasını ve alt-kelime "
            "bölümlemesinin dilbilgisi kurallarına uygun olarak gerçekleştirilmesini gerektirir.\n"
        )
        * 30,
        "Agglutinative_Finnish": (
            "Luonnollisen kielen käsittelyssä agglutinoivien kielten sananmuodostus asettaa erityisiä vaatimuksia. "
            "Suomen kielen taivutuspäätteet ja johdokset muodostavat monimutkaisia rakenteita, joiden oikea "
            "tokenisointi takaa kielimallin optimaalisen suorituskyvyn ja sanaston tehokkaan käytön.\n"
        )
        * 30,
        "Agglutinative_Swahili": (
            "Katika uchakataji wa lugha asilia na teknolojia ya kompyuta, mfumo wa ugawaji maneno una umuhimu mkubwa sana. "
            "Lugha ya Kiswahili hutumia viambishi awali na viambishi tamati kuunda maumbo changamano ya maneno, "
            "ambapo mzizi wa neno huambatanishwa na viwakilishi vya ngeli, nafsi, na nyakati mbalimbali. "
            "Ugawaji sahihi wa vipande vya maneno unahitajika ili kuwezesha miundo ya lugha kuelewa miundo ya kisarufi bila kupoteza maana.\n"
        )
        * 30,
        "Tonal_Yoruba": (
            "Nínú ìmọ̀ ẹ̀rọ ìṣirò àti ìtúpalẹ̀ èdè àdánidá, pínpín àwọn ọ̀rọ̀ sí wẹ́wẹ́ jẹ́ kókó pàtàkì fún àwọn àwòṣe kọ̀mpútà. "
            "Èdè Yorùbá ní àwọn àmì ohùn àti àwọn àmì ìsàlẹ̀ tí ó ń fi ìyàtọ̀ sí ìtumọ̀ ọ̀rọ̀, pẹ̀lú àwọn àfòmọ́ tí ó ń so mọ́ orí ọ̀rọ̀. "
            "Pínpín ọ̀rọ̀ ní ọ̀nà tó péye ń mú kí ẹ̀rọ mọ bí a ṣe ń lo àwọn ìsọ̀rí ọ̀rọ̀ láìsí àdánù kankan nínú ìtumọ̀.\n"
        )
        * 30,
        "Agglutinative_Malayalam": (
            "സ്വാഭാവിക ഭാഷാ പ്രക്രിയയിലും കമ്പ്യൂട്ടർ ശാസ്ത്രത്തിലും ടോക്കണൈസേഷൻ പ്രധാനപ്പെട്ട ഒരു ഘടകമാണ്. "
            "മലയാളം ദ്രാവിഡ ഭാഷാ കുടുംബത്തിലെ പ്രധാനപ്പെട്ട ഒരു ഭാഷയാണ്. "
            "ഈ ഭാഷയിൽ വാക്കുകൾ ചേർത്തുണ്ടാക്കുന്ന രീതിയും വിഭക്തിയും വളരെ സങ്കീർണ്ണമാണ്. "
            "ശരിയായ ടോക്കൺ വിഭജനം ഭാഷാ മോഡലുകളുടെ കൃത്യത വർദ്ധിപ്പിക്കുന്നു.\n"
        )
        * 30,
        "Geez_Amharic": (
            "በተፈጥሮ ቋንቋ ሂደት እና በኮምፒውተር ሳይንስ ውስጥ የቃላት መከፋፈል በጣም አስፈላጊ አካል ነው። "
            "የአማርኛ ቋንቋ በግዕዝ ፊደላት የሚጻፍ ሲሆን የበለጸገ የስነ-ቅርጽ እና የቅጥያ አወቃቀር አለው። "
            "ቃላት ከስርወ-ግስ ተነስተው በርካታ ቅድመ-ቅጥያዎች፣ ውስጠ-ቅጥያዎች እና ድህረ-ቅጥያዎችን በማጣመር ይገነባሉ። "
            "ትክክለኛ የንዑስ ቃላት ክፍፍል የቋንቋ ሞዴሎችን የማስታወስ ብቃት እና የትርጉም ጥራትን ያሻሽላል። "
            "ይህም የፊደላት ውህደት ሳይዛባ የቃላት ፍቺ በትክክል እንዲጠበቅ እና የባይት ብክነትን ለመቀነስ ያስችላል።\n"
        )
        * 30,
    }

    # Training documents are intentionally distinct from every measurement
    # document above. They cover the same broad domains without reusing the
    # repeated benchmark paragraphs.
    TRAINING_CORPUS = [
        "Attention layers map contextual token sequences through learned projections and residual connections.",
        "A tokenizer vocabulary trades embedding-table size against the number of positions consumed by text.",
        "def update_parameters(model, optimizer):\n    optimizer.zero_grad()\n    model.loss().backward()\n    optimizer.step()",
        "fn normalize_batch(values: &mut [f32]) { for value in values { *value = value.tanh(); } }",
        "भाषा प्रौद्योगिकी में प्रशिक्षण पाठ और मूल्यांकन पाठ को अलग रखना आवश्यक है।",
        "機械学習の評価では、学習用文書とテスト用文書を明確に分離します。",
        "يجب فصل بيانات تدريب المفردات عن النصوص المستخدمة لقياس جودة التقطيع.",
        "Türkçe alt sözcük deneylerinde eğitim ve değerlendirme belgeleri birbirinden ayrılır.",
        "Suomenkielisessä kokeessa sanasto opetetaan eri asiakirjoilla kuin ne, joilla tulokset mitataan.",
        "Katika jaribio la lugha, hati za mafunzo hazipaswi kutumiwa tena kama hati za tathmini.",
        "Nínú ìdánwò èdè, a ya àwọn ìwé ìkẹ́kọ̀ọ́ sọ́tọ̀ kúrò ní àwọn ìwé ìdánwò.",
        "പരിശീലന രേഖകളും മൂല്യനിർണ്ണയ രേഖകളും പരീക്ഷണത്തിൽ വേർതിരിച്ചിരിക്കണം.",
        "የስልጠና ሰነዶች ከግምገማ ሰነዶች ተለይተው መቀመጥ አለባቸው።",
    ] * 20 + [
        (
            f"Synthetic smoke-training record {index:04d}: "
            f"identifier_{index:04x} route_{index % 97:02d} shard_{index % 53:02d} "
            f"checksum_{(index * 2654435761) & 0xFFFFFFFF:08x}."
        )
        for index in range(512)
    ]

    def __init__(
        self,
        tokenizer: Optional[CustomTokenizer] = None,
        training_corpus: Optional[List[str]] = None,
        evaluation_corpora: Optional[Dict[str, str]] = None,
    ):
        """
        Initializes the empirical benchmark suite with a pre-trained or corpus-trained tokenizer.

        Args:
            tokenizer: Optional pre-configured CustomTokenizer.
            training_corpus: Documents used only for tokenizer training. Required when a
                pre-configured tokenizer is supplied so overlap can be checked.
            evaluation_corpora: Named documents used only for measurement.
        """
        self.training_corpus = list(training_corpus if training_corpus is not None else self.TRAINING_CORPUS)
        self.evaluation_corpora = dict(evaluation_corpora if evaluation_corpora is not None else self.BENCHMARK_CORPORA)
        if not self.training_corpus or not self.evaluation_corpora:
            raise ValueError("benchmark requires non-empty training and evaluation corpora")
        if set(self.training_corpus).intersection(self.evaluation_corpora.values()):
            raise ValueError("training and evaluation corpora overlap by document identity")
        if tokenizer is not None and training_corpus is None:
            raise ValueError("training_corpus is required when passing a pre-configured tokenizer")

        if tokenizer is None:
            self.tokenizer = CustomTokenizer.train_from_corpus(
                corpus=self.training_corpus,
                target_vocab_size=1000,
                min_edge_log_prob=float("-inf"),
                max_ngram_length=12,
                min_frequency=2,
                byte_fallback=True,
                split_digits=True,
                verbose=False,
            )
            self._require_vocab_size(self.tokenizer, 1000, "UniqToken")
        else:
            self.tokenizer = tokenizer

    @staticmethod
    def _require_vocab_size(tokenizer: Any, target_vocab: int, label: str) -> None:
        actual_vocab = int(tokenizer.vocab_size)
        if actual_vocab != target_vocab:
            raise RuntimeError(f"{label} produced {actual_vocab} pieces; requested budget is {target_vocab}")

    def evaluate_dataset(self, name: str, text: str, warmup: int = 2, iterations: int = 5) -> BenchmarkMetrics:
        raw_bytes = text.encode("utf-8")
        num_bytes = len(raw_bytes)
        num_chars = len(text)
        num_words = max(len(text.split()), 1)

        # Warmup
        for _ in range(warmup):
            _ = self.tokenizer.encode_to_ids(text)

        # 1. Encode Speed & RAM Profiling
        tracemalloc.start()
        t0 = time.perf_counter()
        token_ids: List[int] = []
        for _ in range(iterations):
            token_ids = self.tokenizer.encode_to_ids(text)
        t_encode = (time.perf_counter() - t0) / iterations
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = peak_mem / (1024.0 * 1024.0)

        num_tokens = len(token_ids)
        encode_kbs = (num_bytes / 1024.0) / max(t_encode, 1e-6)
        encode_tokens_sec = num_tokens / max(t_encode, 1e-6)

        # 2. Decode Speed
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = self.tokenizer.decode(token_ids)
        t_decode = (time.perf_counter() - t0) / iterations
        decode_kbs = (num_bytes / 1024.0) / max(t_decode, 1e-6)

        # 3. Offset Mapping Overhead & Fallback Rate
        t0 = time.perf_counter()
        tokens_with_offsets = []
        for _ in range(iterations):
            tokens_with_offsets = self.tokenizer.encode_with_offsets(text)
        t_offsets = (time.perf_counter() - t0) / iterations
        offset_overhead = t_offsets / max(t_encode, 1e-6)

        fallback_tokens = sum(1 for t in tokens_with_offsets if t.text.startswith("<0x") and t.text.endswith(">"))
        fallback_rate_pct = (fallback_tokens / max(num_tokens, 1)) * 100.0

        bytes_per_token = num_bytes / max(num_tokens, 1)
        tokens_per_unicode_character = num_tokens / max(num_chars, 1)

        return BenchmarkMetrics(
            dataset_name=name,
            model_kind="tokenizer_only",
            num_chars=num_chars,
            num_bytes=num_bytes,
            num_words=num_words,
            num_tokens=num_tokens,
            bytes_per_token=round(bytes_per_token, 3),
            tokens_per_unicode_character=round(tokens_per_unicode_character, 3),
            encode_speed_kbs=round(encode_kbs, 2),
            encode_speed_tokens_sec=round(encode_tokens_sec, 1),
            decode_speed_kbs=round(decode_kbs, 2),
            offset_overhead_ratio=round(offset_overhead, 2),
            peak_ram_mb=round(peak_ram_mb, 3),
            fallback_rate_pct=round(fallback_rate_pct, 2),
        )

    def run_all_benchmarks(self) -> List[BenchmarkMetrics]:
        results: List[BenchmarkMetrics] = []
        for name, text in self.evaluation_corpora.items():
            metrics = self.evaluate_dataset(name, text)
            results.append(metrics)
        return results

    def evaluate_payload_sizes(self) -> List[BenchmarkMetrics]:
        """Measure 1 MiB and 10 MiB payloads without making CI runs expensive."""
        seed = self.evaluation_corpora["English_Prose"]
        seed_bytes = len(seed.encode("utf-8"))
        results: List[BenchmarkMetrics] = []
        for size_mib in (1, 10):
            target_bytes = size_mib * 1024 * 1024
            payload = seed * ((target_bytes + seed_bytes - 1) // seed_bytes)
            results.append(
                self.evaluate_dataset(
                    f"English_{size_mib}MiB",
                    payload,
                    warmup=1,
                    iterations=1,
                )
            )
        return results

    def evaluate_indentation_compression(self) -> Dict[str, Any]:
        """
        Measures context token savings with indentation compression enabled during training & inference.
        """
        code_corpus = self.evaluation_corpora["Python_Code"]

        # Use identical training data and vocabulary budgets so the comparison
        # measures indentation compression rather than unrelated model quality.
        plain_tok = CustomTokenizer.train_from_corpus(
            corpus=self.training_corpus,
            target_vocab_size=500,
            min_edge_log_prob=float("-inf"),
            compress_indents=False,
            verbose=False,
        )

        # Train code model with indentation compression enabled.
        code_tok = CustomTokenizer.train_from_corpus(
            corpus=self.training_corpus,
            target_vocab_size=500,
            min_edge_log_prob=float("-inf"),
            compress_indents=True,
            verbose=False,
        )
        self._require_vocab_size(plain_tok, 500, "plain indentation tokenizer")
        self._require_vocab_size(code_tok, 500, "compressed-indentation tokenizer")

        # Plain tokenization (spaces tokenized individually)
        plain_ids = plain_tok.encode_to_ids(code_corpus)

        # CustomTokenizer compresses indentation during normal encoding.
        compressed_ids = code_tok.encode_to_ids(code_corpus)

        token_delta = len(plain_ids) - len(compressed_ids)
        token_reduction = token_delta / len(plain_ids) * 100.0

        return {
            "plain_token_count": len(plain_ids),
            "compressed_token_count": len(compressed_ids),
            "token_delta": token_delta,
            "reduction_percentage": round(token_reduction, 2),
        }

    def evaluate_unigram_vs_bpe(self) -> Dict[str, Any]:
        text = self.evaluation_corpora["English_Prose"] + self.evaluation_corpora["Python_Code"]
        training_text = "\n".join(self.training_corpus)
        chunks = self.tokenizer.pre_tokenizer.pre_tokenize(self.tokenizer.normalizer.normalize(training_text))

        # Train BPE
        bpe_trainer = BPETrainer(target_vocab_size=self.tokenizer.vocab_size, byte_fallback=True)
        bpe_model = bpe_trainer.train(chunks)
        if len(bpe_model.vocab) != self.tokenizer.vocab_size:
            raise RuntimeError(
                f"BPE produced {len(bpe_model.vocab)} pieces; comparison requires {self.tokenizer.vocab_size}"
            )

        # Unigram stats
        t0 = time.perf_counter()
        unigram_tokens = self.tokenizer.encode(text)
        t_unigram = time.perf_counter() - t0

        # BPE stats
        t0 = time.perf_counter()
        bpe_tokens: List[str] = []
        eval_chunks = self.tokenizer.pre_tokenizer.pre_tokenize(self.tokenizer.normalizer.normalize(text))
        for c in eval_chunks:
            bpe_tokens.extend(bpe_model.encode(c))
        t_bpe = time.perf_counter() - t0

        return {
            "unigram_token_count": len(unigram_tokens),
            "bpe_token_count": len(bpe_tokens),
            "unigram_bytes_per_token": round(len(text.encode("utf-8")) / len(unigram_tokens), 3),
            "bpe_bytes_per_token": round(len(text.encode("utf-8")) / len(bpe_tokens), 3),
            "unigram_encode_sec": round(t_unigram, 4),
            "bpe_encode_sec": round(t_bpe, 4),
        }

    def evaluate_cem(self) -> Dict[str, Any]:
        """
        Cross-Entropy Merging post-training compression gain in the fixed-budget
        scenario. Uses a controlled repetitive-phrase corpus (the embedded
        English/Code corpora are far too small to stress a vocab budget): a
        Unigram model trained at a tight vocab limit leaves common words split
        into subwords; CEM then recovers them as single tokens without
        retraining, cutting the token count.
        """
        training_phrases = [
            "the quick brown fox jumps over the lazy dog",
            "the quick fox and the lazy dog",
            "jumping foxes are quick and brown",
            "brown dogs are quick",
        ]
        evaluation_phrases = [
            "the quick brown dog watches a jumping fox",
            "quick foxes pass the brown and lazy dogs",
        ]
        training_text = " ".join(training_phrases) * 30
        text = " ".join(evaluation_phrases) * 15
        chunks = self.tokenizer.pre_tokenizer.pre_tokenize(self.tokenizer.normalizer.normalize(training_text))

        constrained = UnigramTrainer(
            target_vocab_size=300,
            max_ngram_length=6,
            min_frequency=2,
            byte_fallback=True,
            min_edge_log_prob=float("-inf"),
        ).train(chunks, verbose=False)

        before = sum(len(constrained.encode(c)) for c in chunks)
        cem = CrossEntropyMerging(max_merges=200)
        improved = cem.optimize(constrained, chunks)
        after = sum(len(improved.encode(c)) for c in chunks)

        return {
            "cem_merges_applied": len(cem.merges),
            "vocab_before": len(constrained.vocab),
            "vocab_after": len(improved.vocab),
            "token_count_before": before,
            "token_count_after": after,
            "token_reduction_pct": round((before - after) / max(before, 1) * 100.0, 2),
            "bytes_per_token_before": round(len(text.encode("utf-8")) / max(before, 1), 3),
            "bytes_per_token_after": round(len(text.encode("utf-8")) / max(after, 1), 3),
        }

    def evaluate_superbpe(self) -> Dict[str, Any]:
        """
        SuperBPE ('space travel') post-training compression gain. Merges tokens
        that span word boundaries (e.g. ``the\u2581quick``) using the same
        fixed-budget stress corpus as the CEM metric, measured end-to-end
        through the tokenizer pipeline.
        """
        training_phrases = [
            "the quick brown fox jumps over the lazy dog",
            "the quick fox and the lazy dog",
            "jumping foxes are quick and brown",
            "brown dogs are quick",
        ]
        evaluation_phrases = [
            "the quick brown dog watches a jumping fox",
            "quick foxes pass the brown and lazy dogs",
        ]
        training_text = " ".join(training_phrases) * 30
        text = " ".join(evaluation_phrases) * 15
        chunks = self.tokenizer.pre_tokenizer.pre_tokenize(self.tokenizer.normalizer.normalize(training_text))

        constrained = UnigramTrainer(
            target_vocab_size=300,
            max_ngram_length=6,
            min_frequency=2,
            byte_fallback=True,
            min_edge_log_prob=float("-inf"),
        ).train(chunks, verbose=False)

        base_tok = CustomTokenizer(self.tokenizer.normalizer, self.tokenizer.pre_tokenizer, constrained)
        before = len(base_tok.encode(text))

        superbpe = CrossEntropyMerging(max_merges=100, cross_word=True)
        improved = superbpe.optimize(constrained, chunks)
        if not superbpe.merges:
            raise RuntimeError("SuperBPE benchmark configuration was a no-op: no cross-word merges were learned")
        improved_tok = CustomTokenizer(self.tokenizer.normalizer, self.tokenizer.pre_tokenizer, improved)
        after = len(improved_tok.encode(text))

        return {
            "superbpe_merges_applied": len(superbpe.merges),
            "vocab_before": len(constrained.vocab),
            "vocab_after": len(improved.vocab),
            "token_count_before": before,
            "token_count_after": after,
            "token_reduction_pct": round((before - after) / max(before, 1) * 100.0, 2),
            "bytes_per_token_before": round(len(text.encode("utf-8")) / max(before, 1), 3),
            "bytes_per_token_after": round(len(text.encode("utf-8")) / max(after, 1), 3),
        }

    def evaluate_external_baselines(self) -> Dict[str, Any]:
        text = self.evaluation_corpora["English_Prose"]
        results: Dict[str, Any] = {}

        # 1. UniqToken (Rust C-Extension or Python fallback)
        try:
            import uniqtoken_core

            uniqtoken_label = "UniqToken (Rust Core)"
        except ImportError:
            try:
                import uniqtoken_core

                uniqtoken_label = "UniqToken (Rust Core)"
            except ImportError:
                uniqtoken_label = "UniqToken (Pure Python)"

        t0 = time.perf_counter()
        for _ in range(5):
            uniqtoken_tokens = self.tokenizer.encode_to_ids(text)
        t_uniqtoken = (time.perf_counter() - t0) / 5.0
        results[uniqtoken_label] = {
            "tokens": len(uniqtoken_tokens),
            "time_sec": round(t_uniqtoken, 4),
            "tokens_sec": round(len(uniqtoken_tokens) / max(t_uniqtoken, 1e-6), 1),
        }

        # 2. HuggingFace Tokenizers (Rust C-FFI) via our HF Exporter.
        try:
            from tokenizers import Tokenizer
            import json

            with TemporaryDirectory() as tmp_dir:
                hf_json = HuggingFaceExporter.export_to_hf_dict(self.tokenizer)
                path = Path(tmp_dir) / "hf_tok.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(hf_json, f)

                hf_tok = Tokenizer.from_file(str(path))
                t0 = time.perf_counter()
                for _ in range(5):
                    hf_encoded = hf_tok.encode(text)
                t_hf = (time.perf_counter() - t0) / 5.0

                results["HuggingFace (Rust)"] = {
                    "tokens": len(hf_encoded.ids),
                    "time_sec": round(t_hf, 5),
                    "tokens_sec": round(len(hf_encoded.ids) / max(t_hf, 1e-6), 1),
                }
        except Exception as e:
            results["HuggingFace (Rust)"] = {"error": str(e)}

        # 3. SentencePiece trained on the disjoint training corpus.
        try:
            import sentencepiece as spm
        except ImportError as exc:
            results["SentencePiece (Unigram)"] = {"error": f"unavailable: {exc}"}
        else:
            with TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                corpus_path = tmp_path / "corpus.txt"
                training_lines = self.training_corpus
                corpus_path.write_text("\n".join(training_lines), encoding="utf-8")
                model_prefix = tmp_path / "sentencepiece"
                spm.SentencePieceTrainer.train(
                    input=str(corpus_path),
                    model_prefix=str(model_prefix),
                    model_type="unigram",
                    vocab_size=1000,
                    character_coverage=1.0,
                    byte_fallback=True,
                    hard_vocab_limit=True,
                    bos_id=-1,
                    eos_id=-1,
                    pad_id=-1,
                    minloglevel=2,
                )
                sp_processor = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")
                if sp_processor.get_piece_size() != 1000:
                    raise RuntimeError(
                        f"SentencePiece produced {sp_processor.get_piece_size()} pieces; requested budget is 1000"
                    )
                t0 = time.perf_counter()
                for _ in range(5):
                    sp_ids = sp_processor.encode(text, out_type=int)
                t_sentencepiece = (time.perf_counter() - t0) / 5.0
                results["SentencePiece (Unigram)"] = {
                    "tokens": len(sp_ids),
                    "time_sec": round(t_sentencepiece, 5),
                    "tokens_sec": round(len(sp_ids) / max(t_sentencepiece, 1e-6), 1),
                }

        # 4. tiktoken uses a fixed pre-trained vocabulary, so its compression
        # numbers are informational rather than a same-corpus comparison.
        try:
            import tiktoken

            tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            t0 = time.perf_counter()
            for _ in range(5):
                tiktoken_ids = tiktoken_encoder.encode(text)
            t_tiktoken = (time.perf_counter() - t0) / 5.0
            results["tiktoken (cl100k_base)"] = {
                "tokens": len(tiktoken_ids),
                "time_sec": round(t_tiktoken, 5),
                "tokens_sec": round(len(tiktoken_ids) / max(t_tiktoken, 1e-6), 1),
            }
        except Exception as e:
            results["tiktoken (cl100k_base)"] = {"error": str(e)}

        return results

    def print_summary_report(self, include_large_payloads: bool = False) -> None:
        results = self.run_all_benchmarks()

        header = f"{'Dataset':<22} | {'Bytes':<7} | {'Tokens':<7} | {'Bytes/Tok':<10} | {'Tokens/char':<10} | {'Enc KB/s':<10} | {'Tok/sec':<10} | {'RAM (MB)':<9} | {'Fallback %':<11} | {'Offset Overhead':<15}"
        sep = "=" * len(header)
        print(sep)
        print("UNIQTOKEN TOKENIZER EMPIRICAL BENCHMARK REPORT")
        print(sep)
        print(header)
        print("-" * len(header))

        for r in results:
            print(
                f"{r.dataset_name:<22} | {r.num_bytes:<7} | {r.num_tokens:<7} | "
                f"{r.bytes_per_token:<10} | {r.tokens_per_unicode_character:<10} | "
                f"{r.encode_speed_kbs:<10} | {r.encode_speed_tokens_sec:<10} | "
                f"{r.peak_ram_mb:<9} | {r.fallback_rate_pct:<11} | "
                f"{r.offset_overhead_ratio:<15}x"
            )

        if include_large_payloads:
            print("\n" + "=" * 85)
            print("LARGE-PAYLOAD THROUGHPUT")
            print("=" * 85)
            for r in self.evaluate_payload_sizes():
                print(
                    f"  {r.dataset_name:<16} | {r.num_bytes / (1024 * 1024):>5.2f} MiB | "
                    f"{r.num_tokens} tokens | {r.encode_speed_kbs} KB/s | "
                    f"{r.encode_speed_tokens_sec} tok/s"
                )

        print("\n" + "=" * 85)
        print("CODE INDENTATION COMPRESSION CONTEXT SAVINGS")
        print("=" * 85)
        indent_stats = self.evaluate_indentation_compression()
        print(f"  Plain Code Tokens       : {indent_stats['plain_token_count']}")
        print(f"  Compressed Code Tokens  : {indent_stats['compressed_token_count']}")
        if indent_stats["token_delta"] >= 0:
            print(
                f"  Context Capacity Saved  : {indent_stats['token_delta']} tokens "
                f"({indent_stats['reduction_percentage']}% context reduction)"
            )
        else:
            print(
                f"  Token Increase          : {-indent_stats['token_delta']} tokens "
                f"({-indent_stats['reduction_percentage']}% regression)"
            )

        print("\n" + "=" * 85)
        print("HEAD-TO-HEAD: UNIGRAM VS. BPE ON IDENTICAL DATA")
        print("=" * 85)
        cmp_stats = self.evaluate_unigram_vs_bpe()
        print(
            f"  Unigram Token Count     : {cmp_stats['unigram_token_count']} tokens ({cmp_stats['unigram_bytes_per_token']} bytes/tok)"
        )
        print(
            f"  BPE Token Count         : {cmp_stats['bpe_token_count']} tokens ({cmp_stats['bpe_bytes_per_token']} bytes/tok)"
        )
        print(f"  Unigram Time (Trie DAG) : {cmp_stats['unigram_encode_sec']}s")
        print(f"  BPE Time (Rank Merges)  : {cmp_stats['bpe_encode_sec']}s")

        print("\n" + "=" * 85)
        print("POST-TRAINING CROSS-ENTROPY MERGING (CEM)")
        print("=" * 85)
        cem_stats = self.evaluate_cem()
        print(
            f"  Vocab                   : {cem_stats['vocab_before']} -> {cem_stats['vocab_after']} tokens "
            f"({cem_stats['cem_merges_applied']} merges)"
        )
        print(
            f"  Token Count             : {cem_stats['token_count_before']} -> {cem_stats['token_count_after']} "
            f"({cem_stats['token_reduction_pct']}% fewer)"
        )
        print(
            f"  Compression             : {cem_stats['bytes_per_token_before']} -> {cem_stats['bytes_per_token_after']} bytes/tok"
        )

        print("\n" + "=" * 85)
        print("POST-TRAINING SUPERBPE (SPACE TRAVEL)")
        print("=" * 85)
        sbp_stats = self.evaluate_superbpe()
        print(
            f"  Vocab                   : {sbp_stats['vocab_before']} -> {sbp_stats['vocab_after']} tokens "
            f"({sbp_stats['superbpe_merges_applied']} merges)"
        )
        print(
            f"  Token Count             : {sbp_stats['token_count_before']} -> {sbp_stats['token_count_after']} "
            f"({sbp_stats['token_reduction_pct']}% fewer)"
        )
        print(
            f"  Compression             : {sbp_stats['bytes_per_token_before']} -> {sbp_stats['bytes_per_token_after']} bytes/tok"
        )

        print("\n" + "=" * 85)
        print("COMPARATIVE ENGINE BASELINES")
        print("=" * 85)
        baselines = self.evaluate_external_baselines()
        for engine, stats in baselines.items():
            if "error" in stats:
                print(f"  {engine:<24} : {stats['error']}")
            else:
                print(
                    f"  {engine:<24} : {stats['tokens']} tokens | {stats['tokens_sec']} tok/sec ({stats['time_sec']}s)"
                )
        print("=" * 85)

        print("\n" + "=" * 85)
        print("CROSS-LINGUISTIC HELD-OUT TOKENIZATION")
        print("=" * 85)
        cross_rows = self.evaluate_cross_linguistic_baselines()
        for cr in cross_rows:
            print(
                f"  {cr['name']:<24} | UniqTok: {cr['uniq_bytes_per_tok']:>5.2f} B/Tok (tokens/char {cr['uniq_tokens_per_unicode_character']:>5.2f}) | "
                f"tiktoken: {cr['tiktoken_bytes_per_tok']:>5.2f} B/Tok (tokens/char {cr['tiktoken_tokens_per_unicode_character']:>5.2f}) | "
                f"Delta: {cr['compression_delta']:<8} | FB: {cr['fallback_pct']:.1f}%"
            )
        print("=" * 85)

    def evaluate_cross_linguistic_baselines(self) -> List[Dict[str, Any]]:
        """
        Evaluates compression and tokens per whitespace-delimited unit across African and Indic
        corpora against external production BPE tokenizers (tiktoken cl100k_base) using the
        suite's canonical iteration budget (warmup=2, iterations=5).

        Returns:
            List of dictionaries containing comparative metrics for Table 3.
        """
        target_keys = [
            ("Agglutinative_Swahili", "Agglutinative Swahili", "Latin"),
            ("Tonal_Yoruba", "Tonal Yoruba", "Latin + Diacritics"),
            ("Agglutinative_Malayalam", "Agglutinative Malayalam", "Dravidian (മലയാളം)"),
            ("Geez_Amharic", "Ge'ez Amharic", "Ethiopic Fidäl (ግዕዝ)"),
        ]

        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = None

        rows: List[Dict[str, Any]] = []
        for key, name, script in target_keys:
            if key not in self.evaluation_corpora:
                continue
            text = self.evaluation_corpora[key]
            metrics = self.evaluate_dataset(key, text, warmup=2, iterations=5)
            raw_bytes = metrics.num_bytes
            characters = metrics.num_chars
            uniq_toks = metrics.num_tokens
            uniq_bpt = metrics.bytes_per_token
            uniq_fert = metrics.tokens_per_unicode_character
            fallback_pct = metrics.fallback_rate_pct

            if enc is not None:
                tt_toks = len(enc.encode(text))
                tt_bpt = round(raw_bytes / max(tt_toks, 1), 2)
                tt_fert = round(tt_toks / max(characters, 1), 2)
                delta_pct = ((uniq_bpt - tt_bpt) / max(tt_bpt, 1e-6)) * 100.0
                comp_delta = f"{delta_pct:+.1f}%"
            else:
                tt_toks = 0
                tt_bpt = 0.0
                tt_fert = 0.0
                comp_delta = "N/A"

            rows.append(
                {
                    "key": key,
                    "name": name,
                    "script": script,
                    "raw_bytes": raw_bytes,
                    "uniq_tokens": uniq_toks,
                    "uniq_bytes_per_tok": round(uniq_bpt, 2),
                    "uniq_tokens_per_unicode_character": round(uniq_fert, 2),
                    "tiktoken_tokens": tt_toks,
                    "tiktoken_bytes_per_tok": tt_bpt,
                    "tiktoken_tokens_per_unicode_character": tt_fert,
                    "compression_delta": comp_delta,
                    "fallback_pct": fallback_pct,
                }
            )
        return rows

    def evaluate_vocab_scaling(self, vocab_sizes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Evaluates compression and throughput scaling across different vocabulary budgets.

        Args:
            vocab_sizes: Optional list of target vocabulary sizes to evaluate. Defaults to
                [400, 800, 1600, 3200].

        Returns:
            List of dictionaries containing target vocabulary, actual vocabulary, tokens,
            bytes per token, tokens per word, throughput, and fallback rate metrics.
        """
        if vocab_sizes is None:
            vocab_sizes = [400, 800, 1600, 3200]

        combined_text = "\n".join(self.evaluation_corpora.values())
        results: List[Dict[str, Any]] = []

        for vs in vocab_sizes:
            if vs < 1:
                raise ValueError("vocabulary budgets must be positive")
            tok = CustomTokenizer.train_from_corpus(
                corpus=self.training_corpus,
                target_vocab_size=vs,
                min_frequency=1,
                byte_fallback=True,
                min_edge_log_prob=float("-inf"),
                verbose=False,
            )
            self._require_vocab_size(tok, vs, "UniqToken")
            t0 = time.perf_counter()
            tokens = tok.encode(combined_text)
            t_enc = max(time.perf_counter() - t0, 1e-6)
            num_bytes = len(combined_text.encode("utf-8"))
            num_tok = len(tokens)

            fb_tokens = sum(1 for t in tokens if t.startswith("<0x") and t.endswith(">"))
            results.append(
                {
                    "target_vocab": vs,
                    "actual_vocab": tok.vocab_size,
                    "tokens": num_tok,
                    "bytes_per_tok": round(num_bytes / max(num_tok, 1), 3),
                    "tokens_per_unicode_character": round(num_tok / max(len(combined_text), 1), 3),
                    "tok_per_sec": round(num_tok / t_enc, 1),
                    "fallback_rate_pct": round((fb_tokens / max(num_tok, 1)) * 100.0, 2),
                }
            )

        return results

    def export_markdown_report(self, output_path: str) -> None:
        """Exports full benchmark results to a formatted GitHub Markdown document."""
        results = self.run_all_benchmarks()
        lines = [
            "# UniqToken Tokenizer Benchmark Report",
            "",
            "## Multilingual Throughput & Compression",
            "",
            "| Dataset | Bytes | Tokens | Bytes/Tok | Tokens/Unicode char | Enc KB/s | Tok/sec | RAM (MB) | Fallback % | Offset Overhead |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in results:
            lines.append(
                f"| {r.dataset_name} | {r.num_bytes} | {r.num_tokens} | {r.bytes_per_token} | {r.tokens_per_unicode_character} | {r.encode_speed_kbs} | {r.encode_speed_tokens_sec} | {r.peak_ram_mb} | {r.fallback_rate_pct}% | {r.offset_overhead_ratio}x |"
            )

        baselines = self.evaluate_external_baselines()
        lines.extend(
            [
                "",
                "## Comparative Baselines",
                "",
                "| Engine | Tokens | Throughput (tok/sec) | Time (s) |",
                "| :--- | :--- | :--- | :--- |",
            ]
        )
        for engine, stats in baselines.items():
            if "error" not in stats:
                lines.append(f"| {engine} | {stats['tokens']} | {stats['tokens_sec']} | {stats['time_sec']} |")

        cross_rows = self.evaluate_cross_linguistic_baselines()
        lines.extend(
            [
                "",
                "## Cross-Linguistic Held-Out Tokenization",
                "",
                "| Language / Family | Script Family | Raw Bytes | UniqToken Tokens | UniqToken Bytes/Tok | UniqToken Tokens/char | Tiktoken Tokens | Tiktoken Bytes/Tok | Tiktoken Tokens/char | Compression Delta | Fallback Rate |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for cr in cross_rows:
            lines.append(
                f"| **{cr['name']}** | {cr['script']} | {cr['raw_bytes']:,} | {cr['uniq_tokens']:,} | {cr['uniq_bytes_per_tok']:.2f} | {cr['uniq_tokens_per_unicode_character']:.2f} | {cr['tiktoken_tokens']:,} | {cr['tiktoken_bytes_per_tok']:.2f} | {cr['tiktoken_tokens_per_unicode_character']:.2f} | {cr['compression_delta']} | **{cr['fallback_pct']:.1f}%** |"
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def export_latex_report(self, output_path: str) -> None:
        """Exports full benchmark results to a publication-ready LaTeX table."""
        results = self.run_all_benchmarks()
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"\textbf{Dataset} & \textbf{Bytes} & \textbf{Tokens} & \textbf{Bytes/Tok} & \textbf{Tokens/char} & \textbf{Enc KB/s} & \textbf{Tok/s} & \textbf{RAM (MB)} & \textbf{Fallback \%} \\",
            r"\midrule",
        ]
        for r in results:
            clean_name = r.dataset_name.replace("_", r"\_")
            lines.append(
                f"{clean_name} & {r.num_bytes} & {r.num_tokens} & {r.bytes_per_token:.2f} & {r.tokens_per_unicode_character:.2f} & {r.encode_speed_kbs:.1f} & {r.encode_speed_tokens_sec:.1f} & {r.peak_ram_mb:.2f} & {r.fallback_rate_pct:.1f}\\% \\\\"
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{UniqToken Empirical Benchmark Suite Evaluation Across Multilingual Corpora.}",
                r"\label{tab:uniqtoken_benchmarks}",
                r"\end{table*}",
            ]
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UniqToken tokenizer benchmarks.")
    parser.add_argument(
        "--large-payloads",
        action="store_true",
        help="also run the 1 MiB and 10 MiB local throughput workloads",
    )
    parser.add_argument(
        "--export-markdown",
        type=str,
        default=None,
        help="path to write markdown benchmark report",
    )
    parser.add_argument(
        "--export-latex",
        type=str,
        default=None,
        help="path to write LaTeX benchmark table",
    )
    args = parser.parse_args()
    suite = TokenizerBenchmarkSuite()
    suite.print_summary_report(include_large_payloads=args.large_payloads)

    if args.export_markdown:
        suite.export_markdown_report(args.export_markdown)
        print(f"\n[Exporter] Saved Markdown benchmark report to {args.export_markdown}")
    if args.export_latex:
        suite.export_latex_report(args.export_latex)
        print(f"\n[Exporter] Saved LaTeX benchmark table to {args.export_latex}")
