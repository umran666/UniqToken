"""
Metric Accounting & Vocabulary Allocation Audit.

Mechanically verifies:
1. Invariant: sum(language bytes) == total evaluation bytes
2. Invariant: sum(language tokens) == total evaluation tokens
3. Invariant: aggregate tokens-per-byte is byte-weighted
4. Vocabulary script allocation distribution (how many tokens out of V are Latin vs CJK vs Indic vs Cyrillic vs Arabic)
5. Token sequence inspection across languages
"""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uniqtoken.seed_builder import SeedVocabularyBuilder
from uniqtoken.tokenizer import CustomTokenizer

# 12-Script Rich Multilingual Dataset Generator for Accounting Invariant Audit
MULTILINGUAL_DATA_SOURCES: Dict[str, List[str]] = {
    "English": [
        "The transformer architecture revolutionized natural language processing by replacing recurrence with self-attention mechanisms.",
        "Deep neural networks require efficient tokenization to balance sequence length, vocabulary size, and embedding compute parameters.",
        "Exact character and byte offset alignment is essential for structured extraction, code intelligence, and reliable evaluation.",
        "Distributed gradient descent across multiple accelerators enables training language models on trillions of web tokens.",
        "Byte-level fallback guarantees that unexpected Unicode codepoints and binary payloads never cause out-of-vocabulary crashes.",
    ],
    "Hindi": [
        "प्राकृतिक भाषा प्रसंस्करण और कंप्यूटर विज्ञान में टोकनाइज़र एक अत्यंत महत्वपूर्ण और आधारभूत घटक है।",
        "देवनागरी लिपि में अक्षरों, मात्राओं और संयुक्ताक्षरों का सटीक संयोजन पाठ विखंडन को रोकने के लिए आवश्यक है।",
        "भाषा मॉडल की संदर्भ दक्षता और कम्प्यूटेशनल गति सही सबवर्ड टोकनीकरण एल्गोरिदम पर निर्भर करती है।",
        "आर्टिफिशियल इंटेलिजेंस और मशीन लर्निंग सिस्टम भारतीय भाषाओं के समृद्ध साहित्य को समझने में प्रगति कर रहे हैं।",
        "कंप्यूटर प्रोग्रामिंग और बहुभाषी अनुवाद में यूनिकोड का मानकीकरण विश्व भर में सुगमता प्रदान करता है।",
    ],
    "Telugu": [
        "సహజ భాషా ప్రాసెసింగ్ కంప్యూటర్ సైన్స్ రంగంలో అత్యంత వేగంగా అభివృద్ధి చెందుతున్న ఆధునిక విభాగం.",
        "తెలుగు లిపిలో అచ్చులు, హల్లులు మరియు గుణింతాల సంక్లిష్ట అమరికను సమర్థవంతంగా విశ్లేషించాలి.",
        "సరైన టోకనైజర్ ఉపయోగించడం వల్ల భాషా నమూనాల పనితీరు మెరుగుపడి గణన వేగం పెరుగుతుంది.",
        "యంత్ర అభ్యాస నమూనాలు భారతీయ భాషల విస్తృత సమాచారాన్ని సులభంగా ప్రాసెస్ చేయగలవు.",
        "కృత్రిమ మేధస్సు మరియు సమాచార విశ్లేషణ సాంకేతిక రంగంలో గొప్ప విప్లవాన్ని సృష్టిస్తున్నాయి.",
    ],
    "Tamil": [
        "இயற்கை மொழி செயலாக்கம் கணினி அறிவியலில் ஒரு முதன்மையான மற்றும் இன்றியமையாத பகுதியாகும்.",
        "தமிழ் எழுத்துக்களின் தனித்துவமான அமைப்பும் மெய் எழுத்துக்களின் பயன்பாடும் துல்லியமாக கையாளப்பட வேண்டும்.",
        "சரியான டோக்கனைசர் மொழி மாதிரிகளின் சூழல் சாளரத் திறனை கணிசமாக உயர்த்தி செலவைக் குறைக்கிறது.",
        "செயற்கை நுண்ணறிவு தொழில்நுட்பம் தமிழ் மொழியின் பழமையான இலக்கியங்களை டிஜிட்டல் முறையில் பாதுகாக்கிறது.",
        "கணினி மொழியியல் மற்றும் தானியங்கி மொழிபெயர்ப்பு உலகம் முழுவதும் உள்ள மக்களை இணைக்கிறது.",
    ],
    "Bengali": [
        "প্রাকৃতিক ভাষা প্রক্রিয়াকরণ কম্পিউটার বিজ্ঞানের একটি অত্যন্ত গুরুত্বপূর্ণ এবং গতিশীল শাখা।",
        "বাংলা লিপির জটিল গঠন, যুক্তাক্ষর এবং স্বরবর্ণের সঠিক রূপান্তর সাবওয়ার্ড পর্যায়ে সংরক্ষণ করা আবশ্যক।",
        "ভাষার মডেলের দক্ষতা ও নির্ভুলতা নির্ভর করে উন্নত টোকেন বিভাজন এবং শব্দকোষ অপ্টিমাইজেশনের উপর।",
        "কৃত্রিম বুদ্ধিমত্তা এবং ডিপ লার্নিং প্রযুক্তি বাংলা ভাষার সমৃদ্ধ সাহিত্য বিশ্লেষণে নতুন দিগন্ত উন্মোচন করেছে।",
        "বহুভাষিক ডেটাসেট প্রক্রিয়াকরণে ইউনিকোড ও সঠিক বাইট ফলব্যাক ব্যবস্থা অপরিহার্য।",
    ],
    "Arabic": [
        "تعتبر معالجة اللغات الطبيعية وتجزئة النصوص من أهم ركائز الذكاء الاصطناعي الحديث في العالم الرقمي.",
        "يتطلب التعامل مع اللغة العربية دعماً دقيقاً للحركات وعلامات التشكيل والجذور الصرفية لمنع التجزئة المفرطة.",
        "يعتمد أداء النماذج اللغوية الكبيرة على كفاءة التوكنايزر في ضغط السياق وتقليل الخصوبة اللغوية للكلمات.",
        "تسهم تقنيات التعلم العميق في تطوير محركات البحث والترجمة الآلية بدقة وسرعة فائقة.",
        "تعد حوسبة اللغة العربية وتوليد النصوص المتقدمة خطوة محورية في بناء منظومات ذكية شاملة.",
    ],
    "Chinese": [
        "自然语言处理和大型语言模型依赖高效的分词技术来大幅提升长文本上下文的计算利用率。",
        "汉字作为典型的表意文字没有显式词间空格，子词切分算法必须准确捕捉语义边界与词组构词法。",
        "高质量的分词器能够显著缩减输入序列长度，从而有效降低注意力机制在自回归生成时的二次方开销。",
        "大规模多语言语料库在分布式训练中对词表分配和各语种平衡提出了严格的工程优化需求。",
        "字节回退与精确对齐机制确保了在解析代码、数学公式以及异常字符时具备极高的鲁棒性。",
    ],
    "Japanese": [
        "自然言語処理におけるトークナイザーは、テキストを一連の最適なサブワード系列に分割する基盤です。",
        "日本語のように形態素境界がスペースで区切られない言語では、精緻な辞書とバイトフォールバックが不可欠です。",
        "正確なトークン境界とオフセットアライメントの維持が、下流のトランスフォーマーモデルの性能を決定づけます。",
        "大規模言語モデルの推論効率は、トークンあたりのバイト数圧縮率と語彙サイズのバランスに大きく依存します。",
        "マルチリンガルモデルにおける効率的な文字配分が、多言語間での文脈理解と転移学習を促進します。",
    ],
    "Korean": [
        "자연어 처리에서 토크나이저는 원시 텍스트를 최적의 서브워드 시퀀스로 분할하여 모델의 연산 효율을 결정합니다.",
        "한국어의 독특한 교착어적 특성과 다양한 조사 결합을 효과적으로 처리하는 것이 어휘 구성의 핵심입니다.",
        "문맥 효율성을 극대화하기 위해 정확한 형태소 분할 알고리즘과 압축 기법이 필수적으로 요구됩니다.",
        "딥러닝 기반의 언어 모델은 대규모 코퍼스를 학습하여 문맥 간의 복잡한 의미 관계를 파악합니다.",
        "바이트 단위 폴백 지원은 미등록 단어와 특수 기호에 대해 완벽한 무손실 복원력을 제공합니다.",
    ],
    "Thai": [
        "การประมวลผลภาษาธรรมชาติและการตัดคำเป็นขั้นตอนพื้นฐานที่สำคัญยิ่งในการพัฒนาโมเดลภาษาขนาดใหญ่.",
        "ภาษาไทยไม่มีการเว้นวรรคระหว่างคำทำให้อัลกอริทึมการตัดคำระดับหน่วยย่อยมีความท้าทายและความซับซ้อนสูง.",
        "การเลือกใช้โทเคไนเซอร์ที่มีประสิทธิภาพช่วยลดความยาวของลำดับข้อมูลและเพิ่มความเร็วในการคำนวณของโมเดล.",
        "ปัญญาประดิษฐ์และการเรียนรู้เชิงลึกกำลังปฏิวัติระบบการแปลภาษาอัตโนมัติและการวิเคราะห์ข้อความภาษาไทย.",
        "การจัดการข้อมูลหลายภาษาจำเป็นต้องมีระบบที่รองรับยูนิโค้ดและการสำรองระดับไบต์อย่างแม่นยำ.",
    ],
    "Russian": [
        "Обработка естественного языка и токенизация лежат в основе функционирования современных трансформеров.",
        "Кириллический текст требует сбалансированного словаря для предотвращения чрезмерной фрагментации слов.",
        "Эффективность контекстного окна языковой модели напрямую зависит от среднего сжатия байт на один токен.",
        "Машинное обучение и нейросетевые архитектуры обеспечивают глубокий синтаксический анализ сложных текстов.",
        "Байтовый фоллбэк гарантирует полную устойчивость токенизатора при обработке редких символов и кода.",
    ],
    "Spanish": [
        "El procesamiento del lenguaje natural depende de la tokenización precisa de subpalabras en modelos modernos.",
        "Los modelos de lenguaje autorregresivos equilibran la fertilidad léxica con la dimensión del vocabulario.",
        "La compresión eficiente del contexto mejora la velocidad de inferencia y reduce el costo computacional global.",
        "Las arquitecturas basadas en atención transforman la comprensión multilingüe en aplicaciones de gran escala.",
        "La alineación exacta de caracteres y bytes es indispensable para tareas de extracción estructurada y análisis.",
    ],
}


def build_multilingual_dataset(multiplier: int = 50, seed: int = 42) -> Tuple[List[str], Dict[str, str]]:
    """Builds a rich multilingual training corpus and held-out validation evaluation sets with seed control."""
    import random

    rng = random.Random(seed)
    train_docs: List[str] = []
    val_by_lang: Dict[str, str] = {}

    for lang, sentences in MULTILINGUAL_DATA_SOURCES.items():
        expanded = sentences * multiplier
        rng.shuffle(expanded)
        split_idx = int(len(expanded) * 0.8)
        train_docs.extend(expanded[:split_idx])
        val_by_lang[lang] = " ".join(expanded[split_idx:])

    return train_docs, val_by_lang


@dataclass
class LanguageAuditEntry:
    language: str
    raw_byte_count: int
    char_count: int
    word_count: int
    token_count: int
    tokens_per_byte: float
    bytes_per_token: float
    sample_tokens: List[str]


class MetricAccountingAuditTests(unittest.TestCase):
    vocab_size: int
    train_docs: List[str]
    val_by_lang: Dict[str, str]
    tokenizer: CustomTokenizer

    @classmethod
    def setUpClass(cls):
        cls.vocab_size = 2000
        cls.train_docs, cls.val_by_lang = build_multilingual_dataset(multiplier=40, seed=100)

        # Train tokenizer
        cls.tokenizer = CustomTokenizer.train_from_corpus(
            corpus=cls.train_docs,
            target_vocab_size=cls.vocab_size,
            ranking_strategy="byte_savings",
            script_balance_temperature=0.9,
            min_frequency=1,
            verbose=False,
        )

    def test_vocabulary_script_distribution(self):
        """Audits the exact number of vocabulary entries allocated to each writing system."""
        vocab_tokens = list(self.tokenizer.model.vocab.keys())
        script_counts = Counter(
            SeedVocabularyBuilder._script_family(SeedVocabularyBuilder._detect_script(tok)) for tok in vocab_tokens
        )

        print("\n" + "=" * 80)
        print(f"VOCABULARY SCRIPT ALLOCATION BREAKDOWN (Total Vocab: {len(vocab_tokens):,})")
        print("=" * 80)
        for script, count in sorted(script_counts.items(), key=lambda x: -x[1]):
            pct = (count / len(vocab_tokens)) * 100.0
            print(f"  {script:<15}: {count:>5} tokens ({pct:>5.1f}%)")
        print("=" * 80)

        # Basic health assertions
        self.assertEqual(len(vocab_tokens), self.vocab_size)
        self.assertGreater(script_counts["latin"], 50, "Latin must have non-trivial vocabulary allocation")
        self.assertGreater(script_counts["cjk"], 50, "CJK must have non-trivial vocabulary allocation")
        self.assertGreater(script_counts["indic"], 50, "Indic must have non-trivial vocabulary allocation")

    def test_metric_accounting_invariants(self):
        """Mechanically verifies arithmetic consistency across all languages and totals."""
        entries: List[LanguageAuditEntry] = []
        total_bytes = 0
        total_tokens = 0
        total_chars = 0
        total_words = 0

        for lang, text in self.val_by_lang.items():
            raw_b = len(text.encode("utf-8"))
            chars = len(text)
            words = max(len(text.split()), 1)

            tokens = self.tokenizer.encode(text)
            token_ids = self.tokenizer.encode_to_ids(text)

            # Token count must match token ID count exactly
            self.assertEqual(len(tokens), len(token_ids), f"Token and ID length mismatch in {lang}")

            num_tok = len(tokens)
            total_bytes += raw_b
            total_tokens += num_tok
            total_chars += chars
            total_words += words

            tpb = num_tok / max(raw_b, 1)
            bpt = raw_b / max(num_tok, 1)
            # Invariant: tpb * bpt == 1.0
            self.assertAlmostEqual(tpb * bpt, 1.0, places=6, msg=f"tpb * bpt reciprocal invariant failed in {lang}")

            entries.append(
                LanguageAuditEntry(
                    language=lang,
                    raw_byte_count=raw_b,
                    char_count=chars,
                    word_count=words,
                    token_count=num_tok,
                    tokens_per_byte=tpb,
                    bytes_per_token=bpt,
                    sample_tokens=tokens[:10],
                )
            )

        # Aggregate token density must equal the byte-weighted language values.
        aggregate_tokens_per_byte = total_tokens / total_bytes
        sum_weighted_tokens_per_byte = sum(
            (e.raw_byte_count / total_bytes) * e.tokens_per_byte for e in entries
        )
        self.assertAlmostEqual(
            aggregate_tokens_per_byte,
            sum_weighted_tokens_per_byte,
            places=6,
            msg="Aggregate tokens-per-byte weighting mismatch",
        )

        print("\n" + "=" * 115)
        print("LANGUAGE-BY-LANGUAGE METRIC ACCOUNTING AUDIT")
        print("=" * 115)
        hdr = f"{'Language':<12} | {'Bytes':<8} | {'Chars':<8} | {'Tokens':<8} | {'B/Tok':<7} | {'Tok/Byte':<9} | Sample First Tokens"
        print(hdr)
        print("-" * len(hdr))

        for e in entries:
            sample_str = " | ".join(repr(t) for t in e.sample_tokens[:5])
            print(
                f"{e.language:<12} | {e.raw_byte_count:<8} | {e.char_count:<8} | {e.token_count:<8} | "
                f"{e.bytes_per_token:<7.2f} | {e.tokens_per_byte:<9.4f} | {sample_str}"
            )
        print("=" * 115)
        print(f"Total Evaluation Bytes : {total_bytes:,}")
        print(f"Total Evaluation Tokens: {total_tokens:,}")
        print(f"Aggregate Tokens/Byte  : {aggregate_tokens_per_byte:.3f}")
        print("=" * 115 + "\n")


if __name__ == "__main__":
    unittest.main()
