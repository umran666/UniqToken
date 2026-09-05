"""Public package namespace for the UniqToken tokenizer."""

from .batch_collator import BatchCollator, BatchEncoding
from .binary_format import export_binary, load_binary
from .bpe_model import BPEModel
from .bpe_trainer import BPETrainer
from .byte_codec import ByteFallbackEngine
from .cem_merger import CrossEntropyMerging
from .hf_exporter import (
    GGUFExporter,
    HuggingFaceExporter,
    extract_gguf_metadata,
    extract_gguf_scores,
)
from .indentation_compressor import IndentationCompressor
from .pre_tokenizer import Normalizer, PreToken, RegexPreTokenizer
from .security_shield import SecurityShield
from .seed_builder import SeedToken, SeedVocabularyBuilder
from .streaming_decoder import StreamingDecoder
from .tokenizer import CustomTokenizer, Token, TokenizationReport
from .trie import PrefixTrie, TrieNode
from .unigram_lattice import LatticeEdge, UnigramLattice
from .unigram_trainer import UnigramModel, UnigramTrainer
from .vocab_adapter import VocabularyAdapter

__version__ = "1.0.0"

_LAZY_MULTIMODAL = {
    "AudioSegment": "multimodal.audio_codec",
    "ResidualVectorQuantizer": "multimodal.audio_codec",
    "DynamicImagePatcher": "multimodal.image_patcher",
    "ImagePatch": "multimodal.image_patcher",
    "ImageElement": "multimodal.multimodal_tokenizer",
    "MultimodalSequence": "multimodal.multimodal_tokenizer",
    "MultimodalTokenizer": "multimodal.multimodal_tokenizer",
    "HAS_TORCH": "multimodal.neural_codecs",
    "NeuralAudioCodec": "multimodal.neural_codecs",
    "NeuralCodecFacade": "multimodal.neural_codecs",
    "NeuralVisualCodec": "multimodal.neural_codecs",
    "VisualCodebook": "multimodal.visual_codebook",
}

_LAZY_COMPAT = {
    "TiktokenEncoding": "tiktoken_adapter",
    "load_tiktoken_ranks": "tiktoken_adapter",
    "TIKTOKEN_PATTERNS": "tiktoken_adapter",
    "HFByteLevelBPE": "hf_importer",
    "import_hf_tokenizer": "hf_importer",
    "import_hf_unigram": "hf_importer",
    "import_hf_bpe": "hf_importer",
    "import_sentencepiece": "sentencepiece_importer",
    "load_sentencepiece_model": "sentencepiece_importer",
    "parse_sentencepiece_proto": "sentencepiece_importer",
}


def __getattr__(name: str):
    module_name = _LAZY_MULTIMODAL.get(name) or _LAZY_COMPAT.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_MULTIMODAL) | set(_LAZY_COMPAT))


__all__ = [
    "CustomTokenizer",
    "Token",
    "TokenizationReport",
    "Normalizer",
    "RegexPreTokenizer",
    "PreToken",
    "ByteFallbackEngine",
    "UnigramModel",
    "UnigramTrainer",
    "UnigramLattice",
    "LatticeEdge",
    "PrefixTrie",
    "TrieNode",
    "SeedVocabularyBuilder",
    "SeedToken",
    "CrossEntropyMerging",
    "BPEModel",
    "BPETrainer",
    "SecurityShield",
    "StreamingDecoder",
    "IndentationCompressor",
    "VocabularyAdapter",
    "BatchCollator",
    "BatchEncoding",
    "HuggingFaceExporter",
    "GGUFExporter",
    "extract_gguf_metadata",
    "extract_gguf_scores",
    "export_binary",
    "load_binary",
    "MultimodalTokenizer",
    "MultimodalSequence",
    "ImageElement",
    "DynamicImagePatcher",
    "ImagePatch",
    "VisualCodebook",
    "ResidualVectorQuantizer",
    "AudioSegment",
    "NeuralCodecFacade",
    "NeuralVisualCodec",
    "NeuralAudioCodec",
    "HAS_TORCH",
    "TiktokenEncoding",
    "load_tiktoken_ranks",
    "TIKTOKEN_PATTERNS",
    "HFByteLevelBPE",
    "import_hf_tokenizer",
    "import_hf_unigram",
    "import_hf_bpe",
    "import_sentencepiece",
    "load_sentencepiece_model",
    "parse_sentencepiece_proto",
]
