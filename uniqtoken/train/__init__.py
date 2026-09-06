"""Research Engine: train *new* vocabularies with UniqToken's research features.

Unlike :mod:`uniqtoken.compat` (which loads existing models with exact ID
parity), this module changes the vocabulary and token IDs: Unigram lattice
training with EM/Viterbi, script-aware seed building, SuperBPE cross-word
merging, BPE training, and dual-offset composition. Both engines sit on the
shared native Rust core (``crates/uniqtoken_core``).
"""

from ..bpe_model import BPEModel
from ..bpe_trainer import BPETrainer
from ..byte_codec import ByteFallbackEngine
from ..cem_merger import CrossEntropyMerging, SuperBPE
from ..pre_tokenizer import Normalizer, PreToken, RegexPreTokenizer
from ..seed_builder import SeedToken, SeedVocabularyBuilder
from ..tokenizer import CustomTokenizer, Token, TokenizationReport
from ..trie import PrefixTrie, TrieNode
from ..unigram_lattice import LatticeEdge, UnigramLattice
from ..unigram_trainer import UnigramModel, UnigramTrainer
from ..vocab_adapter import VocabularyAdapter

__all__ = [
    "UnigramTrainer",
    "UnigramModel",
    "UnigramLattice",
    "LatticeEdge",
    "SeedVocabularyBuilder",
    "SeedToken",
    "BPETrainer",
    "BPEModel",
    "ByteFallbackEngine",
    "CrossEntropyMerging",
    "SuperBPE",
    "VocabularyAdapter",
    "CustomTokenizer",
    "Token",
    "TokenizationReport",
    "PrefixTrie",
    "TrieNode",
    "Normalizer",
    "RegexPreTokenizer",
    "PreToken",
]
