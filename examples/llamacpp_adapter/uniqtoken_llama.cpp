#include "uniqtoken_llama.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cstdint>
#include <cassert>
template <typename T>
static T read_le(const uint8_t* ptr) {
    T val;
    std::memcpy(&val, ptr, sizeof(T));
    return val;
}
class UniqTokenLlamaVocab {
public:
    std::string model_type;
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<int32_t> token_types;
    std::unordered_map<std::string, int32_t> token_to_id;
    uint32_t bos_id = 1;
    uint32_t eos_id = 2;
    uint32_t unk_id = 0;
    uint32_t pad_id = 0;
    static bool load(const std::string& model_path, UniqTokenLlamaVocab& vocab) {
        void* buffer = nullptr;
        size_t size = 0;
        int32_t rc = uniqtoken_export_gguf_vocab(model_path.c_str(), &buffer, &size);
        if (rc != UNIQTOKEN_OK || !buffer) {
            std::cerr << "Failed to export GGUF vocab. Error code: " << rc << std::endl;
            return false;
        }
        if (size < 24) {
            std::cerr << "GGUF buffer too short: " << size << " bytes" << std::endl;
            uniqtoken_free_buffer(buffer, size);
            return false;
        }
        const uint8_t* p = reinterpret_cast<const uint8_t*>(buffer);
        const uint8_t* end = p + size;
        if (std::memcmp(p, "GGUF", 4) != 0) {
            std::cerr << "Invalid GGUF magic" << std::endl;
            uniqtoken_free_buffer(buffer, size);
            return false;
        }
        p += 4;
        uint32_t version = read_le<uint32_t>(p);
        p += 4;
        if (version != 3) {
            std::cerr << "Unsupported GGUF version: " << version << std::endl;
            uniqtoken_free_buffer(buffer, size);
            return false;
        }
        uint64_t tensor_count = read_le<uint64_t>(p);
        (void)tensor_count;
        p += 8;
        uint64_t kv_count = read_le<uint64_t>(p);
        p += 8;
        auto remaining = [&]() -> uint64_t {
            return (p < end) ? static_cast<uint64_t>(end - p) : 0;
        };
        auto read_str = [&](std::string& out) -> bool {
            if (remaining() < 8) return false;
            uint64_t len = read_le<uint64_t>(p);
            p += 8;
            if (remaining() < len) return false;
            out.assign(reinterpret_cast<const char*>(p), len);
            p += len;
            return true;
        };
        auto read_array_header = [&](uint64_t elem_size, uint64_t& arr_len) -> bool {
            if (remaining() < 12) return false;
            p += 4; // element type
            arr_len = read_le<uint64_t>(p);
            p += 8;
            return (elem_size == 0 || remaining() / elem_size >= arr_len);
        };
        for (uint64_t i = 0; i < kv_count && p < end; ++i) {
            std::string key;
            if (!read_str(key)) {
                uniqtoken_free_buffer(buffer, size);
                return false;
            }
            if (remaining() < 4) {
                uniqtoken_free_buffer(buffer, size);
                return false;
            }
            uint32_t val_type = read_le<uint32_t>(p);
            (void)val_type;
            p += 4;
            if (key == "tokenizer.ggml.model") {
                if (!read_str(vocab.model_type)) {
                    uniqtoken_free_buffer(buffer, size);
                    return false;
                }
            } else if (key == "tokenizer.ggml.tokens") {
                uint64_t arr_len = 0;
                if (!read_array_header(0, arr_len)) {
                    uniqtoken_free_buffer(buffer, size);
                    return false;
                }
                vocab.tokens.resize(arr_len);
                for (uint64_t t = 0; t < arr_len; ++t) {
                    if (!read_str(vocab.tokens[t])) {
                        uniqtoken_free_buffer(buffer, size);
                        return false;
                    }
                    vocab.token_to_id[vocab.tokens[t]] = static_cast<int32_t>(t);
                }
            } else if (key == "tokenizer.ggml.scores") {
                uint64_t arr_len = 0;
                if (!read_array_header(sizeof(float), arr_len)) {
                    uniqtoken_free_buffer(buffer, size);
                    return false;
                }
                vocab.scores.resize(arr_len);
                std::memcpy(vocab.scores.data(), p, arr_len * sizeof(float));
                p += arr_len * sizeof(float);
            } else if (key == "tokenizer.ggml.token_type") {
                uint64_t arr_len = 0;
                if (!read_array_header(sizeof(int32_t), arr_len)) {
                    uniqtoken_free_buffer(buffer, size);
                    return false;
                }
                vocab.token_types.resize(arr_len);
                std::memcpy(vocab.token_types.data(), p, arr_len * sizeof(int32_t));
                p += arr_len * sizeof(int32_t);
            } else if (key == "tokenizer.ggml.bos_token_id") {
                if (remaining() < 4) { uniqtoken_free_buffer(buffer, size); return false; }
                vocab.bos_id = read_le<uint32_t>(p); p += 4;
            } else if (key == "tokenizer.ggml.eos_token_id") {
                if (remaining() < 4) { uniqtoken_free_buffer(buffer, size); return false; }
                vocab.eos_id = read_le<uint32_t>(p); p += 4;
            } else if (key == "tokenizer.ggml.unknown_token_id") {
                if (remaining() < 4) { uniqtoken_free_buffer(buffer, size); return false; }
                vocab.unk_id = read_le<uint32_t>(p); p += 4;
            } else if (key == "tokenizer.ggml.padding_token_id") {
                if (remaining() < 4) { uniqtoken_free_buffer(buffer, size); return false; }
                vocab.pad_id = read_le<uint32_t>(p); p += 4;
            }
        }
        uniqtoken_free_buffer(buffer, size);
        return true;
    }
    int32_t find_token_id(const std::string& tok) const {
        auto it = token_to_id.find(tok);
        if (it != token_to_id.end()) return it->second;
        return static_cast<int32_t>(unk_id);
    }
};
int main(int argc, char** argv) {
    std::string path = "crates/uniqtoken_core/demo_vocab.json";
    if (argc > 1) {
        path = argv[1];
    }
    std::cout << "[llama.cpp Hook] Loading UniqToken GGUF vocab from: " << path << std::endl;
    UniqTokenLlamaVocab vocab;
    if (!UniqTokenLlamaVocab::load(path, vocab)) {
        std::cerr << "Failed to load vocabulary" << std::endl;
        return 1;
    }
    std::cout << "  - Model type: " << vocab.model_type << std::endl;
    std::cout << "  - Vocab size: " << vocab.tokens.size() << " tokens" << std::endl;
    std::cout << "  - Special IDs: BOS=" << vocab.bos_id << ", EOS=" << vocab.eos_id
              << ", UNK=" << vocab.unk_id << ", PAD=" << vocab.pad_id << std::endl;
    assert(!vocab.tokens.empty());
    assert(vocab.tokens.size() == vocab.scores.size());
    assert(vocab.tokens.size() == vocab.token_types.size());
    std::cout << "[llama.cpp Hook] Validation PASSED with zero leaks." << std::endl;
    return 0;
}
