#ifndef UNIQTOKEN_LLAMA_H
#define UNIQTOKEN_LLAMA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define UNIQTOKEN_OK 0
#define UNIQTOKEN_ERR_NULL_PTR -1
#define UNIQTOKEN_ERR_INVALID_PATH -2
#define UNIQTOKEN_ERR_IO -3
#define UNIQTOKEN_ERR_PARSE -4
#define UNIQTOKEN_ERR_SERIALIZE -5
enum llama_token_type {
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
    LLAMA_TOKEN_TYPE_NORMAL       = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
    LLAMA_TOKEN_TYPE_CONTROL      = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED       = 5,
    LLAMA_TOKEN_TYPE_BYTE         = 6,
};
int32_t uniqtoken_export_gguf_vocab(
    const char* model_path,
    void** buffer_out,
    size_t* size_out
);
void uniqtoken_free_buffer(
    void* buffer,
    size_t size
);
#ifdef __cplusplus
}
#endif
#endif // UNIQTOKEN_LLAMA_H
