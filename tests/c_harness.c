/* Raw C99 test harness for the UniqToken tokenizer C-ABI (issue #22).
 *
 * Exercises create/encode/free/destroy against known IDs, plus error paths.
 * Usage:
 *   c_harness            run all checks once (exit 0 on success)
 *   c_harness --loop N   repeat the create/encode/free/destroy cycle N times
 *                        (encode/free leak soak, e.g. under valgrind)
 *
 * Build (from the repository root):
 *   cargo build -p uniqtoken-core --release --no-default-features --features c_abi
 *   cc -std=c99 -Wall -Wextra -Werror tests/c_harness.c -o /tmp/c_harness \
 *     -Icrates/uniqtoken_core/include -Ltarget/release -luniqtoken_core
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "uniqtoken.h"

/* U+2581 METASPACE is spelled with explicit hex escapes so this file behaves
 * identically under any source encoding. */
static const char *kVocabJson =
    "[[\"hello\",-1.0,0],"
    "[\"\xe2\x96\x81world\",-1.0,1],"
    "[\"<|unk|>\",-5.0,2]]";

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

static UniqTokenHandle *make_tokenizer(void) {
    UniqTokenHandle *handle = uniqtoken_create(kVocabJson);
    CHECK(handle != NULL);
    return handle;
}

static int check_roundtrip(void) {
    static const char text[] = "hello world";
    static const uint32_t expected[] = {0, 1};
    UniqTokenHandle *handle = make_tokenizer();
    if (handle == NULL) {
        return 1;
    }
    uint32_t *ids = NULL;
    size_t len = 0;
    int32_t rc = uniqtoken_encode(handle, text, strlen(text), &ids, &len);
    CHECK(rc == UNIQTOKEN_OK);
    CHECK(len == sizeof(expected) / sizeof(expected[0]));
    if (rc == UNIQTOKEN_OK && len == sizeof(expected) / sizeof(expected[0])) {
        CHECK(memcmp(ids, expected, sizeof(expected)) == 0);
    }
    uniqtoken_free_tokens(ids, len);
    uniqtoken_destroy(handle);
    printf("PASS roundtrip hello-world -> [0, 1]\n");
    return failures != 0;
}

static int check_edge_cases(void) {
    UniqTokenHandle *handle = make_tokenizer();
    if (handle == NULL) {
        return 1;
    }
    /* Empty input yields NULL + 0, still UNIQTOKEN_OK. */
    uint32_t *ids = (uint32_t *)0x1;
    size_t len = 42;
    CHECK(uniqtoken_encode(handle, "", 0, &ids, &len) == UNIQTOKEN_OK);
    CHECK(ids == NULL);
    CHECK(len == 0);
    /* Null handle is rejected without crashing. */
    CHECK(uniqtoken_encode(NULL, "hi", 2, &ids, &len) == UNIQTOKEN_ERR_NULL_PTR);
    /* Freeing/destroying null are safe no-ops. */
    uniqtoken_free_tokens(NULL, 0);
    uniqtoken_destroy(NULL);
    uniqtoken_destroy(handle);
    printf("PASS edge cases (empty input, null handle, null free)\n");
    return failures != 0;
}

static int check_bad_vocab(void) {
    static const char *bad[] = {"not json", "[]", "[[]]", "[\"lonely\"]"};
    size_t i;
    for (i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        CHECK(uniqtoken_create(bad[i]) == NULL);
    }
    CHECK(uniqtoken_create(NULL) == NULL);
    printf("PASS malformed vocabularies rejected\n");
    return failures != 0;
}

static int soak(size_t iterations) {
    static const char text[] = "hello world";
    size_t i;
    for (i = 0; i < iterations; ++i) {
        UniqTokenHandle *handle = make_tokenizer();
        if (handle == NULL) {
            return 1;
        }
        uint32_t *ids = NULL;
        size_t len = 0;
        if (uniqtoken_encode(handle, text, strlen(text), &ids, &len) != UNIQTOKEN_OK) {
            uniqtoken_destroy(handle);
            return 1;
        }
        uniqtoken_free_tokens(ids, len);
        uniqtoken_destroy(handle);
    }
    printf("PASS soak (%lu cycles)\n", (unsigned long)iterations);
    return failures != 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--loop") == 0) {
        return soak((size_t)strtoul(argv[2], NULL, 10));
    }
    if (check_roundtrip() != 0 || check_edge_cases() != 0 || check_bad_vocab() != 0) {
        fprintf(stderr, "c_harness: FAILURES (%d)\n", failures);
        return 1;
    }
    printf("c_harness: all checks passed\n");
    return 0;
}
