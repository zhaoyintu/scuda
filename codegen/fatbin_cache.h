#ifndef FATBIN_CACHE_H
#define FATBIN_CACHE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#define RPC_CACHE_FATBINARY_HIT 8001
#define RPC_CACHE_FATBINARY_STORE 8002

struct fatbin_hash_t {
    uint64_t h1, h2;
    bool operator==(const fatbin_hash_t& o) const {
        return h1 == o.h1 && h2 == o.h2;
    }
};

struct fatbin_hash_hasher {
    size_t operator()(const fatbin_hash_t& h) const {
        return h.h1 ^ (h.h2 * 0x9e3779b97f4a7c15ULL);
    }
};

// Compute FNV-1a 128-bit hash (two independent 64-bit halves)
fatbin_hash_t fatbin_compute_hash(const void* data, size_t size);

// Convert hash to 32-char hex string
std::string fatbin_hash_hex(const fatbin_hash_t& hash);

// Client disk cache entry: a parsed PTX function
struct CachedFunction {
    std::string name;
    std::vector<int> arg_sizes;
};

// Save function entries to ~/.scuda/cache/<hash>.bin
void fatbin_cache_save(const fatbin_hash_t& hash,
                       const std::vector<CachedFunction>& funcs);

// Load function entries from disk cache. Returns true if found and valid.
bool fatbin_cache_load(const fatbin_hash_t& hash,
                       std::vector<CachedFunction>& funcs);

// Ensure ~/.scuda/cache/ directory exists (call once at startup)
void fatbin_cache_init();

#endif // FATBIN_CACHE_H
