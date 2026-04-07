#include "fatbin_cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

// Binary file magic for cache files
static const uint32_t CACHE_MAGIC = 0x5CDA0001u;

// ---------------------------------------------------------------------------
// FNV-1a 128-bit hash implemented as two independent 64-bit streams.
//
// h1 uses the classical FNV-1a 64-bit parameters:
//   offset basis = 14695981039346656037  prime = 1099511628211
// h2 uses an alternative set to keep them independent:
//   offset basis = 6364136223846793005   prime = 1442695040888963407
// ---------------------------------------------------------------------------
fatbin_hash_t fatbin_compute_hash(const void* data, size_t size) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);

    uint64_t h1 = 14695981039346656037ULL;
    const uint64_t prime1 = 1099511628211ULL;

    uint64_t h2 = 6364136223846793005ULL;
    const uint64_t prime2 = 1442695040888963407ULL;

    for (size_t i = 0; i < size; ++i) {
        h1 ^= static_cast<uint64_t>(p[i]);
        h1 *= prime1;
        h2 ^= static_cast<uint64_t>(p[i]);
        h2 *= prime2;
    }

    return fatbin_hash_t{h1, h2};
}

// ---------------------------------------------------------------------------
// Convert hash to 32-character lowercase hex string
// ---------------------------------------------------------------------------
std::string fatbin_hash_hex(const fatbin_hash_t& hash) {
    char buf[33];
    snprintf(buf, sizeof(buf),
             "%016llx%016llx",
             (unsigned long long)hash.h1,
             (unsigned long long)hash.h2);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Return path to the cache directory (~/. scuda/cache)
// ---------------------------------------------------------------------------
static std::string cache_dir() {
    const char* home = getenv("HOME");
    if (!home || !home[0])
        home = "/tmp";
    return std::string(home) + "/.scuda/cache";
}

static std::string cache_path(const fatbin_hash_t& hash) {
    return cache_dir() + "/" + fatbin_hash_hex(hash) + ".bin";
}

// ---------------------------------------------------------------------------
// mkdir -p equivalent for two levels (HOME/.scuda/cache)
// ---------------------------------------------------------------------------
void fatbin_cache_init() {
    const char* home = getenv("HOME");
    if (!home || !home[0])
        home = "/tmp";

    std::string scuda_dir = std::string(home) + "/.scuda";
    mkdir(scuda_dir.c_str(), 0755);

    std::string dir = scuda_dir + "/cache";
    mkdir(dir.c_str(), 0755);
}

// ---------------------------------------------------------------------------
// Binary format:
//   [magic:u32][num_funcs:u32]
//   for each function:
//     [name_len:u32][name:bytes (including NUL)]
//     [arg_count:u32][arg_sizes:i32 * arg_count]
// ---------------------------------------------------------------------------
void fatbin_cache_save(const fatbin_hash_t& hash,
                       const std::vector<CachedFunction>& funcs) {
    std::string path = cache_path(hash);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[fatbin_cache] cannot open %s for writing\n",
                path.c_str());
        return;
    }

    uint32_t magic = CACHE_MAGIC;
    uint32_t num_funcs = static_cast<uint32_t>(funcs.size());

    if (fwrite(&magic, sizeof(magic), 1, f) != 1 ||
        fwrite(&num_funcs, sizeof(num_funcs), 1, f) != 1) {
        fclose(f);
        return;
    }

    for (const auto& fn : funcs) {
        uint32_t name_len = static_cast<uint32_t>(fn.name.size() + 1); // incl. NUL
        uint32_t arg_count = static_cast<uint32_t>(fn.arg_sizes.size());

        if (fwrite(&name_len, sizeof(name_len), 1, f) != 1 ||
            fwrite(fn.name.c_str(), name_len, 1, f) != 1 ||
            fwrite(&arg_count, sizeof(arg_count), 1, f) != 1)
            break;

        if (arg_count > 0 &&
            fwrite(fn.arg_sizes.data(), sizeof(int) * arg_count, 1, f) != 1)
            break;
    }

    fclose(f);
}

// ---------------------------------------------------------------------------
bool fatbin_cache_load(const fatbin_hash_t& hash,
                       std::vector<CachedFunction>& funcs) {
    std::string path = cache_path(hash);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return false;

    uint32_t magic = 0, num_funcs = 0;
    bool ok = (fread(&magic, sizeof(magic), 1, f) == 1 &&
               magic == CACHE_MAGIC &&
               fread(&num_funcs, sizeof(num_funcs), 1, f) == 1);

    if (!ok) {
        fclose(f);
        return false;
    }

    funcs.clear();
    funcs.reserve(num_funcs);

    for (uint32_t i = 0; i < num_funcs && ok; ++i) {
        uint32_t name_len = 0, arg_count = 0;

        ok = (fread(&name_len, sizeof(name_len), 1, f) == 1 && name_len > 0);
        if (!ok) break;

        std::string name(name_len, '\0');
        ok = (fread(&name[0], name_len, 1, f) == 1);
        if (!ok) break;
        name.resize(name_len - 1); // strip NUL

        ok = (fread(&arg_count, sizeof(arg_count), 1, f) == 1);
        if (!ok) break;

        std::vector<int> arg_sizes(arg_count);
        if (arg_count > 0)
            ok = (fread(arg_sizes.data(), sizeof(int) * arg_count, 1, f) == 1);
        if (!ok) break;

        CachedFunction fn;
        fn.name = std::move(name);
        fn.arg_sizes = std::move(arg_sizes);
        funcs.push_back(std::move(fn));
    }

    fclose(f);

    if (!ok) {
        funcs.clear();
        return false;
    }

    return true;
}
