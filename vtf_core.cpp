/**
 * VTF (Venkey Tick Format) v1.0 — High-Frequency Option Market Data Engine
 * Default Profile: VTF v1.0 (LZ4-Fast, accel=4)
 * Target Markets: NSE (NIFTY 50) & BSE (SENSEX) Equity Index Options
 *
 * Zero-Dependency Portable Edition:
 *   - Compiles out-of-the-box on Windows (MSVC / MinGW), Linux (GCC / Clang), and macOS.
 *   - Automatically loads system `liblz4` / `libzstd` via `dlopen` / `LoadLibrary` if available on the host,
 *     and includes a built-in RFC-compliant LZ4-Fast (accel=4) compressor/decompressor
 *     fallback so it works on any machine even without external libraries installed.
 */

#ifndef VTF_BUILDING_DLL
#define VTF_BUILDING_DLL
#endif
#include "vtf.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <fcntl.h>
#include <sys/stat.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <unistd.h>
#include <dlfcn.h>
#endif

#if defined(__SSE4_2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
#include <nmmintrin.h>
#include <tmmintrin.h>
#define VTF_HAS_SSE4_2 1
#endif

namespace vtf {

static constexpr uint8_t CODEC_NONE        = VTF_CODEC_NONE;
static constexpr uint8_t CODEC_LZ4_FAST    = VTF_CODEC_LZ4_FAST;
static constexpr uint8_t CODEC_LZ4_DEFAULT = VTF_CODEC_LZ4_DEFAULT;
static constexpr uint8_t CODEC_ZSTD        = VTF_CODEC_ZSTD;

// ============================================================================
// Platform-Independent File I/O Helpers
// ============================================================================

#if defined(_WIN32) || defined(__CYGWIN__)

static inline int os_open_create_trunc(const char* path) {
    return ::_open(path, _O_CREAT | _O_TRUNC | _O_RDWR | _O_BINARY, _S_IREAD | _S_IWRITE);
}

static inline int os_open_read(const char* path) {
    return ::_open(path, _O_RDONLY | _O_BINARY);
}

static inline int os_open_rw(const char* path) {
    return ::_open(path, _O_RDWR | _O_BINARY);
}

static inline void os_close(int fd) {
    if (fd >= 0) ::_close(fd);
}

static inline int64_t os_file_size(int fd) {
    struct __stat64 st{};
    if (::_fstat64(fd, &st) != 0) return -1;
    return static_cast<int64_t>(st.st_size);
}

static inline bool os_truncate(int fd, uint64_t len) {
    return ::_chsize_s(fd, static_cast<__int64>(len)) == 0;
}

static inline bool os_seek(int fd, uint64_t offset) {
    return ::_lseeki64(fd, static_cast<__int64>(offset), SEEK_SET) >= 0;
}

static inline void os_fsync(int fd) {
    ::_commit(fd);
}

static inline bool write_all(int fd, const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    while (len > 0) {
        unsigned int chunk = static_cast<unsigned int>(std::min<size_t>(len, 0x7FFFFFFF));
        int w = ::_write(fd, p, chunk);
        if (w <= 0) return false;
        p += w;
        len -= static_cast<size_t>(w);
    }
    return true;
}

static inline bool pread_all(int fd, void* buf, size_t len, uint64_t offset) {
    char* p = static_cast<char*>(buf);
    if (!os_seek(fd, offset)) return false;
    while (len > 0) {
        unsigned int chunk = static_cast<unsigned int>(std::min<size_t>(len, 0x7FFFFFFF));
        int r = ::_read(fd, p, chunk);
        if (r <= 0) return false;
        p += r;
        len -= static_cast<size_t>(r);
    }
    return true;
}

static inline bool pwrite_all(int fd, const void* buf, size_t len, uint64_t offset) {
    if (!os_seek(fd, offset)) return false;
    return write_all(fd, buf, len);
}

#else

static inline int os_open_create_trunc(const char* path) {
    return ::open(path, O_CREAT | O_TRUNC | O_RDWR, 0644);
}

static inline int os_open_read(const char* path) {
    return ::open(path, O_RDONLY);
}

static inline int os_open_rw(const char* path) {
    return ::open(path, O_RDWR);
}

static inline void os_close(int fd) {
    if (fd >= 0) ::close(fd);
}

static inline int64_t os_file_size(int fd) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) return -1;
    return static_cast<int64_t>(st.st_size);
}

static inline bool os_truncate(int fd, uint64_t len) {
    return ::ftruncate(fd, static_cast<off_t>(len)) == 0;
}

static inline bool os_seek(int fd, uint64_t offset) {
    return ::lseek(fd, static_cast<off_t>(offset), SEEK_SET) >= 0;
}

static inline void os_fsync(int fd) {
    ::fsync(fd);
}

static inline bool write_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    while (len > 0) {
        ssize_t w = ::write(fd, p, len);
        if (w <= 0) return false;
        p += w;
        len -= static_cast<size_t>(w);
    }
    return true;
}

static inline bool pread_all(int fd, void* buf, size_t len, uint64_t offset) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    while (len > 0) {
        ssize_t r = ::pread(fd, p, len, static_cast<off_t>(offset));
        if (r <= 0) return false;
        p += r;
        offset += static_cast<uint64_t>(r);
        len -= static_cast<size_t>(r);
    }
    return true;
}

static inline bool pwrite_all(int fd, const void* buf, size_t len, uint64_t offset) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    while (len > 0) {
        ssize_t w = ::pwrite(fd, p, len, static_cast<off_t>(offset));
        if (w <= 0) return false;
        p += w;
        offset += static_cast<uint64_t>(w);
        len -= static_cast<size_t>(w);
    }
    return true;
}

#endif

// ============================================================================
// Built-in Wire-Compatible LZ4-Fast (accel=4) Fallback + Dynamic System Loader
// ============================================================================

using fn_LZ4_compress_default = int (*)(const char*, char*, int, int);
using fn_LZ4_compress_fast    = int (*)(const char*, char*, int, int, int);
using fn_LZ4_decompress_safe  = int (*)(const char*, char*, int, int);
using fn_LZ4_compressBound    = int (*)(int);

using fn_ZSTD_compress        = size_t (*)(void*, size_t, const void*, size_t, int);
using fn_ZSTD_decompress      = size_t (*)(void*, size_t, const void*, size_t);
using fn_ZSTD_compressBound   = size_t (*)(size_t);
using fn_ZSTD_isError         = unsigned (*)(size_t);

static inline int builtin_LZ4_compressBound(int isize) {
    return isize > 0x7E000000 ? 0 : (isize + (isize / 255) + 16);
}

static inline uint32_t lz4_read32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

static inline uint32_t lz4_hash(uint32_t v) {
    return (v * 2654435761U) >> (32 - 14);
}

static int builtin_LZ4_compress_fast(const char* source, char* dest, int inputSize, int maxOutputSize, int acceleration) {
    if (acceleration < 1) acceleration = 1;
    const uint8_t* ip = reinterpret_cast<const uint8_t*>(source);
    const uint8_t* const ibase = ip;
    const uint8_t* const iend = ip + inputSize;
    const uint8_t* const mflimit = iend - 12;
    const uint8_t* const matchlimit = iend - 5;
    const uint8_t* anchor = ip;

    uint8_t* op = reinterpret_cast<uint8_t*>(dest);
    uint8_t* const oend = op + maxOutputSize;

    if (inputSize < 13) goto _last_literals;

    {
        uint32_t hash_table[1 << 14] = {0};
        ip++;

        while (ip < mflimit) {
            const uint8_t* match = nullptr;
            uint8_t* token = nullptr;
            int step = 1;
            int searchMatchNb = acceleration << 6;

            const uint8_t* forwardIp = ip;
            do {
                ip = forwardIp;
                forwardIp += step;
                step = (searchMatchNb++ >> 6);
                if (forwardIp > mflimit) goto _last_literals;

                uint32_t h = lz4_hash(lz4_read32(ip));
                match = ibase + hash_table[h];
                hash_table[h] = static_cast<uint32_t>(ip - ibase);
            } while ((ip - match > 65535) || (lz4_read32(match) != lz4_read32(ip)));

            while ((ip > anchor) && (match > ibase) && (ip[-1] == match[-1])) {
                ip--;
                match--;
            }

            unsigned litLength = static_cast<unsigned>(ip - anchor);
            token = op++;
            if (op + litLength + (litLength / 255) + 8 > oend) return 0;

            if (litLength >= 15) {
                int len = static_cast<int>(litLength) - 15;
                *token = (15 << 4);
                for (; len >= 255; len -= 255) *op++ = 255;
                *op++ = static_cast<uint8_t>(len);
            } else {
                *token = static_cast<uint8_t>(litLength << 4);
            }

            std::memcpy(op, anchor, litLength);
            op += litLength;

            uint16_t offset = static_cast<uint16_t>(ip - match);
            std::memcpy(op, &offset, 2);
            op += 2;

            ip += 4;
            match += 4;
            const uint8_t* matchStart = ip;
            while (ip < matchlimit && *ip == *match) {
                ip++;
                match++;
            }
            unsigned matchCode = static_cast<unsigned>(ip - matchStart);

            if (op + (matchCode / 255) + 6 > oend) return 0;
            if (matchCode >= 15) {
                *token += 15;
                matchCode -= 15;
                for (; matchCode >= 255; matchCode -= 255) *op++ = 255;
                *op++ = static_cast<uint8_t>(matchCode);
            } else {
                *token += static_cast<uint8_t>(matchCode);
            }

            anchor = ip;
            if (ip >= mflimit) break;
            hash_table[lz4_hash(lz4_read32(ip - 2))] = static_cast<uint32_t>(ip - 2 - ibase);
        }
    }

_last_literals:
    {
        size_t lastRun = static_cast<size_t>(iend - anchor);
        if (op + lastRun + 1 + (lastRun / 255) > oend) return 0;
        if (lastRun >= 15) {
            size_t accum = lastRun - 15;
            *op++ = (15 << 4);
            for (; accum >= 255; accum -= 255) *op++ = 255;
            *op++ = static_cast<uint8_t>(accum);
        } else {
            *op++ = static_cast<uint8_t>(lastRun << 4);
        }
        std::memcpy(op, anchor, lastRun);
        op += lastRun;
    }
    return static_cast<int>(op - reinterpret_cast<uint8_t*>(dest));
}

static int builtin_LZ4_decompress_safe(const char* source, char* dest, int compressedSize, int maxDecompressedSize) {
    const uint8_t* ip = reinterpret_cast<const uint8_t*>(source);
    const uint8_t* const iend = ip + compressedSize;
    uint8_t* op = reinterpret_cast<uint8_t*>(dest);
    uint8_t* const oend = op + maxDecompressedSize;

    while (ip < iend) {
        unsigned token = *ip++;
        size_t length = token >> 4;
        if (length == 15) {
            uint8_t s;
            do {
                if (ip >= iend) return -1;
                s = *ip++;
                length += s;
            } while (s == 255);
        }
        if (op + length > oend || ip + length > iend) return -2;
        std::memcpy(op, ip, length);
        ip += length;
        op += length;

        if (ip == iend) break;
        if (ip + 2 > iend) return -3;

        uint16_t offset;
        std::memcpy(&offset, ip, 2);
        ip += 2;
        if (offset == 0) return -4;
        const uint8_t* match = op - offset;
        if (match < reinterpret_cast<const uint8_t*>(dest)) return -5;

        length = token & 15;
        if (length == 15) {
            uint8_t s;
            do {
                if (ip >= iend) return -6;
                s = *ip++;
                length += s;
            } while (s == 255);
        }
        length += 4;
        if (op + length > oend) return -7;

        if (offset >= length) {
            std::memcpy(op, match, length);
            op += length;
        } else {
            for (size_t i = 0; i < length; ++i) {
                *op++ = *match++;
            }
        }
    }
    return static_cast<int>(op - reinterpret_cast<uint8_t*>(dest));
}

struct CodecDispatch {
    fn_LZ4_compress_default lz4_default   = nullptr;
    fn_LZ4_compress_fast    lz4_fast      = nullptr;
    fn_LZ4_decompress_safe  lz4_decomp    = nullptr;
    fn_LZ4_compressBound    lz4_bound     = nullptr;
    fn_ZSTD_compress        zstd_compress = nullptr;
    fn_ZSTD_decompress      zstd_decomp   = nullptr;
    fn_ZSTD_compressBound   zstd_bound    = nullptr;
    fn_ZSTD_isError         zstd_is_error = nullptr;

    CodecDispatch() {
#if defined(_WIN32) || defined(__CYGWIN__)
        const char* lz4_names[] = {"lz4.dll", "liblz4.dll"};
        for (const char* name : lz4_names) {
            HMODULE h = ::LoadLibraryA(name);
            if (h) {
                lz4_default = reinterpret_cast<fn_LZ4_compress_default>(::GetProcAddress(h, "LZ4_compress_default"));
                lz4_fast    = reinterpret_cast<fn_LZ4_compress_fast>(::GetProcAddress(h, "LZ4_compress_fast"));
                lz4_decomp  = reinterpret_cast<fn_LZ4_decompress_safe>(::GetProcAddress(h, "LZ4_decompress_safe"));
                lz4_bound   = reinterpret_cast<fn_LZ4_compressBound>(::GetProcAddress(h, "LZ4_compressBound"));
                if (lz4_fast && lz4_decomp && lz4_bound) break;
            }
        }
        const char* zstd_names[] = {"zstd.dll", "libzstd.dll"};
        for (const char* name : zstd_names) {
            HMODULE h = ::LoadLibraryA(name);
            if (h) {
                zstd_compress = reinterpret_cast<fn_ZSTD_compress>(::GetProcAddress(h, "ZSTD_compress"));
                zstd_decomp   = reinterpret_cast<fn_ZSTD_decompress>(::GetProcAddress(h, "ZSTD_decompress"));
                zstd_bound    = reinterpret_cast<fn_ZSTD_compressBound>(::GetProcAddress(h, "ZSTD_compressBound"));
                zstd_is_error = reinterpret_cast<fn_ZSTD_isError>(::GetProcAddress(h, "ZSTD_isError"));
                if (zstd_compress && zstd_decomp && zstd_bound && zstd_is_error) break;
            }
        }
#else
        const char* lz4_names[] = {"liblz4.so.1", "liblz4.so", "liblz4.dylib", "/usr/lib/x86_64-linux-gnu/liblz4.so.1"};
        for (const char* name : lz4_names) {
            void* h = ::dlopen(name, RTLD_LAZY | RTLD_LOCAL);
            if (h) {
                lz4_default = reinterpret_cast<fn_LZ4_compress_default>(::dlsym(h, "LZ4_compress_default"));
                lz4_fast    = reinterpret_cast<fn_LZ4_compress_fast>(::dlsym(h, "LZ4_compress_fast"));
                lz4_decomp  = reinterpret_cast<fn_LZ4_decompress_safe>(::dlsym(h, "LZ4_decompress_safe"));
                lz4_bound   = reinterpret_cast<fn_LZ4_compressBound>(::dlsym(h, "LZ4_compressBound"));
                if (lz4_fast && lz4_decomp && lz4_bound) break;
            }
        }
        const char* zstd_names[] = {"libzstd.so.1", "libzstd.so", "libzstd.dylib", "/usr/lib/x86_64-linux-gnu/libzstd.so.1"};
        for (const char* name : zstd_names) {
            void* h = ::dlopen(name, RTLD_LAZY | RTLD_LOCAL);
            if (h) {
                zstd_compress = reinterpret_cast<fn_ZSTD_compress>(::dlsym(h, "ZSTD_compress"));
                zstd_decomp   = reinterpret_cast<fn_ZSTD_decompress>(::dlsym(h, "ZSTD_decompress"));
                zstd_bound    = reinterpret_cast<fn_ZSTD_compressBound>(::dlsym(h, "ZSTD_compressBound"));
                zstd_is_error = reinterpret_cast<fn_ZSTD_isError>(::dlsym(h, "ZSTD_isError"));
                if (zstd_compress && zstd_decomp && zstd_bound && zstd_is_error) break;
            }
        }
#endif
        if (!lz4_fast || !lz4_decomp || !lz4_bound) {
            lz4_fast    = builtin_LZ4_compress_fast;
            lz4_default = [](const char* s, char* d, int is, int os) -> int {
                return builtin_LZ4_compress_fast(s, d, is, os, 1);
            };
            lz4_decomp  = builtin_LZ4_decompress_safe;
            lz4_bound   = builtin_LZ4_compressBound;
        }
    }
};

static const CodecDispatch& codecs() {
    static CodecDispatch cd;
    return cd;
}

// ============================================================================
// Constants, Flags & Wire Headers
// ============================================================================

static constexpr uint32_t VTF_FILE_MAGIC    = 0x31465456u; // "VTF1" little-endian
static constexpr uint32_t VTF_BLOCK_MAGIC   = 0x4B425456u; // "VTBK" little-endian
static constexpr uint32_t VTF_TRAILER_MAGIC = 0x45465456u; // "VTFE" little-endian

static constexpr uint8_t VTF_VERSION_MAJOR  = 1;
static constexpr uint8_t VTF_VERSION_MINOR  = 0;

enum VTFFlags : uint8_t {
    FLAG_HAS_FOOTER        = 0x01,
    FLAG_RECOVERED         = 0x02,
    FLAG_BYTE_STREAM_SPLIT = 0x04,
};

enum VTFTickFlags : uint8_t {
    TICK_FLAG_TRADE          = 0x01,
    TICK_FLAG_BUY_AGGRESSOR  = 0x02,
    TICK_FLAG_SELL_AGGRESSOR = 0x04,
    TICK_FLAG_L2_UPDATE      = 0x08,
    TICK_FLAG_OI_UPDATE      = 0x10,
    TICK_FLAG_GREEKS_VALID   = 0x20
};

#pragma pack(push, 1)

struct VTFBlockHeader {
    uint32_t block_magic;           // "VTBK" (0x4B425456)
    uint32_t block_index;           // 0-based sequential block index
    uint32_t row_count;             // Number of ticks in this block
    uint32_t uncompressed_bytes;    // Size of encoded SoA buffer before LZ4/Zstd
    uint32_t compressed_bytes;      // Size of compressed payload on disk
    int64_t  min_timestamp_ns;      // Minimum timestamp in block
    int64_t  max_timestamp_ns;      // Maximum timestamp in block
    uint64_t inst_bloom_filter;     // 64-bit Bloom filter of instrument_ids
    uint32_t payload_crc32c;        // CRC32C of compressed payload bytes
    uint8_t  reserved[12];          // Reserved padding to 64-byte cache line
    uint32_t block_header_crc32c;   // CRC32C of bytes [0..59] of VTFBlockHeader
};
static_assert(sizeof(VTFBlockHeader) == 64, "VTFBlockHeader must be 64 bytes");

struct VTFFooterTrailer {
    uint64_t index_offset;          // Byte offset where VTFBlockIndexEntry array begins
    uint64_t total_blocks;          // Total valid blocks in file
    uint64_t total_rows;            // Total valid tick rows across all blocks
    uint32_t footer_crc32c;         // CRC32C covering index array + first 24 bytes of trailer
    uint32_t trailer_magic;         // "VTFE" (0x45465456)
};
static_assert(sizeof(VTFFooterTrailer) == 32, "VTFFooterTrailer must be 32 bytes");

#pragma pack(pop)

// ============================================================================
// Hardware CRC32C & Bloom Filters & Delta / ZigZag
// ============================================================================

static uint32_t s_crc32c_table[256];
static bool s_crc32c_initialized = false;

static void init_crc32c_table() {
    if (s_crc32c_initialized) return;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0x82F63B78u : 0u);
        }
        s_crc32c_table[i] = crc;
    }
    s_crc32c_initialized = true;
}

inline uint32_t crc32c_append(uint32_t crc, const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint32_t c = ~crc;
#if defined(VTF_HAS_SSE4_2) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64))
    while (len >= 8) {
        uint64_t v;
        std::memcpy(&v, p, sizeof(uint64_t));
        c = static_cast<uint32_t>(_mm_crc32_u64(c, v));
        p += 8;
        len -= 8;
    }
    while (len > 0) {
        c = _mm_crc32_u8(c, *p++);
        --len;
    }
#else
    if (!s_crc32c_initialized) init_crc32c_table();
    while (len > 0) {
        c = (c >> 8) ^ s_crc32c_table[(c ^ *p++) & 0xFFu];
        --len;
    }
#endif
    return ~c;
}

inline uint32_t crc32c(const void* data, size_t len) {
    return crc32c_append(0u, data, len);
}

inline uint64_t bloom_hash(uint32_t inst_id) {
    uint32_t h1 = inst_id * 0x9E3779B1u;
    uint32_t h2 = (inst_id ^ (inst_id >> 15)) * 0x85EBCA6Bu;
    uint64_t b1 = 1ULL << (h1 & 63u);
    uint64_t b2 = 1ULL << ((h2 >> 16) & 63u);
    return b1 | b2;
}

inline uint64_t zigzag64(int64_t v) {
    return (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63);
}
inline int64_t unzigzag64(uint64_t v) {
    return static_cast<int64_t>((v >> 1) ^ (~(v & 1ULL) + 1ULL));
}

inline uint32_t zigzag32(int32_t v) {
    return (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31);
}
inline int32_t unzigzag32(uint32_t v) {
    return static_cast<int32_t>((v >> 1) ^ (~(v & 1u) + 1u));
}

inline uint16_t zigzag16(int16_t v) {
    return static_cast<uint16_t>((static_cast<uint32_t>(static_cast<uint16_t>(v)) << 1) ^ static_cast<uint32_t>(v >> 15));
}
inline int16_t unzigzag16(uint16_t v) {
    return static_cast<int16_t>((v >> 1) ^ (~(v & 1u) + 1u));
}

struct InstrumentState {
    uint32_t instrument_id;
    int32_t  ltp_paise;
    uint32_t volume;
    uint32_t oi;
    int32_t  oi_change;
    uint32_t bid_qty[5];
    uint32_t ask_qty[5];
    uint16_t bid_orders[5];
    uint16_t ask_orders[5];
    int32_t  iv_bps;
    int16_t  delta_bps;
    int32_t  gamma_scaled;
    int32_t  theta_scaled;
    int32_t  vega_scaled;
};

static constexpr size_t STATE_TABLE_SIZE = 512;

inline size_t inst_slot(uint32_t inst_id) {
    return ((inst_id * 0x19660Du) >> 10) & (STATE_TABLE_SIZE - 1);
}

static constexpr size_t ROW_BYTES = sizeof(VTFTickRow); // 156 bytes
static constexpr size_t ROW_WORDS = ROW_BYTES / 4;      // 39 uint32_t words

void encode_rows_to_soa(
    const VTFTickRow* rows,
    size_t n,
    uint8_t* out_buf,
    std::vector<uint8_t>& scratch,
    int64_t& min_ts,
    int64_t& max_ts,
    uint64_t& bloom
) {
    scratch.resize(n * ROW_BYTES);
    VTFTickRow* d_rows = reinterpret_cast<VTFTickRow*>(scratch.data());

    InstrumentState state_table[STATE_TABLE_SIZE];
    for (size_t s = 0; s < STATE_TABLE_SIZE; ++s) {
        state_table[s].instrument_id = 0xFFFFFFFFu;
    }

    int64_t prev_ts = 0;
    min_ts = (n > 0) ? rows[0].timestamp_ns : 0;
    max_ts = min_ts;
    uint64_t local_bloom = 0ULL;

    for (size_t i = 0; i < n; ++i) {
        const VTFTickRow& r = rows[i];
        VTFTickRow d;

        if (r.timestamp_ns < min_ts) min_ts = r.timestamp_ns;
        if (r.timestamp_ns > max_ts) max_ts = r.timestamp_ns;
        local_bloom |= bloom_hash(r.instrument_id);

        d.timestamp_ns  = static_cast<int64_t>(zigzag64(r.timestamp_ns - prev_ts));
        prev_ts         = r.timestamp_ns;
        d.instrument_id = r.instrument_id;
        d.ltq           = r.ltq;

        d.bid_price_paise[0] = static_cast<int32_t>(zigzag32(r.bid_price_paise[0] - r.ltp_paise));
        d.ask_price_paise[0] = static_cast<int32_t>(zigzag32(r.ask_price_paise[0] - r.bid_price_paise[0]));
        for (int k = 1; k < 5; ++k) {
            d.bid_price_paise[k] = static_cast<int32_t>(zigzag32(r.bid_price_paise[k - 1] - r.bid_price_paise[k]));
            d.ask_price_paise[k] = static_cast<int32_t>(zigzag32(r.ask_price_paise[k] - r.ask_price_paise[k - 1]));
        }

        size_t slot = inst_slot(r.instrument_id);
        InstrumentState& st = state_table[slot];

        if (st.instrument_id == r.instrument_id) {
            d.ltp_paise = static_cast<int32_t>(zigzag32(r.ltp_paise - st.ltp_paise));
            d.volume    = zigzag32(static_cast<int32_t>(r.volume - st.volume));
            d.oi        = zigzag32(static_cast<int32_t>(r.oi - st.oi));
            d.oi_change = static_cast<int32_t>(zigzag32(r.oi_change - st.oi_change));
            for (int k = 0; k < 5; ++k) {
                d.bid_qty[k]    = zigzag32(static_cast<int32_t>(r.bid_qty[k] - st.bid_qty[k]));
                d.ask_qty[k]    = zigzag32(static_cast<int32_t>(r.ask_qty[k] - st.ask_qty[k]));
                d.bid_orders[k] = zigzag16(static_cast<int16_t>(r.bid_orders[k] - st.bid_orders[k]));
                d.ask_orders[k] = zigzag16(static_cast<int16_t>(r.ask_orders[k] - st.ask_orders[k]));
            }
            d.iv_bps       = static_cast<int32_t>(zigzag32(r.iv_bps - st.iv_bps));
            d.delta_bps    = static_cast<int16_t>(zigzag16(static_cast<int16_t>(r.delta_bps - st.delta_bps)));
            d.gamma_scaled = static_cast<int32_t>(zigzag32(r.gamma_scaled - st.gamma_scaled));
            d.theta_scaled = static_cast<int32_t>(zigzag32(r.theta_scaled - st.theta_scaled));
            d.vega_scaled  = static_cast<int32_t>(zigzag32(r.vega_scaled - st.vega_scaled));
        } else {
            d.ltp_paise = static_cast<int32_t>(zigzag32(r.ltp_paise));
            d.volume    = zigzag32(static_cast<int32_t>(r.volume));
            d.oi        = zigzag32(static_cast<int32_t>(r.oi));
            d.oi_change = static_cast<int32_t>(zigzag32(r.oi_change));
            for (int k = 0; k < 5; ++k) {
                d.bid_qty[k]    = zigzag32(static_cast<int32_t>(r.bid_qty[k]));
                d.ask_qty[k]    = zigzag32(static_cast<int32_t>(r.ask_qty[k]));
                d.bid_orders[k] = zigzag16(static_cast<int16_t>(r.bid_orders[k]));
                d.ask_orders[k] = zigzag16(static_cast<int16_t>(r.ask_orders[k]));
            }
            d.iv_bps       = static_cast<int32_t>(zigzag32(r.iv_bps));
            d.delta_bps    = static_cast<int16_t>(zigzag16(r.delta_bps));
            d.gamma_scaled = static_cast<int32_t>(zigzag32(r.gamma_scaled));
            d.theta_scaled = static_cast<int32_t>(zigzag32(r.theta_scaled));
            d.vega_scaled  = static_cast<int32_t>(zigzag32(r.vega_scaled));
            st.instrument_id = r.instrument_id;
        }

        st.ltp_paise = r.ltp_paise;
        st.volume    = r.volume;
        st.oi        = r.oi;
        st.oi_change = r.oi_change;
        for (int k = 0; k < 5; ++k) {
            st.bid_qty[k]    = r.bid_qty[k];
            st.ask_qty[k]    = r.ask_qty[k];
            st.bid_orders[k] = r.bid_orders[k];
            st.ask_orders[k] = r.ask_orders[k];
        }
        st.iv_bps       = r.iv_bps;
        st.delta_bps    = r.delta_bps;
        st.gamma_scaled = r.gamma_scaled;
        st.theta_scaled = r.theta_scaled;
        st.vega_scaled  = r.vega_scaled;

        d.reserved_greek = r.reserved_greek;
        d.flags          = r.flags;
        d.exchange_id    = r.exchange_id;
        d.reserved_pad   = r.reserved_pad;

        d_rows[i] = d;
    }
    bloom = local_bloom;

    const uint32_t* src_words = reinterpret_cast<const uint32_t*>(scratch.data());
    constexpr size_t TILE = 64;
    alignas(64) uint32_t tile_words[ROW_BYTES][TILE / 4];

#if defined(VTF_HAS_SSE4_2)
    const __m128i shuf_mask = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
#endif

    size_t i0 = 0;
    for (; i0 + TILE <= n; i0 += TILE) {
        const uint32_t* tile_src = src_words + i0 * ROW_WORDS;
        for (size_t g = 0; g < TILE / 4; ++g) {
            const uint32_t* r0 = tile_src + (g * 4 + 0) * ROW_WORDS;
            const uint32_t* r1 = tile_src + (g * 4 + 1) * ROW_WORDS;
            const uint32_t* r2 = tile_src + (g * 4 + 2) * ROW_WORDS;
            const uint32_t* r3 = tile_src + (g * 4 + 3) * ROW_WORDS;
            for (size_t w = 0; w < ROW_WORDS; ++w) {
#if defined(VTF_HAS_SSE4_2)
                __m128i v = _mm_setr_epi32(
                    static_cast<int>(r0[w]),
                    static_cast<int>(r1[w]),
                    static_cast<int>(r2[w]),
                    static_cast<int>(r3[w])
                );
                __m128i t = _mm_shuffle_epi8(v, shuf_mask);
                tile_words[w * 4 + 0][g] = static_cast<uint32_t>(_mm_cvtsi128_si32(t));
                tile_words[w * 4 + 1][g] = static_cast<uint32_t>(_mm_extract_epi32(t, 1));
                tile_words[w * 4 + 2][g] = static_cast<uint32_t>(_mm_extract_epi32(t, 2));
                tile_words[w * 4 + 3][g] = static_cast<uint32_t>(_mm_extract_epi32(t, 3));
#else
                uint32_t v0 = r0[w], v1 = r1[w], v2 = r2[w], v3 = r3[w];
                for (size_t b = 0; b < 4; ++b) {
                    uint32_t shift = static_cast<uint32_t>(b * 8);
                    tile_words[w * 4 + b][g] =
                        ((v0 >> shift) & 0xFFu) |
                        (((v1 >> shift) & 0xFFu) << 8) |
                        (((v2 >> shift) & 0xFFu) << 16) |
                        (((v3 >> shift) & 0xFFu) << 24);
                }
#endif
            }
        }
        for (size_t b = 0; b < ROW_BYTES; ++b) {
            std::memcpy(out_buf + b * n + i0, tile_words[b], TILE);
        }
    }

    const uint8_t* src_bytes = scratch.data();
    for (size_t r = i0; r < n; ++r) {
        const uint8_t* row_p = src_bytes + r * ROW_BYTES;
        for (size_t b = 0; b < ROW_BYTES; ++b) {
            out_buf[b * n + r] = row_p[b];
        }
    }
}

void decode_soa_to_rows(
    const uint8_t* in_buf,
    size_t n,
    VTFTickRow* rows,
    std::vector<uint8_t>& scratch
) {
    scratch.resize(n * ROW_BYTES);
    uint32_t* dst_words = reinterpret_cast<uint32_t*>(scratch.data());

    constexpr size_t TILE = 64;
    alignas(64) uint32_t tile_words[ROW_BYTES][TILE / 4];

#if defined(VTF_HAS_SSE4_2)
    const __m128i shuf_mask = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
#endif

    size_t i0 = 0;
    for (; i0 + TILE <= n; i0 += TILE) {
        for (size_t b = 0; b < ROW_BYTES; ++b) {
            std::memcpy(tile_words[b], in_buf + b * n + i0, TILE);
        }
        uint32_t* tile_dst = dst_words + i0 * ROW_WORDS;
        for (size_t g = 0; g < TILE / 4; ++g) {
            uint32_t* r0 = tile_dst + (g * 4 + 0) * ROW_WORDS;
            uint32_t* r1 = tile_dst + (g * 4 + 1) * ROW_WORDS;
            uint32_t* r2 = tile_dst + (g * 4 + 2) * ROW_WORDS;
            uint32_t* r3 = tile_dst + (g * 4 + 3) * ROW_WORDS;
            for (size_t w = 0; w < ROW_WORDS; ++w) {
#if defined(VTF_HAS_SSE4_2)
                __m128i v = _mm_setr_epi32(
                    static_cast<int>(tile_words[w * 4 + 0][g]),
                    static_cast<int>(tile_words[w * 4 + 1][g]),
                    static_cast<int>(tile_words[w * 4 + 2][g]),
                    static_cast<int>(tile_words[w * 4 + 3][g])
                );
                __m128i t = _mm_shuffle_epi8(v, shuf_mask);
                r0[w] = static_cast<uint32_t>(_mm_cvtsi128_si32(t));
                r1[w] = static_cast<uint32_t>(_mm_extract_epi32(t, 1));
                r2[w] = static_cast<uint32_t>(_mm_extract_epi32(t, 2));
                r3[w] = static_cast<uint32_t>(_mm_extract_epi32(t, 3));
#else
                uint32_t b0 = tile_words[w * 4 + 0][g];
                uint32_t b1 = tile_words[w * 4 + 1][g];
                uint32_t b2 = tile_words[w * 4 + 2][g];
                uint32_t b3 = tile_words[w * 4 + 3][g];
                r0[w] = (b0 & 0xFFu) | ((b1 & 0xFFu) << 8) | ((b2 & 0xFFu) << 16) | ((b3 & 0xFFu) << 24);
                r1[w] = ((b0 >> 8) & 0xFFu) | (((b1 >> 8) & 0xFFu) << 8) | (((b2 >> 8) & 0xFFu) << 16) | (((b3 >> 8) & 0xFFu) << 24);
                r2[w] = ((b0 >> 16) & 0xFFu) | (((b1 >> 16) & 0xFFu) << 8) | (((b2 >> 16) & 0xFFu) << 16) | (((b3 >> 16) & 0xFFu) << 24);
                r3[w] = ((b0 >> 24) & 0xFFu) | (((b1 >> 24) & 0xFFu) << 8) | (((b2 >> 24) & 0xFFu) << 16) | (((b3 >> 24) & 0xFFu) << 24);
#endif
            }
        }
    }

    uint8_t* dst_bytes = scratch.data();
    for (size_t r = i0; r < n; ++r) {
        uint8_t* row_p = dst_bytes + r * ROW_BYTES;
        for (size_t b = 0; b < ROW_BYTES; ++b) {
            row_p[b] = in_buf[b * n + r];
        }
    }

    const VTFTickRow* d_rows = reinterpret_cast<const VTFTickRow*>(scratch.data());

    InstrumentState state_table[STATE_TABLE_SIZE];
    for (size_t s = 0; s < STATE_TABLE_SIZE; ++s) {
        state_table[s].instrument_id = 0xFFFFFFFFu;
    }

    int64_t prev_ts = 0;
    for (size_t i = 0; i < n; ++i) {
        const VTFTickRow& d = d_rows[i];
        VTFTickRow r;

        prev_ts += unzigzag64(static_cast<uint64_t>(d.timestamp_ns));
        r.timestamp_ns  = prev_ts;
        r.instrument_id = d.instrument_id;
        r.ltq           = d.ltq;

        size_t slot = inst_slot(r.instrument_id);
        InstrumentState& st = state_table[slot];

        if (st.instrument_id == r.instrument_id) {
            r.ltp_paise = st.ltp_paise + unzigzag32(static_cast<uint32_t>(d.ltp_paise));
            r.volume    = static_cast<uint32_t>(static_cast<int32_t>(st.volume) + unzigzag32(d.volume));
            r.oi        = static_cast<uint32_t>(static_cast<int32_t>(st.oi) + unzigzag32(d.oi));
            r.oi_change = st.oi_change + unzigzag32(static_cast<uint32_t>(d.oi_change));
            for (int k = 0; k < 5; ++k) {
                r.bid_qty[k]    = static_cast<uint32_t>(static_cast<int32_t>(st.bid_qty[k]) + unzigzag32(d.bid_qty[k]));
                r.ask_qty[k]    = static_cast<uint32_t>(static_cast<int32_t>(st.ask_qty[k]) + unzigzag32(d.ask_qty[k]));
                r.bid_orders[k] = static_cast<uint16_t>(static_cast<int16_t>(st.bid_orders[k]) + unzigzag16(d.bid_orders[k]));
                r.ask_orders[k] = static_cast<uint16_t>(static_cast<int16_t>(st.ask_orders[k]) + unzigzag16(d.ask_orders[k]));
            }
            r.iv_bps       = st.iv_bps + unzigzag32(static_cast<uint32_t>(d.iv_bps));
            r.delta_bps    = static_cast<int16_t>(st.delta_bps + unzigzag16(static_cast<uint16_t>(d.delta_bps)));
            r.gamma_scaled = st.gamma_scaled + unzigzag32(static_cast<uint32_t>(d.gamma_scaled));
            r.theta_scaled = st.theta_scaled + unzigzag32(static_cast<uint32_t>(d.theta_scaled));
            r.vega_scaled  = st.vega_scaled + unzigzag32(static_cast<uint32_t>(d.vega_scaled));
        } else {
            r.ltp_paise = unzigzag32(static_cast<uint32_t>(d.ltp_paise));
            r.volume    = static_cast<uint32_t>(unzigzag32(d.volume));
            r.oi        = static_cast<uint32_t>(unzigzag32(d.oi));
            r.oi_change = unzigzag32(static_cast<uint32_t>(d.oi_change));
            for (int k = 0; k < 5; ++k) {
                r.bid_qty[k]    = static_cast<uint32_t>(unzigzag32(d.bid_qty[k]));
                r.ask_qty[k]    = static_cast<uint32_t>(unzigzag32(d.ask_qty[k]));
                r.bid_orders[k] = static_cast<uint16_t>(unzigzag16(d.bid_orders[k]));
                r.ask_orders[k] = static_cast<uint16_t>(unzigzag16(d.ask_orders[k]));
            }
            r.iv_bps       = unzigzag32(static_cast<uint32_t>(d.iv_bps));
            r.delta_bps    = unzigzag16(static_cast<uint16_t>(d.delta_bps));
            r.gamma_scaled = unzigzag32(static_cast<uint32_t>(d.gamma_scaled));
            r.theta_scaled = unzigzag32(static_cast<uint32_t>(d.theta_scaled));
            r.vega_scaled  = unzigzag32(static_cast<uint32_t>(d.vega_scaled));
            st.instrument_id = r.instrument_id;
        }

        r.bid_price_paise[0] = r.ltp_paise + unzigzag32(static_cast<uint32_t>(d.bid_price_paise[0]));
        r.ask_price_paise[0] = r.bid_price_paise[0] + unzigzag32(static_cast<uint32_t>(d.ask_price_paise[0]));
        for (int k = 1; k < 5; ++k) {
            r.bid_price_paise[k] = r.bid_price_paise[k - 1] - unzigzag32(static_cast<uint32_t>(d.bid_price_paise[k]));
            r.ask_price_paise[k] = r.ask_price_paise[k - 1] + unzigzag32(static_cast<uint32_t>(d.ask_price_paise[k]));
        }

        st.ltp_paise = r.ltp_paise;
        st.volume    = r.volume;
        st.oi        = r.oi;
        st.oi_change = r.oi_change;
        for (int k = 0; k < 5; ++k) {
            st.bid_qty[k]    = r.bid_qty[k];
            st.ask_qty[k]    = r.ask_qty[k];
            st.bid_orders[k] = r.bid_orders[k];
            st.ask_orders[k] = r.ask_orders[k];
        }
        st.iv_bps       = r.iv_bps;
        st.delta_bps    = r.delta_bps;
        st.gamma_scaled = r.gamma_scaled;
        st.theta_scaled = r.theta_scaled;
        st.vega_scaled  = r.vega_scaled;

        r.reserved_greek = d.reserved_greek;
        r.flags          = d.flags;
        r.exchange_id    = d.exchange_id;
        r.reserved_pad   = d.reserved_pad;

        rows[i] = r;
    }
}

class VTFWriter {
public:
    VTFWriter() : fd_(-1), codec_(CODEC_LZ4_FAST), block_target_rows_(16384),
                  current_offset_(0), total_rows_(0), closed_(true) {}

    ~VTFWriter() {
        if (!closed_) {
            close();
        }
    }

    int open(const char* path, uint8_t codec_id, uint32_t block_target_rows, uint64_t created_at_ns) {
        if (block_target_rows == 0) block_target_rows = 16384;
        fd_ = os_open_create_trunc(path);
        if (fd_ < 0) return -1;

        codec_ = codec_id;
        block_target_rows_ = block_target_rows;
        row_buffer_.clear();
        row_buffer_.reserve(block_target_rows_);
        index_.clear();
        total_rows_ = 0;
        closed_ = false;

        std::memset(&header_, 0, sizeof(header_));
        header_.magic             = VTF_FILE_MAGIC;
        header_.version_major     = VTF_VERSION_MAJOR;
        header_.version_minor     = VTF_VERSION_MINOR;
        header_.codec_id          = codec_;
        header_.flags             = FLAG_BYTE_STREAM_SPLIT;
        header_.block_target_rows = block_target_rows_;
        header_.created_at_ns     = created_at_ns;
        header_.price_scale       = 100u;
        header_.iv_scale          = 10000u;
        header_.greek_scale       = 10000u;
        header_.gamma_scale       = 1000000u;
        header_.header_crc32c     = crc32c(&header_, sizeof(VTFFileHeader) - sizeof(uint32_t));

        if (!write_all(fd_, &header_, sizeof(header_))) {
            os_close(fd_);
            fd_ = -1;
            closed_ = true;
            return -2;
        }
        current_offset_ = sizeof(VTFFileHeader);
        return 0;
    }

    int append_rows(const VTFTickRow* rows, size_t count) {
        if (closed_ || fd_ < 0) return -1;
        size_t idx = 0;
        while (idx < count) {
            size_t space = block_target_rows_ - row_buffer_.size();
            size_t to_copy = std::min(space, count - idx);
            if (row_buffer_.empty() && to_copy == block_target_rows_) {
                int rc = write_block(rows + idx, block_target_rows_);
                if (rc != 0) return rc;
            } else {
                row_buffer_.insert(row_buffer_.end(), rows + idx, rows + idx + to_copy);
                if (row_buffer_.size() == block_target_rows_) {
                    int rc = write_block(row_buffer_.data(), row_buffer_.size());
                    row_buffer_.clear();
                    if (rc != 0) return rc;
                }
            }
            idx += to_copy;
        }
        return 0;
    }

    int flush() {
        if (closed_ || fd_ < 0) return -1;
        if (!row_buffer_.empty()) {
            int rc = write_block(row_buffer_.data(), row_buffer_.size());
            row_buffer_.clear();
            if (rc != 0) return rc;
        }
        return 0;
    }

    int close() {
        if (closed_ || fd_ < 0) return 0;
        int rc = flush();
        if (rc != 0) {
            os_close(fd_);
            fd_ = -1;
            closed_ = true;
            return rc;
        }

        uint64_t index_offset = current_offset_;
        uint32_t footer_crc = 0u;
        if (!index_.empty()) {
            size_t index_bytes = index_.size() * sizeof(VTFBlockIndexEntry);
            footer_crc = crc32c_append(footer_crc, index_.data(), index_bytes);
            if (!write_all(fd_, index_.data(), index_bytes)) {
                os_close(fd_);
                fd_ = -1;
                closed_ = true;
                return -3;
            }
            current_offset_ += index_bytes;
        }

        VTFFooterTrailer trailer{};
        trailer.index_offset  = index_offset;
        trailer.total_blocks  = index_.size();
        trailer.total_rows    = total_rows_;
        footer_crc            = crc32c_append(footer_crc, &trailer, 24);
        trailer.footer_crc32c = footer_crc;
        trailer.trailer_magic = VTF_TRAILER_MAGIC;

        if (!write_all(fd_, &trailer, sizeof(trailer))) {
            os_close(fd_);
            fd_ = -1;
            closed_ = true;
            return -4;
        }
        current_offset_ += sizeof(trailer);

        header_.flags |= FLAG_HAS_FOOTER;
        header_.header_crc32c = crc32c(&header_, sizeof(VTFFileHeader) - sizeof(uint32_t));
        if (!pwrite_all(fd_, &header_, sizeof(header_), 0)) {
            os_close(fd_);
            fd_ = -1;
            closed_ = true;
            return -5;
        }

        os_fsync(fd_);
        os_close(fd_);
        fd_ = -1;
        closed_ = true;
        return 0;
    }

    void abort_without_footer() {
        if (fd_ >= 0) {
            os_close(fd_);
            fd_ = -1;
        }
        closed_ = true;
    }

private:
    int write_block(const VTFTickRow* rows, size_t n) {
        size_t raw_bytes = n * sizeof(VTFTickRow);
        soa_buf_.resize(raw_bytes);

        int64_t min_ts = 0, max_ts = 0;
        uint64_t bloom = 0;
        encode_rows_to_soa(rows, n, soa_buf_.data(), scratch_buf_, min_ts, max_ts, bloom);

        const auto& cd = codecs();
        const uint8_t* payload_ptr = nullptr;
        size_t comp_bytes = 0;

        if (codec_ == CODEC_NONE) {
            payload_ptr = soa_buf_.data();
            comp_bytes  = raw_bytes;
        } else if (codec_ == CODEC_LZ4_FAST || codec_ == CODEC_LZ4_DEFAULT) {
            int max_dst = cd.lz4_bound(static_cast<int>(raw_bytes));
            comp_buf_.resize(static_cast<size_t>(max_dst));
            int csize = 0;
            if (codec_ == CODEC_LZ4_FAST) {
                csize = cd.lz4_fast(
                    reinterpret_cast<const char*>(soa_buf_.data()),
                    reinterpret_cast<char*>(comp_buf_.data()),
                    static_cast<int>(raw_bytes),
                    max_dst,
                    4 // Default HFT acceleration = 4
                );
            } else {
                csize = cd.lz4_default(
                    reinterpret_cast<const char*>(soa_buf_.data()),
                    reinterpret_cast<char*>(comp_buf_.data()),
                    static_cast<int>(raw_bytes),
                    max_dst
                );
            }
            if (csize <= 0) return -10;
            payload_ptr = comp_buf_.data();
            comp_bytes  = static_cast<size_t>(csize);
        } else if (codec_ == CODEC_ZSTD) {
            if (cd.zstd_compress && cd.zstd_bound && cd.zstd_is_error) {
                size_t max_dst = cd.zstd_bound(raw_bytes);
                comp_buf_.resize(max_dst);
                size_t csize = cd.zstd_compress(comp_buf_.data(), max_dst, soa_buf_.data(), raw_bytes, 3);
                if (cd.zstd_is_error(csize)) return -11;
                payload_ptr = comp_buf_.data();
                comp_bytes  = csize;
            } else {
                return -12;
            }
        } else {
            return -13;
        }

        VTFBlockHeader bh{};
        bh.block_magic        = VTF_BLOCK_MAGIC;
        bh.block_index        = static_cast<uint32_t>(index_.size());
        bh.row_count          = static_cast<uint32_t>(n);
        bh.uncompressed_bytes = static_cast<uint32_t>(raw_bytes);
        bh.compressed_bytes   = static_cast<uint32_t>(comp_bytes);
        bh.min_timestamp_ns   = min_ts;
        bh.max_timestamp_ns   = max_ts;
        bh.inst_bloom_filter  = bloom;
        bh.payload_crc32c     = crc32c(payload_ptr, comp_bytes);
        bh.block_header_crc32c = crc32c(&bh, sizeof(VTFBlockHeader) - sizeof(uint32_t));

        if (!write_all(fd_, &bh, sizeof(bh))) return -20;
        if (!write_all(fd_, payload_ptr, comp_bytes)) return -21;

        VTFBlockIndexEntry ie{};
        ie.block_file_offset = current_offset_;
        ie.compressed_bytes  = static_cast<uint32_t>(comp_bytes);
        ie.row_count         = static_cast<uint32_t>(n);
        ie.min_timestamp_ns  = min_ts;
        ie.max_timestamp_ns  = max_ts;
        ie.inst_bloom_filter = bloom;
        index_.push_back(ie);

        current_offset_ += sizeof(bh) + comp_bytes;
        total_rows_ += n;
        return 0;
    }

    int fd_;
    uint8_t codec_;
    uint32_t block_target_rows_;
    uint64_t current_offset_;
    uint64_t total_rows_;
    bool closed_;
    VTFFileHeader header_;
    std::vector<VTFTickRow> row_buffer_;
    std::vector<VTFBlockIndexEntry> index_;
    std::vector<uint8_t> soa_buf_;
    std::vector<uint8_t> scratch_buf_;
    std::vector<uint8_t> comp_buf_;
};

int recover_vtf_file(const char* path, VTFRecoveryReport* report) {
    if (!report) return -1;
    std::memset(report, 0, sizeof(VTFRecoveryReport));

    int fd = os_open_rw(path);
    if (fd < 0) {
        report->status = -1;
        return -1;
    }

    int64_t fsz = os_file_size(fd);
    if (fsz < static_cast<int64_t>(sizeof(VTFFileHeader))) {
        os_close(fd);
        report->status = -2;
        return -2;
    }
    uint64_t file_size = static_cast<uint64_t>(fsz);
    report->original_file_size = file_size;

    VTFFileHeader hdr{};
    if (!pread_all(fd, &hdr, sizeof(hdr), 0)) {
        os_close(fd);
        report->status = -3;
        return -3;
    }

    uint32_t expected_hdr_crc = crc32c(&hdr, sizeof(VTFFileHeader) - sizeof(uint32_t));
    if (hdr.magic != VTF_FILE_MAGIC || hdr.header_crc32c != expected_hdr_crc) {
        os_close(fd);
        report->status = -4;
        return -4;
    }

    bool footer_intact = false;
    if ((hdr.flags & FLAG_HAS_FOOTER) && file_size >= sizeof(VTFFileHeader) + sizeof(VTFFooterTrailer)) {
        VTFFooterTrailer tr{};
        if (pread_all(fd, &tr, sizeof(tr), file_size - sizeof(VTFFooterTrailer))) {
            if (tr.trailer_magic == VTF_TRAILER_MAGIC &&
                tr.index_offset >= sizeof(VTFFileHeader) &&
                tr.index_offset + tr.total_blocks * sizeof(VTFBlockIndexEntry) + sizeof(VTFFooterTrailer) == file_size) {
                std::vector<VTFBlockIndexEntry> idx(tr.total_blocks);
                size_t idx_bytes = tr.total_blocks * sizeof(VTFBlockIndexEntry);
                if (idx_bytes == 0 || pread_all(fd, idx.data(), idx_bytes, tr.index_offset)) {
                    uint32_t fcrc = 0u;
                    if (idx_bytes > 0) fcrc = crc32c_append(fcrc, idx.data(), idx_bytes);
                    fcrc = crc32c_append(fcrc, &tr, 24);
                    if (fcrc == tr.footer_crc32c) {
                        footer_intact = true;
                    }
                }
            }
        }
    }

    uint64_t offset = sizeof(VTFFileHeader);
    std::vector<VTFBlockIndexEntry> valid_index;
    std::vector<uint8_t> payload_buf;
    uint64_t total_valid_rows = 0;

    while (offset + sizeof(VTFBlockHeader) <= file_size) {
        VTFBlockHeader bh{};
        if (!pread_all(fd, &bh, sizeof(bh), offset)) break;

        if (bh.block_magic != VTF_BLOCK_MAGIC) {
            break;
        }

        uint32_t expected_bh_crc = crc32c(&bh, sizeof(VTFBlockHeader) - sizeof(uint32_t));
        if (bh.block_header_crc32c != expected_bh_crc) {
            footer_intact = false;
            break;
        }

        if (bh.row_count == 0 || bh.uncompressed_bytes != bh.row_count * sizeof(VTFTickRow)) {
            footer_intact = false;
            break;
        }

        uint64_t next_offset = offset + sizeof(VTFBlockHeader) + bh.compressed_bytes;
        if (next_offset > file_size) {
            footer_intact = false;
            break;
        }

        payload_buf.resize(bh.compressed_bytes);
        if (!pread_all(fd, payload_buf.data(), bh.compressed_bytes, offset + sizeof(VTFBlockHeader))) {
            footer_intact = false;
            break;
        }

        uint32_t actual_payload_crc = crc32c(payload_buf.data(), bh.compressed_bytes);
        if (actual_payload_crc != bh.payload_crc32c) {
            footer_intact = false;
            break;
        }

        VTFBlockIndexEntry entry{};
        entry.block_file_offset = offset;
        entry.compressed_bytes  = bh.compressed_bytes;
        entry.row_count         = bh.row_count;
        entry.min_timestamp_ns  = bh.min_timestamp_ns;
        entry.max_timestamp_ns  = bh.max_timestamp_ns;
        entry.inst_bloom_filter = bh.inst_bloom_filter;
        valid_index.push_back(entry);

        total_valid_rows += bh.row_count;
        offset = next_offset;
    }

    if (footer_intact) {
        report->status              = 0;
        report->valid_blocks        = static_cast<uint32_t>(valid_index.size());
        report->valid_rows          = total_valid_rows;
        report->recovered_file_size = file_size;
        report->truncated_bytes     = 0;
        report->had_valid_footer    = 1;
        os_close(fd);
        return 0;
    }

    if (!os_truncate(fd, offset)) {
        os_close(fd);
        report->status = -5;
        return -5;
    }
    if (!os_seek(fd, offset)) {
        os_close(fd);
        report->status = -6;
        return -6;
    }

    uint64_t truncated_bytes = (file_size > offset) ? (file_size - offset) : 0;

    uint64_t index_offset = offset;
    uint32_t footer_crc = 0u;
    if (!valid_index.empty()) {
        size_t idx_bytes = valid_index.size() * sizeof(VTFBlockIndexEntry);
        footer_crc = crc32c_append(footer_crc, valid_index.data(), idx_bytes);
        if (!write_all(fd, valid_index.data(), idx_bytes)) {
            os_close(fd);
            report->status = -7;
            return -7;
        }
        offset += idx_bytes;
    }

    VTFFooterTrailer trailer{};
    trailer.index_offset  = index_offset;
    trailer.total_blocks  = valid_index.size();
    trailer.total_rows    = total_valid_rows;
    footer_crc            = crc32c_append(footer_crc, &trailer, 24);
    trailer.footer_crc32c = footer_crc;
    trailer.trailer_magic = VTF_TRAILER_MAGIC;

    if (!write_all(fd, &trailer, sizeof(trailer))) {
        os_close(fd);
        report->status = -8;
        return -8;
    }
    offset += sizeof(trailer);

    hdr.flags |= (FLAG_HAS_FOOTER | FLAG_RECOVERED);
    hdr.header_crc32c = crc32c(&hdr, sizeof(VTFFileHeader) - sizeof(uint32_t));
    if (!pwrite_all(fd, &hdr, sizeof(hdr), 0)) {
        os_close(fd);
        report->status = -9;
        return -9;
    }

    os_fsync(fd);
    os_close(fd);

    report->status              = 0;
    report->valid_blocks        = static_cast<uint32_t>(valid_index.size());
    report->valid_rows          = total_valid_rows;
    report->recovered_file_size = offset;
    report->truncated_bytes     = truncated_bytes;
    report->had_valid_footer    = 0;
    return 0;
}

class VTFReader {
public:
    VTFReader() : fd_(-1), file_size_(0), total_rows_(0) {}
    ~VTFReader() { close(); }

    int open(const char* path, bool auto_recover = true) {
        close();
        fd_ = os_open_read(path);
        if (fd_ < 0) return -1;

        int64_t fsz = os_file_size(fd_);
        if (fsz < static_cast<int64_t>(sizeof(VTFFileHeader))) {
            close();
            return -2;
        }
        file_size_ = static_cast<uint64_t>(fsz);

        if (!pread_all(fd_, &header_, sizeof(header_), 0)) {
            close();
            return -3;
        }

        uint32_t expected_crc = crc32c(&header_, sizeof(VTFFileHeader) - sizeof(uint32_t));
        if (header_.magic != VTF_FILE_MAGIC || header_.header_crc32c != expected_crc) {
            close();
            return -4;
        }

        bool loaded_footer = false;
        if ((header_.flags & FLAG_HAS_FOOTER) && file_size_ >= sizeof(VTFFileHeader) + sizeof(VTFFooterTrailer)) {
            VTFFooterTrailer tr{};
            if (pread_all(fd_, &tr, sizeof(tr), file_size_ - sizeof(VTFFooterTrailer))) {
                if (tr.trailer_magic == VTF_TRAILER_MAGIC &&
                    tr.index_offset >= sizeof(VTFFileHeader) &&
                    tr.index_offset + tr.total_blocks * sizeof(VTFBlockIndexEntry) + sizeof(VTFFooterTrailer) == file_size_) {
                    index_.resize(tr.total_blocks);
                    size_t idx_bytes = tr.total_blocks * sizeof(VTFBlockIndexEntry);
                    if (idx_bytes == 0 || pread_all(fd_, index_.data(), idx_bytes, tr.index_offset)) {
                        uint32_t fcrc = 0u;
                        if (idx_bytes > 0) fcrc = crc32c_append(fcrc, index_.data(), idx_bytes);
                        fcrc = crc32c_append(fcrc, &tr, 24);
                        if (fcrc == tr.footer_crc32c) {
                            loaded_footer = true;
                            total_rows_ = tr.total_rows;
                        }
                    }
                }
            }
        }

        if (!loaded_footer) {
            close();
            if (!auto_recover) return -5;
            VTFRecoveryReport report{};
            int rc = recover_vtf_file(path, &report);
            if (rc != 0 || report.status != 0) return -6;
            return open(path, false);
        }
        return 0;
    }

    void close() {
        if (fd_ >= 0) {
            os_close(fd_);
            fd_ = -1;
        }
        index_.clear();
        total_rows_ = 0;
        file_size_ = 0;
    }

    uint64_t total_rows() const { return total_rows_; }
    uint64_t total_blocks() const { return index_.size(); }
    const VTFFileHeader& header() const { return header_; }
    const std::vector<VTFBlockIndexEntry>& index() const { return index_; }

    int read_block(size_t block_idx, VTFTickRow* out_rows, size_t max_rows, bool verify_crc = true) {
        if (fd_ < 0 || block_idx >= index_.size()) return -1;
        const VTFBlockIndexEntry& ie = index_[block_idx];
        if (max_rows < ie.row_count) return -2;

        VTFBlockHeader bh{};
        if (!pread_all(fd_, &bh, sizeof(bh), ie.block_file_offset)) return -3;
        if (bh.block_magic != VTF_BLOCK_MAGIC) return -4;

        if (verify_crc) {
            uint32_t expected_bh_crc = crc32c(&bh, sizeof(VTFBlockHeader) - sizeof(uint32_t));
            if (bh.block_header_crc32c != expected_bh_crc) return -5;
        }

        comp_buf_.resize(bh.compressed_bytes);
        if (!pread_all(fd_, comp_buf_.data(), bh.compressed_bytes, ie.block_file_offset + sizeof(VTFBlockHeader))) {
            return -6;
        }

        if (verify_crc) {
            uint32_t actual_payload_crc = crc32c(comp_buf_.data(), bh.compressed_bytes);
            if (actual_payload_crc != bh.payload_crc32c) return -7;
        }

        const uint8_t* soa_ptr = nullptr;
        const auto& cd = codecs();

        if (header_.codec_id == CODEC_NONE) {
            if (bh.compressed_bytes != bh.uncompressed_bytes) return -8;
            soa_ptr = comp_buf_.data();
        } else if (header_.codec_id == CODEC_LZ4_FAST || header_.codec_id == CODEC_LZ4_DEFAULT) {
            soa_buf_.resize(bh.uncompressed_bytes);
            int dsize = cd.lz4_decomp(
                reinterpret_cast<const char*>(comp_buf_.data()),
                reinterpret_cast<char*>(soa_buf_.data()),
                static_cast<int>(bh.compressed_bytes),
                static_cast<int>(bh.uncompressed_bytes)
            );
            if (dsize != static_cast<int>(bh.uncompressed_bytes)) return -9;
            soa_ptr = soa_buf_.data();
        } else if (header_.codec_id == CODEC_ZSTD) {
            soa_buf_.resize(bh.uncompressed_bytes);
            if (!cd.zstd_decomp || !cd.zstd_is_error) return -12;
            size_t dsize = cd.zstd_decomp(
                soa_buf_.data(),
                bh.uncompressed_bytes,
                comp_buf_.data(),
                bh.compressed_bytes
            );
            if (cd.zstd_is_error(dsize) || dsize != bh.uncompressed_bytes) {
                return -11;
            }
            soa_ptr = soa_buf_.data();
        } else {
            return -10;
        }

        decode_soa_to_rows(soa_ptr, bh.row_count, out_rows, scratch_buf_);
        return static_cast<int>(bh.row_count);
    }

    int read_all(VTFTickRow* out_rows, size_t max_rows, bool verify_crc = true) {
        if (max_rows < total_rows_) return -1;
        size_t cursor = 0;
        for (size_t b = 0; b < index_.size(); ++b) {
            int n = read_block(b, out_rows + cursor, max_rows - cursor, verify_crc);
            if (n < 0) return n;
            cursor += static_cast<size_t>(n);
        }
        return static_cast<int>(cursor);
    }

    int64_t query_range(
        int64_t start_ns,
        int64_t end_ns,
        uint32_t filter_instrument,
        VTFTickRow* out_rows,
        size_t max_rows
    ) {
        uint64_t req_bloom = (filter_instrument != 0) ? bloom_hash(filter_instrument) : 0ULL;
        size_t count = 0;
        std::vector<VTFTickRow> block_rows;

        for (size_t b = 0; b < index_.size(); ++b) {
            const VTFBlockIndexEntry& ie = index_[b];
            if (ie.max_timestamp_ns < start_ns || ie.min_timestamp_ns > end_ns) {
                continue;
            }
            if (filter_instrument != 0 && ((ie.inst_bloom_filter & req_bloom) != req_bloom)) {
                continue;
            }

            block_rows.resize(ie.row_count);
            int n = read_block(b, block_rows.data(), ie.row_count, true);
            if (n < 0) return n;

            for (int i = 0; i < n; ++i) {
                const VTFTickRow& r = block_rows[i];
                if (r.timestamp_ns >= start_ns && r.timestamp_ns <= end_ns) {
                    if (filter_instrument == 0 || r.instrument_id == filter_instrument) {
                        if (count >= max_rows) return static_cast<int64_t>(count);
                        out_rows[count++] = r;
                    }
                }
            }
        }
        return static_cast<int64_t>(count);
    }

private:
    int fd_;
    uint64_t file_size_;
    uint64_t total_rows_;
    VTFFileHeader header_;
    std::vector<VTFBlockIndexEntry> index_;
    std::vector<uint8_t> comp_buf_;
    std::vector<uint8_t> soa_buf_;
    std::vector<uint8_t> scratch_buf_;
};

} // namespace vtf

// ============================================================================
// C API Export Functions
// ============================================================================

extern "C" {

VTF_API void* vtf_writer_create() { return new vtf::VTFWriter(); }
VTF_API int vtf_writer_open(void* w, const char* path, uint8_t codec_id, uint32_t block_target_rows, uint64_t created_at_ns) {
    return w ? reinterpret_cast<vtf::VTFWriter*>(w)->open(path, codec_id, block_target_rows, created_at_ns) : -1;
}
VTF_API int vtf_writer_append(void* w, const VTFTickRow* rows, size_t count) {
    return w ? reinterpret_cast<vtf::VTFWriter*>(w)->append_rows(rows, count) : -1;
}
VTF_API int vtf_writer_flush(void* w) { return w ? reinterpret_cast<vtf::VTFWriter*>(w)->flush() : -1; }
VTF_API int vtf_writer_close(void* w) { return w ? reinterpret_cast<vtf::VTFWriter*>(w)->close() : -1; }
VTF_API void vtf_writer_abort_without_footer(void* w) { if (w) reinterpret_cast<vtf::VTFWriter*>(w)->abort_without_footer(); }
VTF_API void vtf_writer_destroy(void* w) { delete reinterpret_cast<vtf::VTFWriter*>(w); }

VTF_API void* vtf_reader_create() { return new vtf::VTFReader(); }
VTF_API int vtf_reader_open(void* r, const char* path, int auto_recover) {
    return r ? reinterpret_cast<vtf::VTFReader*>(r)->open(path, auto_recover != 0) : -1;
}
VTF_API uint64_t vtf_reader_total_rows(void* r) { return r ? reinterpret_cast<vtf::VTFReader*>(r)->total_rows() : 0; }
VTF_API uint64_t vtf_reader_total_blocks(void* r) { return r ? reinterpret_cast<vtf::VTFReader*>(r)->total_blocks() : 0; }
VTF_API int vtf_reader_get_header(void* r, VTFFileHeader* out_hdr) {
    if (!r || !out_hdr) return -1;
    *out_hdr = reinterpret_cast<vtf::VTFReader*>(r)->header();
    return 0;
}
VTF_API int vtf_reader_get_block_index(void* r, VTFBlockIndexEntry* out_entries, size_t max_entries) {
    if (!r || !out_entries) return -1;
    const auto& idx = reinterpret_cast<vtf::VTFReader*>(r)->index();
    size_t n = std::min(idx.size(), max_entries);
    std::memcpy(out_entries, idx.data(), n * sizeof(VTFBlockIndexEntry));
    return static_cast<int>(n);
}
VTF_API int vtf_reader_read_all(void* r, VTFTickRow* out_rows, size_t max_rows, int verify_crc) {
    return r ? reinterpret_cast<vtf::VTFReader*>(r)->read_all(out_rows, max_rows, verify_crc != 0) : -1;
}
VTF_API int vtf_reader_read_block(void* r, size_t block_idx, VTFTickRow* out_rows, size_t max_rows, int verify_crc) {
    return r ? reinterpret_cast<vtf::VTFReader*>(r)->read_block(block_idx, out_rows, max_rows, verify_crc != 0) : -1;
}
VTF_API int64_t vtf_reader_query_range(void* r, int64_t start_ns, int64_t end_ns, uint32_t filter_instrument, VTFTickRow* out_rows, size_t max_rows) {
    return r ? reinterpret_cast<vtf::VTFReader*>(r)->query_range(start_ns, end_ns, filter_instrument, out_rows, max_rows) : -1;
}
VTF_API void vtf_reader_close(void* r) { if (r) reinterpret_cast<vtf::VTFReader*>(r)->close(); }
VTF_API void vtf_reader_destroy(void* r) { delete reinterpret_cast<vtf::VTFReader*>(r); }

VTF_API int vtf_recover_file(const char* path, VTFRecoveryReport* out_report) {
    return vtf::recover_vtf_file(path, out_report);
}

VTF_API uint32_t vtf_crc32c(const void* data, size_t len) {
    return vtf::crc32c(data, len);
}

VTF_API int raw_lz4_binary_write(const char* path, const VTFTickRow* rows, size_t count, size_t block_rows, int accel) {
    int fd = vtf::os_open_create_trunc(path);
    if (fd < 0) return -1;
    uint64_t total = count;
    if (!vtf::write_all(fd, &total, sizeof(total))) { vtf::os_close(fd); return -2; }

    const auto& cd = vtf::codecs();
    std::vector<char> comp_buf;
    size_t idx = 0;
    while (idx < count) {
        size_t n = std::min(block_rows, count - idx);
        int raw_bytes = static_cast<int>(n * sizeof(VTFTickRow));
        int max_dst = cd.lz4_bound(raw_bytes);
        comp_buf.resize(static_cast<size_t>(max_dst));
        int csize = (accel > 1)
            ? cd.lz4_fast(reinterpret_cast<const char*>(rows + idx), comp_buf.data(), raw_bytes, max_dst, accel)
            : cd.lz4_default(reinterpret_cast<const char*>(rows + idx), comp_buf.data(), raw_bytes, max_dst);
        if (csize <= 0) { vtf::os_close(fd); return -3; }
        uint32_t hdr[2] = { static_cast<uint32_t>(n), static_cast<uint32_t>(csize) };
        if (!vtf::write_all(fd, hdr, sizeof(hdr))) { vtf::os_close(fd); return -4; }
        if (!vtf::write_all(fd, comp_buf.data(), static_cast<size_t>(csize))) { vtf::os_close(fd); return -5; }
        idx += n;
    }
    vtf::os_close(fd);
    return 0;
}

VTF_API int raw_lz4_binary_read(const char* path, VTFTickRow* out_rows, size_t max_rows) {
    int fd = vtf::os_open_read(path);
    if (fd < 0) return -1;
    uint64_t total = 0;
    uint64_t off = 0;
    if (!vtf::pread_all(fd, &total, sizeof(total), off)) { vtf::os_close(fd); return -2; }
    off += sizeof(total);
    if (total > max_rows) { vtf::os_close(fd); return -3; }

    const auto& cd = vtf::codecs();
    std::vector<char> comp_buf;
    size_t idx = 0;
    while (idx < total) {
        uint32_t hdr[2] = {0, 0};
        if (!vtf::pread_all(fd, hdr, sizeof(hdr), off)) { vtf::os_close(fd); return -4; }
        off += sizeof(hdr);
        uint32_t n = hdr[0];
        uint32_t csize = hdr[1];
        comp_buf.resize(csize);
        if (!vtf::pread_all(fd, comp_buf.data(), csize, off)) { vtf::os_close(fd); return -5; }
        off += csize;
        int raw_bytes = static_cast<int>(n * sizeof(VTFTickRow));
        int dsize = cd.lz4_decomp(comp_buf.data(), reinterpret_cast<char*>(out_rows + idx), static_cast<int>(csize), raw_bytes);
        if (dsize != raw_bytes) { vtf::os_close(fd); return -6; }
        idx += n;
    }
    vtf::os_close(fd);
    return static_cast<int>(idx);
}

} // extern "C"
