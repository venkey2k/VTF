#pragma once
/**
 * VTF v1.0 (Venkey Tick Format) — C/C++ Public Header (`vtf.hpp`)
 * High-frequency option market data engine for NSE/BSE equity derivatives.
 */
#include <cstdint>
#include <cstddef>

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(VTF_BUILDING_DLL)
    #define VTF_API __declspec(dllexport)
  #elif defined(VTF_STATIC)
    #define VTF_API
  #else
    #define VTF_API __declspec(dllimport)
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define VTF_API __attribute__((visibility("default")))
  #else
    #define VTF_API
  #endif
#endif

extern "C" {

enum VTFCodec : uint8_t {
    VTF_CODEC_NONE        = 0,
    VTF_CODEC_LZ4_FAST    = 1, // Default: Delta + SIMD BSS + LZ4-Fast (accel=4)
    VTF_CODEC_LZ4_DEFAULT = 2,
    VTF_CODEC_ZSTD        = 3,

    CODEC_NONE            = 0,
    CODEC_LZ4_FAST        = 1,
    CODEC_LZ4_DEFAULT     = 2,
    CODEC_ZSTD            = 3
};

#pragma pack(push, 1)
struct VTFTickRow {
    int64_t  timestamp_ns;
    uint32_t instrument_id;
    int32_t  ltp_paise;
    uint32_t ltq;
    uint32_t volume;
    uint32_t oi;
    int32_t  oi_change;
    int32_t  bid_price_paise[5];
    int32_t  ask_price_paise[5];
    uint32_t bid_qty[5];
    uint32_t ask_qty[5];
    uint16_t bid_orders[5];
    uint16_t ask_orders[5];
    int32_t  iv_bps;
    int16_t  delta_bps;
    int16_t  reserved_greek;
    int32_t  gamma_scaled;
    int32_t  theta_scaled;
    int32_t  vega_scaled;
    uint8_t  flags;
    uint8_t  exchange_id;
    uint16_t reserved_pad;
};

struct VTFFileHeader {
    uint32_t magic;
    uint8_t  version_major;
    uint8_t  version_minor;
    uint8_t  codec_id;
    uint8_t  flags;
    uint32_t block_target_rows;
    uint64_t created_at_ns;
    uint32_t price_scale;
    uint32_t iv_scale;
    uint32_t greek_scale;
    uint32_t gamma_scale;
    uint8_t  reserved[24];
    uint32_t header_crc32c;
};

struct VTFBlockIndexEntry {
    uint64_t block_file_offset;
    uint32_t compressed_bytes;
    uint32_t row_count;
    int64_t  min_timestamp_ns;
    int64_t  max_timestamp_ns;
    uint64_t inst_bloom_filter;
};

struct VTFRecoveryReport {
    int32_t  status;
    uint32_t valid_blocks;
    uint64_t valid_rows;
    uint64_t original_file_size;
    uint64_t recovered_file_size;
    uint64_t truncated_bytes;
    uint32_t had_valid_footer;
};
#pragma pack(pop)

VTF_API void*    vtf_writer_create();
VTF_API int      vtf_writer_open(void* w, const char* path, uint8_t codec_id, uint32_t block_target_rows, uint64_t created_at_ns);
VTF_API int      vtf_writer_append(void* w, const VTFTickRow* rows, size_t count);
VTF_API int      vtf_writer_flush(void* w);
VTF_API int      vtf_writer_close(void* w);
VTF_API void     vtf_writer_abort_without_footer(void* w);
VTF_API void     vtf_writer_destroy(void* w);

VTF_API void*    vtf_reader_create();
VTF_API int      vtf_reader_open(void* r, const char* path, int auto_recover);
VTF_API uint64_t vtf_reader_total_rows(void* r);
VTF_API uint64_t vtf_reader_total_blocks(void* r);
VTF_API int      vtf_reader_get_header(void* r, VTFFileHeader* out_hdr);
VTF_API int      vtf_reader_get_block_index(void* r, VTFBlockIndexEntry* out_entries, size_t max_entries);
VTF_API int      vtf_reader_read_all(void* r, VTFTickRow* out_rows, size_t max_rows, int verify_crc);
VTF_API int      vtf_reader_read_block(void* r, size_t block_idx, VTFTickRow* out_rows, size_t max_rows, int verify_crc);
VTF_API int64_t  vtf_reader_query_range(void* r, int64_t start_ns, int64_t end_ns, uint32_t filter_instrument, VTFTickRow* out_rows, size_t max_rows);
VTF_API void     vtf_reader_close(void* r);
VTF_API void     vtf_reader_destroy(void* r);

VTF_API int      vtf_recover_file(const char* path, VTFRecoveryReport* out_report);
VTF_API uint32_t vtf_crc32c(const void* data, size_t len);
VTF_API int      vtf_codec_is_supported(int codec_id);

VTF_API int      raw_lz4_binary_write(const char* path, const VTFTickRow* rows, size_t count, size_t block_rows, int accel);
VTF_API int      raw_lz4_binary_read(const char* path, VTFTickRow* out_rows, size_t max_rows);

} // extern "C"
