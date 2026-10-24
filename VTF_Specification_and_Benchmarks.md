# VTF (Venkey Tick Format) v1.0 — Binary Specification, Architecture & Benchmarks

## 1. Executive Summary & Design Objectives

**VTF (Venkey Tick Format) v1.0** is a custom binary market-data storage and streaming format engineered specifically for high-frequency **NSE (NIFTY 50)** and **BSE (SENSEX)** equity index options tick recording. Recording full option chain streams—comprising trade ticks, 5-level Market-By-Price (L2) order books, Open Interest (OI), Implied Volatility (IV), and Black-Scholes Greeks ($\Delta, \Gamma, \Theta, \mathcal{V}$)—poses three simultaneous systems challenges:

1. **Multiplexed Multi-Strike Entropy**: An option recorder ingests interleaved ticks from 100+ active strike/expiry/call-put contracts simultaneously. Naive row-major binary dumps compressed with LZ4 yield poor compression (\~2.4x) because adjacent structs belong to different strikes with unrelated price levels, volumes, and Greeks.

2. **Columnar Footer Vulnerability & Write Latency**: General-purpose columnar formats like Apache Parquet buffer large row groups, incur high CPU overhead for Dremel/dictionary encoding (\~0.5M–0.6M ticks/sec), and write their `FileMetaData` directory strictly at the end of the file. If a capture process suffers a power loss, kernel panic, or `SIGKILL` before `close()`, an unclosed Parquet or Arrow IPC file is unreadable.

3. **Microstructure Redundancy**: On NSE and BSE option feeds, contract prices move in discrete `0.05 INR` (`5 paise`) increments, L2 order book price ladders step uniformly by `5 paise` per depth level, deeper L2 book quantities (`levels 1..4`) remain unchanged across \~80% of top-of-book updates, and Open Interest only updates every few seconds.

**VTF v1.0 resolves all three bottlenecks via a four-stage pipeline**:

* **Self-Framing PAX Micro-Blocks (16,384 rows/block)**: Every block is prefixed by a 64-byte cache-line-aligned `VTFBlockHeader` protected by two independent hardware SSE4.2 `CRC32C` checksums (`block_header_crc32c` and `payload_crc32c`). Blocks are immediately durable upon flush; if a crash occurs mid-write, `vtf_recover_file()` scans forward, truncates any incomplete tail block via `ftruncate()`, and reconstructs the `VTFFileFooter` index in-place with zero loss of previously flushed blocks.

* **Per-Instrument State Cache (512-Slot Zero-Collision Table)**: Within each block, VTF maintains an L1-cache-resident state tracker keyed by `instrument_id`. Prices, cumulative volume, Open Interest, L2 quantities, L2 order counts, IV, and Greeks are delta-encoded against the **previous tick of the same option contract**, collapsing 90%+ of deltas to `0` or $\pm 1$ tick.

* **Relational Order-Book Ladder Encoding**: Instead of storing ten independent 32-bit prices for the 5-level bid/ask book, VTF encodes `bid_price[0]` relative to `ltp`, `ask_price[0]` as the bid-ask spread relative to `bid_price[0]`, and deeper levels `k = 1..4` as positive ladder steps (`bid[k-1] - bid[k]` and `ask[k] - ask[k-1]`).

* **SIMD SSSE3 `pshufb` 4x4 Byte-Matrix Transposition (Byte-Stream-Split)**: After ZigZag encoding signed deltas into unsigned integers, the upper bytes of 16-bit, 32-bit, and 64-bit fields are almost entirely `0x00`. VTF transposes 64-row tiles into 156 contiguous byte planes using 1-cycle SSSE3 `_mm_shuffle_epi8` instructions before applying `LZ4_compress_fast` or `ZSTD_compress`.

## 2. Versioned Wire Specification (`VTF v1.0`)

All multi-byte integers on disk use **little-endian** byte ordering. A VTF v1.0 file consists of four sequential regions:

```
+-----------------------------------------------------------------------+ Offset 0
| 1. VTFFileHeader (64 bytes, 1 cache line, CRC32C protected)           |
+-----------------------------------------------------------------------+ Offset 64
| 2. Self-Framing Data Blocks (Block 0 .. Block B-1)                    |
|    +---------------------------------------------------------------+  |
|    | VTFBlockHeader (64 bytes, magic "VTBK", header & payload CRC) |  |
|    +---------------------------------------------------------------+  |
|    | Compressed SoA Byte-Stream-Split Payload (variable bytes)     |  |
|    +---------------------------------------------------------------+  |
+-----------------------------------------------------------------------+ Offset = index_offset
| 3. Footer Block Index (B * 40 bytes of VTFBlockIndexEntry)            |
+-----------------------------------------------------------------------+ EOF - 32
| 4. VTFFooterTrailer (32 bytes, magic "VTFE", footer_crc32c)           |
+-----------------------------------------------------------------------+ EOF

```

### 2.1 File Header (`VTFFileHeader` — 64 Bytes at Offset `0x00`)

| Byte Offset | Field Name | C++ Type | Default / Value | Description | 
| ----- | ----- | ----- | ----- | ----- | 
| `0..3` | `magic` | `uint32_t` | `0x31465456` (`"VTF1"`) | Format magic identifier | 
| `4` | `version_major` | `uint8_t` | `1` | Major specification version | 
| `5` | `version_minor` | `uint8_t` | `0` | Minor specification version | 
| `6` | `codec_id` | `uint8_t` | `1` (`LZ4_FAST`) | `0`=None, `1`=LZ4-Fast(accel=4), `2`=LZ4-Default, `3`=Zstd(L3) | 
| `7` | `flags` | `uint8_t` | `0x05` | Bitmask: `0x01`=`HAS_FOOTER`, `0x02`=`RECOVERED`, `0x04`=`BYTE_STREAM_SPLIT` | 
| `8..11` | `block_target_rows` | `uint32_t` | `16384` | Target tick rows per PAX micro-block | 
| `12..19` | `created_at_ns` | `uint64_t` | Epoch ns | File creation UTC timestamp in nanoseconds | 
| `20..23` | `price_scale` | `uint32_t` | `100` | Fixed-point divisor for INR prices (`100` = paise, exact for `0.05` ticks) | 
| `24..27` | `iv_scale` | `uint32_t` | `10000` | Fixed-point divisor for Implied Volatility (`10000` = 1 bps) | 
| `28..31` | `greek_scale` | `uint32_t` | `10000` | Fixed-point divisor for Delta, Theta, and Vega (`0.0001` precision) | 
| `32..35` | `gamma_scale` | `uint32_t` | `1000000` | Fixed-point divisor for Gamma (`0.000001` precision) | 
| `36..59` | `reserved` | `uint8_t[24]` | `0x00 * 24` | Zero-filled reserved bytes for future v1.x extensions | 
| `60..63` | `header_crc32c` | `uint32_t` | `CRC32C(0..59)` | Castagnoli CRC32C checksum over the first 60 bytes of the header | 

### 2.2 Logical Tick Schema (`VTFTickRow` — 156 Bytes In-Memory / NumPy `VTF_TICK_DTYPE`)

Each tick row captures the complete trade, L2 5-level Market-By-Price order book, Open Interest, Implied Volatility, and option Greeks in 156 bytes (exactly 39 `uint32_t` words, zero implicit padding):

| Byte Offset | Field Name | C++ / NumPy Type | Fixed-Point Scale | Wire Transform Inside Block | 
| ----- | ----- | ----- | ----- | ----- | 
| `0..7` | `timestamp_ns` | `int64_t` (`<i8`) | `1 ns` | `zigzag64(ts[i] - ts[i-1])` + 8-plane BSS | 
| `8..11` | `instrument_id` | `uint32_t` (`<u4`) | Contract token | Raw token + 4-plane BSS + 64-bit Bloom insert | 
| `12..15` | `ltp_paise` | `int32_t` (`<i4`) | `INR * 100` | `zigzag32(ltp[i] - inst_state.ltp)` + 4-plane BSS | 
| `16..19` | `ltq` | `uint32_t` (`<u4`) | Contracts | Raw unsigned quantity + 4-plane BSS | 
| `20..23` | `volume` | `uint32_t` (`<u4`) | Contracts | `zigzag32(vol[i] - inst_state.vol)` + 4-plane BSS | 
| `24..27` | `oi` | `uint32_t` (`<u4`) | Contracts | `zigzag32(oi[i] - inst_state.oi)` + 4-plane BSS | 
| `28..31` | `oi_change` | `int32_t` (`<i4`) | Contracts | `zigzag32(oichg[i] - inst_state.oichg)` + 4-plane BSS | 
| `32..51` | `bid_price_paise[5]` | `int32_t[5]` (`<i4`) | `INR * 100` | `k=0`: `bid[0]-ltp`; `k=1..4`: `bid[k-1]-bid[k]` (ZigZag + BSS) | 
| `52..71` | `ask_price_paise[5]` | `int32_t[5]` (`<i4`) | `INR * 100` | `k=0`: `ask[0]-bid[0]`; `k=1..4`: `ask[k]-ask[k-1]` (ZigZag + BSS) | 
| `72..91` | `bid_qty[5]` | `uint32_t[5]` (`<u4`) | Contracts | `zigzag32(bid_q[k] - inst_state.bid_q[k])` + 4-plane BSS | 
| `92..111` | `ask_qty[5]` | `uint32_t[5]` (`<u4`) | Contracts | `zigzag32(ask_q[k] - inst_state.ask_q[k])` + 4-plane BSS | 
| `112..121` | `bid_orders[5]` | `uint16_t[5]` (`<u2`) | Order count | `zigzag16(bid_o[k] - inst_state.bid_o[k])` + 2-plane BSS | 
| `122..131` | `ask_orders[5]` | `uint16_t[5]` (`<u2`) | Order count | `zigzag16(ask_o[k] - inst_state.ask_o[k])` + 2-plane BSS | 
| `132..135` | `iv_bps` | `int32_t` (`<i4`) | `IV * 10,000` | `zigzag32(iv[i] - inst_state.iv)` + 4-plane BSS | 
| `136..137` | `delta_bps` | `int16_t` (`<i2`) | `Delta * 10,000` | `zigzag16(delta[i] - inst_state.delta)` + 2-plane BSS | 
| `138..139` | `reserved_greek` | `int16_t` (`<i2`) | Alignment | 2-plane BSS | 
| `140..143` | `gamma_scaled` | `int32_t` (`<i4`) | `Gamma * 1,000,000` | `zigzag32(gamma[i] - inst_state.gamma)` + 4-plane BSS | 
| `144..147` | `theta_scaled` | `int32_t` (`<i4`) | `Theta * 10,000` | `zigzag32(theta[i] - inst_state.theta)` + 4-plane BSS | 
| `148..151` | `vega_scaled` | `int32_t` (`<i4`) | `Vega * 10,000` | `zigzag32(vega[i] - inst_state.vega)` + 4-plane BSS | 
| `152` | `flags` | `uint8_t` (`<u1`) | Bitmask | Contiguous byte plane | 
| `153` | `exchange_id` | `uint8_t` (`<u1`) | `1`=NSE, `2`=BSE | Contiguous byte plane | 
| `154..155` | `reserved_pad` | `uint16_t` (`<u2`) | Alignment | 2-plane BSS | 

### 2.3 Self-Framing Block Header (`VTFBlockHeader` — 64 Bytes)

Every block is completely self-contained (`state_table` resets at the start of each block so blocks can be decompressed randomly in $O(1)$ without reading prior blocks):

| Byte Offset | Field Name | C++ Type | Description | 
| ----- | ----- | ----- | ----- | 
| `0..3` | `block_magic` | `uint32_t` | `0x4B425456` (`"VTBK"`) sync marker | 
| `4..7` | `block_index` | `uint32_t` | 0-based sequential block number | 
| `8..11` | `row_count` | `uint32_t` | Number of ticks stored in this block ($N \le 16384$) | 
| `12..15` | `uncompressed_bytes` | `uint32_t` | Exact uncompressed size ($N \times 156$ bytes) | 
| `16..19` | `compressed_bytes` | `uint32_t` | Size of compressed payload immediately following this header | 
| `20..27` | `min_timestamp_ns` | `int64_t` | Minimum tick timestamp in block (for predicate pushdown) | 
| `28..35` | `max_timestamp_ns` | `int64_t` | Maximum tick timestamp in block (for predicate pushdown) | 
| `36..43` | `inst_bloom_filter` | `uint64_t` | 64-bit Bloom filter (`h1`, `h2` bit positions) of all `instrument_id`s in block | 
| `44..47` | `payload_crc32c` | `uint32_t` | Hardware CRC32C of the `compressed_bytes` payload | 
| `48..59` | `reserved` | `uint8_t[12]` | Zero padding to 64-byte cache line | 
| `60..63` | `block_header_crc32c` | `uint32_t` | Hardware CRC32C of bytes `0..59` of `VTFBlockHeader` | 

### 2.4 Footer Block Index & Trailer (`VTFBlockIndexEntry` & `VTFFooterTrailer`)

When `VTFWriter.close()` is called (or when `vtf_recover_file()` repairs a crashed file), VTF appends an array of 40-byte `VTFBlockIndexEntry` structs (`block_file_offset`, `compressed_bytes`, `row_count`, `min_timestamp_ns`, `max_timestamp_ns`, `inst_bloom_filter`) followed by a 32-byte `VTFFooterTrailer` (`index_offset`, `total_blocks`, `total_rows`, `footer_crc32c`, and magic `"VTFE"`).

## 3. Crash Recovery State Machine (`vtf_recover_file`)

Unlike Parquet or Arrow IPC—where an unclosed file lacks a footer and is completely unreadable—VTF treats the file as an append-only sequence of self-framing, checksummed blocks:

1. **Header Validation**: Verify `magic == "VTF1"` and `header_crc32c` at offset `0`.

2. **Fast-Path Footer Check**: If `flags & FLAG_HAS_FOOTER` is set, read the last 32 bytes (`VTFFooterTrailer`), verify `trailer_magic == "VTFE"`, and validate `footer_crc32c` over the block index array.

3. **Forward Block Scan & Integrity Audit**: Starting at byte offset `64`, inspect each candidate `VTFBlockHeader`:

   * Verify `block_magic == "VTBK"` and `block_header_crc32c`.

   * Verify `uncompressed_bytes == row_count * 156` and check that `offset + 64 + compressed_bytes <= file_size`.

   * Read the compressed payload and verify `crc32c(payload) == payload_crc32c`.

   * If any check fails (due to a power loss mid-`write()`, torn page, or tail bit-rot), stop the scan at the end of the last verified block (`last_valid_offset`).

4. **In-Place Truncation & Footer Reconstruction**:

   * Execute `ftruncate(fd, last_valid_offset)` to slice off the incomplete/corrupted tail bytes.

   * Write the reconstructed `VTFBlockIndexEntry` array and `VTFFooterTrailer` at `last_valid_offset`.

   * Update `VTFFileHeader.flags |= (FLAG_HAS_FOOTER | FLAG_RECOVERED)`, recompute `header_crc32c`, and `fsync(fd)`.

### Verified Crash Recovery Test Results (`benchmark_vtf.py`)

| Failure Mode Simulated | Flushed Blocks Written | Corrupt / Incomplete Tail Injected | Valid Blocks Recovered | Valid Rows Recovered | Tail Bytes Truncated | Post-Recovery Bit-Exact Match | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Scenario A: `SIGKILL` / Missing Footer** | 3 blocks (49,152 rows) | Footer omitted (`abort_without_footer()`) | **3 / 3** | **49,152 / 49,152** | `0 B` | **100% Exact** | 
| **Scenario B: Mid-Block Power Loss** | 4 blocks (65,536 rows) | Last `6,500 B` of Block #3 sliced off | **3 / 3** | **49,152 / 49,152** | `292,488 B` (Block #3) | **100% Exact** | 
| **Scenario C: Tail Payload Bit-Rot** | 4 blocks (65,536 rows) | Byte `EOF - 500` XORed with `0xFF` | **3 / 3** | **49,152 / 49,152** | `298,988 B` (Block #3) | **100% Exact** | 

## 4. Reproducible Benchmark Results (500,000 NIFTY/SENSEX Option Ticks)

### 4.1 Dataset & Environment Configuration

* **Workload**: `500,000` high-frequency option ticks across `128` contracts (`64` NSE NIFTY 50 strikes + `64` BSE SENSEX strikes, Zipf liquidity distribution, `0.05 INR` tick steps, 5-level L2 depth, OI, IV, and 4 Greeks).

* **Uncompressed Size**: `500,000 rows * 156 bytes/row = 78,000,000 bytes` (`74.39 MiB`).

* **Compiler & Runtime**: `g++ 12.2.0 (-O3 -march=native -funroll-loops -std=c++17)`, `liblz4 1.9.4`, `libzstd 1.5.4`, `PyArrow`, `Python 3.11.2`.

* **Methodology**: Warmup pass followed by 5 timed trials per format; median values reported below.

### 4.2 End-to-End Performance & Compression Comparison

| Format / Codec Configuration | File Size (MiB) | Bytes / Tick | Compression Ratio | Write Time (ms) | Write Speed (M Ticks/s) | Effective Write (MiB/s) | Read Time (ms) | Read Speed (M Ticks/s) | Crash Recovery | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **VTF v1.0 (`LZ4-Fast`, accel=4)** | **8.496** | **17.82 B** | **8.76x** | **141.23** | **3.54 MT/s** | **526.7 MiB/s** | **169.38** | **2.95 MT/s** | **Yes (Block CRC32C)** | 
| **VTF v1.0 (`LZ4-Default`)** | **8.354** | **17.52 B** | **8.90x** | **157.86** | **3.17 MT/s** | **471.2 MiB/s** | **179.46** | **2.79 MT/s** | **Yes (Block CRC32C)** | 
| **VTF v1.0 (`Zstd-L3`)** | **5.067** | **10.63 B** | **14.68x** | **233.21** | **2.14 MT/s** | **319.0 MiB/s** | **200.39** | **2.50 MT/s** | **Yes (Block CRC32C)** | 
| Raw Binary + LZ4 (`Fast`, C++) | 30.627 | 64.23 B | 2.43x | 226.35 | 2.21 MT/s | 328.6 MiB/s | 131.27 | 3.81 MT/s | Partial (No CRC/Index) | 
| Raw Binary + LZ4 (`Default`, C++) | 28.824 | 60.45 B | 2.58x | 254.29 | 1.97 MT/s | 292.5 MiB/s | 126.51 | 3.95 MT/s | Partial (No CRC/Index) | 
| Arrow IPC / Feather v2 (`Uncompressed`) | 72.550 | 152.15 B | 1.03x | 105.73 | 4.73 MT/s | 703.6 MiB/s | 31.64 | 15.80 MT/s | No (Requires Footer) | 
| Arrow IPC / Feather v2 (`LZ4_FRAME`) | 51.027 | 107.01 B | 1.46x | 275.40 | 1.82 MT/s | 270.1 MiB/s | 147.38 | 3.39 MT/s | No (Requires Footer) | 
| Apache Parquet (`Snappy`) | 27.127 | 56.89 B | 2.74x | 789.18 | 0.63 MT/s | 94.3 MiB/s | 161.32 | 3.10 MT/s | No (Requires Footer) | 
| Apache Parquet (`Zstd-L3`) | 23.945 | 50.22 B | 3.11x | 946.12 | 0.53 MT/s | 78.6 MiB/s | 178.46 | 2.80 MT/s | No (Requires Footer) | 

### 4.3 Key Architectural Findings

1. **VTF (`LZ4-Fast`) is both 1.60x faster to write and 3.60x smaller than Raw Binary + LZ4**:
   Because VTF's SIMD SSSE3 4x4 byte-matrix transposition groups the upper zero bytes of per-instrument deltas into contiguous `0x00` byte planes, `LZ4_compress_fast` matches long zero runs in a fraction of the instructions required to hash high-entropy interleaved struct bytes, and writes `8.496 MiB` to disk instead of `30.627 MiB`.

2. **VTF (`LZ4-Fast`) is 5.59x faster to write and 3.19x smaller than Parquet (`Snappy`)**:
   Parquet encodes each column independently without cross-row per-instrument state tracking; when 128 option contracts are interleaved chronologically, Parquet's column deltas jump wildly between different strikes. VTF's 512-slot per-instrument state cache eliminates cross-strike jumps prior to byte-plane splitting.

3. **VTF (`Zstd-L3`) achieves 14.68x compression (`10.63 bytes/tick`, `5.067 MiB`)**:
   For cold archival storage, switching the block codec flag from `CODEC_LZ4_FAST` (`1`) to `CODEC_ZSTD` (`3`) shrinks a 156-byte L2 + Greeks option tick to `10.63 bytes` while still writing **4.06x faster** than Parquet (`Zstd-L3`).

## 5. Usage & Reproduction Guide

### 5.1 Compiling the C++17 Core Library

```
g++ -O3 -march=native -funroll-loops -std=c++17 -shared -fPIC \
    vtf_core.cpp -o libvtf.so \
    /usr/lib/x86_64-linux-gnu/liblz4.so.1 \
    /usr/lib/x86_64-linux-gnu/libzstd.so.1

```

### 5.2 Python API Quickstart (`vtf.py`)

```
import numpy as np
import vtf

# 1. Allocate structured NumPy array (156 bytes/row)
ticks = np.zeros(100_000, dtype=vtf.VTF_TICK_DTYPE)
ticks["timestamp_ns"] = 1_792_554_300_000_000_000 + np.arange(100_000, dtype=np.int64) * 5_000
ticks["instrument_id"] = 1001  # NIFTY 24500 CE
ticks["ltp_paise"] = 18505     # 185.05 INR

# 2. Write with VTFWriter (LZ4-Fast, 16,384 rows per block)
with vtf.VTFWriter("nifty_ticks.vtf", codec=vtf.CODEC_LZ4_FAST, block_rows=16384) as writer:
    writer.append(ticks)

# 3. Read all rows or run a predicate-pushdown time-range + instrument query
with vtf.VTFReader("nifty_ticks.vtf") as reader:
    print(reader.header_info())
    all_rows = reader.read_all(verify_crc=True)
    subset = reader.query(
        start_ns=int(ticks["timestamp_ns"][10_000]),
        end_ns=int(ticks["timestamp_ns"][20_000]),
        instrument_id=1001,
    )

# 4. Recover an unclosed or truncated file in-place after a crash
report = vtf.recover_vtf_file("nifty_ticks.vtf")
print(report)

```

### 5.3 Running the Reproducible Benchmark & Crash-Test Suite

```
python3 benchmark_vtf.py

```