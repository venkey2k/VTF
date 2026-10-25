# VTF (Venkey Tick Format) v1.0

[![CI](https://github.com/venkey2k/VTF/actions/workflows/ci.yml/badge.svg)](https://github.com/venkey2k/VTF/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/nuro-vtf.svg)](https://pypi.org/project/nuro-vtf/)
[![Python versions](https://img.shields.io/pypi/pyversions/nuro-vtf.svg)](https://pypi.org/project/nuro-vtf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VTF (Venkey Tick Format)** is a custom binary market-data storage format and zero-copy streaming engine engineered specifically for high-frequency **NSE (NIFTY 50)** and **BSE (SENSEX)** equity index options.

It captures trades, 5-level Market-By-Price (L2) order books, Open Interest (OI), Implied Volatility (IV), and Black-Scholes Greeks ($\Delta, \Gamma, \Theta, \mathcal{V}$) in a 156-byte compact binary tick representation, striking an optimal balance between **ultra-fast write throughput**, **rapid random query speeds**, and **extreme compression**.

---

## ⚡ Performance Highlights (500,000 Option Ticks Benchmark)

| Format / Configuration | File Size (MiB) | Bytes / Tick | Compression Ratio | Write Speed (MT/s) | Effective Write (MiB/s) | Read Speed (MT/s) | Mid-Write Crash Resilience |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VTF v1.0 (`LZ4-Fast`, accel=4)** | **8.496** | **17.82 B** | **8.76x** | **3.54 MT/s** | **526.7 MiB/s** | **2.95 MT/s** | **Yes (Block CRC32C)** |
| **VTF v1.0 (`Zstd-L3`)** | **5.067** | **10.63 B** | **14.68x** | **2.14 MT/s** | **319.0 MiB/s** | **2.50 MT/s** | **Yes (Block CRC32C)** |
| Apache Parquet (`Snappy`) | 27.127 | 56.89 B | 2.74x | 0.63 MT/s | 94.3 MiB/s | 3.10 MT/s | **No** (Unreadable if unclosed) |
| Apache Parquet (`Zstd-L3`) | 23.945 | 50.22 B | 3.11x | 0.53 MT/s | 78.6 MiB/s | 2.80 MT/s | **No** (Unreadable if unclosed) |
| Arrow IPC / Feather (`LZ4`) | 51.027 | 107.01 B | 1.46x | 1.82 MT/s | 270.1 MiB/s | 3.39 MT/s | **No** (Unreadable if unclosed) |
| Raw Structs + LZ4 | 30.627 | 64.23 B | 2.43x | 2.21 MT/s | 328.6 MiB/s | 3.81 MT/s | Partial (No index/CRC) |

- **5.59x faster writes** and **3.19x smaller files** than Apache Parquet (`Snappy`).
- **100% crash recovery**: Zero loss of flushed blocks if process is terminated mid-write (`SIGKILL`, power failure).
- **Direct NumPy zero-copy integration**: Structured array `VTF_TICK_DTYPE` matches C++ memory layout with 0 serialization copies.

---

## 🚀 Installation

Install pre-built binary wheels directly from PyPI:

```bash
pip install nuro-vtf
```

Or install from source with full CPU-native optimizations:

```bash
git clone https://github.com/venkey2k/VTF.git
cd VTF
pip install -v .
```

*Compatible with Python 3.9–3.14 on Linux (x86_64, aarch64), Windows (x64), and macOS (Apple Silicon & Intel).*

---

## 💡 Quickstart (Python API)

```python
import numpy as np
import vtf

# 1. Allocate structured NumPy buffer matching the 156-byte tick schema
num_ticks = 32_768
ticks = np.zeros(num_ticks, dtype=vtf.VTF_TICK_DTYPE)

ticks["timestamp_ns"] = 1_792_554_300_000_000_000 + np.arange(num_ticks, dtype=np.int64) * 10_000
ticks["instrument_id"] = 1001  # NIFTY 24500 CE token
ticks["exchange_id"] = vtf.EXCHANGE_NSE_NIFTY
ticks["ltp_paise"] = 18505     # 185.05 INR
ticks["ltq"] = 75
ticks["volume"] = 150_000
ticks["oi"] = 1_250_000
ticks["iv_bps"] = 1645         # 16.45% IV
ticks["delta_bps"] = 5120      # +0.5120 Delta

# 2. Write in self-framing PAX micro-blocks (16,384 ticks/block)
with vtf.VTFWriter("nifty_options.vtf", codec=vtf.CODEC_LZ4_FAST) as writer:
    writer.append(ticks)

# 3. Read back or run pushdown range queries
with vtf.VTFReader("nifty_options.vtf") as reader:
    print(reader.header_info())
    
    # Read full stream
    all_ticks = reader.read_all(verify_crc=True)
    
    # Filter by time range and contract token using block-level Bloom filters
    subset = reader.query(
        start_ns=int(ticks["timestamp_ns"][1000]),
        end_ns=int(ticks["timestamp_ns"][5000]),
        instrument_id=1001,
    )
    print(f"Retrieved {len(subset):,} filtered ticks in microseconds.")

# 4. Crash Recovery (in-place repair for unfinalized files)
report = vtf.recover_vtf_file("nifty_options.vtf")
print(f"Valid blocks preserved: {report.valid_blocks}, rows recovered: {report.valid_rows:,}")
```

---

## 🛠️ The VTF Encoding Pipeline

```
Raw Multiplexed Option Ticks (156 B/row, interleaved strikes)
                    │
                    ▼
Stage 1: Per-Instrument State Cache (512-slot hash table)
         Delta-encoded against previous tick of SAME contract
                    │
                    ▼
Stage 2: Relational Order Book Ladder Encoding
         Bid[0] vs LTP, Ask[0] spread, L2-L5 depth step diffs
                    │
                    ▼
Stage 3: ZigZag Variable-Length Delta Transformation
         Maps small positive/negative jumps to small unsigned integers
                    │
                    ▼
Stage 4: SIMD SSSE3 4x4 Byte-Stream-Split (BSS) Transposition
         Clusters upper 0x00 bytes into contiguous byte planes
                    │
                    ▼
Stage 5: High-Speed Block Codec (LZ4-Fast accel=4 / Zstd)
         Instant hardware CRC32C verification on flush
```

---

## 📦 Repository Structure

```
VTF/
├── vtf.py                              # Python API & NumPy ctypes wrapper
├── vtf.hpp                             # C/C++ public API header
├── vtf_core.cpp                        # Cross-platform zero-dependency C++17 engine
├── benchmark_vtf.py                    # 500k tick benchmark suite against Parquet & Arrow
├── example_quickstart.py               # Ready-to-run quickstart example
├── VTF_Specification_and_Benchmarks.md # Complete wire format specification
├── setup.py                            # C++ extension setuptools builder
├── pyproject.toml                      # PEP 517 build & cibuildwheel config
├── MANIFEST.in                         # sdist source manifest (header + core + tools)
├── LICENSE                             # MIT License
└── .github/workflows/
    ├── ci.yml                          # Multi-OS CI testing (Linux, Windows, macOS)
    └── release.yml                     # Automated multi-arch wheel build & PyPI publish
```

---

## 🔬 Benchmark Verification

To run the complete benchmark suite comparing VTF with Parquet and Arrow:

```bash
pip install numpy pyarrow
python benchmark_vtf.py
```

---

## 📄 License

VTF is released under the [MIT License](LICENSE). Copyright (c) 2026 venkey2k.
