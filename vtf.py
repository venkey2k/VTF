"""
VTF (Venkey Tick Format) v1.0 — Python API & Zero-Copy NumPy Interface
===========================================================================
High-frequency option tick + 5-level L2 Order Book + Open Interest + IV + Greeks
storage format engineered for NSE (NIFTY 50) and BSE (SENSEX) derivatives.

Wraps the C++17 core engine (`vtf_core.cpp` / `libvtf.so` / `vtf.dll`) via zero-copy
`ctypes` and structured NumPy arrays (`VTF_TICK_DTYPE`, 156 bytes/row).
"""

from __future__ import annotations

import ctypes
import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# ============================================================================
# Constants & Enums
# ============================================================================

CODEC_NONE: int = 0
CODEC_LZ4_FAST: int = 1
CODEC_LZ4_DEFAULT: int = 2
CODEC_ZSTD: int = 3

CODEC_NAMES: Dict[int, str] = {
    CODEC_NONE: "NONE (Raw SoA + BSS)",
    CODEC_LZ4_FAST: "LZ4_FAST (Delta + BSS + LZ4-Accel-4)",
    CODEC_LZ4_DEFAULT: "LZ4_DEFAULT (Delta + BSS + LZ4-Default)",
    CODEC_ZSTD: "ZSTD (Delta + BSS + Zstd-L3)",
}

FLAG_HAS_FOOTER: int = 0x01
FLAG_RECOVERED: int = 0x02
FLAG_BYTE_STREAM_SPLIT: int = 0x04

TICK_FLAG_TRADE: int = 0x01
TICK_FLAG_BUY_AGGRESSOR: int = 0x02
TICK_FLAG_SELL_AGGRESSOR: int = 0x04
TICK_FLAG_L2_UPDATE: int = 0x08
TICK_FLAG_OI_UPDATE: int = 0x10
TICK_FLAG_GREEKS_VALID: int = 0x20

EXCHANGE_NSE_NIFTY: int = 1
EXCHANGE_BSE_SENSEX: int = 2

# Fixed-point scaling factors
PRICE_SCALE: int = 100        # 0.01 INR (paise)
IV_SCALE: int = 10_000        # 0.0001 (1 bps of implied vol)
GREEK_SCALE: int = 10_000     # 0.0001 for Delta, Theta, Vega
GAMMA_SCALE: int = 1_000_000  # 0.000001 for Gamma

# ============================================================================
# Exact 156-Byte NumPy Structured Dtype Matching C++ `vtf::VTFTickRow`
# ============================================================================

VTF_TICK_DTYPE = np.dtype(
    [
        ("timestamp_ns", "<i8"),           # 0..7    : Exchange/NIC epoch timestamp (ns)
        ("instrument_id", "<u4"),          # 8..11   : Option contract token
        ("ltp_paise", "<i4"),              # 12..15  : Last Traded Price * 100
        ("ltq", "<u4"),                    # 16..19  : Last Traded Quantity
        ("volume", "<u4"),                 # 20..23  : Cumulative daily volume
        ("oi", "<u4"),                     # 24..27  : Open Interest
        ("oi_change", "<i4"),              # 28..31  : Day Open Interest change
        ("bid_price_paise", "<i4", (5,)),  # 32..51  : 5-level L2 Bid Prices (paise)
        ("ask_price_paise", "<i4", (5,)),  # 52..71  : 5-level L2 Ask Prices (paise)
        ("bid_qty", "<u4", (5,)),          # 72..91  : 5-level L2 Bid Quantities
        ("ask_qty", "<u4", (5,)),          # 92..111 : 5-level L2 Ask Quantities
        ("bid_orders", "<u2", (5,)),       # 112..121: 5-level L2 Bid Order Counts
        ("ask_orders", "<u2", (5,)),       # 122..131: 5-level L2 Ask Order Counts
        ("iv_bps", "<i4"),                 # 132..135: Implied Volatility * 10,000
        ("delta_bps", "<i2"),              # 136..137: Option Delta * 10,000
        ("reserved_greek", "<i2"),         # 138..139: Alignment padding
        ("gamma_scaled", "<i4"),           # 140..143: Option Gamma * 1,000,000
        ("theta_scaled", "<i4"),           # 144..147: Option Theta * 10,000
        ("vega_scaled", "<i4"),            # 148..151: Option Vega * 10,000
        ("flags", "<u1"),                  # 152     : VTFTickFlags bitmask
        ("exchange_id", "<u1"),            # 153     : 1=NSE (NIFTY), 2=BSE (SENSEX)
        ("reserved_pad", "<u2"),           # 154..155: Padding to 156 bytes
    ],
    align=False,
)
assert VTF_TICK_DTYPE.itemsize == 156, f"Expected 156 bytes, got {VTF_TICK_DTYPE.itemsize}"


# ============================================================================
# ctypes Struct Definitions Matching C++ Wire Headers
# ============================================================================

class CFileHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version_major", ctypes.c_uint8),
        ("version_minor", ctypes.c_uint8),
        ("codec_id", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("block_target_rows", ctypes.c_uint32),
        ("created_at_ns", ctypes.c_uint64),
        ("price_scale", ctypes.c_uint32),
        ("iv_scale", ctypes.c_uint32),
        ("greek_scale", ctypes.c_uint32),
        ("gamma_scale", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 24),
        ("header_crc32c", ctypes.c_uint32),
    ]


assert ctypes.sizeof(CFileHeader) == 64


class CBlockIndexEntry(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("block_file_offset", ctypes.c_uint64),
        ("compressed_bytes", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("min_timestamp_ns", ctypes.c_int64),
        ("max_timestamp_ns", ctypes.c_int64),
        ("inst_bloom_filter", ctypes.c_uint64),
    ]


assert ctypes.sizeof(CBlockIndexEntry) == 40


class CRecoveryReport(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("status", ctypes.c_int32),
        ("valid_blocks", ctypes.c_uint32),
        ("valid_rows", ctypes.c_uint64),
        ("original_file_size", ctypes.c_uint64),
        ("recovered_file_size", ctypes.c_uint64),
        ("truncated_bytes", ctypes.c_uint64),
        ("had_valid_footer", ctypes.c_uint32),
    ]


@dataclasses.dataclass
class VTFHeaderInfo:
    magic: str
    version: Tuple[int, int]
    codec_id: int
    codec_name: str
    flags: int
    has_footer: bool
    was_recovered: bool
    byte_stream_split: bool
    block_target_rows: int
    created_at_ns: int
    price_scale: int
    iv_scale: int
    greek_scale: int
    gamma_scale: int
    header_crc32c: int


@dataclasses.dataclass
class VTFBlockMeta:
    block_index: int
    block_file_offset: int
    compressed_bytes: int
    uncompressed_bytes: int
    row_count: int
    compression_ratio: float
    bytes_per_row: float
    min_timestamp_ns: int
    max_timestamp_ns: int
    inst_bloom_filter: int


@dataclasses.dataclass
class RecoveryReport:
    status: int
    valid_blocks: int
    valid_rows: int
    original_file_size: int
    recovered_file_size: int
    truncated_bytes: int
    had_valid_footer: bool


# ============================================================================
# Shared Library Loader & Auto-Builder
# ============================================================================

_LIBVTF: Optional[ctypes.CDLL] = None


def _load_libvtf() -> ctypes.CDLL:
    global _LIBVTF
    if _LIBVTF is not None:
        return _LIBVTF

    here = Path(__file__).resolve().parent
    temp_dir = Path(tempfile.gettempdir())
    is_win = sys.platform.startswith("win")
    is_mac = sys.platform == "darwin"

    # Search candidates across library extensions (.dll, .so, .dylib, .pyd)
    candidate_paths: List[Path] = []

    # 1. Search for compiled extensions in current and parent directories
    for pattern in ["*_libvtf*", "*vtf_core*", "libvtf*", "vtf*"]:
        for p in here.glob(f"{pattern}*"):
            if p.suffix in [".so", ".pyd", ".dylib", ".dll"]:
                candidate_paths.append(p)
        for p in here.parent.glob(f"{pattern}*"):
            if p.suffix in [".so", ".pyd", ".dylib", ".dll"]:
                candidate_paths.append(p)

    candidate_names = [
        "vtf.dll" if is_win else "libvtf.so",
        "libvtf.dll",
        "libvtf.so",
        "libvtf.dylib",
        "_libvtf.so",
        "_libvtf.pyd",
    ]
    for name in candidate_names:
        candidate_paths.append(here / name)
        candidate_paths.append(here.parent / name)
        candidate_paths.append(temp_dir / name)

    so_path: Optional[Path] = None
    for p in candidate_paths:
        if p.exists():
            try:
                # Test loading the library
                test_lib = ctypes.CDLL(str(p))
                if hasattr(test_lib, "vtf_writer_create"):
                    so_path = p
                    break
            except Exception:
                continue

    if so_path is None:
        cpp_src = here / "vtf_core.cpp"
        if not cpp_src.exists():
            raise FileNotFoundError(f"Cannot locate vtf_core.cpp at {cpp_src}")

        target_ext = ".dll" if is_win else (".dylib" if is_mac else ".so")
        target_lib = temp_dir / f"vtf{target_ext}"

        # Detect available compilers
        if is_win:
            if shutil.which("cl.exe"):
                cmd = ["cl.exe", "/O2", "/std:c++17", "/EHsc", "/LD", str(cpp_src), f"/Fe:{target_lib}"]
            elif shutil.which("g++"):
                cmd = ["g++", "-O3", "-std=c++17", "-shared", str(cpp_src), "-o", str(target_lib)]
            else:
                raise RuntimeError(
                    "VTF pre-compiled library not found and no C++ compiler (MSVC 'cl.exe' or 'g++') detected on PATH. "
                    "Please install a binary wheel via `pip install nuro-vtf` or install Build Tools."
                )
        elif is_mac:
            cmd = [
                "clang++", "-O3", "-std=c++17", "-shared",
                "-fPIC", str(cpp_src), "-o", str(target_lib)
            ]
        else:
            cmd = [
                "g++", "-O3", "-march=native", "-funroll-loops", "-std=c++17",
                "-shared", "-fPIC", str(cpp_src), "-o", str(target_lib), "-ldl"
            ]

        subprocess.run(cmd, check=True)
        so_path = target_lib

    lib = ctypes.CDLL(str(so_path))

    # Bind C functions
    lib.vtf_writer_create.restype = ctypes.c_void_p
    lib.vtf_writer_open.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint8,
        ctypes.c_uint32,
        ctypes.c_uint64,
    ]
    lib.vtf_writer_open.restype = ctypes.c_int

    lib.vtf_writer_append.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.vtf_writer_append.restype = ctypes.c_int

    lib.vtf_writer_flush.argtypes = [ctypes.c_void_p]
    lib.vtf_writer_flush.restype = ctypes.c_int

    lib.vtf_writer_close.argtypes = [ctypes.c_void_p]
    lib.vtf_writer_close.restype = ctypes.c_int

    lib.vtf_writer_abort_without_footer.argtypes = [ctypes.c_void_p]
    lib.vtf_writer_abort_without_footer.restype = None

    lib.vtf_writer_destroy.argtypes = [ctypes.c_void_p]
    lib.vtf_writer_destroy.restype = None

    lib.vtf_reader_create.restype = ctypes.c_void_p
    lib.vtf_reader_open.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.vtf_reader_open.restype = ctypes.c_int

    lib.vtf_reader_total_rows.argtypes = [ctypes.c_void_p]
    lib.vtf_reader_total_rows.restype = ctypes.c_uint64

    lib.vtf_reader_total_blocks.argtypes = [ctypes.c_void_p]
    lib.vtf_reader_total_blocks.restype = ctypes.c_uint64

    lib.vtf_reader_get_header.argtypes = [ctypes.c_void_p, ctypes.POINTER(CFileHeader)]
    lib.vtf_reader_get_header.restype = ctypes.c_int

    lib.vtf_reader_get_block_index.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(CBlockIndexEntry),
        ctypes.c_size_t,
    ]
    lib.vtf_reader_get_block_index.restype = ctypes.c_int

    lib.vtf_reader_read_all.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.vtf_reader_read_all.restype = ctypes.c_int

    lib.vtf_reader_read_block.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.vtf_reader_read_block.restype = ctypes.c_int

    lib.vtf_reader_query_range.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.vtf_reader_query_range.restype = ctypes.c_int64

    lib.vtf_reader_close.argtypes = [ctypes.c_void_p]
    lib.vtf_reader_close.restype = None

    lib.vtf_reader_destroy.argtypes = [ctypes.c_void_p]
    lib.vtf_reader_destroy.restype = None

    lib.vtf_recover_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(CRecoveryReport)]
    lib.vtf_recover_file.restype = ctypes.c_int

    lib.vtf_crc32c.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.vtf_crc32c.restype = ctypes.c_uint32

    lib.raw_lz4_binary_write.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.raw_lz4_binary_write.restype = ctypes.c_int

    lib.raw_lz4_binary_read.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.raw_lz4_binary_read.restype = ctypes.c_int

    _LIBVTF = lib
    return lib


# ============================================================================
# VTFWriter — High-Speed Streaming & Batch Writer
# ============================================================================

class VTFWriter:
    """
    Streaming and batch writer for VTF v1.0 market data files.

    Parameters
    ----------
    path : str or Path
        Target file path.
    codec : int
        Compression codec (`CODEC_LZ4_FAST`, `CODEC_LZ4_DEFAULT`, `CODEC_ZSTD`, `CODEC_NONE`).
    block_rows : int
        Target rows per self-framing PAX micro-block (default: 16,384).
    """

    def __init__(
        self,
        path: Union[str, Path],
        codec: int = CODEC_LZ4_FAST,
        block_rows: int = 16384,
        created_at_ns: Optional[int] = None,
    ) -> None:
        self._lib = _load_libvtf()
        self.path = str(path)
        self.codec = codec
        self.block_rows = block_rows
        self._handle = self._lib.vtf_writer_create()
        if not self._handle:
            raise MemoryError("Failed to allocate C++ VTFWriter")

        ts_ns = created_at_ns if created_at_ns is not None else time.time_ns()
        rc = self._lib.vtf_writer_open(
            self._handle,
            self.path.encode("utf-8"),
            ctypes.c_uint8(codec),
            ctypes.c_uint32(block_rows),
            ctypes.c_uint64(ts_ns),
        )
        if rc != 0:
            self._lib.vtf_writer_destroy(self._handle)
            self._handle = None
            raise IOError(f"Failed to open VTF file '{self.path}' for writing (rc={rc})")

    def append(self, ticks: np.ndarray) -> None:
        """
        Append a 1D NumPy structured array of `VTF_TICK_DTYPE` rows with zero copy.
        """
        if not self._handle:
            raise RuntimeError("VTFWriter is already closed")
        if ticks.dtype != VTF_TICK_DTYPE:
            ticks = np.asarray(ticks, dtype=VTF_TICK_DTYPE)
        if not ticks.flags["C_CONTIGUOUS"]:
            ticks = np.ascontiguousarray(ticks)
        count = len(ticks)
        if count == 0:
            return
        rc = self._lib.vtf_writer_append(
            self._handle,
            ticks.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(count),
        )
        if rc != 0:
            raise IOError(f"VTFWriter.append failed with error code {rc}")

    def flush(self) -> None:
        """Flush any buffered rows into a self-framing, CRC32C-protected block on disk."""
        if not self._handle:
            raise RuntimeError("VTFWriter is already closed")
        rc = self._lib.vtf_writer_flush(self._handle)
        if rc != 0:
            raise IOError(f"VTFWriter.flush failed with error code {rc}")

    def close(self) -> None:
        """Flush remaining rows, write the Footer Index & Trailer, and update the header flag."""
        if self._handle:
            rc = self._lib.vtf_writer_close(self._handle)
            self._lib.vtf_writer_destroy(self._handle)
            self._handle = None
            if rc != 0:
                raise IOError(f"VTFWriter.close failed with error code {rc}")

    def abort_without_footer(self) -> None:
        """
        Simulate an abrupt process crash (`SIGKILL` / power loss) by closing the OS file
        descriptor immediately after flushed blocks WITHOUT writing the footer index.
        """
        if self._handle:
            self._lib.vtf_writer_abort_without_footer(self._handle)
            self._lib.vtf_writer_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "VTFWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_handle", None):
            try:
                self.close()
            except Exception:
                pass


# ============================================================================
# VTFReader — Zero-Copy Reader & Predicate Pushdown Engine
# ============================================================================

class VTFReader:
    """
    Reader for VTF v1.0 files with automatic crash recovery, CRC32C verification,
    and O(log B) time-range + 64-bit Bloom filter predicate pushdown.
    """

    def __init__(self, path: Union[str, Path], auto_recover: bool = True) -> None:
        self._lib = _load_libvtf()
        self.path = str(path)
        self._handle = self._lib.vtf_reader_create()
        if not self._handle:
            raise MemoryError("Failed to allocate C++ VTFReader")

        rc = self._lib.vtf_reader_open(
            self._handle,
            self.path.encode("utf-8"),
            1 if auto_recover else 0,
        )
        if rc != 0:
            self._lib.vtf_reader_destroy(self._handle)
            self._handle = None
            raise IOError(f"Failed to open VTF file '{self.path}' for reading (rc={rc})")

    @property
    def total_rows(self) -> int:
        return int(self._lib.vtf_reader_total_rows(self._handle))

    @property
    def total_blocks(self) -> int:
        return int(self._lib.vtf_reader_total_blocks(self._handle))

    def header_info(self) -> VTFHeaderInfo:
        ch = CFileHeader()
        rc = self._lib.vtf_reader_get_header(self._handle, ctypes.byref(ch))
        if rc != 0:
            raise IOError(f"Failed to read VTF header (rc={rc})")
        magic_bytes = int(ch.magic).to_bytes(4, "little")
        return VTFHeaderInfo(
            magic=magic_bytes.decode("ascii", errors="replace"),
            version=(int(ch.version_major), int(ch.version_minor)),
            codec_id=int(ch.codec_id),
            codec_name=CODEC_NAMES.get(int(ch.codec_id), f"UNKNOWN({ch.codec_id})"),
            flags=int(ch.flags),
            has_footer=bool(ch.flags & FLAG_HAS_FOOTER),
            was_recovered=bool(ch.flags & FLAG_RECOVERED),
            byte_stream_split=bool(ch.flags & FLAG_BYTE_STREAM_SPLIT),
            block_target_rows=int(ch.block_target_rows),
            created_at_ns=int(ch.created_at_ns),
            price_scale=int(ch.price_scale),
            iv_scale=int(ch.iv_scale),
            greek_scale=int(ch.greek_scale),
            gamma_scale=int(ch.gamma_scale),
            header_crc32c=int(ch.header_crc32c),
        )

    def block_index(self) -> List[VTFBlockMeta]:
        n = self.total_blocks
        if n == 0:
            return []
        arr = (CBlockIndexEntry * n)()
        got = self._lib.vtf_reader_get_block_index(self._handle, arr, n)
        if got < 0:
            raise IOError(f"Failed to retrieve block index (rc={got})")
        out: List[VTFBlockMeta] = []
        for i in range(got):
            e = arr[i]
            uncomp = int(e.row_count) * VTF_TICK_DTYPE.itemsize
            comp = int(e.compressed_bytes)
            out.append(
                VTFBlockMeta(
                    block_index=i,
                    block_file_offset=int(e.block_file_offset),
                    compressed_bytes=comp,
                    uncompressed_bytes=uncomp,
                    row_count=int(e.row_count),
                    compression_ratio=(uncomp / comp) if comp > 0 else 0.0,
                    bytes_per_row=(comp / int(e.row_count)) if e.row_count > 0 else 0.0,
                    min_timestamp_ns=int(e.min_timestamp_ns),
                    max_timestamp_ns=int(e.max_timestamp_ns),
                    inst_bloom_filter=int(e.inst_bloom_filter),
                )
            )
        return out

    def read_all(self, verify_crc: bool = True) -> np.ndarray:
        """Read all ticks in the file into a structured NumPy array (`VTF_TICK_DTYPE`)."""
        n = self.total_rows
        out = np.empty(n, dtype=VTF_TICK_DTYPE)
        if n == 0:
            return out
        rc = self._lib.vtf_reader_read_all(
            self._handle,
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(n),
            1 if verify_crc else 0,
        )
        if rc < 0:
            raise IOError(f"VTFReader.read_all failed (rc={rc}) — CRC or decompression error")
        return out[:rc]

    def read_block(self, block_idx: int, verify_crc: bool = True) -> np.ndarray:
        """Read a single block by index."""
        idx = self.block_index()
        if block_idx < 0 or block_idx >= len(idx):
            raise IndexError(f"Block index {block_idx} out of range [0, {len(idx)})")
        rows = idx[block_idx].row_count
        out = np.empty(rows, dtype=VTF_TICK_DTYPE)
        rc = self._lib.vtf_reader_read_block(
            self._handle,
            ctypes.c_size_t(block_idx),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(rows),
            1 if verify_crc else 0,
        )
        if rc < 0:
            raise IOError(f"VTFReader.read_block({block_idx}) failed (rc={rc})")
        return out[:rc]

    def query(
        self,
        start_ns: int,
        end_ns: int,
        instrument_id: int = 0,
        max_rows: Optional[int] = None,
    ) -> np.ndarray:
        """
        Execute a time-range and optional single-instrument predicate pushdown query.
        Uses the Footer Block Index min/max timestamps and 64-bit Bloom filter to skip
        non-matching blocks without decompressing them.
        """
        cap = max_rows if max_rows is not None else self.total_rows
        if cap == 0:
            return np.empty(0, dtype=VTF_TICK_DTYPE)
        out = np.empty(cap, dtype=VTF_TICK_DTYPE)
        n = self._lib.vtf_reader_query_range(
            self._handle,
            ctypes.c_int64(start_ns),
            ctypes.c_int64(end_ns),
            ctypes.c_uint32(instrument_id),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(cap),
        )
        if n < 0:
            raise IOError(f"VTFReader.query failed with error code {n}")
        return out[:n].copy()

    def to_float_dict(self, rows: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Convert fixed-point integer fields into human-readable floating-point arrays
        (INR prices, decimal IV and Greeks) for analytics or DataFrame conversion.
        """
        if rows is None:
            rows = self.read_all()
        return {
            "timestamp_ns": rows["timestamp_ns"],
            "instrument_id": rows["instrument_id"],
            "ltp": rows["ltp_paise"].astype(np.float64) / PRICE_SCALE,
            "ltq": rows["ltq"],
            "volume": rows["volume"],
            "oi": rows["oi"],
            "oi_change": rows["oi_change"],
            "bid_price": rows["bid_price_paise"].astype(np.float64) / PRICE_SCALE,
            "ask_price": rows["ask_price_paise"].astype(np.float64) / PRICE_SCALE,
            "bid_qty": rows["bid_qty"],
            "ask_qty": rows["ask_qty"],
            "bid_orders": rows["bid_orders"],
            "ask_orders": rows["ask_orders"],
            "iv": rows["iv_bps"].astype(np.float64) / IV_SCALE,
            "delta": rows["delta_bps"].astype(np.float64) / GREEK_SCALE,
            "gamma": rows["gamma_scaled"].astype(np.float64) / GAMMA_SCALE,
            "theta": rows["theta_scaled"].astype(np.float64) / GREEK_SCALE,
            "vega": rows["vega_scaled"].astype(np.float64) / GREEK_SCALE,
            "flags": rows["flags"],
            "exchange_id": rows["exchange_id"],
        }

    def close(self) -> None:
        if self._handle:
            self._lib.vtf_reader_close(self._handle)
            self._lib.vtf_reader_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "VTFReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_handle", None):
            try:
                self.close()
            except Exception:
                pass


def recover_vtf_file(path: Union[str, Path]) -> RecoveryReport:
    """
    Scan an unclosed, truncated, or tail-corrupted VTF file, verify every block's
    header and payload CRC32C checksum, truncate incomplete tail bytes, and rebuild
    the Footer Index & Trailer in-place.
    """
    lib = _load_libvtf()
    rep = CRecoveryReport()
    rc = lib.vtf_recover_file(str(path).encode("utf-8"), ctypes.byref(rep))
    if rc != 0:
        raise IOError(f"VTF file recovery failed on '{path}' (status={rep.status}, rc={rc})")
    return RecoveryReport(
        status=int(rep.status),
        valid_blocks=int(rep.valid_blocks),
        valid_rows=int(rep.valid_rows),
        original_file_size=int(rep.original_file_size),
        recovered_file_size=int(rep.recovered_file_size),
        truncated_bytes=int(rep.truncated_bytes),
        had_valid_footer=bool(rep.had_valid_footer),
    )


def raw_lz4_write(path: Union[str, Path], ticks: np.ndarray, block_rows: int = 16384, accel: int = 1) -> None:
    """Write raw row-major `VTF_TICK_DTYPE` structs compressed with native C++ LZ4."""
    lib = _load_libvtf()
    rc = lib.raw_lz4_binary_write(
        str(path).encode("utf-8"),
        ticks.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(len(ticks)),
        ctypes.c_size_t(block_rows),
        ctypes.c_int(accel),
    )
    if rc != 0:
        raise IOError(f"raw_lz4_write failed (rc={rc})")


def raw_lz4_read(path: Union[str, Path], count: int) -> np.ndarray:
    """Read raw row-major `VTF_TICK_DTYPE` structs decompressed with native C++ LZ4."""
    lib = _load_libvtf()
    out = np.empty(count, dtype=VTF_TICK_DTYPE)
    rc = lib.raw_lz4_binary_read(
        str(path).encode("utf-8"),
        out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(count),
    )
    if rc < 0:
        raise IOError(f"raw_lz4_read failed (rc={rc})")
    return out[:rc]