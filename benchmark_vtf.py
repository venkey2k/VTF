"""
Reproducible Verification & Benchmark Suite for VTF v1.0
=========================================================
Tests and benchmarks VTF v1.0 (LZ4-Fast, LZ4-Default, Zstd) against:
  1. Apache Parquet (Snappy & Zstd) via PyArrow C++ engine
  2. Apache Arrow IPC / Feather v2 (Uncompressed & LZ4_FRAME) via PyArrow C++ engine
  3. Raw LZ4-Compressed Binary Structs (LZ4-Fast & LZ4-Default) via C++ liblz4

Also validates:
  - 100% bit-exact round-trip fidelity across all 22 fields (ticks, 5-level L2 book, OI, IV, Greeks)
  - Time-range and 64-bit Bloom filter predicate pushdown queries
  - 3 realistic crash recovery scenarios (missing footer / SIGKILL, mid-block write truncation,
    and compressed payload bit-rot corruption)
"""

from __future__ import annotations

import dataclasses
import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

import vtf


# ============================================================================
# 1. Realistic NIFTY 50 & SENSEX High-Frequency Option Chain Tick Generator
# ============================================================================

def generate_nifty_sensex_ticks(num_rows: int = 500_000, seed: int = 42) -> np.ndarray:
    """
    Synthesizes a realistic high-frequency multiplexed option chain stream across
    128 active NIFTY 50 (NSE) and SENSEX (BSE) option contracts.
    """
    rng = np.random.default_rng(seed)
    ticks = np.zeros(num_rows, dtype=vtf.VTF_TICK_DTYPE)

    num_instruments = 128
    # 64 NIFTY contracts (tokens 1000..1063), 64 SENSEX contracts (tokens 2000..2063)
    inst_ids = np.concatenate([
        np.arange(1000, 1064, dtype=np.uint32),
        np.arange(2000, 2064, dtype=np.uint32),
    ])

    weights = 1.0 / (np.arange(1, num_instruments + 1) ** 0.65)
    weights /= weights.sum()
    inst_idx = rng.choice(num_instruments, size=num_rows, p=weights)
    ticks["instrument_id"] = inst_ids[inst_idx]
    ticks["exchange_id"] = np.where(inst_idx < 64, vtf.EXCHANGE_NSE_NIFTY, vtf.EXCHANGE_BSE_SENSEX).astype(np.uint8)

    base_ts_ns = 1_792_554_300_000_000_000
    dt_ns = rng.integers(500, 25_000, size=num_rows, dtype=np.int64)
    ticks["timestamp_ns"] = base_ts_ns + np.cumsum(dt_ns)

    init_ltp_paise = rng.integers(2_000, 45_000, size=num_instruments, dtype=np.int32) // 5 * 5
    init_oi = rng.integers(50_000, 5_000_000, size=num_instruments, dtype=np.uint32) // 25 * 25
    init_iv_bps = rng.integers(1_100, 2_800, size=num_instruments, dtype=np.int32)
    init_delta_bps = rng.integers(-9_000, 9_000, size=num_instruments, dtype=np.int16)
    init_gamma = rng.integers(50, 1_500, size=num_instruments, dtype=np.int32)
    init_theta = rng.integers(-250_000, -10_000, size=num_instruments, dtype=np.int32)
    init_vega = rng.integers(15_000, 180_000, size=num_instruments, dtype=np.int32)

    price_steps = rng.choice([-10, -5, 0, 0, 0, 5, 10], size=num_rows).astype(np.int32)
    is_trade = rng.random(num_rows) < 0.28
    lot_size = np.where(inst_idx < 64, 75, 20).astype(np.uint32)
    trade_lots = np.where(is_trade, rng.integers(1, 15, size=num_rows, dtype=np.uint32), 0)
    ltq = trade_lots * lot_size
    ticks["ltq"] = ltq

    for c in range(num_instruments):
        mask = np.flatnonzero(inst_idx == c)
        m = len(mask)
        if m == 0:
            continue

        ltp = np.maximum(50, init_ltp_paise[c] + np.cumsum(price_steps[mask])).astype(np.int32)
        ticks["ltp_paise"][mask] = ltp

        vol = np.cumsum(ltq[mask], dtype=np.uint64).astype(np.uint32)
        ticks["volume"][mask] = vol

        oi_step_events = (rng.random(m) < 0.01) * rng.choice([-75, 0, 75, 150], size=m)
        oi_chg = np.cumsum(oi_step_events, dtype=np.int64).astype(np.int32)
        ticks["oi"][mask] = (int(init_oi[c]) + oi_chg).clip(min=75).astype(np.uint32)
        ticks["oi_change"][mask] = oi_chg

        b0_offset = rng.choice([-5, 0], size=m, p=[0.85, 0.15]).astype(np.int32)
        spread = rng.choice([5, 10, 15], size=m, p=[0.80, 0.15, 0.05]).astype(np.int32)
        b0 = ltp + b0_offset
        a0 = b0 + spread

        bid_p = np.empty((m, 5), dtype=np.int32)
        ask_p = np.empty((m, 5), dtype=np.int32)
        bid_p[:, 0] = b0
        ask_p[:, 0] = a0
        for lvl in range(1, 5):
            bid_p[:, lvl] = b0 - lvl * 5
            ask_p[:, lvl] = a0 + lvl * 5
        ticks["bid_price_paise"][mask] = bid_p
        ticks["ask_price_paise"][mask] = ask_p

        lot = int(lot_size[mask[0]])
        bid_q = np.empty((m, 5), dtype=np.uint32)
        ask_q = np.empty((m, 5), dtype=np.uint32)
        bid_o = np.empty((m, 5), dtype=np.uint16)
        ask_o = np.empty((m, 5), dtype=np.uint16)

        for lvl in range(5):
            update_prob = 0.65 if lvl == 0 else (0.12 if lvl <= 2 else 0.04)
            q_deltas = (rng.random(m) < update_prob) * rng.choice([-2, -1, 1, 2], size=m) * lot
            base_q = (lvl + 2) * 10 * lot
            bq = np.clip(base_q + np.cumsum(q_deltas), lot, 500 * lot).astype(np.uint32)
            aq = np.clip(base_q + np.cumsum(q_deltas[::-1]), lot, 500 * lot).astype(np.uint32)
            bid_q[:, lvl] = bq
            ask_q[:, lvl] = aq
            bid_o[:, lvl] = np.clip(bq // (lot * 2), 1, 200).astype(np.uint16)
            ask_o[:, lvl] = np.clip(aq // (lot * 2), 1, 200).astype(np.uint16)

        ticks["bid_qty"][mask] = bid_q
        ticks["ask_qty"][mask] = ask_q
        ticks["bid_orders"][mask] = bid_o
        ticks["ask_orders"][mask] = ask_o

        greek_upd = rng.random(m) < 0.05
        iv_steps = greek_upd * rng.choice([-2, -1, 1, 2], size=m)
        ticks["iv_bps"][mask] = np.clip(init_iv_bps[c] + np.cumsum(iv_steps), 500, 8000).astype(np.int32)

        d_steps = greek_upd * rng.choice([-5, -2, 2, 5], size=m)
        ticks["delta_bps"][mask] = np.clip(init_delta_bps[c] + np.cumsum(d_steps), -10000, 10000).astype(np.int16)

        g_steps = greek_upd * rng.choice([-1, 0, 1], size=m)
        ticks["gamma_scaled"][mask] = np.clip(init_gamma[c] + np.cumsum(g_steps), 1, 10000).astype(np.int32)

        t_steps = greek_upd * rng.choice([-10, -5, 5, 10], size=m)
        ticks["theta_scaled"][mask] = (init_theta[c] + np.cumsum(t_steps)).astype(np.int32)

        v_steps = greek_upd * rng.choice([-10, -5, 5, 10], size=m)
        ticks["vega_scaled"][mask] = np.clip(init_vega[c] + np.cumsum(v_steps), 100, 500000).astype(np.int32)

    flags = np.full(num_rows, vtf.TICK_FLAG_L2_UPDATE | vtf.TICK_FLAG_GREEKS_VALID, dtype=np.uint8)
    flags = np.where(is_trade, flags | vtf.TICK_FLAG_TRADE | vtf.TICK_FLAG_BUY_AGGRESSOR, flags)
    ticks["flags"] = flags

    return ticks


# ============================================================================
# 2. Convert Structured NumPy Array <-> Flat PyArrow Table (42 columns)
# ============================================================================

def ticks_to_arrow_table(ticks: np.ndarray) -> pa.Table:
    """
    Converts `VTF_TICK_DTYPE` into a flat 42-column PyArrow Table for zero-copy
    Arrow IPC and Parquet serialization.
    """
    arrays = [
        pa.array(ticks["timestamp_ns"], type=pa.int64()),
        pa.array(ticks["instrument_id"], type=pa.uint32()),
        pa.array(ticks["ltp_paise"], type=pa.int32()),
        pa.array(ticks["ltq"], type=pa.uint32()),
        pa.array(ticks["volume"], type=pa.uint32()),
        pa.array(ticks["oi"], type=pa.uint32()),
        pa.array(ticks["oi_change"], type=pa.int32()),
    ]
    names = [
        "timestamp_ns", "instrument_id", "ltp_paise", "ltq", "volume", "oi", "oi_change"
    ]
    for k in range(5):
        arrays.append(pa.array(ticks["bid_price_paise"][:, k], type=pa.int32()))
        names.append(f"bid_price_{k}")
    for k in range(5):
        arrays.append(pa.array(ticks["ask_price_paise"][:, k], type=pa.int32()))
        names.append(f"ask_price_{k}")
    for k in range(5):
        arrays.append(pa.array(ticks["bid_qty"][:, k], type=pa.uint32()))
        names.append(f"bid_qty_{k}")
    for k in range(5):
        arrays.append(pa.array(ticks["ask_qty"][:, k], type=pa.uint32()))
        names.append(f"ask_qty_{k}")
    for k in range(5):
        arrays.append(pa.array(ticks["bid_orders"][:, k], type=pa.uint16()))
        names.append(f"bid_orders_{k}")
    for k in range(5):
        arrays.append(pa.array(ticks["ask_orders"][:, k], type=pa.uint16()))
        names.append(f"ask_orders_{k}")

    arrays.extend([
        pa.array(ticks["iv_bps"], type=pa.int32()),
        pa.array(ticks["delta_bps"], type=pa.int16()),
        pa.array(ticks["gamma_scaled"], type=pa.int32()),
        pa.array(ticks["theta_scaled"], type=pa.int32()),
        pa.array(ticks["vega_scaled"], type=pa.int32()),
        pa.array(ticks["flags"], type=pa.uint8()),
        pa.array(ticks["exchange_id"], type=pa.uint8()),
    ])
    names.extend([
        "iv_bps", "delta_bps", "gamma_scaled", "theta_scaled", "vega_scaled", "flags", "exchange_id"
    ])
    return pa.Table.from_arrays(arrays, names=names)


# ============================================================================
# 3. Correctness, Predicate Pushdown & Crash Recovery Verification
# ============================================================================

def run_correctness_and_crash_tests(ticks: np.ndarray, work_dir: Path) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    subset = ticks[:65_536]

    codec_checks = {}
    for codec_id, label in [
        (vtf.CODEC_LZ4_FAST, "VTF_LZ4_FAST"),
        (vtf.CODEC_LZ4_DEFAULT, "VTF_LZ4_DEFAULT"),
        (vtf.CODEC_ZSTD, "VTF_ZSTD"),
        (vtf.CODEC_NONE, "VTF_NONE"),
    ]:
        fpath = work_dir / f"test_{label}.vtf"
        with vtf.VTFWriter(fpath, codec=codec_id, block_rows=16384) as w:
            w.append(subset)
        with vtf.VTFReader(fpath) as r:
            read_back = r.read_all(verify_crc=True)
            hdr = r.header_info()
            exact_match = bool(np.array_equal(subset, read_back))
            assert exact_match, f"Round-trip mismatch for {label}!"
            codec_checks[label] = {
                "rows_verified": len(read_back),
                "bit_exact": exact_match,
                "blocks": r.total_blocks,
                "file_bytes": fpath.stat().st_size,
                "has_footer": hdr.has_footer,
            }
    results["codec_roundtrip"] = codec_checks

    qpath = work_dir / "test_VTF_LZ4_FAST.vtf"
    t_start = int(subset["timestamp_ns"][20_000])
    t_end = int(subset["timestamp_ns"][45_000])
    target_inst = int(subset["instrument_id"][25_000])

    with vtf.VTFReader(qpath) as r:
        queried = r.query(t_start, t_end, instrument_id=target_inst)

    expected_mask = (
        (subset["timestamp_ns"] >= t_start)
        & (subset["timestamp_ns"] <= t_end)
        & (subset["instrument_id"] == target_inst)
    )
    expected_rows = subset[expected_mask]
    assert np.array_equal(queried, expected_rows), "Predicate pushdown query mismatch!"
    results["predicate_pushdown"] = {
        "matching_rows": len(queried),
        "bit_exact": True,
    }

    # Crash Scenario A: Abrupt Process Termination (`SIGKILL` / Missing Footer)
    crash_a_path = work_dir / "crash_sigkill_no_footer.vtf"
    w_a = vtf.VTFWriter(crash_a_path, codec=vtf.CODEC_LZ4_FAST, block_rows=16384)
    w_a.append(subset[:49_152])
    w_a.flush()
    w_a.abort_without_footer()

    rep_a = vtf.recover_vtf_file(crash_a_path)
    with vtf.VTFReader(crash_a_path, auto_recover=False) as r_a:
        rec_rows_a = r_a.read_all()
        hdr_a = r_a.header_info()
    assert len(rec_rows_a) == 49_152 and np.array_equal(rec_rows_a, subset[:49_152])
    assert hdr_a.has_footer and hdr_a.was_recovered
    results["crash_scenario_a_missing_footer"] = dataclasses.asdict(rep_a)

    # Crash Scenario B: Mid-Block Write Truncation (Power Loss During 4th Block Write)
    crash_b_path = work_dir / "crash_mid_block_truncated.vtf"
    w_b = vtf.VTFWriter(crash_b_path, codec=vtf.CODEC_LZ4_FAST, block_rows=16384)
    w_b.append(subset[:65_536])
    w_b.flush()
    w_b.abort_without_footer()
    orig_b_size = crash_b_path.stat().st_size
    os.truncate(crash_b_path, orig_b_size - 6500)

    rep_b = vtf.recover_vtf_file(crash_b_path)
    with vtf.VTFReader(crash_b_path, auto_recover=False) as r_b:
        rec_rows_b = r_b.read_all()
    assert rep_b.valid_blocks == 3 and rep_b.valid_rows == 49_152
    assert np.array_equal(rec_rows_b, subset[:49_152])
    results["crash_scenario_b_mid_block_truncation"] = dataclasses.asdict(rep_b)

    # Crash Scenario C: Bit-Rot / Corrupted Payload in Final Block
    crash_c_path = work_dir / "crash_corrupt_tail_payload.vtf"
    w_c = vtf.VTFWriter(crash_c_path, codec=vtf.CODEC_LZ4_FAST, block_rows=16384)
    w_c.append(subset[:65_536])
    w_c.flush()
    w_c.abort_without_footer()
    with open(crash_c_path, "r+b") as f:
        f.seek(-500, os.SEEK_END)
        bad_byte = f.read(1)[0] ^ 0xFF
        f.seek(-500, os.SEEK_END)
        f.write(bytes([bad_byte]))

    rep_c = vtf.recover_vtf_file(crash_c_path)
    with vtf.VTFReader(crash_c_path, auto_recover=False) as r_c:
        rec_rows_c = r_c.read_all()
    assert rep_c.valid_blocks == 3 and rep_c.valid_rows == 49_152
    assert np.array_equal(rec_rows_c, subset[:49_152])
    results["crash_scenario_c_payload_bitrot"] = dataclasses.asdict(rep_c)

    return results


# ============================================================================
# 4. Reproducible Performance & Compression Benchmarks
# ============================================================================

def run_benchmarks(ticks: np.ndarray, work_dir: Path, trials: int = 5) -> List[Dict[str, Any]]:
    num_rows = len(ticks)
    raw_bytes = num_rows * vtf.VTF_TICK_DTYPE.itemsize
    raw_mb = raw_bytes / (1024.0 * 1024.0)

    arrow_table = ticks_to_arrow_table(ticks)
    benchmarks: List[Dict[str, Any]] = []

    def record_bench(
        name: str,
        fpath: Path,
        write_fn,
        read_fn,
        crash_safe: str,
    ) -> None:
        write_fn()
        _ = read_fn()

        w_times = []
        r_times = []
        for _ in range(trials):
            if fpath.exists():
                fpath.unlink()
            t0 = time.perf_counter()
            write_fn()
            t1 = time.perf_counter()
            w_times.append(t1 - t0)

            t2 = time.perf_counter()
            _ = read_fn()
            t3 = time.perf_counter()
            r_times.append(t3 - t2)

        w_med = statistics.median(w_times)
        r_med = statistics.median(r_times)
        fsize = fpath.stat().st_size
        fsize_mb = fsize / (1024.0 * 1024.0)

        benchmarks.append({
            "format": name,
            "file_size_bytes": fsize,
            "file_size_mb": round(fsize_mb, 3),
            "bytes_per_tick": round(fsize / num_rows, 2),
            "compression_ratio": round(raw_bytes / fsize, 2),
            "write_ms": round(w_med * 1000.0, 2),
            "write_m_ticks_sec": round((num_rows / w_med) / 1e6, 2),
            "write_mb_sec": round(raw_mb / w_med, 1),
            "read_ms": round(r_med * 1000.0, 2),
            "read_m_ticks_sec": round((num_rows / r_med) / 1e6, 2),
            "read_mb_sec": round(raw_mb / r_med, 1),
            "crash_recovery": crash_safe,
        })

    # 1. VTF v1.0 (LZ4_FAST, accel=4)
    p_vtf_fast = work_dir / "bench_vtf_lz4_fast.vtf"
    record_bench(
        "VTF v1.0 (LZ4-Fast)",
        p_vtf_fast,
        lambda: _vtf_write_helper(p_vtf_fast, ticks, vtf.CODEC_LZ4_FAST, 16384),
        lambda: _vtf_read_helper(p_vtf_fast),
        "Yes (Block CRC32C + Auto-Rebuild)",
    )

    # 2. VTF v1.0 (LZ4_DEFAULT)
    p_vtf_def = work_dir / "bench_vtf_lz4_default.vtf"
    record_bench(
        "VTF v1.0 (LZ4-Default)",
        p_vtf_def,
        lambda: _vtf_write_helper(p_vtf_def, ticks, vtf.CODEC_LZ4_DEFAULT, 16384),
        lambda: _vtf_read_helper(p_vtf_def),
        "Yes (Block CRC32C + Auto-Rebuild)",
    )

    # 3. VTF v1.0 (ZSTD Level 3)
    p_vtf_zstd = work_dir / "bench_vtf_zstd.vtf"
    record_bench(
        "VTF v1.0 (Zstd-L3)",
        p_vtf_zstd,
        lambda: _vtf_write_helper(p_vtf_zstd, ticks, vtf.CODEC_ZSTD, 16384),
        lambda: _vtf_read_helper(p_vtf_zstd),
        "Yes (Block CRC32C + Auto-Rebuild)",
    )

    # 4. Raw Binary + LZ4-Fast (C++ struct blocks)
    p_raw_lz4_fast = work_dir / "bench_raw_lz4_fast.bin"
    record_bench(
        "Raw Binary + LZ4 (Fast)",
        p_raw_lz4_fast,
        lambda: vtf.raw_lz4_write(p_raw_lz4_fast, ticks, block_rows=16384, accel=4),
        lambda: vtf.raw_lz4_read(p_raw_lz4_fast, num_rows),
        "Partial (No CRC / No Index)",
    )

    # 5. Raw Binary + LZ4-Default (C++ struct blocks)
    p_raw_lz4_def = work_dir / "bench_raw_lz4_def.bin"
    record_bench(
        "Raw Binary + LZ4 (Default)",
        p_raw_lz4_def,
        lambda: vtf.raw_lz4_write(p_raw_lz4_def, ticks, block_rows=16384, accel=1),
        lambda: vtf.raw_lz4_read(p_raw_lz4_def, num_rows),
        "Partial (No CRC / No Index)",
    )

    # 6. Apache Arrow IPC / Feather v2 (Uncompressed)
    p_arrow_uncomp = work_dir / "bench_arrow_uncompressed.arrow"
    record_bench(
        "Arrow IPC (Uncompressed)",
        p_arrow_uncomp,
        lambda: feather.write_feather(arrow_table, p_arrow_uncomp, compression="uncompressed", chunksize=16384),
        lambda: feather.read_table(p_arrow_uncomp),
        "No (Requires EOF Footer)",
    )

    # 7. Apache Arrow IPC / Feather v2 (LZ4_FRAME)
    p_arrow_lz4 = work_dir / "bench_arrow_lz4.arrow"
    record_bench(
        "Arrow IPC (LZ4_FRAME)",
        p_arrow_lz4,
        lambda: feather.write_feather(arrow_table, p_arrow_lz4, compression="lz4", chunksize=16384),
        lambda: feather.read_table(p_arrow_lz4),
        "No (Requires EOF Footer)",
    )

    # 8. Apache Parquet (Snappy)
    p_pq_snappy = work_dir / "bench_parquet_snappy.parquet"
    record_bench(
        "Parquet (Snappy)",
        p_pq_snappy,
        lambda: pq.write_table(arrow_table, p_pq_snappy, compression="SNAPPY", row_group_size=65536),
        lambda: pq.read_table(p_pq_snappy),
        "No (Unreadable Without Footer)",
    )

    # 9. Apache Parquet (ZSTD Level 3)
    p_pq_zstd = work_dir / "bench_parquet_zstd.parquet"
    record_bench(
        "Parquet (Zstd-L3)",
        p_pq_zstd,
        lambda: pq.write_table(arrow_table, p_pq_zstd, compression="ZSTD", compression_level=3, row_group_size=65536),
        lambda: pq.read_table(p_pq_zstd),
        "No (Unreadable Without Footer)",
    )

    return benchmarks


def _vtf_write_helper(path: Path, ticks: np.ndarray, codec: int, block_rows: int) -> None:
    with vtf.VTFWriter(path, codec=codec, block_rows=block_rows) as w:
        w.append(ticks)


def _vtf_read_helper(path: Path) -> np.ndarray:
    with vtf.VTFReader(path) as r:
        return r.read_all(verify_crc=True)


def main() -> None:
    print("=" * 92)
    print("VTF v1.0 — High-Frequency NIFTY/SENSEX Option Market Data Format Benchmark & Test Suite")
    print("=" * 92)

    num_rows = 500_000
    print(f"[1/3] Generating {num_rows:,} realistic NIFTY 50 & SENSEX option ticks (156 B/tick = {num_rows * 156 / (1024**2):.2f} MB)...")
    ticks = generate_nifty_sensex_ticks(num_rows=num_rows, seed=42)

    with tempfile.TemporaryDirectory(prefix="vtf_bench_") as tmpdir:
        work_dir = Path(tmpdir)

        print("[2/3] Running bit-exact round-trip, predicate pushdown, and 3 crash recovery tests...")
        test_report = run_correctness_and_crash_tests(ticks, work_dir)
        print("      -> All 4 codecs passed 100% bit-exact verification across all 22 fields.")
        print(f"      -> Predicate pushdown verified ({test_report['predicate_pushdown']['matching_rows']} rows matched).")
        print("      -> Crash Scenario A (SIGKILL / Missing Footer): RECOVERED", test_report["crash_scenario_a_missing_footer"])
        print("      -> Crash Scenario B (Mid-Block Write Truncation): RECOVERED", test_report["crash_scenario_b_mid_block_truncation"])
        print("      -> Crash Scenario C (Tail Payload Bit-Rot): RECOVERED", test_report["crash_scenario_c_payload_bitrot"])

        print("[3/3] Running 5-trial reproducible benchmarks across VTF, Raw LZ4, Arrow IPC, and Parquet...")
        bench_rows = run_benchmarks(ticks, work_dir, trials=5)

    print("\n" + "-" * 112)
    header = (
        f"{'Format':<27} | {'Size (MB)':>9} | {'B/Tick':>7} | {'Ratio':>6} | "
        f"{'Write (ms)':>10} | {'Write (MT/s)':>12} | {'Write (MB/s)':>12} | "
        f"{'Read (ms)':>9} | {'Read (MT/s)':>11}"
    )
    print(header)
    print("-" * 112)
    for b in bench_rows:
        print(
            f"{b['format']:<27} | {b['file_size_mb']:>9.3f} | {b['bytes_per_tick']:>7.2f} | "
            f"{b['compression_ratio']:>5.2f}x | {b['write_ms']:>10.2f} | {b['write_m_ticks_sec']:>12.2f} | "
            f"{b['write_mb_sec']:>12.1f} | {b['read_ms']:>9.2f} | {b['read_m_ticks_sec']:>11.2f}"
        )
    print("-" * 112)

    out_json = Path(__file__).resolve().parent.parent / "intermediate" / "benchmark_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"tests": test_report, "benchmarks": bench_rows}, f, indent=2)
    print(f"\nSaved detailed JSON results to {out_json}")


if __name__ == "__main__":
    main()