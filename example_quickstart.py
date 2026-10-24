#!/usr/bin/env python3
"""
Quickstart Example: Recording & Querying NIFTY/SENSEX Option Ticks with VTF v1.0 (LZ4-Fast, accel=4)
"""

import numpy as np
import vtf


def main() -> None:
    out_file = "sample_nifty_sensex_lz4_fast.vtf"
    num_ticks = 32_768  # 2 blocks of 16,384 ticks

    # 1. Populate structured NumPy tick buffer (156 bytes/row)
    ticks = np.zeros(num_ticks, dtype=vtf.VTF_TICK_DTYPE)
    ticks["timestamp_ns"] = 1_792_554_300_000_000_000 + np.arange(num_ticks, dtype=np.int64) * 10_000
    ticks["instrument_id"] = np.where(np.arange(num_ticks) % 2 == 0, 1001, 2001).astype(np.uint32)
    ticks["exchange_id"] = np.where(np.arange(num_ticks) % 2 == 0, vtf.EXCHANGE_NSE_NIFTY, vtf.EXCHANGE_BSE_SENSEX).astype(np.uint8)
    ticks["ltp_paise"] = 24505 + (np.arange(num_ticks, dtype=np.int32) % 5) * 5
    ticks["ltq"] = 75
    ticks["volume"] = np.cumsum(ticks["ltq"], dtype=np.uint32)
    ticks["oi"] = 1_250_000
    ticks["oi_change"] = 45_000

    for lvl in range(5):
        ticks["bid_price_paise"][:, lvl] = ticks["ltp_paise"] - (lvl + 1) * 5
        ticks["ask_price_paise"][:, lvl] = ticks["ltp_paise"] + (lvl + 1) * 5
        ticks["bid_qty"][:, lvl] = (lvl + 1) * 150
        ticks["ask_qty"][:, lvl] = (lvl + 1) * 150
        ticks["bid_orders"][:, lvl] = (lvl + 1) * 3
        ticks["ask_orders"][:, lvl] = (lvl + 1) * 3

    ticks["iv_bps"] = 1645         # 16.45% IV
    ticks["delta_bps"] = 5120      # +0.5120 Delta
    ticks["gamma_scaled"] = 415    # 0.000415 Gamma
    ticks["theta_scaled"] = -142500
    ticks["vega_scaled"] = 84200
    ticks["flags"] = vtf.TICK_FLAG_TRADE | vtf.TICK_FLAG_L2_UPDATE | vtf.TICK_FLAG_GREEKS_VALID

    # 2. Write using VTF v1.0 (LZ4-Fast, accel=4)
    with vtf.VTFWriter(out_file, codec=vtf.CODEC_LZ4_FAST, block_rows=16384) as writer:
        writer.append(ticks)

    # 3. Inspect and read back
    with vtf.VTFReader(out_file) as reader:
        hdr = reader.header_info()
        print(f"File: {out_file}")
        print(f"Codec: {hdr.codec_name} | Blocks: {reader.total_blocks} | Rows: {reader.total_rows:,}")
        for blk in reader.block_index():
            print(
                f"  Block #{blk.block_index}: {blk.row_count:,} rows | "
                f"{blk.compressed_bytes:,} B ({blk.bytes_per_row:.2f} B/tick, {blk.compression_ratio:.2f}x)"
            )
        read_rows = reader.read_all(verify_crc=True)
        assert np.array_equal(ticks, read_rows), "Round-trip verification failed!"
        print("100% bit-exact round-trip verified!")


if __name__ == "__main__":
    main()
