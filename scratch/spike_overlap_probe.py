"""Diagnose whether open_stream's background decode overlaps the consumer reduce.

Python-only, no Rust rebuild. Times next() (recv + bytes build) separately from
the bincount reduce, per chunk.

  - If producer runs ahead of a slower consumer: next() returns fast (chunk
    already buffered), so t_recv << decode_total and t_recv + t_reduce ~= max,
    i.e. they overlap.
  - If t_recv ~= full decode time AND t_reduce ~= 2.8s and the two SUM to the
    full wall, decode and reduce are serial — no overlap, and lever #1 has room.

  pixi run -e py313 python scratch/spike_overlap_probe.py tests/golden/hlca_core.h5ad --chunk-size 50000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import picklerick as pk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--chunk-size", type=int, default=50000)
    args = ap.parse_args()

    it = iter(pk.open_stream(args.path, chunk_size=args.chunk_size))
    gs = None
    t_recv = t_astype = t_bincount = 0.0
    nchunks = 0
    wall0 = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        try:
            ch = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        if gs is None:
            gs = np.zeros(ch.n_vars, dtype=np.float64)
        w = ch.data.astype(np.float64)
        t2 = time.perf_counter()
        gs += np.bincount(ch.indices, weights=w, minlength=ch.n_vars)
        t3 = time.perf_counter()
        t_recv += t1 - t0
        t_astype += t2 - t1
        t_bincount += t3 - t2
        nchunks += 1
    wall = time.perf_counter() - wall0

    print(f"chunks            {nchunks}")
    print(f"sum(next/recv)    {t_recv:8.2f}s   <- decode + bytes build (GIL released on recv)")
    print(f"sum(astype f64)   {t_astype:8.2f}s")
    print(f"sum(bincount)     {t_bincount:8.2f}s")
    print(f"reduce total      {t_astype + t_bincount:8.2f}s")
    print(f"wall              {wall:8.2f}s")
    serial = t_recv + t_astype + t_bincount
    print(f"recv+reduce (sum) {serial:8.2f}s")
    print(
        f"\noverlap factor    {serial / wall:5.2f}x "
        f"(1.0 = fully serial/no overlap; >1 = decode hidden behind reduce)"
    )
    print(f"checksum          {float(gs.sum()):.0f}")


if __name__ == "__main__":
    main()
