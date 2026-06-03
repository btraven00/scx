"""Spike lever #1: overlap decode with reduce via a Python prefetch thread.

open_stream.__next__ releases the GIL while the Rust reader decodes a chunk
(py.allow_threads(recv)), and np.bincount releases the GIL during compute. So
if a worker thread pulls the *next* chunk while the main thread reduces the
current one, decode and reduce run concurrently — without any Rust change.

Compares plain sequential consumption vs a thread-prefetched generator.

  pixi run -e py313 python scratch/spike_overlap_prefetch.py tests/golden/hlca_core.h5ad --chunk-size 50000
"""

from __future__ import annotations

import argparse
import queue
import threading
import time

import numpy as np

import picklerick as pk

_SENTINEL = object()


def prefetched(stream, depth: int = 2):
    """Yield chunks from `stream`, decoding ahead in a background thread."""
    q: queue.Queue = queue.Queue(maxsize=depth)

    def worker():
        try:
            for ch in stream:
                q.put(ch)
        except Exception as e:  # propagate to consumer
            q.put(e)
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def reduce_over(chunks, n_vars0=None):
    gs = None
    n = 0
    for ch in chunks:
        if gs is None:
            gs = np.zeros(ch.n_vars, dtype=np.float64)
        gs += np.bincount(ch.indices, weights=ch.data.astype(np.float64), minlength=ch.n_vars)
        n += ch.nrows
    return float(gs.sum()), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--chunk-size", type=int, default=50000)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    t0 = time.perf_counter()
    s_plain, n1 = reduce_over(pk.open_stream(args.path, chunk_size=args.chunk_size))
    t_plain = time.perf_counter() - t0

    t0 = time.perf_counter()
    s_pf, n2 = reduce_over(
        prefetched(pk.open_stream(args.path, chunk_size=args.chunk_size), depth=args.depth)
    )
    t_pf = time.perf_counter() - t0

    print(f"plain          {t_plain:7.2f}s   rows={n1} sum={s_plain:.0f}")
    print(f"prefetched(d{args.depth}) {t_pf:7.2f}s   rows={n2} sum={s_pf:.0f}")
    print(f"speedup        {t_plain / t_pf:7.2f}x")
    assert abs(s_plain - s_pf) < 1e-3 * abs(s_plain), "checksum drift!"


if __name__ == "__main__":
    main()
