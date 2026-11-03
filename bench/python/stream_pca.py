"""Small local bench: time until a streaming PCA result, picklerick vs anndata backed.

    pixi run -e py install-picklerick-release   # ALWAYS release: debug decode is ~3x slower
    pixi run -e py python bench/python/stream_pca.py tests/golden/hlca_core.h5ad --n-obs 20000

Takes the first --n-obs cells of X from any CSR h5ad and writes two copies
(uncompressed, gzip-4) to --workdir (tmpfs by default, so reads stay in RAM).
Each arm runs in its own process (clean peak RSS) and streams X twice through
the same reducer: pass 1 picks the top-variance genes, pass 2 accumulates their
Gram matrix with dsyrk, then eigh. Arms:

  readonly_*  stream only, no math: pure read throughput
  pca_*       read + reducer; `wait` is time blocked on the next chunk

The omni version of this (100k cells, denet + GNU time, scanpy reference) is
/data/ben/scx-bench/pca/pca_arm2.py; this one is for quick local A/B of the read path.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

K, N_HVG = 50, 2000


def make_inputs(src: str, n_obs: int, workdir: Path) -> dict[str, Path]:
    import anndata as ad
    import h5py
    import pandas as pd
    import scipy.sparse as sp

    out = {"raw": workdir / f"x{n_obs}.h5ad", "gz": workdir / f"x{n_obs}.gzip.h5ad"}
    if all(p.exists() for p in out.values()):
        return out
    workdir.mkdir(parents=True, exist_ok=True)
    with h5py.File(src) as f:  # slice X by indptr: no full load of the source
        g = f["X"]
        ip = g["indptr"][: n_obs + 1].astype("int64")
        a, b = int(ip[0]), int(ip[-1])
        X = sp.csr_matrix(
            (g["data"][a:b], g["indices"][a:b], ip - a), shape=(n_obs, int(g.attrs["shape"][1]))
        )
    A = ad.AnnData(X, obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]))
    A.write_h5ad(out["raw"])
    A.write_h5ad(out["gz"], compression="gzip", compression_opts=4)
    return out


def arm(name: str, path: str, cs: int) -> dict:
    import numpy as np
    import scipy.sparse as sp
    from scipy.linalg.blas import dsyrk

    t0 = time.perf_counter()
    wait = 0.0
    if "picklerick" in name:
        import picklerick as pk

        def chunks():
            for c in pk.open_stream(path, chunk_size=cs):
                yield c.to_scipy()
    else:
        import anndata as ad

        A = ad.read_h5ad(path, backed="r")

        def chunks():
            for X, _, _ in A.chunked_X(cs):
                yield X

    def timed():
        nonlocal wait
        it = chunks()
        while True:
            t = time.perf_counter()
            try:
                x = next(it)
            except StopIteration:
                return
            wait += time.perf_counter() - t
            yield x

    t_open = time.perf_counter() - t0
    res: dict = {}
    if name.startswith("readonly"):
        res["nnz"] = sum(X.nnz for X in timed())
    else:
        n, s1, s2 = 0, 0.0, 0.0
        for X in timed():
            d = sp.csr_array(X).astype(np.float64)
            s1 = s1 + np.asarray(d.sum(0)).ravel()
            s2 = s2 + np.asarray(d.multiply(d).sum(0)).ravel()
            n += d.shape[0]
        mu = s1 / n
        sel = np.sort(np.argsort(s2 / n - mu * mu)[::-1][:N_HVG])
        G = np.zeros((len(sel), len(sel)), order="F")
        for X in timed():
            Xd = sp.csr_array(X)[:, sel].astype(np.float64).toarray()
            G = dsyrk(1.0, Xd, beta=1.0, c=G, trans=1, overwrite_c=1)  # upper triangle
        G = np.triu(G) + np.triu(G, 1).T
        m = mu[sel]
        w = np.linalg.eigvalsh((G - n * np.outer(m, m)) / (n - 1))[::-1][:K]
        res["top_eigs"] = w[:5].round(6).tolist()
    res.update(
        arm=name,
        wall=round(time.perf_counter() - t0, 3),
        open=round(t_open, 3),
        wait=round(wait, 3),
        peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    )
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", nargs="?", help="CSR h5ad to slice X from")
    p.add_argument("--n-obs", type=int, default=20000)
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--workdir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "scx-stream-pca")
    p.add_argument("--arms", default="readonly_backed,readonly_picklerick,pca_backed,pca_picklerick")
    p.add_argument("--_arm", nargs=2, help=argparse.SUPPRESS)  # internal: run one arm in this process
    a = p.parse_args()
    if a._arm:
        print(json.dumps(arm(a._arm[0], a._arm[1], a.chunk_size)))
        return
    if not a.src:
        p.error("src is required")
    inputs = make_inputs(a.src, a.n_obs, a.workdir)
    print(f"# {a.n_obs} cells, chunk {a.chunk_size}, {a.reps} reps, median; RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'all')}")
    print(f"{'file':5} {'arm':22} {'wall s':>7} {'wait s':>7} {'peak MB':>8}  check")
    for kind, path in inputs.items():
        subprocess.run(["cat", str(path)], stdout=subprocess.DEVNULL, check=True)  # warm cache
        for name in a.arms.split(","):
            runs = []
            for _ in range(a.reps):
                out = subprocess.run(
                    [sys.executable, __file__, "--_arm", name, str(path), "--chunk-size", str(a.chunk_size)],
                    capture_output=True, text=True, check=True,
                )
                runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
            runs.sort(key=lambda r: r["wall"])
            r = runs[len(runs) // 2]
            check = r.get("top_eigs", r.get("nnz"))
            print(f"{kind:5} {name:22} {r['wall']:7.2f} {r['wait']:7.2f} {r['peak_mb']:8}  {check}")


if __name__ == "__main__":
    main()
