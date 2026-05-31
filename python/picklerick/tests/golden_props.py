"""Format-/tool-neutral 'golden property' digests for correctness testing.

The idea: extract ground-truth properties of each fixture ONCE with the
canonical reader (anndata / anndataR / BPCells) in a heavy oracle environment,
serialise them to JSON, and commit that. Tests then read the same fixture with
picklerick and assert it reproduces the committed digests — no oracle stack
needed at test time.

Correctness hinges on the digest being **canonical**: the same logical matrix
must hash identically regardless of which tool produced it or how it was stored
(CSR vs CSC, f32 vs f64, index order). The canonical form is:

  * matrix  -> CSR, 0-based, column indices ascending within each row, duplicates
               summed; digest = sha256( indptr<i8 LE || indices<i4 LE || data<f8 LE )
  * index   -> sha256 of "\n".join(values) UTF-8
  * column  -> dtype tag + sha256 of the canonical bytes (numeric widened to the
               canonical width; categorical = categories + integer codes)

Widening f32->f64 / i32->i64 is exact, so a correct reader of losslessly-stored
values always lands on the same digest. This module is imported by both the
extraction script and the test, so the contract is defined in exactly one place.
"""

from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp

SCHEMA_VERSION = 1


def _sha(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def canonical_csr(x) -> sp.csr_matrix:
    """Coerce any sparse/dense matrix to canonical CSR (sorted, deduped)."""
    m = sp.csr_matrix(x)
    m.sum_duplicates()
    m.sort_indices()
    return m


def digest_matrix(x) -> dict:
    """Digest a count matrix. Bit-exact-strength but compact (one hash)."""
    m = canonical_csr(x)
    indptr = np.ascontiguousarray(m.indptr, dtype="<i8")
    indices = np.ascontiguousarray(m.indices, dtype="<i4")
    data = np.ascontiguousarray(m.data, dtype="<f8")
    # A few exact spot values aid debugging when a hash mismatches.
    spot = []
    if m.shape[0] and m.nnz:
        r0_end = int(m.indptr[1])
        spot = [
            [int(j), float(v)]
            for j, v in zip(m.indices[:r0_end][:5], m.data[:r0_end][:5])
        ]
    return {
        "shape": [int(m.shape[0]), int(m.shape[1])],
        "nnz": int(m.nnz),
        "digest": _sha(indptr.tobytes(), indices.tobytes(), data.tobytes()),
        "row0_spot": spot,
    }


def digest_index(values) -> str:
    return _sha("\n".join(str(v) for v in values).encode("utf-8"))


def digest_column(series) -> dict:
    """Digest one obs/var column by dtype family (numeric / categorical / other)."""
    arr = np.asarray(series)
    if hasattr(series, "cat") or str(getattr(series, "dtype", "")) == "category":
        cats = list(series.cat.categories) if hasattr(series, "cat") else sorted(set(arr))
        codes = np.asarray(series.cat.codes, dtype="<i8") if hasattr(series, "cat") else None
        return {
            "kind": "categorical",
            "digest": _sha(
                "\n".join(map(str, cats)).encode("utf-8"),
                codes.tobytes() if codes is not None else b"",
            ),
        }
    if np.issubdtype(arr.dtype, np.floating):
        return {"kind": "float", "digest": _sha(np.ascontiguousarray(arr, "<f8").tobytes())}
    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        return {"kind": "int", "digest": _sha(np.ascontiguousarray(arr, "<i8").tobytes())}
    return {"kind": "str", "digest": digest_index(arr)}


def build_properties(adata, *, include_tables: bool = True) -> dict:
    """Build the full property dict for an AnnData-like object."""
    props: dict = {
        "schema": SCHEMA_VERSION,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "X": digest_matrix(adata.X),
        "obs_names": digest_index(adata.obs_names),
        "var_names": digest_index(adata.var_names),
    }
    if include_tables:
        props["obs"] = {c: digest_column(adata.obs[c]) for c in adata.obs.columns}
        props["var"] = {c: digest_column(adata.var[c]) for c in adata.var.columns}
        props["obsm"] = {
            k: digest_matrix(adata.obsm[k]) if sp.issparse(adata.obsm[k]) else
            {"shape": list(np.asarray(adata.obsm[k]).shape),
             "digest": _sha(np.ascontiguousarray(np.asarray(adata.obsm[k]), "<f8").tobytes())}
            for k in adata.obsm.keys()
        }
    return props


def matrix_from_stream_chunks(chunks, n_vars: int) -> sp.csr_matrix:
    """Reassemble a CSR matrix from picklerick open_stream chunks (row order)."""
    data = np.concatenate([np.asarray(c.data) for c in chunks])
    indices = np.concatenate([np.asarray(c.indices) for c in chunks])
    indptr = [0]
    offset = 0
    for c in chunks:
        local = np.asarray(c.indptr)
        indptr.extend((local[1:] + offset).tolist())
        offset += int(local[-1])
    n_obs = sum(int(c.nrows) for c in chunks)
    return sp.csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.int64)), shape=(n_obs, n_vars)
    )
