#!/usr/bin/env python3
"""Snapshot an .h5ad with anndata, the format's own reader, into the neutral
layout snapshot.R also writes, so compare.py can check R- and Python-read
files property by property:

  obs_names.txt  var_names.txt                 one name per line
  obs/<k>.* var/<k>.*  per column k (0-based, column order):
      .name  .class (int|float|bool|string|categorical)  .na (uint8, 1 = NA)
      .f64 (float64 LE; int/float/bool)  .txt (string)
      .codes (int32 LE, 0-based, -1 = NA) .levels.txt .ordered (categorical)
  mats/<k>.*  per matrix: .name .shape ("n_obs n_vars") .dtype .var_names.txt
      .indptr (float64) .indices (float64) .data (float64): CSR, cells x genes
      int columns also: .i64 (int64 LE, exact) .dtype (stored width, e.g. int32)
  raw_var/<k>.*  raw.var columns, as obs/var
  obsp/<k>.* varp/<k>.*  pairwise matrices, as mats (no .var_names.txt)
  obsm/<k>.*  per cell embedding: .name .shape ("n_obs n_dims")
      .f64 (float64 LE, row-major, cells x dims)
  varm/<k>.*  per gene loading, as obsm, plus .rows.txt (its gene names)
  uns/<k>.*  per leaf of uns, nested keys joined with "/": .name
      .kind (int|float|bool|string|other) .shape (space-separated, empty = scalar)
      .f64 (numeric, row-major) or .txt (strings, separated by \x1e; other: the type)
  reader.txt  "anndata" or "R": compare.py is lenient where R has no equivalent

    snapshot.py <file.h5ad> <out_dir>
"""
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def lines(p, xs):
    Path(p).write_text("".join(f"{x}\n" for x in xs))


def levels_kind(dt):
    """The type of a categorical's levels: int, float, bool or str."""
    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    if pd.api.types.is_integer_dtype(dt):
        return "int"
    if pd.api.types.is_float_dtype(dt):
        return "float"
    return "str"


def column(d, k, name, s):
    p = d / str(k)
    lines(f"{p}.name", [name])
    na = s.isna().to_numpy()
    na.astype(np.uint8).tofile(f"{p}.na")
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        cls = "categorical"
        s.cat.codes.to_numpy().astype("<i4").tofile(f"{p}.codes")
        lines(f"{p}.levels.txt", [str(c) for c in s.cat.categories])
        lines(f"{p}.ordered", [int(dt.ordered)])
        lines(f"{p}.levels_kind", [levels_kind(dt.categories.dtype)])
    elif pd.api.types.is_bool_dtype(dt):
        cls = "bool"
    elif pd.api.types.is_integer_dtype(dt):
        cls = "int"
    elif pd.api.types.is_float_dtype(dt):
        cls = "float"
    else:
        cls = "string"
        lines(f"{p}.txt", ["" if m else str(v) for v, m in zip(s, na)])
    if cls in ("int", "float", "bool"):
        v = s.astype("float64").to_numpy(dtype="<f8", copy=True)
        v[na] = 0
        v.tofile(f"{p}.f64")
    if cls == "int":  # float64 is exact only up to 2^53
        lines(f"{p}.dtype", [str(dt).lower()])
        s.fillna(0).to_numpy(dtype="<i8").tofile(f"{p}.i64")
    lines(f"{p}.class", [cls])


def matrix(d, k, name, m, var_names):
    p = d / str(k)
    dtype = str(m.dtype)
    m = sp.csr_matrix(m)
    m.sort_indices()
    lines(f"{p}.name", [name])
    lines(f"{p}.shape", [f"{m.shape[0]} {m.shape[1]}"])
    lines(f"{p}.dtype", [dtype])
    lines(f"{p}.var_names.txt", var_names)
    for part in ("indptr", "indices", "data"):
        getattr(m, part).astype("<f8").tofile(f"{p}.{part}")


def embedding(d, k, name, m, rows=None):
    p = d / str(k)
    m = np.asarray(m.toarray() if sp.issparse(m) else m, dtype="<f8")
    lines(f"{p}.name", [name])
    if rows is not None:
        lines(f"{p}.rows.txt", rows)
    lines(f"{p}.shape", [f"{m.shape[0]} {m.shape[1]}"])
    np.ascontiguousarray(m).tofile(f"{p}.f64")


def uns_leaves(x, path=""):
    """-> (path, kind, array) per leaf. Tables and record arrays give one leaf
    per column, a sparse matrix its three CSR arrays: readers disagree on the
    container, not on the values."""
    sub = lambda k: f"{path}/{k}" if path else str(k)
    if isinstance(x, Mapping):
        for k, v in x.items():
            yield from uns_leaves(v, sub(k))
    elif isinstance(x, pd.DataFrame):
        for c in x.columns:
            col = x[c]
            yield from uns_leaves(col.astype(object).to_numpy() if isinstance(col.dtype, pd.CategoricalDtype)
                                  else col.to_numpy(), sub(c))
    elif sp.issparse(x):
        m = sp.csr_matrix(x)
        m.sort_indices()
        for part in ("data", "indices", "indptr"):
            yield sub(part), "float" if part == "data" else "int", getattr(m, part)
    elif isinstance(x, np.ndarray) and x.dtype.names:
        for f in x.dtype.names:
            yield from uns_leaves(x[f], sub(f))
    else:
        try:
            a = np.asarray(x)
        except Exception:
            a = np.asarray(None)
        kind = {"b": "bool", "i": "int", "u": "int", "f": "float", "U": "string", "S": "string", "T": "string"}.get(a.dtype.kind)
        if a.dtype.kind == "O" and all(isinstance(v, str) or v is None or v is pd.NA or v != v for v in a.ravel()):
            kind = "string" if a.size and any(isinstance(v, str) for v in a.ravel()) else None
        yield path, kind or "other", a if kind else np.asarray(type(x).__name__)


def uns(d, tree):
    for k, (path, kind, a) in enumerate(uns_leaves(tree)):
        p = d / str(k)
        lines(f"{p}.name", [path])
        lines(f"{p}.kind", [kind])
        lines(f"{p}.shape", [" ".join(map(str, a.shape))])
        if kind in ("string", "other"):
            Path(f"{p}.txt").write_text("\x1e".join(
                v.decode() if isinstance(v, bytes) else v if isinstance(v, str) else "" for v in a.ravel().tolist()))
        else:
            np.ascontiguousarray(a, dtype="<f8").tofile(f"{p}.f64")


def main(f, out):
    import anndata
    write(anndata.read_h5ad(f), out)


def write(a, out):
    out = Path(out)
    for sub in ("obs", "var", "raw_var", "mats", "obsp", "varp", "obsm", "varm", "uns"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    lines(out / "reader.txt", ["anndata"])
    lines(out / "obs_names.txt", a.obs_names)
    lines(out / "var_names.txt", a.var_names)
    for slot, df in (("obs", a.obs), ("var", a.var)):
        for k, c in enumerate(df.columns):
            column(out / slot, k, str(c), df[c])
    mats = [("X", a.X, a.var_names)] if a.X is not None else []
    # anndata >= 0.13 lists X as layers[None]; it is already "X".
    mats += [(f"layers/{n}", a.layers[n], a.var_names) for n in a.layers if n is not None]
    if a.raw is not None:
        mats.append(("raw/X", a.raw.X, a.raw.var_names))
    for k, (n, m, v) in enumerate(mats):
        matrix(out / "mats", k, n, m, v)
    # A dataframe in obsm has no single numeric form: not compared.
    embs = [(n, m) for n, m in a.obsm.items() if not isinstance(m, pd.DataFrame) and np.ndim(m) == 2]
    for k, (n, m) in enumerate(embs):
        embedding(out / "obsm", k, n, m)
    embs = [(n, m) for n, m in a.varm.items() if not isinstance(m, pd.DataFrame) and np.ndim(m) == 2]
    for k, (n, m) in enumerate(embs):
        embedding(out / "varm", k, n, m, a.var_names)
    for slot, pairs in (("obsp", a.obsp), ("varp", a.varp)):
        for k, (n, m) in enumerate(pairs.items()):
            matrix(out / slot, k, n, m, [])
    if a.raw is not None:
        for k, c in enumerate(a.raw.var.columns):
            column(out / "raw_var", k, str(c), a.raw.var[c])
    uns(out / "uns", a.uns)


if __name__ == "__main__":
    main(*sys.argv[1:3])
