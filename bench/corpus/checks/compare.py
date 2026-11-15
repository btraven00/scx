#!/usr/bin/env python3
"""Compare two snapshots (snapshot.py / snapshot.R layout), input vs output,
property by property. Each property is pass, note or fail.

Fail: cell or gene names changed or reordered; an input cell embedding
(obsm) with no equal output embedding; an input obs/var column
missing; a column's type class changed (int/float/bool/string/categorical,
`ordered` and the levels' type included); values or NA mask changed; an input matrix with no
equal matrix in the output. Integer-valued matrices must match exactly, float
matrices within rtol 1e-6 (a f64 -> f32 downcast passes, and is noted).
Integer columns are compared as int64 when both snapshots hold them (exact
above 2^53); the same values in a narrower stored width are a note.
Also checked, when the input has them: raw.var columns (anndata outputs only),
gene loadings (varm, rows matched by gene name; a row the output lacks must be
all zero or NaN), pairwise matrices (obsp, varp) and every leaf of uns (kind
string/numeric, size, values). An uns leaf missing from an R-read output is a
note (Seurat's misc has no equivalent for much of uns), from an anndata-read
output a failure.
Note: categorical level order, typed levels read back by snapshot.R (R
levels are always strings), columns only in the output, float32 storage.
A passing matrix records where it landed (e.g. RNA/counts = X): slot names
differ by format, so the mapping is data for the report, not a verdict.

    compare.py <input_snapshot> <output_snapshot>   -> JSON list on stdout
"""
import json, sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

RTOL = 1e-6


def names(p):
    t = Path(p).read_text()
    return t.split("\n")[:-1] if t else []


def load(d):
    d = Path(d)
    snap = {"obs_names": names(d / "obs_names.txt"), "var_names": names(d / "var_names.txt"),
            "reader": (d / "reader.txt").read_text().strip() if (d / "reader.txt").exists() else None}
    for slot in ("obs", "var", "raw_var"):
        cols = {}
        for f in sorted((d / slot).glob("*.name"), key=lambda f: int(f.stem)):
            p = str(f)[: -len(".name")]
            cls = Path(p + ".class").read_text().strip()
            na = np.fromfile(p + ".na", dtype=np.uint8).astype(bool)
            if cls == "categorical":
                codes = np.fromfile(p + ".codes", dtype="<i4")
                lv = names(p + ".levels.txt")
                vals = np.array([lv[c] if c >= 0 else "" for c in codes], dtype=object)
                kind = Path(p + ".levels_kind")  # absent: snapshot.R (R levels are always strings)
                c = {"levels": lv, "ordered": Path(p + ".ordered").read_text().strip() == "1",
                     "levels_kind": kind.read_text().strip() if kind.exists() else None}
            elif cls == "string":
                vals, c = np.array(names(p + ".txt"), dtype=object), {}
            else:
                vals, c = np.fromfile(p + ".f64", dtype="<f8"), {}
                if Path(p + ".i64").exists():
                    c = {"i64": np.fromfile(p + ".i64", dtype="<i8"), "dtype": Path(p + ".dtype").read_text().strip()}
            cols[f.read_text().rstrip("\n")] = {"class": cls, "na": na, "values": vals, **c}
        snap[slot] = cols
    for slot in ("mats", "obsp", "varp"):
        mats = {}
        for f in sorted((d / slot).glob("*.name"), key=lambda f: int(f.stem)):
            p = str(f)[: -len(".name")]
            n_obs, n_vars = map(int, Path(p + ".shape").read_text().split())
            parts = [np.fromfile(f"{p}.{x}", dtype="<f8") for x in ("data", "indices", "indptr")]
            m = sp.csr_matrix((parts[0], parts[1].astype(np.int64), parts[2].astype(np.int64)), shape=(n_obs, n_vars))
            mats[f.read_text().rstrip("\n")] = {"m": m, "var_names": names(p + ".var_names.txt") if slot == "mats" else [],
                                                "dtype": Path(p + ".dtype").read_text().strip()}
        snap[slot] = mats
    for slot in ("obsm", "varm"):
        embs = {}
        for f in sorted((d / slot).glob("*.name"), key=lambda f: int(f.stem)):
            p = str(f)[: -len(".name")]
            shape = tuple(map(int, Path(p + ".shape").read_text().split()))
            m = np.fromfile(p + ".f64", dtype="<f8").reshape(shape)
            embs[f.read_text().rstrip("\n")] = m if slot == "obsm" else {"m": m, "rows": names(p + ".rows.txt")}
        snap[slot] = embs
    leaves = {}
    for f in sorted((d / "uns").glob("*.name"), key=lambda f: int(f.stem)):
        p = str(f)[: -len(".name")]
        kind = Path(p + ".kind").read_text().strip()
        shape = tuple(map(int, Path(p + ".shape").read_text().split()))
        if kind in ("string", "other"):
            t = Path(p + ".txt").read_text()
            vals = np.array(t.split("\x1e") if t or shape != (0,) else [], dtype=object)
        else:
            vals = np.fromfile(p + ".f64", dtype="<f8")
        leaves[f.read_text().rstrip("\n")] = {"kind": kind, "shape": shape, "values": vals}
    snap["uns"] = leaves
    return snap


def r(prop, status, detail=""):
    return {"property": prop, "status": status, "detail": detail}


def same_names(prop, a, b):
    """-> (result, permutation of b onto a, or None to align by position)."""
    if a == b:
        return r(prop, "pass"), None
    if sorted(a) == sorted(b):
        pos = {x: i for i, x in enumerate(b)}
        return r(prop, "fail", "same names, different order"), np.array([pos[x] for x in a]) if len(set(a)) == len(a) else None
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    return r(prop, "fail", f"{len(a)} vs {len(b)} names; {len(diff)} differ by position, e.g. {diff[:3]}"), None


def columns(slot, a, b, perm, r_reader=False):
    out = []
    for name, ca in a.items():
        prop = f"{slot}/{name}"
        cb = b.get(name)
        if cb is None:
            out.append(r(prop, "fail", "column dropped"))
            continue
        if len(cb["values"]) != len(ca["values"]):
            out.append(r(prop, "fail", f"length {len(ca['values'])} -> {len(cb['values'])}"))
            continue
        take = (lambda x: x[perm]) if perm is not None else (lambda x: x)
        vb, nb = take(cb["values"]), take(cb["na"])
        # R has no 64-bit integer: one that does not fit int32 can only come back as a double
        wide = r_reader and ca["class"] == "int" and cb["class"] == "float" and \
            np.abs(ca["values"]).max(initial=0) > 2 ** 31 - 1
        if not wide and (ca["class"] != cb["class"] or ca.get("ordered") != cb.get("ordered")):
            out.append(r(prop, "fail", f"type {ca['class']}{' ordered' if ca.get('ordered') else ''} -> "
                                       f"{cb['class']}{' ordered' if cb.get('ordered') else ''}"))
            continue
        ka, kb = ca.get("levels_kind"), cb.get("levels_kind")
        if ka and kb and ka != kb:
            out.append(r(prop, "fail", f"levels {ka} -> {kb}"))
            continue
        if not np.array_equal(ca["na"], nb):
            out.append(r(prop, "fail", f"NA mask differs in {int((ca['na'] != nb).sum())} rows"))
            continue
        ok = ~ca["na"]
        exact = "i64" in ca and "i64" in cb
        if exact:
            bad = take(cb["i64"])[ok] != ca["i64"][ok]
        elif ca["class"] == "float" or wide:
            bad = ~np.isclose(vb[ok], ca["values"][ok], rtol=RTOL, atol=0, equal_nan=True)
        else:
            bad = vb[ok] != ca["values"][ok]
        if bad.any():
            i = np.flatnonzero(ok)[np.flatnonzero(bad)[:3]]
            out.append(r(prop, "fail", f"{int(bad.sum())} values differ, e.g. "
                                       f"{[(str(ca['values'][j]), str(vb[j])) for j in i]}"))
        elif wide:
            out.append(r(prop, "note", "integers beyond int32 read back as doubles (R has no 64-bit integer)"))
        elif exact and ca["dtype"] != cb["dtype"]:
            out.append(r(prop, "note", f"stored as {cb['dtype']} (input {ca['dtype']}), values equal"))
        elif "i64" in ca and not exact and ok.any() and np.abs(ca["i64"][ok]).max() > 2 ** 53:
            out.append(r(prop, "note", "values above 2^53 compared as float64 (the reader has no int64)"))
        elif ka not in (None, "str") and kb is None:
            out.append(r(prop, "note", f"{ka} levels read back as strings (the reader has only string levels)"))
        elif ca["class"] == "categorical" and ca["levels"] != cb["levels"]:
            out.append(r(prop, "note", "level order differs" if sorted(ca["levels"]) == sorted(cb["levels"])
                         else f"levels {len(ca['levels'])} -> {len(cb['levels'])} (unused levels changed)"))
        else:
            out.append(r(prop, "pass"))
    extra = [n for n in b if n not in a]
    if extra:
        out.append(r(f"{slot}/+", "note", f"only in output: {extra}"))
    return out


def aligned(mb, rows, var_a):
    """mb's matrix with rows permuted onto a, columns onto var_a; None if impossible."""
    m = mb["m"] if rows is None else mb["m"][rows]
    if mb["var_names"] == var_a:
        return m
    if sorted(mb["var_names"]) == sorted(var_a) and len(set(var_a)) == len(var_a):
        pos = {x: i for i, x in enumerate(mb["var_names"])}
        return m[:, [pos[x] for x in var_a]]
    return m if m.shape[1] == len(var_a) else None  # names changed: by position


def equal(a, b):
    """-> (equal?, detail). Exact when a is integer-valued, else rtol."""
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    a, b = a.tocsr(), b.tocsr()
    a.sort_indices(); b.sort_indices()
    integral = np.array_equal(a.data, np.round(a.data))
    d = (a - b).tocsr()
    d.eliminate_zeros()
    if integral:
        return d.nnz == 0, f"{d.nnz} entries differ (exact, integer-valued)"
    over = (abs(d) - abs(a).maximum(abs(b)) * RTOL).tocsr()
    n = int((over.data > 0).sum())
    return n == 0, f"{n} entries differ beyond rtol {RTOL}"


def matrices(a, b, rows):
    out = []
    for name, ma in a["mats"].items():
        hits, misses = [], []
        for nb, mb in b["mats"].items():
            m = aligned(mb, rows, ma["var_names"])
            if m is None or m.shape != ma["m"].shape:
                continue
            ok, detail = equal(ma["m"], m)
            (hits if ok else misses).append(nb if ok else f"{nb}: {detail}")
        out.append(r(f"matrix/{name}", "pass", f"= {hits}") if hits else
                   r(f"matrix/{name}", "fail", "; ".join(misses) or "no output matrix of the same shape"))
    for nb, mb in b["mats"].items():
        if mb["dtype"] in ("float32",):
            out.append(r(f"matrix/{nb}.dtype", "note", "stored as float32"))
    return out


def embeddings(a, b, rows):
    """Each input cell embedding must equal (rtol) one in the output, rows
    aligned. Names differ by format (X_umap / umap), so the landing name is
    data, as for matrices; an output-only embedding is a note."""
    out, used = [], set()
    for name, ea in a["obsm"].items():
        hits = []
        for nb, eb in b["obsm"].items():
            eb = eb if rows is None else eb[rows]
            if eb.shape == ea.shape and np.allclose(ea, eb, rtol=RTOL, atol=0, equal_nan=True):
                hits.append(nb)
        used.update(hits)
        out.append(r(f"obsm/{name}", "pass", f"= {hits}") if hits else
                   r(f"obsm/{name}", "fail", f"no equal output embedding; output has "
                                             f"{ {n: e.shape for n, e in b['obsm'].items()} }"))
    extra = [n for n in b["obsm"] if n not in used]
    if extra:
        out.append(r("obsm/+", "note", f"only in output: {extra}"))
    return out


def loadings(a, b):
    """Each input gene loading (varm) must equal one in the output on the
    genes both name; a gene the output lacks must be all zero or NaN in the
    input (Seurat keeps loadings for variable genes only, anndata pads)."""
    out = []
    for name, la in a["varm"].items():
        hits = []
        for nb, lb in b["varm"].items():
            if lb["m"].shape[1] != la["m"].shape[1] or not la["rows"] or not lb["rows"]:
                continue
            pos = {x: i for i, x in enumerate(lb["rows"])}
            there = np.array([x in pos for x in la["rows"]])
            if not there.any():
                continue
            rest = la["m"][~there]
            mb = lb["m"][[pos[x] for x, t in zip(la["rows"], there) if t]]
            if np.allclose(la["m"][there], mb, rtol=RTOL, atol=0, equal_nan=True) and \
                    not np.nan_to_num(rest).any():
                hits.append(nb)
        out.append(r(f"varm/{name}", "pass", f"= {hits}") if hits else
                   r(f"varm/{name}", "fail", f"no equal output loading; output has "
                                             f"{ {n: l['m'].shape for n, l in b['varm'].items()} }"))
    return out


def pairwise(slot, a, b, perm):
    """Each input obsp / varp matrix must equal one in the output, both axes
    aligned. One that only matches transposed fails, and says so."""
    out = []
    for name, ma in a[slot].items():
        hits, misses = [], []
        for nb, mb in b[slot].items():
            m = mb["m"] if perm is None else mb["m"][perm][:, perm]
            if m.shape != ma["m"].shape:
                continue
            ok, detail = equal(ma["m"], m)
            if not ok and equal(ma["m"], m.T.tocsr())[0]:
                detail = "equal to the transpose"
            (hits if ok else misses).append(nb if ok else f"{nb}: {detail}")
        out.append(r(f"{slot}/{name}", "pass", f"= {hits}") if hits else
                   r(f"{slot}/{name}", "fail", "; ".join(misses) or "no output matrix of the same shape"))
    return out


def uns(a, b):
    numeric = ("int", "float", "bool")
    squeeze = lambda s: tuple(n for n in s if n != 1)
    out = []
    for path, ua in a["uns"].items():
        prop = f"uns/{path}"
        ub = b["uns"].get(path)
        if ua["kind"] == "other":
            out.append(r(prop, "note", f"not compared ({''.join(ua['values'])})"))
        elif ub is None:
            out.append(r(prop, "note", "entry dropped (the reader's format has no place for it)")
                       if b["reader"] == "R" else r(prop, "fail", "entry dropped"))
        elif (ua["kind"] in numeric) != (ub["kind"] in numeric):
            out.append(r(prop, "fail", f"type {ua['kind']} -> {ub['kind']}"))
        elif ua["values"].size != ub["values"].size:
            out.append(r(prop, "fail", f"shape {ua['shape']} -> {ub['shape']}"))
        elif not (np.allclose(ua["values"], ub["values"], rtol=RTOL, atol=0, equal_nan=True)
                  if ua["kind"] in numeric else np.array_equal(ua["values"], ub["values"])):
            out.append(r(prop, "fail", "values differ"))
        elif ua["kind"] != ub["kind"] or squeeze(ua["shape"]) != squeeze(ub["shape"]):
            out.append(r(prop, "note", f"{ua['kind']} {ua['shape']} read back as {ub['kind']} {ub['shape']}, values equal"))
        else:
            out.append(r(prop, "pass"))
    extra = [n for n in b["uns"] if n not in a["uns"]]
    if extra:
        out.append(r("uns/+", "note", f"only in output: {extra[:20]}"))
    return out


def compare(a, b):
    res, rows = same_names("obs_names", a["obs_names"], b["obs_names"])
    out = [res]
    res, vperm = same_names("var_names", a["var_names"], b["var_names"])
    out.append(res)
    out += columns("obs", a["obs"], b["obs"], rows, b["reader"] == "R")
    out += columns("var", a["var"], b["var"], vperm, b["reader"] == "R")
    out += matrices(a, b, rows)
    out += embeddings(a, b, rows)
    if b["reader"] != "R":  # Seurat keeps feature metadata per assay: no raw.var to read back
        out += columns("raw_var", a["raw_var"], b["raw_var"], None)
    out += loadings(a, b)
    out += pairwise("obsp", a, b, rows)
    out += pairwise("varp", a, b, vperm)
    out += uns(a, b)
    return out


if __name__ == "__main__":
    print(json.dumps(compare(load(sys.argv[1]), load(sys.argv[2])), indent=1))
