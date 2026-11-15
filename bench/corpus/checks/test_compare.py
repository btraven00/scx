"""snapshot.py + compare.py on synthetic corruptions (the h5ad read itself is
exercised by the slice run). Run with the readers env:
    pixi run -e test python bench/corpus/checks/test_compare.py
"""
import sys, tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
import compare, snapshot


def base():
    rng = np.random.default_rng(0)
    X = sp.random(20, 8, density=0.3, format="csr", random_state=0)
    X.data = np.round(X.data * 10) + 1  # counts
    obs = pd.DataFrame({
        "cond": pd.Categorical(rng.choice(["ctrl", "kd"], 20)),
        "n": rng.integers(0, 100, 20),
        "score": rng.random(20),
        "flag": rng.random(20) > 0.5,
        "donor": pd.array(rng.choice(["a", "b", None], 20), dtype=object),
    }, index=[f"bc{i}" for i in range(20)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(8)]),
                      obsm={"X_umap": rng.random((20, 2))})


def verdicts(a, b):
    with tempfile.TemporaryDirectory() as t:
        for x, d in ((a, "a"), (b, "b")):
            snapshot.write(x, f"{t}/{d}")  # in memory: write_h5ad would re-categorise strings
        return {p["property"]: p["status"] for p in compare.compare(compare.load(f"{t}/a"), compare.load(f"{t}/b"))}


a = base()
v = verdicts(a, base())
assert set(v.values()) == {"pass"}, v

b = base(); b.obs_names = [f"Cell_{i + 1}" for i in range(20)]
v = verdicts(a, b)
assert v["obs_names"] == "fail" and v["obs/cond"] == "pass" and v["matrix/X"] == "pass", v  # aligned by position

b = base(); b.obs.loc[b.obs_names[:5], "cond"] = "kd"; b.obs.loc[b.obs_names[5:10], "cond"] = "ctrl"
assert verdicts(a, b)["obs/cond"] == "fail"

b = base(); del b.obs["score"]
assert verdicts(a, b)["obs/score"] == "fail"

b = base(); b.obs["n"] = b.obs["n"].astype(float)
assert verdicts(a, b)["obs/n"] == "fail"  # int -> float: type class changed

b = base(); b.obs["cond"] = b.obs["cond"].astype(str)
assert verdicts(a, b)["obs/cond"] == "fail"  # categorical -> string

b = base(); b.obs["cond"] = b.obs["cond"].cat.reorder_categories(["kd", "ctrl"])
assert verdicts(a, b)["obs/cond"] == "note"  # level order only

# Typed levels (CELLxGENE var/feature_length, a category of ints).
a2 = base(); a2.obs["len"] = pd.Categorical(np.where(a2.obs["flag"], 517, 1015))
b = a2.copy(); b.obs["len"] = b.obs["len"].cat.rename_categories(str)
v = verdicts(a2, b)
assert v["obs/len"] == "fail", v  # int levels -> string levels
assert verdicts(a2, a2.copy())["obs/len"] == "pass"
with tempfile.TemporaryDirectory() as t:  # an R snapshot records no level type
    snapshot.write(a2, f"{t}/a"); snapshot.write(b, f"{t}/b")
    for f in Path(f"{t}/b/obs").glob("*.levels_kind"):
        f.unlink()
    v = {p["property"]: p["status"] for p in compare.compare(compare.load(f"{t}/a"), compare.load(f"{t}/b"))}
    assert v["obs/len"] == "note" and v["obs/cond"] == "pass", v

b = base(); b.X = b.X.astype(np.float32)
v = verdicts(a, b)
assert v["matrix/X"] == "pass" and v["matrix/X.dtype"] == "note", v

b = base(); b.X = b.X.copy(); b.X.data[0] += 1
assert verdicts(a, b)["matrix/X"] == "fail"

b = base()[::-1].copy()
v = verdicts(a, b)
assert v["obs_names"] == "fail" and v["obs/n"] == "pass" and v["matrix/X"] == "pass", v  # reordered, realigned

b = base(); b.obsm["umap"] = b.obsm.pop("X_umap")
assert verdicts(a, b)["obsm/X_umap"] == "pass"  # renamed: matched by value

b = base(); b.obsm["X_umap"] = b.obsm["X_umap"] + 1e-3
assert verdicts(a, b)["obsm/X_umap"] == "fail"

b = base(); del b.obsm["X_umap"]
assert verdicts(a, b)["obsm/X_umap"] == "fail"

b = base(); b.obs["extra"] = 1
assert verdicts(a, b)["obs/+"] == "note"

# Mutation control: each corruption a converter can make silently must fail, on a
# file with every slot filled; the untouched file must pass.
def rich():
    rng = np.random.default_rng(1)
    a = base()
    a.obs["big"] = np.arange(20, dtype=np.int64) + 2 ** 53  # float64 cannot tell these apart
    a.obs["stage"] = pd.Categorical(rng.choice(["early", "late"], 20), categories=["early", "late"], ordered=True)
    a.layers["counts"] = a.X * 3  # its own values: a matrix is matched by value, wherever it lands
    a.raw = a.copy()
    a.raw.var["symbol"] = [f"S{j}" for j in range(8)]
    a.varm["PCs"] = rng.random((8, 3))
    a.obsp["distances"] = sp.random(20, 20, density=0.2, format="csr", random_state=2)  # not symmetric
    a.obsm["sparse"] = sp.random(20, 4, density=0.5, format="csr", random_state=3)
    a.uns = {"neighbors": {"params": {"n_neighbors": 15, "method": "umap"}},
             "colors": np.array(["#ff0000", "#00ff00"]), "pca": {"variance": rng.random(3)},
             "rank": np.rec.fromarrays([np.array(["g1", "g2"]), np.array([1.5, 2.5])], names=["names", "scores"])}
    return a


def raw_var(b, f):
    var = b.raw.var.copy(); f(var)
    b.raw = ad.AnnData(X=b.raw.X, var=var, obs=pd.DataFrame(index=b.obs_names))


a = rich()
v = verdicts(a, rich())
assert set(v.values()) == {"pass"}, {k: s for k, s in v.items() if s != "pass"}
mutations = {  # name -> (corruption, property that must fail)
    "int64 truncated": (lambda b: b.obs.__setitem__("big", b.obs["big"].astype(np.float64).astype(np.int64) + 1), "obs/big"),
    "int64 wrapped to int32": (lambda b: b.obs.__setitem__("big", b.obs["big"].astype(np.int32)), "obs/big"),
    "ordered flag lost": (lambda b: b.obs.__setitem__("stage", b.obs["stage"].cat.as_unordered()), "obs/stage"),
    "layer dropped": (lambda b: b.layers.pop("counts"), "matrix/layers/counts"),
    "raw.var column dropped": (lambda b: raw_var(b, lambda var: var.pop("symbol")), "raw_var/symbol"),
    "varm dropped": (lambda b: b.varm.pop("PCs"), "varm/PCs"),
    "varm changed": (lambda b: b.varm.__setitem__("PCs", b.varm["PCs"] * 2), "varm/PCs"),
    "obsp dropped": (lambda b: b.obsp.pop("distances"), "obsp/distances"),
    "obsp transposed": (lambda b: b.obsp.__setitem__("distances", b.obsp["distances"].T.tocsr()), "obsp/distances"),
    "sparse obsm dropped": (lambda b: b.obsm.pop("sparse"), "obsm/sparse"),
    "uns key dropped": (lambda b: b.uns.pop("colors"), "uns/colors"),
    "uns nested value changed": (lambda b: b.uns["neighbors"]["params"].__setitem__("n_neighbors", 30),
                                 "uns/neighbors/params/n_neighbors"),
    "uns number became text": (lambda b: b.uns["neighbors"]["params"].__setitem__("n_neighbors", "15"),
                               "uns/neighbors/params/n_neighbors"),
    "uns record array dropped": (lambda b: b.uns.pop("rank"), "uns/rank/"),
}
for what, (corrupt, prop) in mutations.items():
    b = rich(); corrupt(b)
    v = verdicts(a, b)
    assert [s for k, s in v.items() if k.startswith(prop)] and all(s == "fail" for k, s in v.items() if k.startswith(prop)), (what, v)
    assert [k for k, s in v.items() if s == "fail" and not k.startswith(prop)] == [], (what, v)

# Not failures: a narrower integer with the same values, a record array read as a dict of arrays.
b = rich(); b.obs["n"] = b.obs["n"].astype(np.int32)
assert verdicts(a, b)["obs/n"] == "note"
b = rich(); b.uns["rank"] = {"names": np.array(["g1", "g2"]), "scores": np.array([1.5, 2.5])}
v = verdicts(a, b)
assert v["uns/rank/names"] == v["uns/rank/scores"] == "pass", v
# An R-read output has no raw.var, and a missing uns leaf is a note there.
with tempfile.TemporaryDirectory() as t:
    b = rich(); del b.uns["colors"]; raw_var(b, lambda var: var.pop("symbol"))
    snapshot.write(a, f"{t}/a"); snapshot.write(b, f"{t}/b")
    Path(f"{t}/b/reader.txt").write_text("R\n")
    v = {p["property"]: p["status"] for p in compare.compare(compare.load(f"{t}/a"), compare.load(f"{t}/b"))}
    assert v["uns/colors"] == "note" and "raw_var/symbol" not in v and "fail" not in v.values(), v
print("ok")
