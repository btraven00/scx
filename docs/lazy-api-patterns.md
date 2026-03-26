# Future Ideas: Lazy Access Patterns for SCX

This note records a few future-looking ideas around lazy or low-overhead data
access in SCX.

It is intentionally brief and non-committal. The current product focus remains:

- robust H5Seurat ↔ H5AD interoperability
- bounded-memory conversion
- reproducible benchmarking
- thin R/Python bindings over the core engine

The ideas below are **not** blockers for the Python bindings milestone.

---

## Current product stance

SCX is an interop engine first, not a general lazy data platform.

That means:

- `read_*()` should stay simple and return normal ecosystem-native objects
- `write_*()` should stay conventional
- `convert()` should remain the primary engine operation
- exploratory low-overhead reopen paths should not complicate the main API

---

## Why lazy ideas are still interesting

Even though lazy access is not a near-term product goal, it is relevant for
research and benchmarking because it may help:

- reduce overhead in repeated benchmark loops
- isolate format read costs from write costs
- inspect intermediate representations more cheaply
- explore out-of-memory access patterns for very large datasets

In that sense, lazy access is best viewed as a **future systems/research
direction**, not as the current product center.

---

## Main design lesson from the broader ecosystem

Modern storage ecosystems usually separate:

- an **eager convenience API**
- from a **storage-facing lazy API**

Examples include dataset handles, scanners, chunked arrays, and batch iterators.

If SCX explores this space later, it should likely follow the same pattern:
keep eager `read_*()` simple, and add any lazy behavior through a separate,
explicit interface rather than overloading the main read path.

---

## Most plausible future direction

If SCX later adds a lazy or low-overhead access path, the cleanest approach is
likely:

1. keep the main bindings eager and boring;
2. use the internal `.npy` snapshot as a low-overhead checkpoint format;
3. expose any future lazy access through an explicit handle such as
   `open_dataset(...)`;
4. prefer batch iteration over row-wise iteration.

This would align well with the current product thesis while keeping lazy work
contained.

---

## Why the `.npy` snapshot is the natural experiment surface

The internal snapshot format is attractive for exploratory work because it is:

- simple
- easy to inspect
- lower-overhead than HDF5 for some benchmark tasks
- a natural fit for NumPy-backed reopen paths

So if SCX ever explores lazy access, the first serious experiment should likely
be built on top of the internal snapshot path, not on top of a more ambitious
new backend.

---

## Constraints to keep in mind

Any future lazy API should respect the same product guardrails as the rest of
SCX:

- do not turn SCX into a general data platform
- do not block core conversion milestones
- do not promise universal zero-copy behavior
- do not force feature symmetry across all language bindings
- do not expand the public API unless the benchmarking/interoperability value is clear

---

## Open questions for later

If this area becomes worth revisiting, the main questions are:

- Should lazy access remain internal/experimental, or become a small public API?
- Should snapshot creation be explicit, or hidden behind caching?
- Should the first lazy interface expose batches, chunks, or something else?
- How much metadata should load eagerly versus lazily?
- Which research use cases would justify the maintenance burden?

---

## Bottom line

Lazy access is a valid future idea for SCX, especially as a benchmarking and
systems experiment around the internal `.npy` snapshot format.

For now, it remains exactly that: a future idea.

The near-term priority is still a small, credible, well-tested interop engine
with thin language bindings.