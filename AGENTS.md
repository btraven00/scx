# AGENTS.md

## Purpose

SCX is a lean, format-to-format interoperability engine for single-cell data, optimized for reproducible benchmarking of conversion correctness, throughput, and memory use.

This repository is intentionally **not** trying to become a general single-cell platform. Contributors should preserve a narrow product thesis, prefer boring interfaces, and resist feature creep unless a proposed change materially strengthens the core interoperability and benchmarking story.

---

## Product Principles

### 1. Engine first
SCX exists first and foremost as a conversion engine.

Prioritize:
- correctness of format translation
- bounded-memory conversion behavior
- reproducible benchmarking
- clear observability and diagnostics
- small, testable interfaces

Do not prioritize broad platform ambitions over core engine quality.

### 2. Interop over platform
SCX moves data between ecosystems; it does not replace them.

That means:
- use native ecosystem objects at the edges when helpful
- avoid re-implementing large analysis frameworks
- avoid turning SCX into a general-purpose storage or compute platform

### 3. Bindings are thin
Language bindings should remain thin wrappers over the engine.

Bindings may provide ergonomic helpers, but should not:
- fork core conversion semantics
- introduce large amounts of language-specific business logic
- create pressure for perfect feature symmetry across R, Python, and any future language

### 4. Benchmarkability matters
A major feature should improve at least one of:
- conversion correctness confidence
- throughput measurement
- memory measurement
- debugging and reproducibility of performance claims

If a feature is interesting but hard to benchmark or justify against the core thesis, it belongs in design notes or experiments rather than the mainline roadmap.

### 5. Boring public API
Keep the user-facing API small, stable, and easy to explain.

Prefer:
- a few commands and functions with clear semantics
- explicit behavior over cleverness
- conventional names and defaults
- narrow FFI surfaces

Avoid:
- overlapping abstractions
- hidden magic
- exposing internal implementation details as public commitments too early

### 6. Experimental ideas stay experimental
Ideas such as lazy dataset handles, mmap-backed reopen paths, custom backed matrix abstractions, or new storage backends are valid areas of exploration, but they are not release blockers for the core engine.

Document them clearly, but do not let them destabilize the main product.

---

## Scope Guardrails

### In scope
The following are the core responsibilities of SCX:

- efficient conversion between supported single-cell formats
- correctness-preserving translation of matrices and metadata
- bounded-memory streaming where feasible
- a CLI suitable for reproducible benchmarking
- thin R/Python bindings that expose the engine ergonomically
- fixtures, tests, and benchmark workflows that support research claims

### Out of scope for the core product
The following are intentionally not core goals:

- becoming a full alternative to AnnData, Scanpy, Seurat, or BPCells
- becoming a general lazy matrix engine
- supporting every single-cell or cloud-native format
- guaranteeing perfect API symmetry across languages
- building a distributed or cloud-native data platform
- making universal zero-copy claims across all workflows
- turning internal snapshot/debug formats into mandatory user-facing workflows

These areas may be explored later, but should not drive near-term architecture.

---

## Roadmap Discipline

When proposing a feature, ask:

1. Does it strengthen the interoperability + benchmarking thesis?
2. Does it preserve a small public API?
3. Can it be tested with existing fixtures or straightforward new fixtures?
4. Can its memory/time behavior be explained simply?
5. If we had to remove it later, would the core product remain coherent?

If the answer to the first four is weak, or the answer to the last is no, the feature probably belongs in a design note rather than in the implementation roadmap.

---

## Design Preferences for Contributors

Because the project should remain understandable and maintainable, contributors should generally prefer:

- simple data flow over abstraction-heavy architectures
- narrow traits/interfaces over generalized plugin systems
- explicit resource ownership and lifecycle
- integration tests over clever internal machinery
- clear errors over clever recoveries
- stable defaults over many optional modes

Avoid premature generalization unless there is a demonstrated need in the current product scope.

---

## Internal Formats and Exploratory Work

Internal formats and exploratory mechanisms are allowed when they reduce benchmarking overhead or improve debugging.

Examples:
- intermediate snapshot formats used to isolate read/write costs
- fixture-generation helpers
- debug/inspection workflows

However, these should be framed as:
- internal tooling,
- experimental infrastructure, or
- research support machinery

They should not be treated as primary end-user product surfaces unless the product thesis is explicitly revised.

---

## Language Binding Policy

Bindings should reflect ecosystem-native expectations without forcing artificial parity.

For example:
- R may expose R-specific convenience functions
- Python may expose Python-specific convenience functions

What must remain aligned across bindings is the core conversion semantics and defaults, not every convenience wrapper.

---

## Default Contributor Heuristic

When in doubt, choose the option that is:
- smaller in scope,
- easier to benchmark,
- easier to test,
- easier to explain,
- and easier to delete later.

---

## Build & Packaging Gotchas

Two non-obvious facts that have each caused a build failure. Full detail in
`docs/packaging.md`; the short version for agents:

### The R package must vendor HDF5 *statically*
`picklerick-r` compiles to a Rust `staticlib` that R links into `picklerick.so`
via `src/Makevars` (which passes no `-lhdf5`). A Rust staticlib only bundles
**statically**-linked native deps, and cargo's dynamic-link directives do not
reach R's separate `.so` link step. So linking HDF5 *dynamically* (system
`libhdf5`) leaves its symbols undefined at load (`undefined symbol:
H5Sselect_elements`). The `conda.recipe/r-picklerick` recipe therefore must
**not** set `HDF5_DIR` or depend on system `hdf5`, and must include `cmake` +
a C++ compiler for the from-source HDF5 build. (CLI/Python packages *can* use
system HDF5 via `--no-default-features`; the R staticlib cannot.)

### scx-core is a pinned git dependency for the R package
`r/picklerick` depends on `scx-core` via a git dependency pinned to a rev on
`btraven00/scx` — not a path dep and not an in-tree copy. There is no vendored
`r/picklerick/src/rust/scx-core` and no `sync-scx-core.sh`; do not reintroduce
them. The committed `r/picklerick/src/rust/Cargo.lock` is required (keep
`hdf5-metno` pinned to a single-ndarray combination, currently `0.12.4`, or a
fresh resolve pulls ndarray 0.17 → `Selection`/`SliceInfo` skew). Bump the
pinned rev/tag when the R bindings need new scx-core changes.

That bias is usually the correct one for SCX.