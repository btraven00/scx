# Why scx reads `.rds` but does not write it

Status: decided, won't fix (2026-11-05).

scx reads Seurat `.rds` / `.rda` files (counts and meta.data, v3/v4 `Assay`
and v5 `Assay5`). It does not write them, and a writer will not be added.
This note explains why and lists the routes into Seurat that scx supports
instead.

## `.rds` is not an interchange format

`saveRDS` serializes an R object from memory. Its only specification is R's
`src/main/serialize.c`. The format describes R values (vectors, attributes,
S4 slots), not single-cell data. A Seurat `.rds` file is valid only if its
objects match the class definitions of the SeuratObject version that loads
it. Those classes are SeuratObject's internals and change between releases.
For example, v5 replaced `Assay` with `Assay5`, added layers and `LogMap`
cells/features, and changed which slots are required.

To write a Seurat `.rds`, scx would have to build those objects by hand,
outside R: every slot, class attribute, `package_version` and factor, for
each SeuratObject release. Mistakes don't show up at write time. The file
loads, and the object is wrong, or fails `validObject()`, or breaks the
first Seurat function that touches a slot. Getting meta.data column types
or missing values slightly wrong is the usual way this goes.

Reading is different. The GEO archive and other public data are full of
Seurat `.rds` files. Reading them connects scx to existing data, and a
reader can reject what it does not understand with a clear error. A writer
only produces more files that only R can open.

## What to use instead

| Goal | Route |
|---|---|
| Seurat object in an R session | picklerick `read_h5ad(path, as = "Seurat")`: SeuratObject's own constructor builds the object, so it is valid by construction |
| On-disk matrix that Seurat v5 can use | `scx convert in.h5ad out.bpcells`, then `BPCells::open_matrix_dir()` as a layer of an `Assay5` |
| Seurat-style HDF5 file | `scx convert in.h5ad out.h5seurat` (dgCMatrix or BPCells-backed) |
| A `.rds` file | Any of the above, then `saveRDS()` in R |

Seurat v5 keeps layers in memory as `dgCMatrix` by default. BPCells-backed
layers are an opt-in for data that doesn't fit in memory. In both cases, R
builds the object and SeuratObject checks it.

## Benchmarking tools that write `.rds`

A conversion that ends in `saveRDS()` spends much of its time on gzip
compression, which runs on one thread. On a 100k-cell HLCA subset,
`saveRDS` took about 116 of the 182 seconds of an h5ad → Seurat `.rds`
conversion. Reading the result back with `readRDS` also loads the whole
object into memory. When comparing scx with such tools, report the
read/convert phase separately from the write, or compare against a
BPCells-backed Seurat output. Otherwise the comparison mostly measures
`.rds` compression.

## Reopening this decision

A Seurat `.rds` writer would need all of the following:

- a release-tracking test matrix: write with scx, then `readRDS()` and
  `validObject()` under each supported SeuratObject release, in CI;
- a user need that no route in the table above covers;
- a maintainer willing to follow SeuratObject's class changes.
