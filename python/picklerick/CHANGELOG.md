# Changelog

## [0.7.0](https://github.com/btraven00/scx/compare/picklerick-v0.6.0...picklerick-v0.7.0) (2026-11-11)


### Miscellaneous Chores

* **picklerick:** Synchronize scx versions

## [0.6.0](https://github.com/btraven00/scx/compare/picklerick-v0.5.0...picklerick-v0.6.0) (2026-11-11)


### ⚠ BREAKING CHANGES

* **uns:** UnsTable has a new public field `arrays`. Struct literals need `..Default::default()`.

### Features

* AnnData Zarr v3 output and row-range reads (open_stream rows=/where=) ([0b78f11](https://github.com/btraven00/scx/commit/0b78f114364201e3be68b0f099609d8c9a4d8420))
* bixverse store input in the Python and R packages ([e10fede](https://github.com/btraven00/scx/commit/e10fede357755031f42f61935d88669ae3136f45))
* **cli:** show gene-major X in inspect; budget the transpose with --transpose-mem ([f5bda4e](https://github.com/btraven00/scx/commit/f5bda4ed4fa8f1f14ecbdc1b3caeca6f90907992))
* **h5mu:** read one MuData modality ([42e1abd](https://github.com/btraven00/scx/commit/42e1abdfd95834aa2d38e13847a576953d56fb55))
* **picklerick:** convert(gene_major=True), with tutorial sections on per-gene work ([e2d585d](https://github.com/btraven00/scx/commit/e2d585d2d7745619cac70ca0ebf8fc31339a6cd6))
* **picklerick:** inspect(path, names=True) returns obs and var names ([9944359](https://github.com/btraven00/scx/commit/994435989a50edc162208bdbf008acf875308b48))
* **picklerick:** MatrixChunk.to_scipy() without copying the arrays ([febc890](https://github.com/btraven00/scx/commit/febc89072b04f9abaf8a76453a584e05932a15db))
* **picklerick:** stream gene-major X by gene with open_stream(axis="var") ([03981b0](https://github.com/btraven00/scx/commit/03981b001bf17f9d2da44b2b0be217b1aba91cff))
* write the matrix to layers/&lt;name&gt; instead of /X (--x-slot for h5ad) ([bcfc791](https://github.com/btraven00/scx/commit/bcfc791ec63ad59b95a13c89ec0bb6a78717fd10))


### Bug Fixes

* **cli:** default --assay to the reader's choice, not RNA ([d0e3325](https://github.com/btraven00/scx/commit/d0e3325eeb51f02b3e847536eceed9b37bf79d88))
* **convert:** keep AnnData raw in h5ad and Zarr output ([8222696](https://github.com/btraven00/scx/commit/8222696a349fbbb3722a25b717e3b9a7baf43120))
* **core:** refuse gene-major input before writing, and transpose BPCells h5seurat ([156ae97](https://github.com/btraven00/scx/commit/156ae97890106ce8770d5d7d2254c9b8a5cd0253))
* **picklerick:** bring the native stub up to date with gene-major streaming ([fa46a32](https://github.com/btraven00/scx/commit/fa46a326f61e8b0af5b02e7e032cbf56152d1ef6))
* **picklerick:** install on Python 3.10 and later ([06c0ece](https://github.com/btraven00/scx/commit/06c0eced56e0682e6a99262abeb97ec8274cffb8))
* put h5ad raw.X into the h5seurat counts slot ([a257104](https://github.com/btraven00/scx/commit/a257104d8c71ecc3460e0ea4321dc7c8c236032f))
* **raw:** keep raw.var's columns, not just its index ([f6a5e61](https://github.com/btraven00/scx/commit/f6a5e61498b160dd761bf22576ee78bfa4a458c5))
* **uns:** carry large numeric uns arrays typed, not as JSON ([20ef448](https://github.com/btraven00/scx/commit/20ef448ba811e01de7060c6d63f92b8a8f337823))


### Performance Improvements

* **cli:** hash the source with blake3 by default (--source-hash) ([7247a06](https://github.com/btraven00/scx/commit/7247a06a8cf4e3581c1a4680671162ddad6c6b6e))
* **picklerick:** import anndata only when read() needs it ([3d588f0](https://github.com/btraven00/scx/commit/3d588f03dc8c3620eb08b3bb8e4643699c39f8a5))
* **picklerick:** skip the n_obs row mask when open_stream gets only rows= ([b841d6e](https://github.com/btraven00/scx/commit/b841d6e5d41777a8c85fa53817cfc8104ce18ff9))

## [0.5.0](https://github.com/btraven00/scx/compare/picklerick-v0.4.1...picklerick-v0.5.0) (2026-10-07)


### Miscellaneous Chores

* **picklerick:** Synchronize scx versions

## [0.4.1](https://github.com/btraven00/scx/compare/picklerick-v0.4.0...picklerick-v0.4.1) (2026-10-06)


### Miscellaneous Chores

* **picklerick:** Synchronize scx versions

## [0.4.0](https://github.com/btraven00/scx/compare/picklerick-v0.3.0...picklerick-v0.4.0) (2026-10-05)


### Features

* inspect stats in CLI + inspect() Python binding ([bc055be](https://github.com/btraven00/scx/commit/bc055be6fd1137f0ea96f81b8ce195c45bc4b309))
* **inspect:** add nnz/cell quartiles for layers and obsp ([e3b34ec](https://github.com/btraven00/scx/commit/e3b34ec1888db97314e8da307f855eee6b8ff468))
* **inspect:** show 'H5Seurat (BPCells)' for BPCells-backed h5seurat files ([a1ab313](https://github.com/btraven00/scx/commit/a1ab3135eb28207e05b1f3ed6d3f39313ed83cd0))
* python bindings ([8f01eb7](https://github.com/btraven00/scx/commit/8f01eb75bb1c98cf67282f0beca03712e9d1776c))
* **stream:** implement pk.open_stream() — Python streaming matrix iterator ([adb2783](https://github.com/btraven00/scx/commit/adb27834e4ede6c02c65eb7858a79223059deb7d))


### Bug Fixes

* **inspect:** show X nnz/cell quartiles for H5AD (and all formats) ([97655e0](https://github.com/btraven00/scx/commit/97655e0590893ad8509596117b21a32615130033))
* **picklerick:** resolve clippy warnings and profile placement ([9464c8e](https://github.com/btraven00/scx/commit/9464c8e6f78956bd27aae9876c254e66f967111c))
* suppress unused chunk_size warning in scx_inspect_native ([28a3ae7](https://github.com/btraven00/scx/commit/28a3ae79422f0982be932541881b322bcb5c1448))
