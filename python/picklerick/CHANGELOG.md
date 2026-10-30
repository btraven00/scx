# Changelog

## [0.6.0](https://github.com/btraven00/scx/compare/picklerick-v0.5.0...picklerick-v0.6.0) (2026-10-30)


### Features

* AnnData Zarr v3 output and row-range reads (open_stream rows=/where=) ([0b78f11](https://github.com/btraven00/scx/commit/0b78f114364201e3be68b0f099609d8c9a4d8420))
* bixverse store input in the Python and R packages ([e10fede](https://github.com/btraven00/scx/commit/e10fede357755031f42f61935d88669ae3136f45))
* **picklerick:** inspect(path, names=True) returns obs and var names ([9944359](https://github.com/btraven00/scx/commit/994435989a50edc162208bdbf008acf875308b48))
* **picklerick:** MatrixChunk.to_scipy() without copying the arrays ([febc890](https://github.com/btraven00/scx/commit/febc89072b04f9abaf8a76453a584e05932a15db))
* write the matrix to layers/&lt;name&gt; instead of /X (--x-slot for h5ad) ([bcfc791](https://github.com/btraven00/scx/commit/bcfc791ec63ad59b95a13c89ec0bb6a78717fd10))


### Bug Fixes

* **picklerick:** install on Python 3.10 and later ([06c0ece](https://github.com/btraven00/scx/commit/06c0eced56e0682e6a99262abeb97ec8274cffb8))

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
