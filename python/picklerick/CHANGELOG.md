# Changelog

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
