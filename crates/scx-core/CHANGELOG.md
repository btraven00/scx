# Changelog

## [0.3.0](https://github.com/btraven00/scx/compare/scx-core-v0.2.0...scx-core-v0.3.0) (2026-07-02)


### Features

* add --source-url flag to scx convert ([9cb7da2](https://github.com/btraven00/scx/commit/9cb7da24ed8e23ed00521bafecbde7b54898eca4))
* add provenance tracking with byte-level reproducibility ([b3be139](https://github.com/btraven00/scx/commit/b3be13992ac3015cee818fb812e5adb68582fc45))
* add scx validate command with YAML schema and multi-format support ([caae0a2](https://github.com/btraven00/scx/commit/caae0a25b1f53b3d4fd4be9ed427488917801ef2))
* **bpcells:** BP-128 decoder, BpcellsDatasetReader, HDF5 routing, and test suite ([bd5602d](https://github.com/btraven00/scx/commit/bd5602dd3a25a90dc45d046dec265e635423f462))
* gate SeuratDisk scaffold behind --seuratdisk-compat flag ([ccf44eb](https://github.com/btraven00/scx/commit/ccf44eb23c2373cad5406aaacdd4b029a884f7c4))
* **inspect:** add 10x HDF5 and plain HDF5 fallback support ([d2b2525](https://github.com/btraven00/scx/commit/d2b252523c5383c5c43005d17ba75eb03f1dc213))
* **merge:** chunk 1 — open_for_append, merge module scaffold, slot selector parser ([ccc4df0](https://github.com/btraven00/scx/commit/ccc4df0c206a75e00097c90e768310c09106a19e))
* **merge:** chunk 2 — create-mode merge, provenance write, CLI, inspect slots ([d3f72fc](https://github.com/btraven00/scx/commit/d3f72fc982970793e45ed830ea868cbcabf37d67))
* **merge:** chunk 3 — obs/var column + obsm/varm dense patch application ([85c523b](https://github.com/btraven00/scx/commit/85c523ba0d3812ca62848492fdd72f17d7aa099f))
* **merge:** implement append mode and shared patch loop ([a005d6c](https://github.com/btraven00/scx/commit/a005d6cf082e7a44bd740b012baf225f809d2d22))
* **merge:** obsp + uns slot patches; reject varp ([04f24c4](https://github.com/btraven00/scx/commit/04f24c468c56e6a722b3b9e748e8a371df23b886))
* **picklerick:** feature parity with anndataR — dense X/layers, nullable columns, write_h5seurat, read_seurat/read_sce ([7685754](https://github.com/btraven00/scx/commit/7685754e436023a5c4323ff79291c4ae7ffb8019))
* **provenance:** show provenance in inspect, add --source-sha256 ([02b2e47](https://github.com/btraven00/scx/commit/02b2e47f850ee00e8e2d60ff0c8ff0cd0d101852))
* **scx-core:** public Rust write API (`scx_core::api::write`) ([a4b6c41](https://github.com/btraven00/scx/commit/a4b6c41e701aab9ce0e26962c0f632d9c85d761f))
* **scx-core:** streaming BPCells writer (no O(nnz) buffer) ([a10e28a](https://github.com/btraven00/scx/commit/a10e28abb378918c3b17954c45a4ce3df4c674fe))
* **snapshot:** stream NPY output, drop materialise_dataset ([d4e2036](https://github.com/btraven00/scx/commit/d4e20360d79689da4685b7757c84b5f22c160306))
* **stream:** implement pk.open_stream() — Python streaming matrix iterator ([adb2783](https://github.com/btraven00/scx/commit/adb27834e4ede6c02c65eb7858a79223059deb7d))
* **tenx:** stream 10x HDF5 through convert/snapshot/validate ([012ac0f](https://github.com/btraven00/scx/commit/012ac0f62bdfb712d6183f1e8b2b51d3b0eabf72))
* **tests:** integration tests for scx merge + export; fix HDF5 provenance ([751187d](https://github.com/btraven00/scx/commit/751187d242c91e597998b1396a9642f9d2019016))


### Bug Fixes

* add key attr to assay group; fix double-create of pre-built groups ([e892644](https://github.com/btraven00/scx/commit/e8926444dc8338db29c226cc5bbeedee2a1a42ce))
* add project root attr required by newer SeuratDisk; expose --project flag ([9da6a2a](https://github.com/btraven00/scx/commit/9da6a2abbee2f626edd1adfafaebb0ee58d82278))
* always create graphs group in H5Seurat output ([e542629](https://github.com/btraven00/scx/commit/e542629c59c47eb0a1ea14fac9ea44e86ac2b1cb))
* always create reductions group in H5Seurat output ([c1954b7](https://github.com/btraven00/scx/commit/c1954b78ee8009cec13e68dfd0173f16304abe43))
* create images group required by newer SeuratDisk (spatial slot) ([356a776](https://github.com/btraven00/scx/commit/356a776c08b942987f9877dd1351406748242021))
* **detect:** recognise lean BPCells-mode h5seurat output ([8d73e4f](https://github.com/btraven00/scx/commit/8d73e4fc959d8edfe6f7e98221bd45a3389f247b))
* **h5ad:** read unsigned obs columns and u32 X (round-trip the writer's output) ([647c896](https://github.com/btraven00/scx/commit/647c8961d54d14fc04dbebdef69a49a1f8db4692))
* **inspect:** show X nnz/cell quartiles for H5AD (and all formats) ([97655e0](https://github.com/btraven00/scx/commit/97655e0590893ad8509596117b21a32615130033))
* **inspect:** show X nnz/cell stats for H5Seurat ([8759b68](https://github.com/btraven00/scx/commit/8759b68a02d436373ae66e3f2205569e95d23d81))
* per-run BPCells encoding — chunk boundaries must align with run boundaries ([be14141](https://github.com/btraven00/scx/commit/be14141bd6ecdafc3d2adc55ed79faa66bbd07c4))
* pre-create all SeuratDisk-required top-level groups in one pass ([be6b40c](https://github.com/btraven00/scx/commit/be6b40c6f98cb03ed7b6c163b5355ade9d59d55f))
* pre-create all top-level groups required by SeuratDisk (misc) ([e2fa100](https://github.com/btraven00/scx/commit/e2fa100b42a17bd3aa5e12ceff7d6d783a702dd9))
* replace deprecated into_raw_vec() with into_raw_vec_and_offset().0 ([02cb291](https://github.com/btraven00/scx/commit/02cb2919930eaf9a77f83ddf9d9005c9bcdfe8be))
* resolve all clippy warnings ([783755d](https://github.com/btraven00/scx/commit/783755d1bd706ed0e17e2284e56d978eeb97f5af))
* write version and active.assay root attrs required by SeuratDisk ([94dc106](https://github.com/btraven00/scx/commit/94dc106af1ec559deb55688f041b62c487f0776b))


### Performance Improvements

* **npy,h5ad:** eliminate redundant copies during NPY→h5ad conversion ([51f18aa](https://github.com/btraven00/scx/commit/51f18aa1204d8c7e1c7f87242e1478f11e4c111c))
* **npy:** mmap NPY files instead of fs::read, eliminating two copies ([45466cd](https://github.com/btraven00/scx/commit/45466cd2be58916fc7fd1783d70fd005abde14bd))
* Rayon parallelism for BPCells decode and write-path type conversion ([3b9543d](https://github.com/btraven00/scx/commit/3b9543de68fcf26e211cbf45791ea188947f4e70))
