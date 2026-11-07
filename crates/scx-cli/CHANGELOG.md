# Changelog

## [0.6.0](https://github.com/btraven00/scx/compare/scx-cli-v0.5.0...scx-cli-v0.6.0) (2026-11-07)


### ⚠ BREAKING CHANGES

* **cli:** provenance `source.sha256` / `output.sha256` are now `source.hash` / `output.hash` with an algorithm prefix.
* **core:** ColumnData::Categorical has a new `ordered` field. Constructors must set it, and patterns that name fields must add `..`. In bixverse: add `..` to the Categorical pattern in src/rust/src/single_cell/r_sc_h5ad_arrow.rs, and set `ordered` in src/rust/src/single_cell/r_sc_h5ad.rs (factor_to_categorical), e.g.
* **core:** ColumnData has a new Int64 variant, so exhaustive matches on it need an extra arm. In bixverse that is one edit: src/rust/src/single_cell/r_sc_h5ad_arrow.rs needs a `ColumnData::Int64(v) => Arc::new(Int64Array::try_new(v.into(), nulls)?)` arm.

### Features

* add scx diff to compare two datasets slot by slot ([b1fe5b8](https://github.com/btraven00/scx/commit/b1fe5b8a9c0add9c92810f3d595d212a999b8a6a))
* AnnData Zarr v3 output and row-range reads (open_stream rows=/where=) ([0b78f11](https://github.com/btraven00/scx/commit/0b78f114364201e3be68b0f099609d8c9a4d8420))
* bixverse store reader (v3 and v4) behind the bixverse feature ([41aadd1](https://github.com/btraven00/scx/commit/41aadd1968803d01427b9c57189c6b6608948c84))
* **cli:** add convert --no-output-hash ([3be3958](https://github.com/btraven00/scx/commit/3be3958e5cef4fe266debc8dace1caf34dd7ad66))
* **cli:** convert --where COL=VALUE keeps only the matching cells ([a60d3d5](https://github.com/btraven00/scx/commit/a60d3d5f969420354d50c49d64af5b091f45c309))
* **cli:** convert reads stdin with `-` ([11add67](https://github.com/btraven00/scx/commit/11add677a20648bd58a78cb2db2f4c1eb5d80463))
* **cli:** inspect asks before downloading a URL over 1 GB ([d7febf7](https://github.com/btraven00/scx/commit/d7febf7ef37eb284baa392c40205031f2c7cf547))
* **cli:** provenance records source file name and algo-tagged hashes ([d2ecd86](https://github.com/btraven00/scx/commit/d2ecd868bec88fbd1bd36ee152fd5f31691e3d25))
* **cli:** scx --version, naming the net and zarr features ([4135bb9](https://github.com/btraven00/scx/commit/4135bb9ae112123620a6299bfbc1efb149f63ca4))
* **cli:** show gene-major X in inspect; budget the transpose with --transpose-mem ([f5bda4e](https://github.com/btraven00/scx/commit/f5bda4ed4fa8f1f14ecbdc1b3caeca6f90907992))
* **cli:** write obkit phase anchors when OBKIT_DIR is set ([61110ad](https://github.com/btraven00/scx/commit/61110adb5fa2ad480a101a0b84d5f3fc9d4e1599))
* **core:** read gzipped .rds.gz, and show what a non-Seurat .rds holds ([e76647c](https://github.com/btraven00/scx/commit/e76647cd87c076fdbc82c0d6031158fa3ec350f0))
* **core:** read per-cell-list Parquet beyond Tahoe's column names and arrow widths ([53cc756](https://github.com/btraven00/scx/commit/53cc756358ead47f4374a05c9045fa69da307969))
* **core:** read Seurat objects from .rds/.rda files ([3798c8b](https://github.com/btraven00/scx/commit/3798c8b490e2d34832c3bfed358079055a7bf981))
* **core:** write X gene-major with bounded memory ([6312be9](https://github.com/btraven00/scx/commit/6312be9e4cf4c3683c5bb235553c4de0cc66cf76))
* read any single-file format from a URL ([af6581d](https://github.com/btraven00/scx/commit/af6581d59eacbc72bcfcb816d4f42f7a59a55159))
* record the source hash of remote .rds inputs ([a290a74](https://github.com/btraven00/scx/commit/a290a7497375e299c231cedee18fb117582c8ea4))
* write the matrix to layers/&lt;name&gt; instead of /X (--x-slot for h5ad) ([bcfc791](https://github.com/btraven00/scx/commit/bcfc791ec63ad59b95a13c89ec0bb6a78717fd10))


### Bug Fixes

* **cli:** default --assay to the reader's choice, not RNA ([d0e3325](https://github.com/btraven00/scx/commit/d0e3325eeb51f02b3e847536eceed9b37bf79d88))
* **cli:** inspect .rds URLs instead of failing to detect them ([8250c90](https://github.com/btraven00/scx/commit/8250c908e44d4e8578065d68a6ebf87a3b29adce))
* **cli:** say a missing input file does not exist ([aadc07f](https://github.com/btraven00/scx/commit/aadc07f8b01645af9e252a8ab260970b5932572e))
* **core:** keep int64 obs/var columns instead of clamping to i32 ([4f5d30a](https://github.com/btraven00/scx/commit/4f5d30afd2f832437f2b0f6c8e2fca7e74380799))
* **core:** keep the ordered flag on categoricals ([ab1f93b](https://github.com/btraven00/scx/commit/ab1f93b9a91a6be06cab5ba860c7a2ff53bd6c20))
* **core:** read CSC layers and obsp instead of failing the whole file ([e9133bf](https://github.com/btraven00/scx/commit/e9133bfb3f9a3eda6f108f851b7b74cc276fb319))
* **core:** refuse gene-major input before writing, and transpose BPCells h5seurat ([156ae97](https://github.com/btraven00/scx/commit/156ae97890106ce8770d5d7d2254c9b8a5cd0253))
* gate the zarr merge test on the zarr feature ([ba16452](https://github.com/btraven00/scx/commit/ba1645243ff5ab673e8515d273cb6c7e2115dc04))
* **h5seurat:** copy chunks into datasets with the source's extents ([432dc76](https://github.com/btraven00/scx/commit/432dc76564f54f7b71eacbb3d18fd439e336f6d5))
* make scx validate reject unknown schema keys and exit 2 on errors ([1af32eb](https://github.com/btraven00/scx/commit/1af32eb37839963220b5eca52b7076ecfa12c937))
* put h5ad raw.X into the h5seurat counts slot ([a257104](https://github.com/btraven00/scx/commit/a257104d8c71ecc3460e0ea4321dc7c8c236032f))
* refuse to merge into a timestamped HDF5 file without opt-in ([ea5ed58](https://github.com/btraven00/scx/commit/ea5ed5868cbfd6ec61cad2773f49abbd8092845a))
* scx-cli bixverse-parquet enables bixverse; test the factory, sidecars and CLI ([a103372](https://github.com/btraven00/scx/commit/a1033721abbf682931d8c05295a90bbf68b39084))
* write merge added_at to the sidecar, not uns ([778e819](https://github.com/btraven00/scx/commit/778e819f10853167f0b4077f5e0c1d9aeb9727e1))


### Performance Improvements

* check rows while copying, find chunks in one pass, hash in the background ([8518587](https://github.com/btraven00/scx/commit/851858774f675094487068e6f4194a6b5ec46453))
* **cli:** hash blake3 output with parallel pread, not a whole-file mmap ([c986f28](https://github.com/btraven00/scx/commit/c986f284de0d26801bce58979ef480e1b50b16fa))
* **cli:** hash the source with blake3 by default (--source-hash) ([7247a06](https://github.com/btraven00/scx/commit/7247a06a8cf4e3581c1a4680671162ddad6c6b6e))
* convert holds one metadata frame at a time and hands freed memory back ([0cb70ef](https://github.com/btraven00/scx/commit/0cb70eff35c18857053fcafcdd41645a37f93212))
* copy an h5ad's X and raw.X into a dgCMatrix h5seurat as stored ([48eaa7d](https://github.com/btraven00/scx/commit/48eaa7dc8c313a58660e832f344f7781323c6d68))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * scx-core bumped from 0.5.0 to 0.6.0

## [0.5.0](https://github.com/btraven00/scx/compare/scx-cli-v0.4.1...scx-cli-v0.5.0) (2026-10-07)


### Miscellaneous Chores

* **scx-cli:** Synchronize scx versions


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * scx-core bumped from 0.4.1 to 0.5.0

## [0.4.1](https://github.com/btraven00/scx/compare/scx-cli-v0.4.0...scx-cli-v0.4.1) (2026-10-06)


### Miscellaneous Chores

* **scx-cli:** Synchronize scx versions


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * scx-core bumped from 0.4.0 to 0.4.1

## [0.4.0](https://github.com/btraven00/scx/compare/scx-cli-v0.3.0...scx-cli-v0.4.0) (2026-10-05)


### Features

* **concat:** obs-axis concatenation with anndata.concat() semantics ([913e908](https://github.com/btraven00/scx/commit/913e9086bfac86c862dd4ac97a51c3ea9af18f2b))
* **convert:** --only/--exclude slot filter for h5ad output ([bd29ecc](https://github.com/btraven00/scx/commit/bd29eccd8fdb6889de3e4625419f04b924ca532e))
* **zarr:** read AnnData Zarr stores (v2 and v3) ([34c986f](https://github.com/btraven00/scx/commit/34c986f80c41600421b7134ab449d851975a0f64))


### Bug Fixes

* **cli:** don't leave a partial output when convert fails ([d2ee056](https://github.com/btraven00/scx/commit/d2ee05640af18b9d72728838b3da6f15e97ab1bf))
* **core:** carry missing values through readers, writers, concat, merge and export ([33d096b](https://github.com/btraven00/scx/commit/33d096b2162b66b376f30bbac95fac86266ce657))
* **core:** open merge patch sources through the reader factory ([2c5aa03](https://github.com/btraven00/scx/commit/2c5aa030763b08a01233fc5b0fa1d2459f052aa7))
* fallback for no-X in file ([7fe856e](https://github.com/btraven00/scx/commit/7fe856ee76a964be6e6295ef09b3974db979f36c))
* **h5ad:** read anndata 0.13 nullable-string-array; survive NaN in inspect ([b55378c](https://github.com/btraven00/scx/commit/b55378c9b6f90400e57f78a43819bcc70c16813f))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * scx-core bumped from 0.3.0 to 0.4.0

## [0.3.0](https://github.com/btraven00/scx/compare/scx-cli-v0.2.0...scx-cli-v0.3.0) (2026-07-02)


### Features

* add --source-url flag to scx convert ([9cb7da2](https://github.com/btraven00/scx/commit/9cb7da24ed8e23ed00521bafecbde7b54898eca4))
* add provenance tracking with byte-level reproducibility ([b3be139](https://github.com/btraven00/scx/commit/b3be13992ac3015cee818fb812e5adb68582fc45))
* add scx validate command with YAML schema and multi-format support ([caae0a2](https://github.com/btraven00/scx/commit/caae0a25b1f53b3d4fd4be9ed427488917801ef2))
* auto-detect X/counts slot assignment for H5Seurat output ([663b29c](https://github.com/btraven00/scx/commit/663b29c8a719ba8ed75e7a0eda640eb1a69b007a))
* **bpcells:** BP-128 decoder, BpcellsDatasetReader, HDF5 routing, and test suite ([bd5602d](https://github.com/btraven00/scx/commit/bd5602dd3a25a90dc45d046dec265e635423f462))
* **cli:** colorize inspect output (owo-colors, TTY-aware) ([a50fa06](https://github.com/btraven00/scx/commit/a50fa06a07941dd40b76dc5c03c27708a7de2710))
* **export:** add scx export subcommand (obs/var/obsm → CSV or Parquet) ([c077769](https://github.com/btraven00/scx/commit/c077769466555ee13b2bd46fff70c547df298878))
* gate SeuratDisk scaffold behind --seuratdisk-compat flag ([ccf44eb](https://github.com/btraven00/scx/commit/ccf44eb23c2373cad5406aaacdd4b029a884f7c4))
* inspect stats in CLI + inspect() Python binding ([bc055be](https://github.com/btraven00/scx/commit/bc055be6fd1137f0ea96f81b8ce195c45bc4b309))
* **inspect:** add 10x HDF5 and plain HDF5 fallback support ([d2b2525](https://github.com/btraven00/scx/commit/d2b252523c5383c5c43005d17ba75eb03f1dc213))
* **inspect:** add nnz/cell quartiles for layers and obsp ([e3b34ec](https://github.com/btraven00/scx/commit/e3b34ec1888db97314e8da307f855eee6b8ff468))
* **inspect:** detect binary 0/1 columns and show counts instead of quartiles ([a2153f9](https://github.com/btraven00/scx/commit/a2153f9714850728ddbf1fb251256adef66aa3f9))
* **inspect:** show 'H5Seurat (BPCells)' for BPCells-backed h5seurat files ([a1ab313](https://github.com/btraven00/scx/commit/a1ab3135eb28207e05b1f3ed6d3f39313ed83cd0))
* **merge:** chunk 2 — create-mode merge, provenance write, CLI, inspect slots ([d3f72fc](https://github.com/btraven00/scx/commit/d3f72fc982970793e45ed830ea868cbcabf37d67))
* **merge:** obsp + uns slot patches; reject varp ([04f24c4](https://github.com/btraven00/scx/commit/04f24c468c56e6a722b3b9e748e8a371df23b886))
* **provenance:** show provenance in inspect, add --source-sha256 ([02b2e47](https://github.com/btraven00/scx/commit/02b2e47f850ee00e8e2d60ff0c8ff0cd0d101852))
* **snapshot:** stream NPY output, drop materialise_dataset ([d4e2036](https://github.com/btraven00/scx/commit/d4e20360d79689da4685b7757c84b5f22c160306))
* **stream:** implement pk.open_stream() — Python streaming matrix iterator ([adb2783](https://github.com/btraven00/scx/commit/adb27834e4ede6c02c65eb7858a79223059deb7d))
* **tenx:** stream 10x HDF5 through convert/snapshot/validate ([012ac0f](https://github.com/btraven00/scx/commit/012ac0f62bdfb712d6183f1e8b2b51d3b0eabf72))
* **tests:** integration tests for scx merge + export; fix HDF5 provenance ([751187d](https://github.com/btraven00/scx/commit/751187d242c91e597998b1396a9642f9d2019016))


### Bug Fixes

* add project root attr required by newer SeuratDisk; expose --project flag ([9da6a2a](https://github.com/btraven00/scx/commit/9da6a2abbee2f626edd1adfafaebb0ee58d82278))
* **inspect:** show (dense) for layers without indptr; fix H5Seurat X stats ([c39be14](https://github.com/btraven00/scx/commit/c39be14b09e2f5ac11700e964e8d12497c07b77f))
* **inspect:** show X nnz/cell quartiles for H5AD (and all formats) ([97655e0](https://github.com/btraven00/scx/commit/97655e0590893ad8509596117b21a32615130033))
* resolve all clippy warnings ([783755d](https://github.com/btraven00/scx/commit/783755d1bd706ed0e17e2284e56d978eeb97f5af))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * scx-core bumped from 0.1.0 to 0.3.0
