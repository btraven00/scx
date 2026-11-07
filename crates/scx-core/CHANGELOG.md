# Changelog

## [0.6.0](https://github.com/btraven00/scx/compare/scx-core-v0.5.0...scx-core-v0.6.0) (2026-11-07)


### ⚠ BREAKING CHANGES

* **cli:** provenance `source.sha256` / `output.sha256` are now `source.hash` / `output.hash` with an algorithm prefix.
* **core:** ColumnData::Categorical has a new `ordered` field. Constructors must set it, and patterns that name fields must add `..`. In bixverse: add `..` to the Categorical pattern in src/rust/src/single_cell/r_sc_h5ad_arrow.rs, and set `ordered` in src/rust/src/single_cell/r_sc_h5ad.rs (factor_to_categorical), e.g.
* **core:** ColumnData has a new Int64 variant, so exhaustive matches on it need an extra arm. In bixverse that is one edit: src/rust/src/single_cell/r_sc_h5ad_arrow.rs needs a `ColumnData::Int64(v) => Arc::new(Int64Array::try_new(v.into(), nulls)?)` arm.

### Features

* add scx diff to compare two datasets slot by slot ([b1fe5b8](https://github.com/btraven00/scx/commit/b1fe5b8a9c0add9c92810f3d595d212a999b8a6a))
* AnnData Zarr v3 output and row-range reads (open_stream rows=/where=) ([0b78f11](https://github.com/btraven00/scx/commit/0b78f114364201e3be68b0f099609d8c9a4d8420))
* bixverse store reader (v3 and v4) behind the bixverse feature ([41aadd1](https://github.com/btraven00/scx/commit/41aadd1968803d01427b9c57189c6b6608948c84))
* bixverse-sc-data 0.9.1 without the analysis stack ([b606174](https://github.com/btraven00/scx/commit/b606174ef7f1b4d1d096335f0163e493434c5a63))
* **cli:** convert reads stdin with `-` ([11add67](https://github.com/btraven00/scx/commit/11add677a20648bd58a78cb2db2f4c1eb5d80463))
* **cli:** inspect asks before downloading a URL over 1 GB ([d7febf7](https://github.com/btraven00/scx/commit/d7febf7ef37eb284baa392c40205031f2c7cf547))
* **cli:** provenance records source file name and algo-tagged hashes ([d2ecd86](https://github.com/btraven00/scx/commit/d2ecd868bec88fbd1bd36ee152fd5f31691e3d25))
* **cli:** show gene-major X in inspect; budget the transpose with --transpose-mem ([f5bda4e](https://github.com/btraven00/scx/commit/f5bda4ed4fa8f1f14ecbdc1b3caeca6f90907992))
* **core:** keep R ordered factors as ordered categoricals when reading .rds ([5d5e61b](https://github.com/btraven00/scx/commit/5d5e61b22a74dfad1667bf96e78d0bdbffb9f4b4))
* **core:** read gzipped .rds.gz, and show what a non-Seurat .rds holds ([e76647c](https://github.com/btraven00/scx/commit/e76647cd87c076fdbc82c0d6031158fa3ec350f0))
* **core:** read per-cell-list Parquet beyond Tahoe's column names and arrow widths ([53cc756](https://github.com/btraven00/scx/commit/53cc756358ead47f4374a05c9045fa69da307969))
* **core:** read Seurat objects from .rds/.rda files ([3798c8b](https://github.com/btraven00/scx/commit/3798c8b490e2d34832c3bfed358079055a7bf981))
* **core:** SubsetReader, a dataset restricted to cells matching obs filters ([be7f736](https://github.com/btraven00/scx/commit/be7f73633b25d38ac8cc13e9bbd074424c24697d))
* **core:** write X gene-major with bounded memory ([6312be9](https://github.com/btraven00/scx/commit/6312be9e4cf4c3683c5bb235553c4de0cc66cf76))
* **h5ad:** read the pre-0.7 anndata layout ([e9f98b9](https://github.com/btraven00/scx/commit/e9f98b95e302bf4c21e132ee0197798c533fd1b9))
* **mtx:** read a GEO-prefixed MEX directory and the feature-type column ([15852e7](https://github.com/btraven00/scx/commit/15852e78accb30a5e0ad62cf59c943d8feb0f170))
* **picklerick:** stream gene-major X by gene with open_stream(axis="var") ([03981b0](https://github.com/btraven00/scx/commit/03981b001bf17f9d2da44b2b0be217b1aba91cff))
* **rds:** read reductions and graphs ([3edee55](https://github.com/btraven00/scx/commit/3edee5518be0f1a4c91fb882ab72d66e0cb4261f))
* read any single-file format from a URL ([af6581d](https://github.com/btraven00/scx/commit/af6581d59eacbc72bcfcb816d4f42f7a59a55159))
* record the source hash of remote .rds inputs ([a290a74](https://github.com/btraven00/scx/commit/a290a7497375e299c231cedee18fb117582c8ea4))
* write the matrix to layers/&lt;name&gt; instead of /X (--x-slot for h5ad) ([bcfc791](https://github.com/btraven00/scx/commit/bcfc791ec63ad59b95a13c89ec0bb6a78717fd10))


### Bug Fixes

* **cli:** inspect .rds URLs instead of failing to detect them ([8250c90](https://github.com/btraven00/scx/commit/8250c908e44d4e8578065d68a6ebf87a3b29adce))
* **cli:** say a missing input file does not exist ([aadc07f](https://github.com/btraven00/scx/commit/aadc07f8b01645af9e252a8ab260970b5932572e))
* **core:** align h5seurat varm to var by feature name and skip other-assay reductions ([33115d4](https://github.com/btraven00/scx/commit/33115d47c779096496eb6f6902f6e864c88bea37))
* **core:** decode legacy h5ad `__categories` columns as categoricals ([a38dc80](https://github.com/btraven00/scx/commit/a38dc807119dedd028fca0c659586738d1c9c502))
* **core:** decode lzf and name unsupported HDF5 filters ([a1192ef](https://github.com/btraven00/scx/commit/a1192ef5eb583d5beef6cc6d5c6e8eb851787abb))
* **core:** keep int64 obs/var columns instead of clamping to i32 ([4f5d30a](https://github.com/btraven00/scx/commit/4f5d30afd2f832437f2b0f6c8e2fca7e74380799))
* **core:** keep the ordered flag on categoricals ([ab1f93b](https://github.com/btraven00/scx/commit/ab1f93b9a91a6be06cab5ba860c7a2ff53bd6c20))
* **core:** read CSC layers and obsp instead of failing the whole file ([e9133bf](https://github.com/btraven00/scx/commit/e9133bfb3f9a3eda6f108f851b7b74cc276fb319))
* **core:** refuse gene-major input before writing, and transpose BPCells h5seurat ([156ae97](https://github.com/btraven00/scx/commit/156ae97890106ce8770d5d7d2254c9b8a5cd0253))
* create the copied h5seurat counts group without a timestamp ([a0ecb87](https://github.com/btraven00/scx/commit/a0ecb87b70c85e934c902f3fea4672fc7c5735a6))
* **h5seurat:** copy chunks into datasets with the source's extents ([432dc76](https://github.com/btraven00/scx/commit/432dc76564f54f7b71eacbb3d18fd439e336f6d5))
* keep the source dtype for h5seurat layers and graphs ([4ab140b](https://github.com/btraven00/scx/commit/4ab140bbf9cc0c987fc05ff9488e45dc39a78797))
* keep varm keys through h5seurat, and put PCs with the pca reduction ([b4b8dc4](https://github.com/btraven00/scx/commit/b4b8dc4077f2d9398704c7481193cb7adadffdb9))
* log the real nnz when h5seurat X was copied ([9b26da5](https://github.com/btraven00/scx/commit/9b26da5ceaeb1cf12de3001018bbf047d34bda93))
* make HDF5 writer output byte-reproducible ([6435884](https://github.com/btraven00/scx/commit/64358841257b7f9155429e125dbaa4bc47e57e6e))
* make scx validate reject unknown schema keys and exit 2 on errors ([1af32eb](https://github.com/btraven00/scx/commit/1af32eb37839963220b5eca52b7076ecfa12c937))
* **parquet:** read null obs values as missing, not as 0, "" or false ([9079e9e](https://github.com/btraven00/scx/commit/9079e9e8486c977155bb7b1eb9d83c8b82f2a318))
* put h5ad raw.X into the h5seurat counts slot ([a257104](https://github.com/btraven00/scx/commit/a257104d8c71ecc3460e0ea4321dc7c8c236032f))
* **rds:** read the assay's other layers and its feature metadata ([5d389c7](https://github.com/btraven00/scx/commit/5d389c7ddd9a277ef5870b2ea7a76a01e19847f0))
* read square h5seurat embeddings the right way round ([ad999bd](https://github.com/btraven00/scx/commit/ad999bd3cf0ef61034b32171bfab83a8e11b99b5))
* refuse to merge into a timestamped HDF5 file without opt-in ([ea5ed58](https://github.com/btraven00/scx/commit/ea5ed5868cbfd6ec61cad2773f49abbd8092845a))
* reject bixverse TSV rows with extra fields ([06c8ae2](https://github.com/btraven00/scx/commit/06c8ae210b7f22aaba55bf34ea2f41c9c71bef9e))
* reject obsm/varm keys that collide as H5Seurat reductions ([5ac0b47](https://github.com/btraven00/scx/commit/5ac0b4728d8e50dcd3c60327699d9e7ade68a0ad))
* scx-cli bixverse-parquet enables bixverse; test the factory, sidecars and CLI ([a103372](https://github.com/btraven00/scx/commit/a1033721abbf682931d8c05295a90bbf68b39084))
* write a missing h5seurat string as "", not its placeholder ([2f7f33d](https://github.com/btraven00/scx/commit/2f7f33dcc0771d7b91cae8e44a7d133376948cf2))
* write BPCells h5seurat embeddings as (n_comps, n_obs) ([dd3ec05](https://github.com/btraven00/scx/commit/dd3ec057ebe3228beeabcd9ef5c9a328c3d24e8d))
* write merge added_at to the sidecar, not uns ([778e819](https://github.com/btraven00/scx/commit/778e819f10853167f0b4077f5e0c1d9aeb9727e1))
* **zarr:** read and write the dtypes and uns values h5ad does ([47a2e60](https://github.com/btraven00/scx/commit/47a2e6059eb43298dd6f59eb50cd730e120ee0b6))


### Performance Improvements

* also copy unfiltered chunked matrices as stored ([812a057](https://github.com/btraven00/scx/commit/812a057953ceff47cda2884ec399874dc72a95ec))
* **bpcells:** cut the h5seurat writer's streaming peak by a third ([b1ae0cd](https://github.com/btraven00/scx/commit/b1ae0cd4e858a0f71135c77f0b5c0619664e6d55))
* **bpcells:** encode BP-128 pieces in parallel when streaming ([042c0cc](https://github.com/btraven00/scx/commit/042c0cc0be5f9f4bfa087f6be0094b3545939b7d))
* check rows while copying, find chunks in one pass, hash in the background ([8518587](https://github.com/btraven00/scx/commit/851858774f675094487068e6f4194a6b5ec46453))
* **cli:** hash the source with blake3 by default (--source-hash) ([7247a06](https://github.com/btraven00/scx/commit/7247a06a8cf4e3581c1a4680671162ddad6c6b6e))
* convert holds one metadata frame at a time and hands freed memory back ([0cb70ef](https://github.com/btraven00/scx/commit/0cb70eff35c18857053fcafcdd41645a37f93212))
* copy an h5ad's X and raw.X into a dgCMatrix h5seurat as stored ([48eaa7d](https://github.com/btraven00/scx/commit/48eaa7dc8c313a58660e832f344f7781323c6d68))
* deflate h5ad sparse chunks in parallel on compressed write ([05b2121](https://github.com/btraven00/scx/commit/05b2121dce2f625683abb92b9f8eeb7de2ae7308))
* **h5ad:** keep one indexed reader across raw/X, layer and obsp blocks ([93e1691](https://github.com/btraven00/scx/commit/93e1691fd9190bb623d5debce24eccc6f88bd072))
* **h5ad:** open the file once per matrix stream, not per block ([9252c32](https://github.com/btraven00/scx/commit/9252c3298410c40298364ec66aa2ab9ad4d240ea))
* **h5ad:** read only a row range's part of X/indptr ([4221432](https://github.com/btraven00/scx/commit/4221432a336d30c47754d325235d10099ecba3d1))
* **h5:** inflate each chunk in one call into a buffer of its full size ([98a3bea](https://github.com/btraven00/scx/commit/98a3beae139be90647bb14ef6598efe5487d7133))
* **h5:** inflate HDF5 chunks straight into the output Vec ([e65d41f](https://github.com/btraven00/scx/commit/e65d41fab6435e7666e68c9434f58503e2dcf306))
* **h5:** inflate with zlib-rs instead of miniz_oxide ([35ecdd1](https://github.com/btraven00/scx/commit/35ecdd16cf01ec46e51fd8a986fe862a9b6ff3f1))
* **h5:** madvise large X read buffers to huge pages ([c7a4069](https://github.com/btraven00/scx/commit/c7a40695bfd763b66fe11b4df07cb873d74cc213))
* **h5:** pread chunks inside the parallel inflate, off the HDF5 lock ([c45e75a](https://github.com/btraven00/scx/commit/c45e75a16727c9504552ccf362321e3a818c745e))
* **h5:** read unfiltered chunks with pread, in parallel and in place ([a62d823](https://github.com/btraven00/scx/commit/a62d82308a94f55d74d8a186eae737806672aba9))
* **h5:** read unfiltered X slices without a copy or conversion ([7940e13](https://github.com/btraven00/scx/commit/7940e13ea442161b10ffa3e5c57400bccf74865a))
* **h5seurat:** check dgCMatrix rows beside the chunk copy ([8744f8f](https://github.com/btraven00/scx/commit/8744f8f8b0bbfbf3a4ebe3ba2bb4a90d756ac66f))
* **h5seurat:** check every chunk at once, and compare X beside the copy ([98dcfce](https://github.com/btraven00/scx/commit/98dcfce3d1d890d6262e4637e57579481bf5e737))
* **h5seurat:** check X's rows once and overlap chunk writes in the copy ([08fb0d9](https://github.com/btraven00/scx/commit/08fb0d98fdc3f97425cfe2561855ad5bdb65a158))
* **h5seurat:** read only a row range's part of the dgCMatrix indptr ([568b25c](https://github.com/btraven00/scx/commit/568b25c0ddc3ce0d746d4c71106b4bb64925c537))
* read row ranges directly from h5seurat (dgCMatrix) and 10x ([6e974b4](https://github.com/btraven00/scx/commit/6e974b4c398eb6302e28b1c427d60b109bc5513f))
* **tenx:** read only a row range's part of matrix/indptr ([27713ea](https://github.com/btraven00/scx/commit/27713ead18024abafbc78acee0bd43f3c24e0947))
* **zarr:** read only a row range's part of X/indptr ([d768ef0](https://github.com/btraven00/scx/commit/d768ef0822c93ce4f159cc883b53ff84712091ff))

## [0.5.0](https://github.com/btraven00/scx/compare/scx-core-v0.4.1...scx-core-v0.5.0) (2026-10-07)


### ⚠ BREAKING CHANGES

* **core:** ScxError (core and api) is #[non_exhaustive]; match it with a wildcard arm. api::ScxError's Hdf5(String), NotImplemented and Other variants are gone; ScxError::Net(String) and ScxError::Zarr(String) are now Net { source, .. } and Zarr { path, source, .. }. H5AdOptions and BpcellsOptions can no longer be built with a struct literal outside scx-core:

### Bug Fixes

* **core:** h5ad keeps layer dtypes, rejects CSC slots, errors instead of skipping ([700d147](https://github.com/btraven00/scx/commit/700d1470e86ac19ed1ad0995d9be83da0232ecb7))


### Code Refactoring

* **core:** one error type; the three api builders share one core ([d4dc9ca](https://github.com/btraven00/scx/commit/d4dc9cad28ecb97a8d75ecef376e73ecb2eede3f))

## [0.4.1](https://github.com/btraven00/scx/compare/scx-core-v0.4.0...scx-core-v0.4.1) (2026-10-06)


### Bug Fixes

* **core:** enable bytemuck's alloc feature, build without default features ([403b2ce](https://github.com/btraven00/scx/commit/403b2cea9d847bc3931965c3663bb02e089f646d))

## [0.4.0](https://github.com/btraven00/scx/compare/scx-core-v0.3.0...scx-core-v0.4.0) (2026-10-05)


### Features

* **concat:** obs-axis concatenation with anndata.concat() semantics ([913e908](https://github.com/btraven00/scx/commit/913e9086bfac86c862dd4ac97a51c3ea9af18f2b))
* **zarr:** read AnnData Zarr stores (v2 and v3) ([34c986f](https://github.com/btraven00/scx/commit/34c986f80c41600421b7134ab449d851975a0f64))


### Bug Fixes

* **core:** carry missing values through readers, writers, concat, merge and export ([33d096b](https://github.com/btraven00/scx/commit/33d096b2162b66b376f30bbac95fac86266ce657))
* **core:** keep uns arrays and bools; write uns to H5Seurat misc ([4d8e552](https://github.com/btraven00/scx/commit/4d8e552830d4b8eb4d0358af6f9e2d8beefb69a9))
* **core:** open merge patch sources through the reader factory ([2c5aa03](https://github.com/btraven00/scx/commit/2c5aa030763b08a01233fc5b0fa1d2459f052aa7))
* **core:** open Seurat v5 H5Seurat files (assays/&lt;assay&gt;/layers/&lt;layer&gt;) ([ff7d91a](https://github.com/btraven00/scx/commit/ff7d91aececfb5b337fc3861aebf39ea5df4d021))
* **core:** read and write H5Seurat dgCMatrix layer dims as [genes, cells] ([4f6444e](https://github.com/btraven00/scx/commit/4f6444ef7ce53d80995c293d9f6817aad7160680))
* **core:** read index and value datasets the same way in every HDF5 reader ([c79ec67](https://github.com/btraven00/scx/commit/c79ec67a8bdaeace6a7780815ccf20d39bdb3e0b))
* **core:** remove undefined behaviour in the parallel BPCells decoders ([91433b2](https://github.com/btraven00/scx/commit/91433b27dd5f52154f9853969947531dfa1ba0cf))
* **core:** set anndata encodings on nullable values and mask ([5129af3](https://github.com/btraven00/scx/commit/5129af3e6c6098f0f8a436a38d7f787b9a7c554c))
* **core:** store integer matrices as packed uint32 in the BPCells writers ([ba2a9e4](https://github.com/btraven00/scx/commit/ba2a9e4bcc4355d25ec2496c1dacceffdea00582))
* fallback for no-X in file ([7fe856e](https://github.com/btraven00/scx/commit/7fe856ee76a964be6e6295ef09b3974db979f36c))
* **h5ad:** don't list the layer serving as X as a layer too ([7d4b00c](https://github.com/btraven00/scx/commit/7d4b00cabb4522cb8fc809431b3e838135658505))
* **h5ad:** drop obs/var column named `_index` (collides with frame index) ([dc7e6d6](https://github.com/btraven00/scx/commit/dc7e6d63c8e4d9c2f4dd00705e6b08ddfa0f7a89))
* **h5ad:** read anndata 0.13 nullable-string-array; survive NaN in inspect ([b55378c](https://github.com/btraven00/scx/commit/b55378c9b6f90400e57f78a43819bcc70c16813f))
* **h5ad:** sort unsorted CSR column indices on read ([2ad49d1](https://github.com/btraven00/scx/commit/2ad49d1c121ca4a34019c604f75732bd79be9914))
* **h5ad:** widen categorical codes to i32 past 32767 levels ([48a0cd0](https://github.com/btraven00/scx/commit/48a0cd0f6c4e9034e93893fceae55237b50d5388))
* **h5ad:** write bool columns as H5T_ENUM, not a plain integer ([67f3241](https://github.com/btraven00/scx/commit/67f32410618d4b6b7ceecf4f5728cf3652291bdd))
* **h5:** read fixed-length string datasets instead of silently returning empty ([c830e28](https://github.com/btraven00/scx/commit/c830e284182200a0701dfde7ba79e5a666323de7))
* **h5seurat:** --seuratdisk-compat must attribute reduction groups ([6dc451c](https://github.com/btraven00/scx/commit/6dc451cb0a55bacf5641de47adec1bc4766f0871))
* **tenx:** accept the Cell Ranger v2 /matrix/genes layout ([f21d6f4](https://github.com/btraven00/scx/commit/f21d6f49d6465191f33dd11a5f8faa4b5a100693))
* **zarr:** don't list the layer serving as X as a layer too ([8c19ec1](https://github.com/btraven00/scx/commit/8c19ec1533139c650ec6c3b01a965671caa61d48))
* **zarr:** sort unsorted CSR column indices on read ([f389e7b](https://github.com/btraven00/scx/commit/f389e7baa144d47c5ecab5aa8c8df081a42b1391))


### Performance Improvements

* **core:** move chunk vectors into the streaming builders ([2b02032](https://github.com/btraven00/scx/commit/2b020320fe63320c77c149b9886108553730d762))
* **h5:** parallel-inflate shuffle+deflate, and use it from the 10x reader ([1d3fe94](https://github.com/btraven00/scx/commit/1d3fe94c7fef97cf005d122ade9ee8c23d695b6c))

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
