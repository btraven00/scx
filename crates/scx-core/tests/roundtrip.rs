//! Round-trip matrix: every tiny fixture (tests/fixtures/tiny/) through every
//! writer, read back with scx's own readers and compared with expected.json.
//!
//! Checks what every writer must keep: shape, cell and gene names, and the
//! matrix values. Metadata that only some writers carry is checked in
//! `metadata_*` tests below.
//!
//! `#[ignore = "bug: ..."]` cases fail today (`cargo test -- --ignored`); the
//! fix un-ignores them.

use futures::StreamExt;
use scx_core::bpcells::BpcellsDirWriter;
use scx_core::dtype::DataType;
use scx_core::h5ad::H5AdWriter;
use scx_core::h5bpcells::BpcellsH5Writer;
use scx_core::h5seurat::H5SeuratWriter;
use scx_core::npy::{NpyIrWriter, SlotFilter};
use scx_core::stream::{DatasetReader, DatasetWriter};
use serde_json::Value;
use std::path::{Path, PathBuf};

const CHUNK: usize = 4; // < n_obs, so X always spans two chunks

fn tiny(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny")
        .join(name)
}

fn expected() -> Value {
    serde_json::from_str(&std::fs::read_to_string(tiny("expected.json")).unwrap()).unwrap()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect()
}

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect()
}

async fn open(path: &Path, layer: &str) -> Box<dyn DatasetReader + Send> {
    let opts = scx_core::OpenOptions {
        layer: Some(layer.to_string()),
        ..scx_core::OpenOptions::new(CHUNK)
    };
    scx_core::open(path.to_str().unwrap(), &opts)
        .await
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

async fn dense_x(reader: &mut (dyn DatasetReader + Send)) -> Vec<Vec<f64>> {
    let (n_obs, n_vars) = reader.shape();
    let mut out = vec![vec![0.0; n_vars]; n_obs];
    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        let values = chunk.data.data.to_f64();
        for r in 0..chunk.nrows {
            for k in chunk.data.indptr[r] as usize..chunk.data.indptr[r + 1] as usize {
                out[chunk.row_offset + r][chunk.data.indices[k] as usize] = values[k];
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
enum Out {
    H5ad,
    /// SeuratDisk dgCMatrix layout (`scx convert --dgcmatrix`).
    H5SeuratDgc,
    /// BPCells-backed H5Seurat (`scx convert` default for .h5seurat).
    H5SeuratBpcells,
    BpcellsDir,
    Npy,
}

/// The same steps as `scx convert`: metadata, layers, obsp, then X. For
/// H5Seurat, X goes to the `data` slot when the source has a `counts` layer
/// (the CLI's `--x-slot auto`), and a layer with the X slot's name is skipped.
async fn write(reader: &mut (dyn DatasetReader + Send), out: Out, path: &Path) -> String {
    if let Out::Npy = out {
        NpyIrWriter::stream(path, &mut *reader, &SlotFilter::all(), CHUNK)
            .await
            .unwrap();
        return "counts".into();
    }
    let (n, m) = reader.shape();
    let layers = reader.layer_metas().await.unwrap();
    let x_slot = if layers.iter().any(|l| l.name == "counts") {
        "data"
    } else {
        "counts"
    };
    let mut w: Box<dyn DatasetWriter> = match out {
        Out::H5ad => Box::new(H5AdWriter::create(path, n, m, DataType::F32).unwrap()),
        Out::H5SeuratDgc => Box::new(
            H5SeuratWriter::create(path, n, m, DataType::F32, None, Some(x_slot), None, false)
                .unwrap(),
        ),
        Out::H5SeuratBpcells => Box::new(
            BpcellsH5Writer::create(path, n, m, DataType::F32, None, Some(x_slot), None, false)
                .unwrap(),
        ),
        Out::BpcellsDir => Box::new(BpcellsDirWriter::create(path, n, m).unwrap()),
        Out::Npy => unreachable!(),
    };
    let seurat = matches!(out, Out::H5SeuratDgc | Out::H5SeuratBpcells);

    w.write_obs(&reader.obs().await.unwrap()).await.unwrap();
    w.write_var(&reader.var().await.unwrap()).await.unwrap();
    let obsm = reader.obsm().await.unwrap();
    if !obsm.map.is_empty() {
        w.write_obsm(&obsm).await.unwrap();
    }
    w.write_uns(&reader.uns().await.unwrap()).await.unwrap();
    let varm = reader.varm().await.unwrap();
    if !varm.map.is_empty() {
        w.write_varm(&varm).await.unwrap();
    }
    for meta in &layers {
        if seurat && meta.name == x_slot {
            continue;
        }
        w.begin_sparse("layers", &meta.name, meta).await.unwrap();
        let mut s = reader.layer_stream(meta, CHUNK);
        while let Some(c) = s.next().await {
            w.write_sparse_chunk(&c.unwrap()).await.unwrap();
        }
        w.end_sparse().await.unwrap();
    }
    for meta in &reader.obsp_metas().await.unwrap() {
        w.begin_sparse("obsp", &meta.name, meta).await.unwrap();
        let mut s = reader.obsp_stream(meta, CHUNK);
        while let Some(c) = s.next().await {
            w.write_sparse_chunk(&c.unwrap()).await.unwrap();
        }
        w.end_sparse().await.unwrap();
    }
    let mut s = reader.x_stream();
    while let Some(c) = s.next().await {
        w.write_x_chunk(&c.unwrap()).await.unwrap();
    }
    w.finalize().await.unwrap();
    if seurat {
        x_slot.into()
    } else {
        "counts".into()
    }
}

fn out_path(dir: &Path, out: Out) -> PathBuf {
    dir.join(match out {
        Out::H5ad => "out.h5ad",
        Out::H5SeuratDgc | Out::H5SeuratBpcells => "out.h5seurat",
        Out::BpcellsDir => "out_bpcells",
        Out::Npy => "out_npy",
    })
}

/// Convert `src` with `out`, read it back, compare against expected.json.
/// `slot` names the matrix the source stores as X ("X" or "counts").
/// Convert `src` with `out` and open the result. The temp dir must outlive
/// the reader.
async fn convert(src: &str, out: Out) -> (tempfile::TempDir, Box<dyn DatasetReader + Send>) {
    let dir = tempfile::tempdir().unwrap();
    let path = out_path(dir.path(), out);
    let read_layer = {
        let mut reader = open(&tiny(src), "counts").await;
        write(&mut *reader, out, &path).await
    };
    let back = open(&path, &read_layer).await;
    (dir, back)
}

async fn roundtrip(src: &str, slot: &str, out: Out) {
    let e = expected();
    let (_dir, mut back) = convert(src, out).await;
    let ctx = format!("{src} -> {out:?}");
    assert_eq!(back.shape(), (6, 5), "{ctx}: shape");
    assert_eq!(
        back.obs().await.unwrap().index,
        strings(&e["obs_names"]),
        "{ctx}: obs names"
    );
    assert_eq!(
        back.var().await.unwrap().index,
        strings(&e["var_names"]),
        "{ctx}: var names"
    );
    assert_eq!(dense_x(&mut *back).await, matrix(&e[slot]), "{ctx}: X");
}

macro_rules! cases {
    ($( $(#[$attr:meta])* $name:ident: $src:literal, $slot:literal, $out:ident; )*) => {$(
        #[tokio::test]
        $(#[$attr])*
        async fn $name() {
            roundtrip($src, $slot, Out::$out).await;
        }
    )*};
}

// Sources the readers handle today. H5Seurat v4 is read with layer "counts";
// v5 (assays/RNA/layers/counts) can't be opened yet, see tiny_fixtures.rs.
cases! {
    h5ad_to_h5ad: "tiny.h5ad", "X", H5ad;
    h5ad_to_h5seurat_dgc: "tiny.h5ad", "X", H5SeuratDgc;
    h5ad_to_h5seurat_bpcells: "tiny.h5ad", "X", H5SeuratBpcells;
    h5ad_to_bpcells_dir: "tiny.h5ad", "X", BpcellsDir;
    h5ad_to_npy: "tiny.h5ad", "X", Npy;

    dense_to_h5ad: "tiny_dense.h5ad", "X", H5ad;
    dense_to_h5seurat_dgc: "tiny_dense.h5ad", "X", H5SeuratDgc;
    dense_to_h5seurat_bpcells: "tiny_dense.h5ad", "X", H5SeuratBpcells;
    dense_to_bpcells_dir: "tiny_dense.h5ad", "X", BpcellsDir;
    dense_to_npy: "tiny_dense.h5ad", "X", Npy;

    tenx_to_h5ad: "tiny_10x.h5", "counts", H5ad;
    tenx_to_h5seurat_dgc: "tiny_10x.h5", "counts", H5SeuratDgc;
    tenx_to_h5seurat_bpcells: "tiny_10x.h5", "counts", H5SeuratBpcells;
    tenx_to_bpcells_dir: "tiny_10x.h5", "counts", BpcellsDir;
    tenx_to_npy: "tiny_10x.h5", "counts", Npy;

    mtx_to_h5ad: "tiny_mtx", "counts", H5ad;
    mtx_to_h5seurat_dgc: "tiny_mtx", "counts", H5SeuratDgc;
    mtx_to_h5seurat_bpcells: "tiny_mtx", "counts", H5SeuratBpcells;
    mtx_to_bpcells_dir: "tiny_mtx", "counts", BpcellsDir;
    mtx_to_npy: "tiny_mtx", "counts", Npy;

    h5seurat_to_h5ad: "tiny_v4.h5seurat", "counts", H5ad;
    h5seurat_to_h5seurat_dgc: "tiny_v4.h5seurat", "counts", H5SeuratDgc;
    h5seurat_to_h5seurat_bpcells: "tiny_v4.h5seurat", "counts", H5SeuratBpcells;
    h5seurat_to_bpcells_dir: "tiny_v4.h5seurat", "counts", BpcellsDir;
    h5seurat_to_npy: "tiny_v4.h5seurat", "counts", Npy;
}

// Metadata, for the writers that carry all of it: tiny.h5ad through h5ad and
// through H5Seurat (dgCMatrix).

async fn dense_layer(reader: &mut (dyn DatasetReader + Send), name: &str) -> Vec<Vec<f64>> {
    let metas = reader.layer_metas().await.unwrap();
    let meta = metas.iter().find(|m| m.name == name).unwrap_or_else(|| {
        panic!(
            "layer {name} missing; have {:?}",
            metas.iter().map(|m| &m.name).collect::<Vec<_>>()
        )
    });
    let mut out = vec![vec![0.0; meta.shape.1]; meta.shape.0];
    let mut s = reader.layer_stream(meta, CHUNK);
    while let Some(c) = s.next().await {
        let c = c.unwrap();
        let values = c.data.data.to_f64();
        for r in 0..c.nrows {
            for k in c.data.indptr[r] as usize..c.data.indptr[r + 1] as usize {
                out[c.row_offset + r][c.data.indices[k] as usize] = values[k];
            }
        }
    }
    out
}

async fn check_cell_type(back: &mut (dyn DatasetReader + Send)) {
    let obs = back.obs().await.unwrap();
    let col = obs
        .columns
        .iter()
        .find(|c| c.name == "cell_type")
        .expect("cell_type");
    let scx_core::ir::ColumnData::Categorical { codes, levels } = &col.data else {
        panic!("cell_type is {}", col.data.dtype_str());
    };
    let got: Vec<Option<&str>> = (0..codes.len())
        .map(|i| (!col.is_na(i)).then(|| levels[codes[i] as usize].as_str()))
        .collect();
    assert_eq!(
        got,
        [Some("B"), Some("T"), None, Some("B"), Some("NK"), Some("T")]
    );
}

async fn check_uns(back: &mut (dyn DatasetReader + Send)) {
    let uns = back.uns().await.unwrap();
    let e = expected();
    for (key, want) in e["uns"].as_object().unwrap() {
        assert_eq!(uns.raw.get(key), Some(want), "uns.{key}: {}", uns.raw);
    }
}

#[tokio::test]
async fn h5ad_to_h5ad_keeps_layers_obsm_obsp() {
    let e = expected();
    let (_dir, mut back) = convert("tiny.h5ad", Out::H5ad).await;
    assert_eq!(
        dense_layer(&mut *back, "counts").await,
        matrix(&e["counts"])
    );
    assert_eq!(
        back.obsm().await.unwrap().map["X_pca"].data,
        matrix(&e["obsm"]["X_pca"]).concat()
    );
    assert_eq!(back.varm().await.unwrap().map["PCs"].shape, (5, 2));
    let obsp = back.obsp_metas().await.unwrap();
    assert_eq!(
        obsp.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
        ["connectivities"]
    );
}

#[tokio::test]
async fn h5ad_to_h5ad_keeps_categorical_na() {
    // Passes only because the reader's u32::MAX code wraps back to -1 on
    // write; the NA fix should keep it passing.
    let (_dir, mut back) = convert("tiny.h5ad", Out::H5ad).await;
    check_cell_type(&mut *back).await;
}

#[tokio::test]
async fn h5ad_to_h5ad_keeps_uns() {
    let (_dir, mut back) = convert("tiny.h5ad", Out::H5ad).await;
    check_uns(&mut *back).await;
}

#[tokio::test]
async fn h5ad_to_h5seurat_keeps_counts_and_obsm() {
    let e = expected();
    // Read back with layer "data" (where X went); counts is then a layer.
    let (_dir, mut back) = convert("tiny.h5ad", Out::H5SeuratDgc).await;
    assert_eq!(
        dense_layer(&mut *back, "counts").await,
        matrix(&e["counts"])
    );
    assert!(back.obsm().await.unwrap().map.contains_key("X_pca"));
}

#[tokio::test]
async fn h5ad_to_h5seurat_keeps_categorical_na() {
    let (_dir, mut back) = convert("tiny.h5ad", Out::H5SeuratDgc).await;
    check_cell_type(&mut *back).await;
}

#[tokio::test]
async fn h5ad_to_h5seurat_keeps_uns() {
    let (_dir, mut back) = convert("tiny.h5ad", Out::H5SeuratDgc).await;
    check_uns(&mut *back).await;
}

#[tokio::test]
async fn bpcells_rejects_negative_integers() {
    use scx_core::dtype::TypedVec;
    use scx_core::ir::{MatrixChunk, SparseMatrixCSR};
    let dir = tempfile::tempdir().unwrap();
    let mut w = BpcellsDirWriter::create(&dir.path().join("m"), 1, 2).unwrap();
    let chunk = MatrixChunk {
        row_offset: 0,
        nrows: 1,
        data: SparseMatrixCSR {
            shape: (1, 2),
            indptr: vec![0, 2],
            indices: vec![0, 1],
            data: TypedVec::I32(vec![3, -1]),
        },
    };
    let err = w.write_x_chunk(&chunk).await.unwrap_err();
    assert!(err.to_string().contains("negative value (-1)"), "{err}");
}
