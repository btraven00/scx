use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures::StreamExt;
use tempfile::{NamedTempFile, TempDir};
use tokio::runtime::Builder;

use scx_core::{
    dtype::{DataType, TypedVec},
    h5::ScxH5Reader,
    h5ad::H5AdWriter,
    ir::{SparseMatrixCSR, SingleCellDataset},
    npy::{NpyIrReader, NpyIrWriter, SlotFilter},
    stream::{DatasetReader, DatasetWriter},
};

const GOLDEN: &str = "../../tests/golden/pbmc3k.h5";

fn golden_exists() -> bool {
    std::path::Path::new(GOLDEN).exists()
}

/// Benchmark: stream the count matrix at different chunk sizes.
/// Measures wall-time for the read → CSC→CSR → write pipeline.
fn bench_stream_chunk_sizes(c: &mut Criterion) {
    if !golden_exists() {
        eprintln!("SKIP: golden file not found at {GOLDEN}");
        return;
    }

    let rt = Builder::new_multi_thread().enable_all().build().unwrap();

    // Probe n_obs and total nnz once for throughput annotation
    let (n_obs, total_nnz) = rt.block_on(async {
        let mut r = ScxH5Reader::open(GOLDEN, 1000).unwrap();
        let (n_obs, _) = r.shape();
        let mut nnz = 0usize;
        let mut s = r.x_stream();
        while let Some(c) = s.next().await {
            nnz += c.unwrap().data.indices.len();
        }
        (n_obs, nnz)
    });

    let mut group = c.benchmark_group("stream_chunk_size");
    group.throughput(Throughput::Elements(n_obs as u64));

    for chunk_size in [100, 500, 1000, 2700] {
        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &chunk_size,
            |b, &chunk_size| {
                b.to_async(&rt).iter(|| async move {
                    let mut reader = ScxH5Reader::open(GOLDEN, chunk_size).unwrap();
                    let mut stream = reader.x_stream();
                    let mut n = 0usize;
                    while let Some(chunk) = stream.next().await {
                        n += chunk.unwrap().data.indices.len();
                    }
                    assert_eq!(n, total_nnz);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: full round-trip read → write h5ad at different chunk sizes.
fn bench_roundtrip(c: &mut Criterion) {
    if !golden_exists() {
        eprintln!("SKIP: golden file not found at {GOLDEN}");
        return;
    }

    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let (n_obs, n_vars) = rt.block_on(async {
        let r = ScxH5Reader::open(GOLDEN, 1000).unwrap();
        r.shape()
    });

    let mut group = c.benchmark_group("roundtrip_chunk_size");
    group.throughput(Throughput::Elements(n_obs as u64));
    // Roundtrip is slower — fewer samples needed
    group.sample_size(10);

    for chunk_size in [500, 2700] {
        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &chunk_size,
            |b, &chunk_size| {
                b.to_async(&rt).iter(|| async move {
                    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();

                    let mut reader = ScxH5Reader::open(GOLDEN, chunk_size).unwrap();
                    let obs  = reader.obs().await.unwrap();
                    let var  = reader.var().await.unwrap();
                    let obsm = reader.obsm().await.unwrap();
                    let uns  = reader.uns().await.unwrap();

                    let layers = reader.layers().await.unwrap();
                    let obsp   = reader.obsp().await.unwrap();
                    let varp   = reader.varp().await.unwrap();
                    let varm   = reader.varm().await.unwrap();

                    let mut writer = H5AdWriter::create(tmp.path(), n_obs, n_vars, DataType::F32).unwrap();
                    writer.write_obs(&obs).await.unwrap();
                    writer.write_var(&var).await.unwrap();
                    writer.write_obsm(&obsm).await.unwrap();
                    writer.write_uns(&uns).await.unwrap();
                    writer.write_layers(&layers).await.unwrap();
                    writer.write_obsp(&obsp).await.unwrap();
                    writer.write_varp(&varp).await.unwrap();
                    writer.write_varm(&varm).await.unwrap();

                    let mut stream = reader.x_stream();
                    while let Some(chunk) = stream.next().await {
                        writer.write_x_chunk(&chunk.unwrap()).await.unwrap();
                    }
                    writer.finalize().await.unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: metadata-only read (obs + var + obsm), no matrix.
fn bench_metadata_read(c: &mut Criterion) {
    if !golden_exists() {
        eprintln!("SKIP: golden file not found at {GOLDEN}");
        return;
    }

    let rt = Builder::new_multi_thread().enable_all().build().unwrap();

    c.bench_function("metadata_read", |b| {
        b.to_async(&rt).iter(|| async {
            let mut reader = ScxH5Reader::open(GOLDEN, 1000).unwrap();
            let _ = reader.obs().await.unwrap();
            let _ = reader.var().await.unwrap();
            let _ = reader.obsm().await.unwrap();
        });
    });
}

// ---------------------------------------------------------------------------
// NPY snapshot benchmarks
// ---------------------------------------------------------------------------

/// Materialise the golden ScxH5 fixture into a SingleCellDataset.
/// Called once per benchmark function (outside the timing loop).
async fn materialise_golden() -> (SingleCellDataset, usize) {
    let mut reader = ScxH5Reader::open(GOLDEN, 5000).unwrap();
    let (n_obs, n_vars) = reader.shape();
    let x_dtype = reader.dtype();
    let obs    = reader.obs().await.unwrap();
    let var    = reader.var().await.unwrap();
    let obsm   = reader.obsm().await.unwrap();
    let uns    = reader.uns().await.unwrap();
    let layers = reader.layers().await.unwrap();
    let obsp   = reader.obsp().await.unwrap();
    let varp   = reader.varp().await.unwrap();
    let varm   = reader.varm().await.unwrap();

    let mut x_indptr: Vec<u64> = vec![0];
    let mut x_indices: Vec<u32> = Vec::new();
    let mut x_f32: Vec<f32> = Vec::new();
    let mut x_f64: Vec<f64> = Vec::new();
    let mut x_i32: Vec<i32> = Vec::new();
    let mut x_u32: Vec<u32> = Vec::new();
    let mut total_nnz = 0usize;

    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        total_nnz += chunk.data.indices.len();
        x_indices.extend_from_slice(&chunk.data.indices);
        match &chunk.data.data {
            TypedVec::F32(v) => x_f32.extend_from_slice(v),
            TypedVec::F64(v) => x_f64.extend_from_slice(v),
            TypedVec::I32(v) => x_i32.extend_from_slice(v),
            TypedVec::U32(v) => x_u32.extend_from_slice(v),
        }
        let base = *x_indptr.last().unwrap();
        for &p in &chunk.data.indptr[1..] {
            x_indptr.push(base + p);
        }
    }

    let x_data = match x_dtype {
        DataType::F32 => TypedVec::F32(x_f32),
        DataType::F64 => TypedVec::F64(x_f64),
        DataType::I32 => TypedVec::I32(x_i32),
        DataType::U32 => TypedVec::U32(x_u32),
    };

    let dataset = SingleCellDataset {
        x: SparseMatrixCSR { shape: (n_obs, n_vars), indptr: x_indptr, indices: x_indices, data: x_data },
        x_dtype,
        obs, var, obsm, uns, layers, obsp, varp, varm,
    };
    (dataset, total_nnz)
}

/// Benchmark: write a full IR snapshot to a temp directory.
/// Measures raw serialisation throughput: how fast can we flush an in-memory
/// IR to disk as NPY files (no HDF5 overhead).
fn bench_npy_write(c: &mut Criterion) {
    if !golden_exists() {
        eprintln!("SKIP: golden file not found at {GOLDEN}");
        return;
    }

    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let (dataset, _) = rt.block_on(materialise_golden());
    let n_obs = dataset.x.shape.0;

    let mut group = c.benchmark_group("npy");
    group.throughput(Throughput::Elements(n_obs as u64));
    group.sample_size(10);

    group.bench_function("write_snapshot", |b| {
        b.iter(|| {
            let dir = TempDir::new().unwrap();
            NpyIrWriter::write(dir.path(), &dataset, &SlotFilter::all()).unwrap();
        });
    });

    group.finish();
}

/// Benchmark: read a full IR snapshot from a temp directory.
/// Measures deserialisation throughput: how fast can we reconstruct the IR
/// from NPY files (no HDF5 overhead).
fn bench_npy_read(c: &mut Criterion) {
    if !golden_exists() {
        eprintln!("SKIP: golden file not found at {GOLDEN}");
        return;
    }

    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let (dataset, _) = rt.block_on(materialise_golden());
    let n_obs = dataset.x.shape.0;

    // Write the snapshot once outside the timing loop.
    let snap_dir = TempDir::new().unwrap();
    let snap_path = snap_dir.path().to_owned();
    NpyIrWriter::write(&snap_path, &dataset, &SlotFilter::all()).unwrap();

    let mut group = c.benchmark_group("npy");
    group.throughput(Throughput::Elements(n_obs as u64));
    group.sample_size(10);

    group.bench_function("read_snapshot", |b| {
        b.iter(|| {
            NpyIrReader::open(&snap_path, 5000).unwrap().into_dataset()
        });
    });

    group.finish();
}

/// Benchmark: NPY snapshot → H5AD.
///
/// This is the key isolation benchmark: it measures H5AD write speed with the
/// H5Seurat read overhead removed from the loop.  Compare against
/// `bench_roundtrip` (H5 → H5AD) to quantify how much of the roundtrip time
/// is I/O format overhead vs. the H5AD write path itself.
fn bench_npy_to_h5ad(c: &mut Criterion) {
    if !golden_exists() {
        eprintln!("SKIP: golden file not found at {GOLDEN}");
        return;
    }

    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let (dataset, total_nnz) = rt.block_on(materialise_golden());
    let (n_obs, n_vars) = dataset.x.shape;
    let x_dtype = dataset.x_dtype;

    // Write the snapshot once outside the timing loop.
    let snap_dir = TempDir::new().unwrap();
    let snap_path = snap_dir.path().to_owned();
    NpyIrWriter::write(&snap_path, &dataset, &SlotFilter::all()).unwrap();
    drop(dataset);

    let mut group = c.benchmark_group("npy");
    group.throughput(Throughput::Elements(n_obs as u64));
    group.sample_size(10);

    for chunk_size in [500, 2700] {
        let snap_path = snap_path.clone();
        group.bench_with_input(
            BenchmarkId::new("to_h5ad_chunk", chunk_size),
            &chunk_size,
            |b, &chunk_size| {
                let snap_path = snap_path.clone();
                b.to_async(&rt).iter(move || {
                    let snap_path = snap_path.clone();
                    async move {
                    let snap_path = &snap_path;
                    let tmp = NamedTempFile::with_suffix(".h5ad").unwrap();

                    let mut reader = NpyIrReader::open(snap_path, chunk_size).unwrap();
                    let obs    = reader.obs().await.unwrap();
                    let var    = reader.var().await.unwrap();
                    let obsm   = reader.obsm().await.unwrap();
                    let uns    = reader.uns().await.unwrap();
                    let layers = reader.layers().await.unwrap();
                    let obsp   = reader.obsp().await.unwrap();
                    let varp   = reader.varp().await.unwrap();
                    let varm   = reader.varm().await.unwrap();

                    let mut writer = H5AdWriter::create(tmp.path(), n_obs, n_vars, x_dtype).unwrap();
                    writer.write_obs(&obs).await.unwrap();
                    writer.write_var(&var).await.unwrap();
                    writer.write_obsm(&obsm).await.unwrap();
                    writer.write_uns(&uns).await.unwrap();
                    writer.write_layers(&layers).await.unwrap();
                    writer.write_obsp(&obsp).await.unwrap();
                    writer.write_varp(&varp).await.unwrap();
                    writer.write_varm(&varm).await.unwrap();

                    let mut stream = reader.x_stream();
                    let mut nnz = 0usize;
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk.unwrap();
                        nnz += chunk.data.indices.len();
                        writer.write_x_chunk(&chunk).await.unwrap();
                    }
                    writer.finalize().await.unwrap();
                    assert_eq!(nnz, total_nnz);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: stream X from a NPY snapshot at different chunk sizes.
/// Compare against `bench_stream_chunk_sizes` (H5-backed) to quantify the
/// per-chunk overhead difference between NPY (memory-slicing) and HDF5 (I/O).
fn bench_npy_stream(c: &mut Criterion) {
    if !golden_exists() {
        eprintln!("SKIP: golden file not found at {GOLDEN}");
        return;
    }

    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let (dataset, total_nnz) = rt.block_on(materialise_golden());
    let n_obs = dataset.x.shape.0;

    let snap_dir = TempDir::new().unwrap();
    let snap_path = snap_dir.path().to_owned();
    NpyIrWriter::write(&snap_path, &dataset, &SlotFilter::from_only("X")).unwrap();
    drop(dataset);

    let mut group = c.benchmark_group("npy");
    group.throughput(Throughput::Elements(n_obs as u64));

    for chunk_size in [100, 500, 1000, 2700] {
        let snap_path = snap_path.clone();
        group.bench_with_input(
            BenchmarkId::new("stream_chunk", chunk_size),
            &chunk_size,
            |b, &chunk_size| {
                let snap_path = snap_path.clone();
                b.to_async(&rt).iter(move || {
                    let snap_path = snap_path.clone();
                    async move {
                    let mut reader = NpyIrReader::open(&snap_path, chunk_size).unwrap();
                    let mut stream = reader.x_stream();
                    let mut n = 0usize;
                    while let Some(chunk) = stream.next().await {
                        n += chunk.unwrap().data.indices.len();
                    }
                    assert_eq!(n, total_nnz);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_stream_chunk_sizes,
    bench_roundtrip,
    bench_metadata_read,
    bench_npy_write,
    bench_npy_read,
    bench_npy_to_h5ad,
    bench_npy_stream,
);
criterion_main!(benches);
