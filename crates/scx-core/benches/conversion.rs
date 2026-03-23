use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures::StreamExt;
use tempfile::NamedTempFile;
use tokio::runtime::Builder;

use scx_core::{
    dtype::DataType,
    h5::ScxH5Reader,
    h5ad::H5AdWriter,
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

criterion_group!(
    benches,
    bench_stream_chunk_sizes,
    bench_roundtrip,
    bench_metadata_read,
);
criterion_main!(benches);
