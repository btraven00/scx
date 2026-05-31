//! Criterion guard for the h5ad X decode path (parallel inflate, h5_chunk.rs).
//!
//! Streams the count matrix of an h5ad fixture and sums nnz, exercising
//! `ad_read_chunk` → `read_x_indices`/`read_x_data` → the parallel-inflate fast
//! path for deflate-chunked datasets. A regression here means the fast path
//! broke or silently fell back to libhdf5's single-threaded decode.
//!
//! norman_subset (~79 MB, many gzip chunks) actually exercises parallel inflate;
//! pbmc3k is small (single chunk) and just guards the code path. Each benches
//! SKIPs when its fixture is absent.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures::StreamExt;
use tokio::runtime::Builder;

use scx_core::{h5ad::H5AdReader, stream::DatasetReader};

const FIXTURES: &[(&str, &str)] = &[
    ("norman", "../../tests/golden/norman_subset.h5ad"),
    ("pbmc3k", "../../tests/golden/pbmc3k_reference.h5ad"),
];

fn bench_x_decode(c: &mut Criterion) {
    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    let mut group = c.benchmark_group("h5ad_x_decode");

    for (name, path) in FIXTURES {
        if !std::path::Path::new(path).exists() {
            eprintln!("SKIP h5ad_x_decode/{name}: fixture not found at {path}");
            continue;
        }
        // Probe total nnz once (also the throughput unit).
        let total_nnz = rt.block_on(async {
            let mut r = H5AdReader::open(path, 5000).unwrap();
            let mut nnz = 0usize;
            let mut s = r.x_stream();
            while let Some(c) = s.next().await {
                nnz += c.unwrap().data.indices.len();
            }
            nnz
        });
        group.throughput(Throughput::Elements(total_nnz as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), path, |b, path| {
            b.to_async(&rt).iter(|| async move {
                let mut reader = H5AdReader::open(path, 5000).unwrap();
                let mut stream = reader.x_stream();
                let mut nnz = 0usize;
                while let Some(chunk) = stream.next().await {
                    nnz += chunk.unwrap().data.indices.len();
                }
                assert_eq!(nnz, total_nnz);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_x_decode);
criterion_main!(benches);
