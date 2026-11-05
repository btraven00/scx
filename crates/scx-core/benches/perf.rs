//! Release perf suite: the read/write hot paths on seeded synthetic data.
//!
//! No fixture needed, so it runs on any checkout. Writers and the reader run
//! at 1 and 8 rayon threads: equal times mean a path is serial. Results are
//! collected per release by `bench/release/run.sh` (see docs/releasing.md).
//!
//! `cargo bench -p scx-core --bench perf`

use std::path::Path;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures::StreamExt;
use tempfile::TempDir;
use tokio::runtime::{Builder, Runtime};

use scx_core::{
    bpcells::{decode_d1z, encode_d1z, encode_for},
    dtype::{DataType, TypedVec},
    h5ad::{H5AdReader, H5AdWriter},
    h5bpcells::BpcellsH5Writer,
    h5seurat::H5SeuratWriter,
    ir::{MatrixChunk, ObsTable, SparseMatrixCSR, VarTable},
    stream::{DatasetReader, DatasetWriter},
};

// HLCA-like shape: ~28k genes, ~1.5k nonzeros per cell.
const N_CELLS: usize = 10_000;
const N_GENES: usize = 28_000;
const CHUNK_CELLS: usize = 1_000;
const THREADS: [usize; 2] = [1, 8];

/// Seeded xorshift64: deterministic across machines and releases.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Cell-major CSR chunks of a synthetic count matrix (f32, sorted genes).
fn synth_chunks() -> Vec<MatrixChunk> {
    let mut rng = Rng(0x5c_c0ffee);
    (0..N_CELLS)
        .step_by(CHUNK_CELLS)
        .map(|row_offset| {
            let mut indptr = vec![0u64];
            let (mut indices, mut data) = (Vec::new(), Vec::new());
            for _ in 0..CHUNK_CELLS {
                let nnz = 500 + (rng.next() % 2000) as usize;
                let max_step = (2 * N_GENES / nnz).max(2) as u64;
                let mut g = rng.next() % max_step;
                while (g as usize) < N_GENES {
                    indices.push(g as u32);
                    data.push((1 + rng.next() % 7) as f32);
                    g += 1 + rng.next() % max_step;
                }
                indptr.push(indices.len() as u64);
            }
            MatrixChunk {
                row_offset,
                nrows: CHUNK_CELLS,
                data: SparseMatrixCSR {
                    shape: (CHUNK_CELLS, N_GENES),
                    indptr,
                    indices,
                    data: TypedVec::F32(data),
                },
            }
        })
        .collect()
}

fn nnz(chunks: &[MatrixChunk]) -> u64 {
    chunks.iter().map(|c| c.data.indices.len() as u64).sum()
}

fn tables() -> (ObsTable, VarTable) {
    let obs = ObsTable {
        index: (0..N_CELLS).map(|i| format!("c{i}")).collect(),
        ..Default::default()
    };
    let var = VarTable {
        index: (0..N_GENES).map(|i| format!("g{i}")).collect(),
        ..Default::default()
    };
    (obs, var)
}

/// Run `f` with rayon limited to `threads`, on the calling thread, so the
/// limit covers every `par_iter` inside it.
fn with_threads<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

fn current_thread_rt() -> Runtime {
    Builder::new_current_thread().enable_all().build().unwrap()
}

fn bench_bp128(c: &mut Criterion) {
    let chunks = synth_chunks();
    let indices: Vec<u32> = chunks.iter().flat_map(|c| c.data.indices.clone()).collect();
    let counts: Vec<u32> = indices.iter().map(|&g| 1 + g % 7).collect();
    let (data, idx, starts) = encode_d1z(&indices).unwrap();
    let idx32: Vec<u32> = idx.iter().map(|&w| w as u32).collect();

    let mut g = c.benchmark_group("bp128");
    g.throughput(Throughput::Elements(indices.len() as u64));
    g.bench_function("encode_d1z", |b| b.iter(|| encode_d1z(&indices).unwrap()));
    g.bench_function("encode_for", |b| b.iter(|| encode_for(&counts)));
    g.bench_function("decode_d1z", |b| {
        b.iter(|| decode_d1z(&data, &idx32, &starts, indices.len()))
    });
    g.finish();
}

fn make_writer(kind: &str, path: &Path) -> Box<dyn DatasetWriter> {
    let dt = DataType::F32;
    match kind {
        "bpcells" => Box::new(
            BpcellsH5Writer::create(path, N_CELLS, N_GENES, dt, None, None, None, false).unwrap(),
        ),
        "h5seurat_dgc" => Box::new(
            H5SeuratWriter::create(path, N_CELLS, N_GENES, dt, None, None, None, false).unwrap(),
        ),
        "h5ad" => Box::new(H5AdWriter::create_at(path, N_CELLS, N_GENES, dt, None, "X").unwrap()),
        _ => unreachable!(),
    }
}

async fn write_x(kind: &str, path: &Path, chunks: &[MatrixChunk]) {
    let (obs, var) = tables();
    let mut w = make_writer(kind, path);
    w.write_obs(&obs).await.unwrap();
    w.write_var(&var).await.unwrap();
    for chunk in chunks {
        w.write_x_chunk(chunk).await.unwrap();
    }
    w.finalize().await.unwrap();
}

fn bench_write_x(c: &mut Criterion) {
    let chunks = synth_chunks();
    let rt = current_thread_rt();
    let dir = TempDir::new().unwrap();

    let mut g = c.benchmark_group("write_x");
    g.throughput(Throughput::Elements(nnz(&chunks)));
    g.sample_size(10);
    for kind in ["bpcells", "h5seurat_dgc", "h5ad"] {
        for t in THREADS {
            let path = dir.path().join(format!("{kind}.out"));
            g.bench_function(BenchmarkId::new(kind, format!("t{t}")), |b| {
                b.iter(|| {
                    let _ = std::fs::remove_file(&path);
                    with_threads(t, || rt.block_on(write_x(kind, &path, &chunks)))
                })
            });
        }
    }
    g.finish();
}

fn bench_read_x(c: &mut Criterion) {
    let chunks = synth_chunks();
    let total = nnz(&chunks);
    let rt = current_thread_rt();
    let dir = TempDir::new().unwrap();

    let mut g = c.benchmark_group("read_x");
    g.throughput(Throughput::Elements(total));
    g.sample_size(10);
    for (enc, level) in [("raw", None), ("gzip", Some(4))] {
        // Written once, outside the timing loop.
        let path = dir.path().join(format!("{enc}.h5ad"));
        rt.block_on(async {
            let (obs, var) = tables();
            let mut w =
                H5AdWriter::create_at(&path, N_CELLS, N_GENES, DataType::F32, level, "X").unwrap();
            w.write_obs(&obs).await.unwrap();
            w.write_var(&var).await.unwrap();
            for chunk in &chunks {
                w.write_x_chunk(chunk).await.unwrap();
            }
            w.finalize().await.unwrap();
        });
        for t in THREADS {
            g.bench_function(BenchmarkId::new(enc, format!("t{t}")), |b| {
                b.iter(|| {
                    with_threads(t, || {
                        rt.block_on(async {
                            let mut r = H5AdReader::open(&path, 5000).unwrap();
                            let mut s = r.x_stream();
                            let mut n = 0u64;
                            while let Some(chunk) = s.next().await {
                                n += chunk.unwrap().data.indices.len() as u64;
                            }
                            assert_eq!(n, total);
                        })
                    })
                })
            });
        }
    }
    g.finish();
}

criterion_group!(benches, bench_bp128, bench_write_x, bench_read_x);
criterion_main!(benches);
