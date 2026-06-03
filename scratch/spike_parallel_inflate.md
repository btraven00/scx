# Parallel-inflate spike — WORKING reference code (2026-05-31)

Proven on hlca: serial HDF5 decode vs raw H5Dread_chunk + rayon/flate2
parallel inflate. X/data 7.78s -> 1.59s (4.9x, bit-exact); X/indices
13.64s -> 2.74s (5.0x). Full X read ~21s -> ~4.3s on 16 cores.

## Deps added to python/picklerick/Cargo.toml (temporary):
```toml
hdf5 = { package = "hdf5-metno", version = "0.12" }
hdf5-sys = { package = "hdf5-metno-sys", version = "0.11" }
flate2 = "1"   # default miniz_oxide backend; zlib-ng could be faster
rayon = "1"
```

## Key facts learned
- Vendored HDF5 is 2.0.0 => H5Dread_chunk2 (6 args incl in/out buf_size).
- ChunkInfo { offset: Vec<hsize_t>, filter_mask, addr, size(bytes) }.
- chunk_info(index) is NOT in logical order — sort by offset[0].
- HDF5 pads the last chunk to full chunk size — truncate to dataset len.
- gzip filter writes zlib streams (RFC1950) => flate2::read::ZlibDecoder.
- Raw reads are cheap (~0.3-0.7s) and can stay under the hdf5 global lock;
  inflate runs OUTSIDE the lock => parallelizes (the whole point).

## Spike #[pyfunction] (from python/picklerick/rust/src/lib.rs):
```rust
// TEMPORARY SPIKE: parallel inflate of an h5ad dataset's gzip chunks.
//
// Baseline = let libhdf5 decode the whole 1-D dataset (single-threaded, under
// the hdf5-metno global lock). Candidate = read each raw gzip chunk via
// H5Dread_chunk (cheap, serial) then inflate the chunks in parallel with rayon
// + flate2 OUTSIDE the lock. Reports both timings + a correctness flag.
// Remove with its deps if the spike is not pursued.
// ---------------------------------------------------------------------------
#[pyfunction]
fn _spike_parallel_inflate(py: Python<'_>, path: &str, ds_path: &str) -> PyResult<PyObject> {
    use std::io::Read;
    use std::time::Instant;

    use flate2::read::ZlibDecoder;
    use rayon::prelude::*;

    let (t_serial, t_raw, t_inflate, n_chunks, n_elems, ok) = py
        .allow_threads(|| -> anyhow::Result<(f64, f64, f64, usize, usize, bool)> {
            let file = hdf5::File::open(path)?;
            let ds = file.dataset(ds_path)?;
            let n = ds
                .num_chunks()
                .ok_or_else(|| anyhow::anyhow!("{ds_path} is not chunked"))?;
            let mut infos: Vec<_> = (0..n)
                .map(|i| ds.chunk_info(i).expect("chunk_info"))
                .collect();
            // chunk_info(index) is not guaranteed in logical order — sort by the
            // 1-D element offset so concatenation reconstructs the dataset.
            infos.sort_by_key(|ci| ci.offset[0]);

            // (1) Baseline: whole-dataset HDF5 decode (single-thread, locked).
            let t0 = Instant::now();
            let whole = ds.read_raw::<f32>()?;
            let t_serial = t0.elapsed().as_secs_f64();

            // (2a) Raw compressed chunk reads (serial, cheap).
            let dsid = ds.id();
            let t1 = Instant::now();
            let raw: Vec<Vec<u8>> = infos
                .iter()
                .map(|ci| {
                    let mut buf = vec![0u8; ci.size as usize];
                    let mut fmask: u32 = 0;
                    // Vendored HDF5 is 2.0.0 → H5Dread_chunk2 (extra in/out buf_size).
                    let mut bsize: usize = buf.len();
                    let rc = unsafe {
                        hdf5_sys::h5d::H5Dread_chunk(
                            dsid,
                            0, // H5P_DEFAULT
                            ci.offset.as_ptr(),
                            &mut fmask,
                            buf.as_mut_ptr().cast(),
                            &mut bsize,
                        )
                    };
                    assert!(rc >= 0, "H5Dread_chunk failed");
                    buf
                })
                .collect();
            let t_raw = t1.elapsed().as_secs_f64();

            // (2b) Parallel inflate (rayon, flate2 — outside the HDF5 lock).
            let t2 = Instant::now();
            let inflated: Vec<Vec<f32>> = raw
                .par_iter()
                .map(|cbuf| {
                    let mut d = ZlibDecoder::new(&cbuf[..]);
                    let mut out: Vec<u8> = Vec::new();
                    d.read_to_end(&mut out).expect("inflate");
                    out.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect()
                })
                .collect();
            let t_inflate = t2.elapsed().as_secs_f64();

            // Concatenate inflated chunks (offset order) and truncate the
            // padded tail of the last chunk to the dataset length.
            let mut flat: Vec<f32> = Vec::with_capacity(whole.len());
            for v in &inflated {
                flat.extend_from_slice(v);
            }
            flat.truncate(whole.len());
            let ok = flat.len() == whole.len() && flat == whole;
            Ok((t_serial, t_raw, t_inflate, n, whole.len(), ok))
        })
        .map_err(py_err)?;

    let d = pyo3::types::PyDict::new_bound(py);
    d.set_item("serial_hdf5_s", t_serial)?;
    d.set_item("raw_read_s", t_raw)?;
    d.set_item("parallel_inflate_s", t_inflate)?;
    d.set_item("candidate_total_s", t_raw + t_inflate)?;
    d.set_item("speedup", t_serial / (t_raw + t_inflate))?;
    d.set_item("n_chunks", n_chunks)?;
    d.set_item("n_elems", n_elems)?;
    d.set_item("correct", ok)?;
    Ok(d.into_any().unbind())
}
```
