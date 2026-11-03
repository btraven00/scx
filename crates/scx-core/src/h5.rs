//! HDF5 building blocks shared by the format readers: typed reads of index
//! and value datasets that accept every integer width and signedness the
//! formats use.

use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
use hdf5::File;
use std::path::Path;

use crate::{
    dtype::{DataType, TypedVec},
    error::{Result, ScxError},
    h5_chunk,
    ir::MatrixChunk,
};

// ---------------------------------------------------------------------------
// Untimed object creation shared by the HDF5 writers
// ---------------------------------------------------------------------------

// HDF5 stamps each group's object header with its creation time by default,
// so converting the same input twice gives different bytes. hdf5-metno
// already turns this off for datasets; these two cover the root group and
// every other group. Writers must create files and groups through them;
// clippy.toml bans the timed `File::create` / `Group::create_group`.

/// `File::create` with no timestamp on the root group.
pub(crate) fn create_file(path: impl AsRef<Path>) -> hdf5::Result<File> {
    File::with_options()
        .with_fcpl(|p| p.obj_track_times(false))
        .create(path)
}

/// `parent.create_group(name)` with no timestamp on the group.
pub(crate) fn create_group(parent: &hdf5::Group, name: &str) -> hdf5::Result<hdf5::Group> {
    parent
        .create_group_builder()
        .with_gcpl(|p| p.obj_track_times(false))
        .create(name)
}

/// Paths of the groups in `root` (itself included) that carry a timestamp,
/// as h5py/anndata writes them. HDF5 has no call to remove one.
pub(crate) fn timed_groups(root: &hdf5::Group) -> hdf5::Result<Vec<String>> {
    fn walk(g: &hdf5::Group, timed: &mut Vec<String>) -> hdf5::Result<()> {
        let i = g.loc_info()?;
        if i.ctime != 0 || i.mtime != 0 || i.btime != 0 {
            timed.push(g.name());
        }
        for c in g.groups()? {
            walk(&c, timed)?;
        }
        Ok(())
    }
    let mut timed = Vec::new();
    walk(root, &mut timed)?;
    Ok(timed)
}

/// Test helper: panic if any group in `path` carries a timestamp.
#[cfg(test)]
pub(crate) fn assert_untimed(path: impl AsRef<Path>) {
    let timed = timed_groups(&File::open(path).unwrap()).unwrap();
    assert!(timed.is_empty(), "groups with timestamps: {timed:?}");
}

/// Test helper: add a timestamped group `name` to the file at `path`, as h5py
/// writes them. Needs the v1.8+ format: with libhdf5 1.10's default (earliest)
/// format a group gets a v1 object header, which has no room for times.
#[cfg(test)]
pub(crate) fn add_timed_group(path: impl AsRef<Path>, name: &str) {
    File::with_options()
        .with_fapl(|p| p.libver_v18())
        .open_rw(path)
        .unwrap()
        .create_group(name)
        .unwrap();
}

// ---------------------------------------------------------------------------
// Typed reads shared by the HDF5 readers (h5ad, H5Seurat, 10x)
// ---------------------------------------------------------------------------

/// Fail with a clear error when `ds` is compressed with a filter libhdf5 has
/// no decoder for: hdf5plugin's blosc, zstd, bitshuffle, ... Those show up as
/// `Filter::User`; lzf is compiled in, so it parses as `Filter::LZF`.
pub(crate) fn check_filters(ds: &hdf5::Dataset) -> Result<()> {
    for f in ds.filters() {
        if let hdf5::filters::Filter::User(id, _) = f {
            let name = match id {
                32001 => "blosc",
                32004 => "lz4",
                32008 => "bitshuffle",
                32015 => "zstd",
                32026 => "blosc2",
                _ => "unknown",
            };
            return Err(ScxError::UnsupportedFilter(format!(
                "dataset {} uses HDF5 filter {name} (id {id}), which scx can't decode; supported: {}",
                ds.name(),
                crate::error::SUPPORTED_FILTERS
            )));
        }
    }
    Ok(())
}

/// Read an index dataset (indptr, shape, ...) as u64, whatever its integer
/// width or signedness. Float is accepted too: rhdf5 writes R doubles.
pub(crate) fn read_u64(ds: &hdf5::Dataset) -> Result<Vec<u64>> {
    read_u64_range(ds, 0..ds.size())
}

/// [`read_u64`] of elements `range` of a 1-D dataset.
pub(crate) fn read_u64_range(
    ds: &hdf5::Dataset,
    range: std::ops::Range<usize>,
) -> Result<Vec<u64>> {
    check_filters(ds)?;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Unsigned(_) => read_vec::<u64>(ds, range),
        TypeDescriptor::Integer(_) => read_vec::<i64>(ds, range)?
            .into_iter()
            .map(|x| {
                u64::try_from(x).map_err(|_| {
                    ScxError::InvalidFormat(format!("negative index {x} in {}", ds.name()))
                })
            })
            .collect(),
        TypeDescriptor::Float(_) => Ok(read_vec::<f64>(ds, range)?
            .into_iter()
            .map(|x| x as u64)
            .collect()),
        other => Err(ScxError::InvalidFormat(format!(
            "{}: expected integers, found {other:?}",
            ds.name()
        ))),
    }
}

/// Read `range` of an integer index dataset (indices) as u32. A 4-byte,
/// deflate-only chunked dataset (the common case) is inflated in parallel;
/// column indices are non-negative, so reading signed storage as u32 is exact.
pub(crate) fn read_u32_range(
    ds: &hdf5::Dataset,
    range: std::ops::Range<usize>,
) -> Result<Vec<u32>> {
    check_filters(ds)?;
    let four = ds.dtype()?.size() == 4;
    match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_) if four => {
            if let Some(plan) = h5_chunk::chunk_plan(ds) {
                return h5_chunk::read_range_typed(ds, range.start, range.end, plan);
            }
            // Read as stored, without libhdf5's i32 -> u32 conversion pass.
            match ds.dtype()?.to_descriptor()? {
                TypeDescriptor::Integer(_) => Ok(bytemuck::cast_vec(read_vec::<i32>(ds, range)?)),
                _ => read_vec::<u32>(ds, range),
            }
        }
        // Other widths (int64 from scipy, ...): libhdf5 converts.
        TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_) => read_vec::<u32>(ds, range),
        other => Err(ScxError::InvalidFormat(format!(
            "{}: expected integers, found {other:?}",
            ds.name()
        ))),
    }
}

/// The DataType a matrix values dataset is read as. Integers of any width map
/// to I32 (signed) or U32 (unsigned): counts fit in 32 bits.
pub(crate) fn value_dtype(ds: &hdf5::Dataset) -> Result<DataType> {
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::Float(FloatSize::U4) => DataType::F32,
        TypeDescriptor::Float(_) => DataType::F64,
        TypeDescriptor::Integer(_) => DataType::I32,
        TypeDescriptor::Unsigned(_) | TypeDescriptor::Boolean => DataType::U32,
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "{}: unsupported matrix value type {other:?}",
                ds.name()
            )))
        }
    })
}

/// Rows `rows` of a compressed-sparse group (`{group}/indices`, `data`),
/// given its full `indptr`, as a chunk with `n_cols` columns.
pub(crate) fn read_csr_rows(
    file: &File,
    group: &str,
    indptr: &[u64],
    rows: std::ops::Range<usize>,
    n_cols: usize,
    dtype: DataType,
) -> Result<MatrixChunk> {
    CsrReader::open(file, group, dtype, false)?.rows(indptr, rows, n_cols)
}

/// The `indices` and `data` of a compressed-sparse group, opened once for a
/// stream of row blocks: deflate datasets get a [`h5_chunk::ChunkReader`]
/// each, which keeps its chunk lookups across blocks.
pub(crate) struct CsrReader {
    indices: hdf5::Dataset,
    data: hdf5::Dataset,
    dtype: DataType,
    fast_indices: Option<h5_chunk::ChunkReader>,
    fast_data: Option<h5_chunk::ChunkReader>,
}

impl CsrReader {
    /// `whole`: the stream will read every row, so list all chunk locations
    /// up front rather than per block.
    pub(crate) fn open(file: &File, group: &str, dtype: DataType, whole: bool) -> Result<Self> {
        let indices = file.dataset(&format!("{group}/indices"))?;
        let data = file.dataset(&format!("{group}/data"))?;
        check_filters(&indices)?;
        check_filters(&data)?;
        // The cases read_u32_range and read_values decode in parallel.
        let fast_indices = match indices.dtype()?.to_descriptor()? {
            TypeDescriptor::Integer(IntSize::U4) | TypeDescriptor::Unsigned(IntSize::U4) => {
                h5_chunk::ChunkReader::new(&indices, whole)?
            }
            _ => None,
        };
        let fast_data = if value_dtype(&data)? == dtype && data.dtype()?.size() == dtype.size() {
            h5_chunk::ChunkReader::new(&data, whole)?
        } else {
            None
        };
        Ok(Self {
            indices,
            data,
            dtype,
            fast_indices,
            fast_data,
        })
    }

    /// Rows `rows` of the group, given `indptr` over at least `..=rows.end`,
    /// as a chunk with `n_cols` columns. `indices` and `data` decode at once.
    pub(crate) fn rows(
        &self,
        indptr: &[u64],
        rows: std::ops::Range<usize>,
        n_cols: usize,
    ) -> Result<MatrixChunk> {
        let nnz = indptr[rows.start] as usize..indptr[rows.end] as usize;
        let (indices, data) = if nnz.is_empty() {
            (Vec::new(), TypedVec::empty(self.dtype))
        } else {
            let indices = || match &self.fast_indices {
                Some(r) => r.read_typed(nnz.start, nnz.end),
                None => read_u32_range(&self.indices, nnz.clone()),
            };
            let data = || match &self.fast_data {
                Some(r) => read_values_with(r, self.dtype, nnz.clone()),
                None => read_values(&self.data, self.dtype, nnz.clone()),
            };
            // libhdf5 reads serialize on its lock, so only overlap ours.
            let (i, d) = if self.fast_indices.is_some() || self.fast_data.is_some() {
                rayon::join(indices, data)
            } else {
                (indices(), data())
            };
            (i?, d?)
        };
        Ok(crate::sparse::csr_chunk(
            indptr, rows, n_cols, indices, data,
        ))
    }
}

fn read_values_with(
    r: &h5_chunk::ChunkReader,
    dtype: DataType,
    range: std::ops::Range<usize>,
) -> Result<TypedVec> {
    let (a, b) = (range.start, range.end);
    Ok(match dtype {
        DataType::F32 => TypedVec::F32(r.read_typed(a, b)?),
        DataType::F64 => TypedVec::F64(r.read_typed(a, b)?),
        DataType::I32 => TypedVec::I32(r.read_typed(a, b)?),
        DataType::U32 => TypedVec::U32(r.read_typed(a, b)?),
    })
}

/// Read `range` of a values dataset as `dtype`. When that is the stored type
/// and the dataset is deflate-only chunked, chunks are inflated in parallel;
/// otherwise HDF5 reads and converts.
pub(crate) fn read_values(
    ds: &hdf5::Dataset,
    dtype: DataType,
    range: std::ops::Range<usize>,
) -> Result<TypedVec> {
    check_filters(ds)?;
    if value_dtype(ds)? == dtype && ds.dtype()?.size() == dtype.size() {
        if let Some(plan) = h5_chunk::chunk_plan(ds) {
            return read_values_parallel(ds, dtype, range, plan);
        }
    }
    Ok(match dtype {
        DataType::F32 => TypedVec::F32(read_vec(ds, range)?),
        DataType::F64 => TypedVec::F64(read_vec(ds, range)?),
        DataType::I32 => TypedVec::I32(read_vec(ds, range)?),
        DataType::U32 => TypedVec::U32(read_vec(ds, range)?),
    })
}

/// `range` of a 1-D dataset as `T`, read by libhdf5 (converting if the stored
/// type differs) straight into a [`h5_chunk::zeroed_vec_huge`] buffer.
pub(crate) fn read_vec<T: hdf5::H5Type + bytemuck::Pod>(
    ds: &hdf5::Dataset,
    range: std::ops::Range<usize>,
) -> Result<Vec<T>> {
    let mut v = h5_chunk::zeroed_vec_huge::<T>(range.len());
    if v.is_empty() {
        return Ok(v);
    }
    let fspace = ds.space()?.select(range.clone())?;
    let mspace = hdf5::Dataspace::try_new(v.len())?;
    let mtype = hdf5::Datatype::from_type::<T>()?;
    // Raw hdf5-sys FFI: hold the process-global HDF5 lock (see h5_chunk.rs).
    let rc = hdf5::sync::sync(|| unsafe {
        hdf5_sys::h5d::H5Dread(
            ds.id(),
            mtype.id(),
            mspace.id(),
            fspace.id(),
            hdf5_sys::h5p::H5P_DEFAULT,
            v.as_mut_ptr().cast(),
        )
    });
    if rc < 0 {
        return Err(ScxError::InvalidFormat(format!(
            "{}: H5Dread of elements {}..{} failed",
            ds.name(),
            range.start,
            range.end
        )));
    }
    Ok(v)
}

/// `range` of `ds` as `dtype`, decoded in parallel by `plan`. The caller has
/// checked that `dtype` has the stored element layout.
pub(crate) fn read_values_parallel(
    ds: &hdf5::Dataset,
    dtype: DataType,
    range: std::ops::Range<usize>,
    plan: h5_chunk::ChunkPlan,
) -> Result<TypedVec> {
    read_values_with(
        &h5_chunk::ChunkReader::with_plan(ds, plan, false)?,
        dtype,
        range,
    )
}

// ---------------------------------------------------------------------------
// Generic tree walk, for `scx inspect` on HDF5 files of no known format
// ---------------------------------------------------------------------------

/// A node in the HDF5 file tree.
pub struct H5Node {
    pub name: String,
    pub kind: H5NodeKind,
}

pub enum H5NodeKind {
    Dataset {
        shape: Vec<usize>,
        dtype: String,
    },
    Group {
        children: Vec<H5Node>,
        /// Number of children that were omitted due to depth limit.
        truncated: usize,
    },
}

/// Walk the root of an HDF5 file up to `max_depth` levels deep.
pub fn walk_h5(path: &Path, max_depth: usize) -> Result<Vec<H5Node>> {
    let file = File::open(path)?;
    let root = file
        .group("/")
        .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
    walk_group(&file, &root, max_depth)
}

fn walk_group(file: &File, grp: &hdf5::Group, depth: usize) -> Result<Vec<H5Node>> {
    let names = grp.member_names().unwrap_or_default();
    let mut nodes = Vec::with_capacity(names.len());

    for name in &names {
        let full_path = {
            let grp_name = grp.name();
            if grp_name == "/" {
                format!("/{name}")
            } else {
                format!("{grp_name}/{name}")
            }
        };

        let is_group = file.group(&full_path).is_ok() && file.dataset(&full_path).is_err();

        let kind = if is_group {
            if depth == 0 {
                H5NodeKind::Group {
                    children: Vec::new(),
                    truncated: file
                        .group(&full_path)
                        .ok()
                        .and_then(|g| g.member_names().ok())
                        .map(|v| v.len())
                        .unwrap_or(0),
                }
            } else {
                let child_grp = file
                    .group(&full_path)
                    .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
                let children = walk_group(file, &child_grp, depth - 1)?;
                H5NodeKind::Group {
                    children,
                    truncated: 0,
                }
            }
        } else {
            match file.dataset(&full_path) {
                Ok(ds) => {
                    let shape = ds.shape();
                    let dtype = dtype_str(&ds);
                    H5NodeKind::Dataset { shape, dtype }
                }
                Err(_) => continue,
            }
        };

        nodes.push(H5Node {
            name: name.clone(),
            kind,
        });
    }

    nodes.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(nodes)
}

fn dtype_str(ds: &hdf5::Dataset) -> String {
    match ds.dtype().and_then(|d| d.to_descriptor()) {
        Ok(TypeDescriptor::Float(s)) => format!("f{}", (s as usize) * 8),
        Ok(TypeDescriptor::Integer(s)) => format!("i{}", (s as usize) * 8),
        Ok(TypeDescriptor::Unsigned(s)) => format!("u{}", (s as usize) * 8),
        Ok(TypeDescriptor::Boolean) => "bool".into(),
        Ok(TypeDescriptor::VarLenUnicode) => "str".into(),
        Ok(TypeDescriptor::VarLenAscii) => "str".into(),
        Ok(TypeDescriptor::FixedAscii(n)) => format!("str[{n}]"),
        Ok(TypeDescriptor::FixedUnicode(n)) => format!("str[{n}]"),
        _ => "?".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("scx_h5_{name}_{}.h5", std::process::id()))
    }

    /// HDF5 creates missing parents of a nested path with the leaf's gcpl,
    /// so they are untimed too; writers rely on this for `graphs/<name>`.
    /// A timed group is caught, which proves the check bites.
    #[test]
    fn nested_create_group_is_untimed() {
        let path = tmp("nested_untimed");
        create_group(&create_file(&path).unwrap(), "a/b/c").unwrap();
        assert_untimed(&path);
        add_timed_group(&path, "timed");
        let caught = std::panic::catch_unwind(|| assert_untimed(&path));
        let _ = std::fs::remove_file(&path);
        assert!(caught.is_err(), "assert_untimed missed a timed root");
    }

    /// h5py's `compression="lzf"` (anndata's second option) must decode.
    #[test]
    fn lzf_values_read_back() {
        let path = tmp("lzf");
        let vals: Vec<f32> = (0..5000).map(|i| (i % 97) as f32 * 0.5).collect();
        File::create(&path)
            .unwrap()
            .new_dataset::<f32>()
            .shape([vals.len()])
            .chunk([1000])
            .lzf()
            .create("data")
            .unwrap()
            .write(&vals)
            .unwrap();
        let ds = File::open(&path).unwrap().dataset("data").unwrap();
        let got = read_values(&ds, DataType::F32, 100..4100).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(matches!(got, TypedVec::F32(v) if v == vals[100..4100]));
    }

    /// Unfiltered reads hand back libhdf5's own Vec and read 4-byte indices
    /// as stored; both must equal what a converting libhdf5 read gives.
    #[test]
    fn unfiltered_reads_match_converting_reads() {
        let path = tmp("unfiltered");
        let n = 3000;
        {
            let f = File::create(&path).unwrap();
            let i32s: Vec<i32> = (0..n as i32).map(|i| i * 7919 % 33_000).collect();
            let i64s: Vec<i64> = i32s.iter().map(|&x| x as i64).collect();
            let u32s: Vec<u32> = i32s.iter().map(|&x| x as u32).collect();
            let f32s: Vec<f32> = (0..n).map(|i| i as f32 * 0.37).collect();
            // Contiguous, plus one chunked but unfiltered.
            f.new_dataset_builder()
                .with_data(&i32s)
                .create("i32")
                .unwrap();
            f.new_dataset_builder()
                .with_data(&i64s)
                .create("i64")
                .unwrap();
            f.new_dataset_builder()
                .with_data(&u32s)
                .create("u32")
                .unwrap();
            f.new_dataset_builder()
                .chunk(1000)
                .with_data(&f32s)
                .create("f32")
                .unwrap();
        }
        let f = File::open(&path).unwrap();
        for (a, b) in [(0, n), (999, 1001), (1500, 2700), (5, 6)] {
            for name in ["i32", "i64", "u32"] {
                let ds = f.dataset(name).unwrap();
                let want = ds.read_slice_1d::<u32, _>(s![a..b]).unwrap().to_vec();
                assert_eq!(read_u32_range(&ds, a..b).unwrap(), want, "{name} {a}..{b}");
            }
            let ds = f.dataset("f32").unwrap();
            let want = ds.read_slice_1d::<f32, _>(s![a..b]).unwrap().to_vec();
            let got = read_values(&ds, DataType::F32, a..b).unwrap();
            assert!(matches!(got, TypedVec::F32(v) if v == want), "f32 {a}..{b}");
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Reads of 4 MB and more go into a hugepage-advised buffer (and the
    /// parallel inflate output does too); the values must not change.
    #[test]
    fn large_reads_into_hugepage_buffers_match() {
        let path = tmp("large");
        let n = 1_500_000; // 6 MB of f32
        let vals: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
        {
            let f = File::create(&path).unwrap();
            f.new_dataset_builder()
                .with_data(&vals)
                .create("plain")
                .unwrap();
            f.new_dataset_builder()
                .chunk(100_000)
                .deflate(1)
                .with_data(&vals)
                .create("gz")
                .unwrap();
        }
        let f = File::open(&path).unwrap();
        for name in ["plain", "gz"] {
            let ds = f.dataset(name).unwrap();
            for r in [0..n, 12_345..n - 6789] {
                let got = read_values(&ds, DataType::F32, r.clone()).unwrap();
                assert!(
                    matches!(got, TypedVec::F32(v) if v == vals[r.clone()]),
                    "{name} {r:?}"
                );
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A dataset filtered by zstd (32015), which this build has no decoder for.
    /// hdf5-metno refuses to create one, so set the filter as optional through
    /// the C API and store the chunk raw, claiming the filter was applied.
    fn write_zstd_dataset(path: &Path) {
        use hdf5_sys::{h5d, h5p, h5s, h5t, h5z};
        let f = File::create(path).unwrap();
        let dims = [100u64];
        let buf = vec![1f32; 100];
        hdf5::sync::sync(|| unsafe {
            let dcpl = h5p::H5Pcreate(*h5p::H5P_CLS_DATASET_CREATE);
            h5p::H5Pset_chunk(dcpl, 1, dims.as_ptr());
            h5p::H5Pset_filter(dcpl, 32015, h5z::H5Z_FLAG_OPTIONAL, 0, std::ptr::null());
            let space = h5s::H5Screate_simple(1, dims.as_ptr(), std::ptr::null());
            let name = c"data";
            let ds = h5d::H5Dcreate2(
                f.id(),
                name.as_ptr(),
                *h5t::H5T_NATIVE_FLOAT,
                space,
                h5p::H5P_DEFAULT,
                dcpl,
                h5p::H5P_DEFAULT,
            );
            assert!(ds >= 0);
            let offset = [0u64];
            let w = h5d::H5Dwrite_chunk(
                ds,
                h5p::H5P_DEFAULT,
                0,
                offset.as_ptr(),
                400,
                buf.as_ptr().cast(),
            );
            assert!(w >= 0);
            h5d::H5Dclose(ds);
            h5s::H5Sclose(space);
            h5p::H5Pclose(dcpl);
        });
    }

    /// Regression: an unsupported filter leaked libhdf5's compiled-in plugin
    /// directory ("can't open directory (/home/runner/...)") as the error.
    #[test]
    fn unsupported_filter_is_named_not_leaked() {
        let path = tmp("zstd");
        write_zstd_dataset(&path);
        let ds = File::open(&path).unwrap().dataset("data").unwrap();

        let err = read_values(&ds, DataType::F32, 0..100)
            .unwrap_err()
            .to_string();
        assert!(err.contains("zstd (id 32015)"), "{err}");
        assert!(err.contains("/data"), "{err}");

        // Any other read path goes through `From<hdf5::Error>`.
        let err = ScxError::from(ds.read_raw::<f32>().unwrap_err()).to_string();
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("can't decode"), "{err}");
        assert!(!err.contains("plugin"), "{err}");
    }

    #[test]
    fn walk_h5_tree_and_depth_truncation() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny/tiny_10x.h5");
        let nodes = walk_h5(&p, 3).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "matrix");
        let H5NodeKind::Group {
            children,
            truncated,
        } = &nodes[0].kind
        else {
            panic!("matrix should be a group");
        };
        assert_eq!(*truncated, 0);
        let names: Vec<&str> = children.iter().map(|n| n.name.as_str()).collect();
        for expected in ["barcodes", "data", "features", "indices", "indptr", "shape"] {
            assert!(names.contains(&expected), "missing {expected}: {names:?}");
        }

        // depth 0: the matrix group's children are omitted but counted.
        let shallow = walk_h5(&p, 0).unwrap();
        let H5NodeKind::Group {
            children,
            truncated,
        } = &shallow[0].kind
        else {
            panic!("matrix should be a group");
        };
        assert!(children.is_empty());
        assert!(*truncated > 0);
    }
}
