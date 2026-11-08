//! 10x Visium: a Space Ranger `outs/` directory, read as its filtered 10x HDF5
//! matrix plus the `spatial/` folder, laid out the way `scanpy.read_visium`
//! does:
//!
//!   spatial/tissue_positions.csv     → obs `in_tissue`, `array_row`, `array_col`;
//!                                      obsm `spatial` = (pxl_col, pxl_row) in full-res pixels
//!   spatial/scalefactors_json.json   → uns `spatial/<library>/scalefactors`
//!   spatial/tissue_{hires,lowres}_image.png → uns `spatial/<library>/images/{hires,lowres}`
//!
//! `<library>` is the matrix file's `library_ids` attribute, `metadata` its
//! `chemistry_description` and `software_version`. Images stay in their PNG
//! sample type (uint8, as CELLxGENE stores them), not scanpy's float32 in [0, 1].
//!
//! Visium HD reads one bin size at a time: pass a bin directory, e.g.
//! `outs/binned_outputs/square_008um`, which has the same layout with
//! `tissue_positions.parquet`.
//!
//! TODO: the full-resolution microscope image, which Space Ranger does not
//! copy into `outs/`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::error::{Result, ScxError};
use crate::ir::{Column, ColumnData, DenseMatrix, UnsArray, UnsData, UnsTable};

pub(crate) const MATRIX: &str = "filtered_feature_bc_matrix.h5";

/// Whether `dir` looks like a Space Ranger `outs/` directory.
pub(crate) fn is_outs_dir(dir: &Path) -> bool {
    dir.join(MATRIX).is_file() && dir.join("spatial").is_dir()
}

/// obs columns `in_tissue`, `array_row`, `array_col` and obsm `spatial`
/// (pxl_col, pxl_row in full-resolution pixels) for `barcodes`, in order.
/// Space Ranger 2+ writes `tissue_positions.csv` with a header, 1.x
/// `tissue_positions_list.csv` without, Visium HD
/// `tissue_positions.parquet`; all have the same six columns.
///
/// Rows are matched as they are read, straight into the output columns: a
/// 2 um Visium HD bin lists ~11M positions for 6.3M filtered barcodes, and
/// the matrix lists its barcodes in no particular order.
pub(crate) fn obs_and_spatial(
    spatial: &Path,
    barcodes: &[String],
) -> Result<(Vec<Column>, DenseMatrix)> {
    let path = [
        "tissue_positions.parquet",
        "tissue_positions.csv",
        "tissue_positions_list.csv",
    ]
    .iter()
    .map(|f| spatial.join(f))
    .find(|p| p.is_file())
    .ok_or_else(|| {
        ScxError::InvalidFormat(format!("no tissue_positions in {}", spatial.display()))
    })?;
    // Fixed seeds: barcodes are not adversarial, and SipHash took 4 s here.
    let mut index: HashMap<&[u8], usize, _> = HashMap::with_capacity_and_hasher(
        barcodes.len(),
        ahash::RandomState::with_seeds(1, 2, 3, 4),
    );
    index.extend(barcodes.iter().enumerate().map(|(i, b)| (b.as_bytes(), i)));
    let n = barcodes.len();
    // `in_tissue` is 0 or 1; -1 marks a barcode no row matched.
    let (mut in_tissue, mut row, mut col) = (vec![-1i64; n], vec![0i64; n], vec![0i64; n]);
    let mut xy = vec![0f64; 2 * n];
    let mut keep = |b: &[u8], v: [f64; 5]| {
        if let Some(&i) = index.get(b) {
            in_tissue[i] = v[0] as i64;
            row[i] = v[1] as i64;
            col[i] = v[2] as i64;
            xy[2 * i] = v[4];
            xy[2 * i + 1] = v[3];
        }
    };
    if path.extension().is_some_and(|e| e == "parquet") {
        scan_parquet(&path, &mut keep)?;
    } else {
        scan_csv(&path, &mut keep)?;
    }
    if let Some(i) = in_tissue.iter().position(|&t| t < 0) {
        return Err(ScxError::InvalidFormat(format!(
            "barcode {} not in {}",
            barcodes[i],
            path.display()
        )));
    }
    let int = |name: &str, v| Column::new(name.to_string(), ColumnData::Int64(v));
    Ok((
        vec![
            int("in_tissue", in_tissue),
            int("array_row", row),
            int("array_col", col),
        ],
        DenseMatrix {
            shape: (n, 2),
            data: xy,
        },
    ))
}

/// Call `f` with each row's barcode and `[in_tissue, array_row, array_col,
/// pxl_row, pxl_col]`.
fn scan_csv(path: &Path, f: &mut dyn FnMut(&[u8], [f64; 5])) -> Result<()> {
    let text = std::fs::read_to_string(path)?;
    let bad = |line: &str| ScxError::InvalidFormat(format!("{}: bad row '{line}'", path.display()));
    for line in text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with("barcode"))
    {
        let mut fields = line.split(',');
        let barcode = fields.next().unwrap_or_default();
        let mut n = [0f64; 5];
        for x in &mut n {
            *x = fields
                .next()
                .and_then(|s| s.trim().parse().ok())
                .ok_or_else(|| bad(line))?;
        }
        if fields.next().is_some() {
            return Err(bad(line));
        }
        f(barcode.as_bytes(), n);
    }
    Ok(())
}

/// As [`scan_csv`]. The six columns are read in step, `BATCH` rows at a
/// time: the record API allocates every row's column names (35 s for a 2 um
/// Visium HD bin), and a whole row group of it is ~11M rows.
fn scan_parquet(path: &Path, f: &mut dyn FnMut(&[u8], [f64; 5])) -> Result<()> {
    use parquet::basic::Type;
    use parquet::column::reader::{get_typed_column_reader, ColumnReaderImpl};
    use parquet::data_type::{
        ByteArrayType, DataType, DoubleType, FloatType, Int32Type, Int64Type,
    };
    use parquet::errors::Result as PqResult;
    use parquet::file::reader::{FileReader, RowGroupReader, SerializedFileReader};

    const BATCH: usize = 1 << 16;
    /// A numeric column, whatever its physical type, read as f64.
    enum Num {
        I32(ColumnReaderImpl<Int32Type>, Vec<i32>),
        I64(ColumnReaderImpl<Int64Type>, Vec<i64>),
        F32(ColumnReaderImpl<FloatType>, Vec<f32>),
        F64(ColumnReaderImpl<DoubleType>, Vec<f64>),
    }
    fn next<T: DataType>(r: &mut ColumnReaderImpl<T>, v: &mut Vec<T::T>) -> PqResult<()> {
        v.clear();
        r.read_records(BATCH, None, None, v).map(|_| ())
    }
    impl Num {
        // Space Ranger's uint8/uint32 are INT32 underneath; positions fit.
        fn next(&mut self, out: &mut Vec<f64>) -> PqResult<()> {
            out.clear();
            match self {
                Num::I32(r, v) => next(r, v).map(|_| out.extend(v.iter().map(|&x| x as f64))),
                Num::I64(r, v) => next(r, v).map(|_| out.extend(v.iter().map(|&x| x as f64))),
                Num::F32(r, v) => next(r, v).map(|_| out.extend(v.iter().map(|&x| x as f64))),
                Num::F64(r, v) => next(r, v).map(|_| out.extend_from_slice(v)),
            }
        }
    }
    fn typed<T: DataType>(rg: &dyn RowGroupReader, i: usize) -> PqResult<ColumnReaderImpl<T>> {
        Ok(get_typed_column_reader::<T>(rg.get_column_reader(i)?))
    }

    let bad = |what: String| ScxError::InvalidFormat(format!("{}: {what}", path.display()));
    let err = |e: parquet::errors::ParquetError| bad(e.to_string());
    let reader = SerializedFileReader::new(std::fs::File::open(path)?).map_err(err)?;
    let schema = reader.metadata().file_metadata().schema_descr_ptr();
    let col = |name: &str| {
        (0..schema.num_columns())
            .find(|&i| schema.column(i).name() == name)
            .ok_or_else(|| bad(format!("no column '{name}'")))
    };
    let bc = col("barcode")?;
    if schema.column(bc).physical_type() != Type::BYTE_ARRAY {
        return Err(bad("barcode is not a string column".into()));
    }
    let nums = [
        col("in_tissue")?,
        col("array_row")?,
        col("array_col")?,
        col("pxl_row_in_fullres")?,
        col("pxl_col_in_fullres")?,
    ];
    let (mut barcodes, mut n) = (Vec::new(), <[Vec<f64>; 5]>::default());
    for g in 0..reader.num_row_groups() {
        let rg = reader.get_row_group(g).map_err(err)?;
        let mut bc_reader = typed::<ByteArrayType>(&*rg, bc).map_err(err)?;
        let mut readers = nums
            .iter()
            .map(|&i| {
                Ok(match schema.column(i).physical_type() {
                    Type::INT32 => Num::I32(typed(&*rg, i).map_err(err)?, Vec::new()),
                    Type::INT64 => Num::I64(typed(&*rg, i).map_err(err)?, Vec::new()),
                    Type::FLOAT => Num::F32(typed(&*rg, i).map_err(err)?, Vec::new()),
                    Type::DOUBLE => Num::F64(typed(&*rg, i).map_err(err)?, Vec::new()),
                    t => {
                        return Err(bad(format!(
                            "column {} is {t}, not a number",
                            schema.column(i).name()
                        )))
                    }
                })
            })
            .collect::<Result<Vec<Num>>>()?;
        loop {
            next(&mut bc_reader, &mut barcodes).map_err(err)?;
            if barcodes.is_empty() {
                break;
            }
            for (r, out) in readers.iter_mut().zip(&mut n) {
                r.next(out).map_err(err)?;
                if out.len() != barcodes.len() {
                    return Err(bad("position columns differ in length".into()));
                }
            }
            for (k, b) in barcodes.iter().enumerate() {
                f(b.data(), [n[0][k], n[1][k], n[2][k], n[3][k], n[4][k]]);
            }
        }
    }
    Ok(())
}

/// uns `spatial/<library>`: scale factors, metadata and the PNG images.
pub(crate) fn uns(spatial: &Path, matrix: &Path) -> Result<UnsTable> {
    let file = hdf5::File::open(matrix)?;
    let attr = |name| crate::h5_str::read_str_attr(&file, name).ok();
    let library = crate::h5_str::read_str_array_attr(&file, "library_ids")
        .ok()
        .and_then(|v| v.into_iter().next())
        .unwrap_or_else(|| "library".into());

    let mut lib = Map::new();
    let sf = spatial.join("scalefactors_json.json");
    if sf.is_file() {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&sf)?)
            .map_err(|e| ScxError::InvalidFormat(format!("{}: {e}", sf.display())))?;
        lib.insert("scalefactors".into(), v);
    }
    let mut meta = Map::new();
    for k in ["chemistry_description", "software_version"] {
        if let Some(v) = attr(k) {
            meta.insert(k.into(), json!(v));
        }
    }
    lib.insert("metadata".into(), Value::Object(meta));

    let mut arrays = std::collections::BTreeMap::new();
    for res in ["hires", "lowres"] {
        let png = spatial.join(format!("tissue_{res}_image.png"));
        if png.is_file() {
            arrays.insert(format!("spatial/{library}/images/{res}"), read_png(&png)?);
        }
    }
    Ok(UnsTable {
        raw: json!({ "spatial": { library: lib } }),
        arrays,
    })
}

/// A PNG as `[height, width, channels]`, palette and sub-byte grey expanded.
fn read_png(path: &PathBuf) -> Result<UnsArray> {
    let err = |e: png::DecodingError| ScxError::InvalidFormat(format!("{}: {e}", path.display()));
    let mut dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    dec.set_transformations(png::Transformations::EXPAND);
    let mut rd = dec.read_info().map_err(err)?;
    let mut buf = vec![0u8; rd.output_buffer_size().unwrap_or(0)];
    let info = rd.next_frame(&mut buf).map_err(err)?;
    buf.truncate(info.buffer_size());
    let shape = vec![
        info.height as usize,
        info.width as usize,
        info.color_type.samples(),
    ];
    let data = match info.bit_depth {
        png::BitDepth::Sixteen => UnsData::U16(
            buf.as_chunks::<2>()
                .0
                .iter()
                .map(|&b| u16::from_be_bytes(b))
                .collect(),
        ),
        _ => UnsData::U8(buf),
    };
    Ok(UnsArray { shape, data })
}
