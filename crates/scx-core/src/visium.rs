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

/// One spot's (or Visium HD bin's) row of the positions table.
struct Spot {
    in_tissue: i64,
    array_row: i64,
    array_col: i64,
    pxl_row: f64,
    pxl_col: f64,
}

/// The positions of `barcodes`, in order. Space Ranger 2+ writes
/// `tissue_positions.csv` with a header, 1.x `tissue_positions_list.csv`
/// without, Visium HD `tissue_positions.parquet`; all have the same six
/// columns.
fn positions(spatial: &Path, barcodes: &[String]) -> Result<Vec<Spot>> {
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
    let rows = if path.extension().is_some_and(|e| e == "parquet") {
        read_parquet(&path)?
    } else {
        read_csv(&path)?
    };
    join(rows, barcodes)
        .map_err(|b| ScxError::InvalidFormat(format!("barcode {b} not in {}", path.display())))
}

/// `rows` picked out in `barcodes` order; on failure, the missing barcode.
/// Space Ranger lists the filtered barcodes in the positions' order, so one
/// forward walk usually does; a hash join covers any other order.
fn join(rows: Vec<(String, Spot)>, barcodes: &[String]) -> std::result::Result<Vec<Spot>, String> {
    let mut want = barcodes.iter().peekable();
    if rows
        .iter()
        .filter(|(b, _)| want.next_if(|w| *w == b).is_some())
        .count()
        == barcodes.len()
    {
        let mut want = barcodes.iter().peekable();
        return Ok(rows
            .into_iter()
            .filter(|(b, _)| want.next_if(|w| *w == b).is_some())
            .map(|(_, s)| s)
            .collect());
    }
    let mut by_barcode: HashMap<String, Spot> = rows.into_iter().collect();
    barcodes
        .iter()
        .map(|b| by_barcode.remove(b).ok_or_else(|| b.clone()))
        .collect()
}

fn read_csv(path: &Path) -> Result<Vec<(String, Spot)>> {
    let text = std::fs::read_to_string(path)?;
    let bad = |line: &str| ScxError::InvalidFormat(format!("{}: bad row '{line}'", path.display()));
    text.lines()
        .filter(|l| !l.is_empty() && !l.starts_with("barcode"))
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let n: Vec<f64> = f
                .get(1..6)
                .filter(|_| f.len() == 6)
                .ok_or_else(|| bad(line))?
                .iter()
                .map(|s| s.trim().parse::<f64>())
                .collect::<std::result::Result<_, _>>()
                .map_err(|_| bad(line))?;
            Ok((f[0].to_string(), spot(&n)))
        })
        .collect()
}

/// `[in_tissue, array_row, array_col, pxl_row, pxl_col]` as a [`Spot`].
fn spot(n: &[f64]) -> Spot {
    Spot {
        in_tissue: n[0] as i64,
        array_row: n[1] as i64,
        array_col: n[2] as i64,
        pxl_row: n[3],
        pxl_col: n[4],
    }
}

/// The six position columns, read whole per row group (the record API
/// allocates every row's column names: 35 s for a 2 um Visium HD bin).
fn read_parquet(path: &Path) -> Result<Vec<(String, Spot)>> {
    use parquet::basic::Type;
    use parquet::column::reader::get_typed_column_reader;
    use parquet::data_type::{
        ByteArrayType, DataType, DoubleType, FloatType, Int32Type, Int64Type,
    };
    use parquet::file::reader::{FileReader, RowGroupReader, SerializedFileReader};

    let bad = |what: String| ScxError::InvalidFormat(format!("{}: {what}", path.display()));
    let err = |e: parquet::errors::ParquetError| bad(e.to_string());
    fn all<T: DataType>(rg: &dyn RowGroupReader, i: usize) -> parquet::errors::Result<Vec<T::T>> {
        let mut r = get_typed_column_reader::<T>(rg.get_column_reader(i)?);
        let mut v = Vec::new();
        r.read_records(usize::MAX, None, None, &mut v)?;
        Ok(v)
    }

    let reader = SerializedFileReader::new(std::fs::File::open(path)?).map_err(err)?;
    let schema = reader.metadata().file_metadata().schema_descr_ptr();
    let col = |name: &str| {
        (0..schema.num_columns())
            .find(|&i| schema.column(i).name() == name)
            .ok_or_else(|| bad(format!("no column '{name}'")))
    };
    let bc = col("barcode")?;
    let nums = [
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    .map(col);
    let mut barcodes: Vec<String> = Vec::new();
    let mut n: [Vec<f64>; 5] = Default::default();
    for g in 0..reader.num_row_groups() {
        let rg = reader.get_row_group(g).map_err(err)?;
        if schema.column(bc).physical_type() != Type::BYTE_ARRAY {
            return Err(bad("barcode is not a string column".into()));
        }
        for b in all::<ByteArrayType>(&*rg, bc).map_err(err)? {
            barcodes.push(b.as_utf8().map_err(err)?.to_string());
        }
        for (out, i) in n.iter_mut().zip(&nums) {
            let i = *i.as_ref().map_err(|e| bad(e.to_string()))?;
            // Space Ranger's uint8/uint32 are INT32 underneath; positions fit.
            match schema.column(i).physical_type() {
                Type::INT32 => out.extend(
                    all::<Int32Type>(&*rg, i)
                        .map_err(err)?
                        .iter()
                        .map(|&x| x as f64),
                ),
                Type::INT64 => out.extend(
                    all::<Int64Type>(&*rg, i)
                        .map_err(err)?
                        .iter()
                        .map(|&x| x as f64),
                ),
                Type::FLOAT => out.extend(
                    all::<FloatType>(&*rg, i)
                        .map_err(err)?
                        .iter()
                        .map(|&x| x as f64),
                ),
                Type::DOUBLE => out.extend(all::<DoubleType>(&*rg, i).map_err(err)?),
                t => {
                    return Err(bad(format!(
                        "column {} is {t}, not a number",
                        schema.column(i).name()
                    )))
                }
            }
        }
    }
    if n.iter().any(|c| c.len() != barcodes.len()) {
        return Err(bad("position columns differ in length".into()));
    }
    Ok(barcodes
        .into_iter()
        .enumerate()
        .map(|(k, b)| (b, spot(&[n[0][k], n[1][k], n[2][k], n[3][k], n[4][k]])))
        .collect())
}

/// obs columns and obsm `spatial` for `barcodes`.
pub(crate) fn obs_and_spatial(
    spatial: &Path,
    barcodes: &[String],
) -> Result<(Vec<Column>, DenseMatrix)> {
    let spots = positions(spatial, barcodes)?;
    let col = |name: &str, f: fn(&Spot) -> i64| {
        Column::new(
            name.to_string(),
            ColumnData::Int64(spots.iter().map(f).collect()),
        )
    };
    let columns = vec![
        col("in_tissue", |s| s.in_tissue),
        col("array_row", |s| s.array_row),
        col("array_col", |s| s.array_col),
    ];
    let xy = DenseMatrix {
        shape: (spots.len(), 2),
        data: spots.iter().flat_map(|s| [s.pxl_col, s.pxl_row]).collect(),
    };
    Ok((columns, xy))
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
