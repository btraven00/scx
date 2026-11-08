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
//! TODO: Visium HD (`tissue_positions.parquet`, binned outputs) and the
//! full-resolution microscope image, which Space Ranger does not copy into `outs/`.

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

/// One spot's row of `tissue_positions.csv`.
struct Spot {
    in_tissue: i64,
    array_row: i64,
    array_col: i64,
    pxl_row: i64,
    pxl_col: i64,
}

/// The positions of `barcodes`, in order. Space Ranger 2+ writes
/// `tissue_positions.csv` with a header, 1.x `tissue_positions_list.csv`
/// without; both have the same six columns.
fn positions(spatial: &Path, barcodes: &[String]) -> Result<Vec<Spot>> {
    let path = ["tissue_positions.csv", "tissue_positions_list.csv"]
        .iter()
        .map(|f| spatial.join(f))
        .find(|p| p.is_file())
        .ok_or_else(|| {
            ScxError::InvalidFormat(format!("no tissue_positions.csv in {}", spatial.display()))
        })?;
    let text = std::fs::read_to_string(&path)?;
    let bad = |line: &str| ScxError::InvalidFormat(format!("{}: bad row '{line}'", path.display()));
    let mut spots = HashMap::new();
    for line in text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with("barcode"))
    {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 6 {
            return Err(bad(line));
        }
        let n: Vec<i64> = f[1..]
            .iter()
            .map(|s| s.trim().parse::<f64>().map(|x| x.round() as i64))
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| bad(line))?;
        spots.insert(
            f[0],
            Spot {
                in_tissue: n[0],
                array_row: n[1],
                array_col: n[2],
                pxl_row: n[3],
                pxl_col: n[4],
            },
        );
    }
    barcodes
        .iter()
        .map(|b| {
            spots.remove(b.as_str()).ok_or_else(|| {
                ScxError::InvalidFormat(format!("barcode {b} not in {}", path.display()))
            })
        })
        .collect()
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
        data: spots
            .iter()
            .flat_map(|s| [s.pxl_col as f64, s.pxl_row as f64])
            .collect(),
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
