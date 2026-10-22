//! Reader for bixverse binary sparse stores (file versions 3 and 4).
//!
//! A store is a cell-major `counts_cells.bin`. bixverse keeps obs and var in
//! DuckDB, not in the `.bin`, so they are read from a sidecar next to it:
//! `obs`/`var` as `.parquet` (feature `bixverse-parquet`, lossless), else
//! `.tsv.gz`, else `.tsv`. The first column is the index. Without a sidecar the
//! index is synthetic (`cell_{i}`, `gene_{i}`).
//!
//! TSV columns are typed by inference: integer, float, boolean (`TRUE`/`FALSE`),
//! else string, with a string column that repeats a value read as categorical.
//! `NA` and the empty string are missing. Values must not contain tabs or
//! newlines; there is no quoting.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use bixverse_sc_data::data_io::{DataLayerReturn, ParallelSparseReader, SingleCellReading};
use flate2::read::MultiGzDecoder;

use crate::dtype::{DataType, TypedVec};
use crate::error::{Result, ScxError};
use crate::ir::{Column, ColumnData, MatrixChunk, ObsTable, SparseMatrixCSR, VarTable};
use crate::stream::{row_chunks, ChunkStream, DatasetReader};

/// Name of the cell-major store inside a store directory.
const CELLS_BIN: &str = "counts_cells.bin";

/// Which count layer ends up in `X`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BixverseLayer {
    /// Raw counts.
    Raw,
    /// The stored normalised layer. Lossy (f16 in v3).
    Norm,
}

/// Streams a bixverse cell-major store as CSR row chunks.
pub struct BixverseReader {
    reader: ParallelSparseReader,
    dir: PathBuf,
    layer: BixverseLayer,
    n_obs: usize,
    n_vars: usize,
    chunk_size: usize,
}

fn store_err(e: impl std::fmt::Display) -> ScxError {
    ScxError::InvalidFormat(format!("bixverse store: {e}"))
}

impl BixverseReader {
    /// Open a store from its `counts_cells.bin` or the directory holding it.
    pub fn open<P: AsRef<Path>>(path: P, chunk_size: usize, layer: BixverseLayer) -> Result<Self> {
        let path = path.as_ref();
        let bin = if path.is_dir() {
            path.join(CELLS_BIN)
        } else {
            path.to_path_buf()
        };
        let bin_str = bin
            .to_str()
            .ok_or_else(|| store_err("path is not valid UTF-8"))?;
        let reader = ParallelSparseReader::new(bin_str).map_err(store_err)?;
        if !reader.is_cell_based() {
            return Err(ScxError::InvalidFormat(format!(
                "'{}' is a gene-based store; open the cell-based counts_cells.bin",
                bin.display()
            )));
        }
        let header = reader.get_header();
        Ok(Self {
            n_obs: header.total_cells,
            n_vars: header.total_genes,
            dir: bin.parent().map(Path::to_path_buf).unwrap_or_default(),
            reader,
            layer,
            chunk_size,
        })
    }

    /// The sidecar table `stem` ("obs" or "var"), if one exists.
    fn sidecar(&self, stem: &str) -> Result<Option<Table>> {
        #[cfg(feature = "bixverse-parquet")]
        {
            let p = self.dir.join(format!("{stem}.parquet"));
            if p.is_file() {
                return read_parquet_table(&p).map(Some);
            }
        }
        for ext in ["tsv.gz", "tsv"] {
            let p = self.dir.join(format!("{stem}.{ext}"));
            if p.is_file() {
                return read_tsv_table(&p).map(Some);
            }
        }
        Ok(None)
    }

    /// The sidecar table, or an index-only table when there is none.
    fn table(&self, stem: &str, prefix: &str, n: usize) -> Result<Table> {
        let table = match self.sidecar(stem)? {
            Some(t) => t,
            None => {
                tracing::warn!(
                    "no {stem} sidecar next to '{}'; using a synthetic index",
                    self.dir.display()
                );
                Table {
                    index: (0..n).map(|i| format!("{prefix}_{i}")).collect(),
                    columns: Vec::new(),
                }
            }
        };
        if table.index.len() != n {
            return Err(ScxError::InvalidFormat(format!(
                "{stem} sidecar has {} rows, the store has {n}",
                table.index.len()
            )));
        }
        Ok(table)
    }
}

#[async_trait]
impl DatasetReader for BixverseReader {
    fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }

    fn dtype(&self) -> DataType {
        DataType::F32
    }

    async fn obs(&mut self) -> Result<ObsTable> {
        let Table { index, columns } = self.table("obs", "cell", self.n_obs)?;
        Ok(ObsTable { index, columns })
    }

    async fn var(&mut self) -> Result<VarTable> {
        let Table { index, columns } = self.table("var", "gene", self.n_vars)?;
        Ok(VarTable { index, columns })
    }

    fn x_stream(&mut self) -> ChunkStream<'_> {
        let reader = &self.reader;
        let (n_obs, n_vars) = (self.n_obs, self.n_vars);
        let layer = self.layer;
        let want = match layer {
            BixverseLayer::Raw => DataLayerReturn::Raw,
            BixverseLayer::Norm => DataLayerReturn::Norm,
        };

        row_chunks(n_obs, self.chunk_size, move |rows| {
            let cells: Vec<usize> = rows.clone().collect();
            let csr = reader
                .read_cells_sparse::<f32>(&cells, &want)
                .map_err(store_err)?;
            let data = match layer {
                BixverseLayer::Raw => csr.data,
                BixverseLayer::Norm => csr
                    .data_2
                    .ok_or_else(|| store_err("normalised layer missing"))?,
            };
            Ok(MatrixChunk {
                row_offset: rows.start,
                nrows: cells.len(),
                data: SparseMatrixCSR {
                    shape: (cells.len(), n_vars),
                    indptr: csr.indptr.into_iter().map(u64::from).collect(),
                    indices: csr.indices,
                    data: TypedVec::F32(data),
                },
            })
        })
    }
}

//////////////
// Sidecars //
//////////////

/// An obs or var table before it is split into an index and columns.
struct Table {
    index: Vec<String>,
    columns: Vec<Column>,
}

fn is_missing(v: &str) -> bool {
    v.is_empty() || v == "NA"
}

fn parse_bool(v: &str) -> Option<bool> {
    match v {
        "TRUE" | "True" | "true" => Some(true),
        "FALSE" | "False" | "false" => Some(false),
        _ => None,
    }
}

/// Type one TSV column. Missing values keep a placeholder under the mask.
fn infer_column(name: &str, values: Vec<String>) -> Column {
    let missing: Vec<bool> = values.iter().map(|v| is_missing(v)).collect();
    let present = || {
        values
            .iter()
            .zip(&missing)
            .filter(|(_, m)| !**m)
            .map(|(v, _)| v)
    };
    let any_present = present().next().is_some();

    let data = if any_present && present().all(|v| v.parse::<i32>().is_ok()) {
        ColumnData::Int(values.iter().map(|v| v.parse().unwrap_or(0)).collect())
    } else if any_present && present().all(|v| v.parse::<i64>().is_ok()) {
        ColumnData::Int64(values.iter().map(|v| v.parse().unwrap_or(0)).collect())
    } else if any_present && present().all(|v| v.parse::<f64>().is_ok()) {
        ColumnData::Float(
            values
                .iter()
                .map(|v| v.parse().unwrap_or(f64::NAN))
                .collect(),
        )
    } else if any_present && present().all(|v| parse_bool(v).is_some()) {
        ColumnData::Bool(
            values
                .iter()
                .map(|v| parse_bool(v).unwrap_or(false))
                .collect(),
        )
    } else {
        let levels: Vec<String> = present()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let n_present = missing.iter().filter(|m| !**m).count();
        if !levels.is_empty() && levels.len() < n_present {
            let code: BTreeMap<&str, u32> = levels
                .iter()
                .enumerate()
                .map(|(i, l)| (l.as_str(), i as u32))
                .collect();
            let codes = values
                .iter()
                .map(|v| code.get(v.as_str()).copied().unwrap_or(0))
                .collect();
            ColumnData::Categorical {
                codes,
                levels,
                ordered: false,
            }
        } else {
            ColumnData::String(values.iter().map(|v| (*v).clone()).collect())
        }
    };
    Column::with_missing(name, data, missing)
}

fn read_tsv_table(path: &Path) -> Result<Table> {
    let file = File::open(path)?;
    let gz = path.extension().and_then(|e| e.to_str()) == Some("gz");
    let reader: Box<dyn BufRead> = if gz {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let bad = |msg: String| ScxError::InvalidFormat(format!("{}: {msg}", path.display()));
    let mut lines = reader.lines();
    let header = lines.next().ok_or_else(|| bad("empty file".into()))??;
    let names: Vec<String> = header.split('\t').map(str::to_string).collect();

    let mut cols: Vec<Vec<String>> = vec![Vec::new(); names.len()];
    for (i, line) in lines.enumerate() {
        let line = line?;
        let mut n = 0;
        for (col, field) in cols.iter_mut().zip(line.split('\t')) {
            col.push(field.to_string());
            n += 1;
        }
        if n != names.len() {
            return Err(bad(format!(
                "row {} has {n} fields, expected {}",
                i + 1,
                names.len()
            )));
        }
    }

    let mut cols = cols.into_iter();
    let index = cols.next().unwrap_or_default();
    let columns = names[1..]
        .iter()
        .zip(cols)
        .map(|(name, values)| infer_column(name, values))
        .collect();
    Ok(Table { index, columns })
}

#[cfg(feature = "bixverse-parquet")]
fn read_parquet_table(path: &Path) -> Result<Table> {
    use arrow::array::{
        Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray,
    };
    use arrow::compute::{cast, concat};
    use arrow::datatypes::{DataType as Arrow, Float64Type, Int32Type, Int64Type, UInt32Type};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let bad =
        |e: &dyn std::fmt::Display| ScxError::InvalidFormat(format!("{}: {e}", path.display()));
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(File::open(path)?).map_err(|e| bad(&e))?;
    let schema = builder.schema().clone();
    let batches = builder
        .build()
        .map_err(|e| bad(&e))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| bad(&e))?;

    let column = |j: usize| -> Result<ArrayRef> {
        let parts: Vec<&dyn Array> = batches.iter().map(|b| b.column(j).as_ref()).collect();
        if parts.is_empty() {
            return Ok(arrow::array::new_empty_array(schema.field(j).data_type()));
        }
        concat(&parts).map_err(|e| bad(&e))
    };
    let strings = |a: &ArrayRef| -> Result<StringArray> {
        let a = cast(a, &Arrow::Utf8).map_err(|e| bad(&e))?;
        Ok(a.as_string::<i32>().clone())
    };

    let index = strings(&column(0)?)?
        .iter()
        .map(|v| v.unwrap_or("").to_string())
        .collect();

    let mut columns = Vec::new();
    for j in 1..schema.fields().len() {
        let field = schema.field(j);
        let arr = column(j)?;
        let missing: Vec<bool> = (0..arr.len()).map(|i| arr.is_null(i)).collect();
        let data = match field.data_type() {
            Arrow::Boolean => {
                let a: &BooleanArray = arr.as_boolean();
                ColumnData::Bool(a.iter().map(|v| v.unwrap_or(false)).collect())
            }
            Arrow::Float16 | Arrow::Float32 | Arrow::Float64 => {
                let a = cast(&arr, &Arrow::Float64).map_err(|e| bad(&e))?;
                let a: &Float64Array = a.as_primitive::<Float64Type>();
                ColumnData::Float(a.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            }
            Arrow::Int64 | Arrow::UInt32 | Arrow::UInt64 => {
                // u64 beyond i64::MAX casts to null and would read as missing;
                // widen to float rather than lose it.
                let as_i64 = cast(&arr, &Arrow::Int64).map_err(|e| bad(&e))?;
                if as_i64.null_count() == arr.null_count() {
                    let a: &Int64Array = as_i64.as_primitive::<Int64Type>();
                    ColumnData::Int64(a.iter().map(|v| v.unwrap_or(0)).collect())
                } else {
                    let a = cast(&arr, &Arrow::Float64).map_err(|e| bad(&e))?;
                    let a: &Float64Array = a.as_primitive::<Float64Type>();
                    ColumnData::Float(a.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
                }
            }
            t if t.is_integer() => {
                let a = cast(&arr, &Arrow::Int32).map_err(|e| bad(&e))?;
                let a: &Int32Array = a.as_primitive::<Int32Type>();
                ColumnData::Int(a.iter().map(|v| v.unwrap_or(0)).collect())
            }
            Arrow::Dictionary(_, _) => {
                let dict = cast(
                    &arr,
                    &Arrow::Dictionary(Box::new(Arrow::UInt32), Box::new(Arrow::Utf8)),
                )
                .map_err(|e| bad(&e))?;
                let d = dict.as_dictionary::<UInt32Type>();
                let levels = strings(d.values())?
                    .iter()
                    .map(|v| v.unwrap_or("").to_string())
                    .collect();
                let codes = (0..d.len())
                    .map(|i| if d.is_null(i) { 0 } else { d.keys().value(i) })
                    .collect();
                ColumnData::Categorical {
                    codes,
                    levels,
                    ordered: false,
                }
            }
            _ => ColumnData::String(
                strings(&arr)?
                    .iter()
                    .map(|v| v.unwrap_or("").to_string())
                    .collect(),
            ),
        };
        columns.push(Column::with_missing(field.name().clone(), data, missing));
    }
    Ok(Table { index, columns })
}
