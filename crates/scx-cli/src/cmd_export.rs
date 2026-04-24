use std::path::{Path, PathBuf};

use polars::prelude::*;
use scx_core::{
    detect::{sniff, sniff_dir, Format},
    h5ad::H5AdReader,
    h5seurat::open_h5seurat,
    ir::{ColumnData, ObsTable, VarTable},
    stream::DatasetReader,
};

pub struct ExportArgs {
    /// Input file (any supported format).
    pub input: String,
    /// Slot to export: `obs`, `var`, or `obsm/<name>`.
    pub slot: String,
    /// Output path. Format inferred from extension (.csv or .parquet).
    pub output: String,
    /// Seurat assay (H5Seurat only).
    pub assay: String,
    /// Seurat layer (H5Seurat only).
    pub layer: String,
}

pub async fn run_export(args: ExportArgs) -> anyhow::Result<()> {
    let input_path = Path::new(&args.input);
    let fmt = sniff_dir(input_path)
        .or_else(|| sniff(input_path))
        .or_else(|| match input_path.extension().and_then(|e| e.to_str()) {
            Some("h5seurat") => Some(Format::H5Seurat),
            _ => Some(Format::H5Ad),
        });

    let mut reader: Box<dyn DatasetReader + Send> = match fmt {
        Some(Format::H5Seurat) => {
            open_h5seurat(input_path, 1, Some(&args.assay), Some(&args.layer))?
        }
        _ => Box::new(H5AdReader::open(input_path, 1)?),
    };

    let mut df = build_dataframe(&mut *reader, &args.slot).await?;

    let output_path = PathBuf::from(&args.output);
    match output_path.extension().and_then(|e| e.to_str()) {
        Some("parquet") => write_parquet(&mut df, &output_path)?,
        _ => write_csv(&mut df, &output_path)?,
    }

    eprintln!(
        "exported {} rows × {} cols → {}",
        df.height(),
        df.width(),
        args.output
    );
    Ok(())
}

async fn build_dataframe(
    reader: &mut dyn DatasetReader,
    slot: &str,
) -> anyhow::Result<DataFrame> {
    match slot {
        "obs" => {
            let obs = reader.obs().await?;
            obs_to_dataframe(obs)
        }
        "var" => {
            let var = reader.var().await?;
            var_to_dataframe(var)
        }
        s if s.starts_with("obsm/") => {
            let name = &s["obsm/".len()..];
            let obs = reader.obs().await?;
            let obsm = reader.obsm().await?;
            let mat = obsm
                .map
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("obsm entry '{name}' not found"))?;
            let (nrows, ncols) = mat.shape;
            let mut columns: Vec<Column> = Vec::with_capacity(ncols + 1);
            columns.push(Column::new("index".into(), obs.index.as_slice()));
            for c in 0..ncols {
                let col_vals: Vec<f64> = (0..nrows).map(|r| mat.data[r * ncols + c]).collect();
                columns.push(Column::new(
                    format!("dim_{c}").into(),
                    col_vals.as_slice(),
                ));
            }
            Ok(DataFrame::new(columns)?)
        }
        other => anyhow::bail!(
            "unsupported slot '{other}'; use obs, var, or obsm/<name>"
        ),
    }
}

fn obs_to_dataframe(obs: ObsTable) -> anyhow::Result<DataFrame> {
    let mut columns: Vec<Column> = Vec::with_capacity(obs.columns.len() + 1);
    columns.push(Column::new("index".into(), obs.index.as_slice()));
    for col in &obs.columns {
        columns.push(column_data_to_column(&col.name, &col.data)?);
    }
    Ok(DataFrame::new(columns)?)
}

fn var_to_dataframe(var: VarTable) -> anyhow::Result<DataFrame> {
    let mut columns: Vec<Column> = Vec::with_capacity(var.columns.len() + 1);
    columns.push(Column::new("index".into(), var.index.as_slice()));
    for col in &var.columns {
        columns.push(column_data_to_column(&col.name, &col.data)?);
    }
    Ok(DataFrame::new(columns)?)
}

fn column_data_to_column(name: &str, data: &ColumnData) -> anyhow::Result<Column> {
    let name: PlSmallStr = name.into();
    Ok(match data {
        ColumnData::Int(v) => Column::new(name, v.as_slice()),
        ColumnData::Float(v) => Column::new(name, v.as_slice()),
        ColumnData::String(v) => Column::new(name, v.as_slice()),
        ColumnData::Bool(v) => Column::new(name, v.as_slice()),
        ColumnData::Categorical { codes, levels } => {
            let decoded: Vec<&str> = codes
                .iter()
                .map(|&c| {
                    levels
                        .get(c as usize)
                        .map(|s| s.as_str())
                        .unwrap_or("")
                })
                .collect();
            Column::new(name, decoded.as_slice())
        }
    })
}

fn write_csv(df: &mut DataFrame, path: &Path) -> anyhow::Result<()> {
    let mut file = std::fs::File::create(path)
        .map_err(|e| anyhow::anyhow!("cannot create '{}': {e}", path.display()))?;
    CsvWriter::new(&mut file)
        .finish(df)
        .map_err(|e| anyhow::anyhow!("CSV write error: {e}"))?;
    Ok(())
}

fn write_parquet(df: &mut DataFrame, path: &Path) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)
        .map_err(|e| anyhow::anyhow!("cannot create '{}': {e}", path.display()))?;
    ParquetWriter::new(file)
        .finish(df)
        .map_err(|e| anyhow::anyhow!("Parquet write error: {e}"))?;
    Ok(())
}
