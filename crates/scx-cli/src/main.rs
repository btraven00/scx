use std::path::Path;

use clap::Parser;
use futures::StreamExt;
use scx_core::{
    detect,
    detect::Format,
    dtype::DataType,
    ir::ColumnData,
    h5::ScxH5Reader,
    h5ad::{H5AdReader, H5AdWriter},
    h5seurat::{H5SeuratReader, H5SeuratWriter},
    stream::{DatasetReader, DatasetWriter},
};

#[derive(Parser)]
#[command(name = "scx", about = "Single-cell streaming interop engine")]
enum Cli {
    /// Convert between single-cell formats
    ///
    /// Input auto-detected by content:
    ///   .h5seurat  — SeuratDisk H5Seurat (Seurat v3/v4)
    ///   .h5ad      — AnnData H5AD (CSR X only)
    ///   .h5        — SCX internal HDF5 schema
    ///
    /// Output format selected by extension:
    ///   .h5ad      — AnnData H5AD  (default)
    ///   .h5seurat  — SeuratDisk H5Seurat
    Convert {
        /// Input file
        input: String,

        /// Output file (.h5ad or .h5seurat)
        output: String,

        /// Cells per streaming chunk
        #[arg(long, default_value = "5000")]
        chunk_size: usize,

        /// Output data type [f32, f64, i32, u32]
        #[arg(long, default_value = "f32")]
        dtype: String,

        /// Seurat assay to convert (H5Seurat only)
        #[arg(long, default_value = "RNA")]
        assay: String,

        /// Seurat layer to convert (H5Seurat only)
        #[arg(long, default_value = "counts")]
        layer: String,
    },

    /// Inspect a single-cell file
    ///
    /// Prints format, shape, and a summary of every slot (obs, var, obsm,
    /// layers, obsp, varp, varm, uns) without converting.
    Inspect {
        /// Input file
        input: String,

        /// Seurat assay (H5Seurat only)
        #[arg(long, default_value = "RNA")]
        assay: String,

        /// Seurat layer (H5Seurat only)
        #[arg(long, default_value = "counts")]
        layer: String,
    },
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    if let Err(e) = run().await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli {
        Cli::Inspect { input, assay, layer } => {
            let input_path = Path::new(&input);
            let fmt = detect::sniff(input_path).or_else(|| {
                match input_path.extension().and_then(|e| e.to_str()) {
                    Some("h5seurat") => Some(Format::H5Seurat),
                    Some("h5ad")     => Some(Format::H5Ad),
                    _                => Some(Format::ScxH5),
                }
            });

            let chunk = 1000; // only used for reader init, not streaming
            match fmt {
                Some(Format::H5Seurat) => {
                    let mut r = H5SeuratReader::open(input_path, chunk, Some(&assay), Some(&layer))?;
                    inspect(&mut r, &input, "H5Seurat").await?;
                }
                Some(Format::H5Ad) => {
                    let mut r = H5AdReader::open(input_path, chunk)?;
                    inspect(&mut r, &input, "H5AD").await?;
                }
                Some(Format::ScxH5) | None => {
                    let mut r = ScxH5Reader::open(input_path, chunk)?;
                    inspect(&mut r, &input, "SCX H5").await?;
                }
            }
        }

        Cli::Convert { input, output, chunk_size, dtype, assay, layer } => {
            let out_dtype = match dtype.as_str() {
                "f32" => DataType::F32,
                "f64" => DataType::F64,
                "i32" => DataType::I32,
                "u32" => DataType::U32,
                other => anyhow::bail!("unknown dtype '{other}': use f32, f64, i32, u32"),
            };

            let input_path = Path::new(&input);

            // Detect format by content; fall back to extension only for
            // files that don't match any HDF5 fingerprint (e.g. non-HDF5).
            let fmt = detect::sniff(input_path).or_else(|| {
                match input_path.extension().and_then(|e| e.to_str()) {
                    Some("h5seurat") => Some(Format::H5Seurat),
                    Some("h5ad")     => Some(Format::H5Ad),
                    _                => Some(Format::ScxH5),
                }
            });

            match fmt {
                Some(Format::H5Seurat) => {
                    tracing::info!(path = %input, "detected format: H5Seurat");
                    let mut reader = H5SeuratReader::open(input_path, chunk_size, Some(&assay), Some(&layer))?;
                    convert_with_reader(&mut reader, Path::new(&output), out_dtype, &assay, &layer).await?;
                }
                Some(Format::H5Ad) => {
                    tracing::info!(path = %input, "detected format: H5AD");
                    let mut reader = H5AdReader::open(input_path, chunk_size)?;
                    convert_with_reader(&mut reader, Path::new(&output), out_dtype, &assay, &layer).await?;
                }
                Some(Format::ScxH5) | None => {
                    tracing::info!(path = %input, "detected format: SCX H5 (internal)");
                    let mut reader = ScxH5Reader::open(input_path, chunk_size)?;
                    convert_with_reader(&mut reader, Path::new(&output), out_dtype, &assay, &layer).await?;
                }
            }
        }
    }

    Ok(())
}

async fn convert_with_reader(
    reader: &mut dyn DatasetReader,
    output: &Path,
    out_dtype: DataType,
    out_assay: &str,
    out_layer: &str,
) -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let (n_obs, n_vars) = reader.shape();

    let is_h5seurat = output.extension().and_then(|e| e.to_str()) == Some("h5seurat");

    tracing::info!(
        output = %output.display(),
        n_obs, n_vars,
        dtype = %out_dtype,
        format = if is_h5seurat { "h5seurat" } else { "h5ad" },
        "starting conversion"
    );

    let obs    = reader.obs().await?;
    let var    = reader.var().await?;
    let obsm   = reader.obsm().await?;
    let uns    = reader.uns().await?;
    let layers = reader.layers().await?;
    let obsp   = reader.obsp().await?;
    let varp   = reader.varp().await?;
    let varm   = reader.varm().await?;

    tracing::info!(
        obs_cols = obs.columns.len(),
        var_cols = var.columns.len(),
        embeddings = obsm.map.len(),
        "metadata loaded in {:.2?}", t0.elapsed()
    );

    let mut writer: Box<dyn DatasetWriter> = if is_h5seurat {
        Box::new(H5SeuratWriter::create(output, n_obs, n_vars, out_dtype, Some(out_assay), Some(out_layer))?)
    } else {
        Box::new(H5AdWriter::create(output, n_obs, n_vars, out_dtype)?)
    };

    writer.write_obs(&obs).await?;
    writer.write_var(&var).await?;
    writer.write_obsm(&obsm).await?;
    writer.write_uns(&uns).await?;
    writer.write_layers(&layers).await?;
    writer.write_obsp(&obsp).await?;
    writer.write_varp(&varp).await?;
    writer.write_varm(&varm).await?;

    let t_x = std::time::Instant::now();
    let mut stream = reader.x_stream();
    let mut total_nnz = 0usize;
    let mut n_chunks  = 0usize;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        total_nnz += chunk.data.indices.len();
        n_chunks  += 1;
        writer.write_x_chunk(&chunk).await?;
    }

    tracing::info!(
        n_chunks,
        total_nnz,
        throughput_cells_s = (n_obs as f64 / t_x.elapsed().as_secs_f64()) as u64,
        "matrix streamed in {:.2?}", t_x.elapsed()
    );

    writer.finalize().await?;

    tracing::info!(
        total = ?t0.elapsed(),
        output = %output.display(),
        "conversion complete"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Inspect
// ---------------------------------------------------------------------------

fn col_type_str(data: &ColumnData) -> &'static str {
    match data {
        ColumnData::Float(_)         => "float64",
        ColumnData::Int(_)           => "int32",
        ColumnData::Bool(_)          => "bool",
        ColumnData::String(_)        => "string",
        ColumnData::Categorical { .. } => "categorical",
    }
}

fn cat_levels_preview(data: &ColumnData) -> String {
    if let ColumnData::Categorical { levels, .. } = data {
        let n = levels.len();
        let preview: Vec<&str> = levels.iter().take(5).map(|s| s.as_str()).collect();
        if n > 5 {
            format!("{} levels [{}  ...]", n, preview.join(", "))
        } else {
            format!("{} levels [{}]", n, preview.join(", "))
        }
    } else {
        String::new()
    }
}

async fn inspect(reader: &mut dyn DatasetReader, path: &str, format_name: &str) -> anyhow::Result<()> {
    let (n_obs, n_vars) = reader.shape();

    println!("File   : {path}");
    println!("Format : {format_name}");
    println!("Shape  : {n_obs} obs × {n_vars} vars");
    println!("X dtype: {}", reader.dtype());
    println!();

    // obs
    let obs = reader.obs().await?;
    println!("obs ({} columns):", obs.columns.len());
    if obs.columns.is_empty() {
        println!("  (none)");
    }
    for col in &obs.columns {
        let extra = cat_levels_preview(&col.data);
        if extra.is_empty() {
            println!("  {:<30} {}", col.name, col_type_str(&col.data));
        } else {
            println!("  {:<30} {}  — {}", col.name, col_type_str(&col.data), extra);
        }
    }
    println!();

    // var
    let var = reader.var().await?;
    println!("var ({} columns):", var.columns.len());
    if var.columns.is_empty() {
        println!("  (none)");
    }
    for col in &var.columns {
        let extra = cat_levels_preview(&col.data);
        if extra.is_empty() {
            println!("  {:<30} {}", col.name, col_type_str(&col.data));
        } else {
            println!("  {:<30} {}  — {}", col.name, col_type_str(&col.data), extra);
        }
    }
    println!();

    // obsm
    let obsm = reader.obsm().await?;
    println!("obsm ({} keys):", obsm.map.len());
    let mut keys: Vec<_> = obsm.map.keys().collect();
    keys.sort();
    for k in keys {
        let m = &obsm.map[k];
        println!("  {:<30} ({}, {})", k, m.shape.0, m.shape.1);
    }
    if obsm.map.is_empty() { println!("  (none)"); }
    println!();

    // varm
    let varm = reader.varm().await?;
    println!("varm ({} keys):", varm.map.len());
    let mut keys: Vec<_> = varm.map.keys().collect();
    keys.sort();
    for k in keys {
        let m = &varm.map[k];
        println!("  {:<30} ({}, {})", k, m.shape.0, m.shape.1);
    }
    if varm.map.is_empty() { println!("  (none)"); }
    println!();

    // layers
    let layers = reader.layers().await?;
    println!("layers ({} keys):", layers.map.len());
    let mut keys: Vec<_> = layers.map.keys().collect();
    keys.sort();
    for k in keys {
        let m = &layers.map[k];
        let nnz = m.indices.len();
        println!("  {:<30} {} × {}  nnz={}", k, m.shape.0, m.shape.1, nnz);
    }
    if layers.map.is_empty() { println!("  (none)"); }
    println!();

    // obsp
    let obsp = reader.obsp().await?;
    println!("obsp ({} keys):", obsp.map.len());
    let mut keys: Vec<_> = obsp.map.keys().collect();
    keys.sort();
    for k in keys {
        let m = &obsp.map[k];
        let nnz = m.indices.len();
        println!("  {:<30} {} × {}  nnz={}", k, m.shape.0, m.shape.1, nnz);
    }
    if obsp.map.is_empty() { println!("  (none)"); }
    println!();

    // varp
    let varp = reader.varp().await?;
    println!("varp ({} keys):", varp.map.len());
    let mut keys: Vec<_> = varp.map.keys().collect();
    keys.sort();
    for k in keys {
        let m = &varp.map[k];
        let nnz = m.indices.len();
        println!("  {:<30} {} × {}  nnz={}", k, m.shape.0, m.shape.1, nnz);
    }
    if varp.map.is_empty() { println!("  (none)"); }
    println!();

    // uns
    let uns = reader.uns().await?;
    if uns.raw.is_null() {
        println!("uns: (empty)");
    } else if let Some(obj) = uns.raw.as_object() {
        println!("uns ({} keys):", obj.len());
        let mut keys: Vec<_> = obj.keys().collect();
        keys.sort();
        for k in keys {
            let v = &obj[k];
            let summary = match v {
                serde_json::Value::Array(a) => format!("array [{}]", a.len()),
                serde_json::Value::Object(o) => format!("dict  ({} keys)", o.len()),
                serde_json::Value::String(s) => {
                    if s.len() > 60 { format!("\"{}...\"", &s[..57]) }
                    else { format!("\"{}\"", s) }
                }
                other => format!("{other}"),
            };
            println!("  {:<30} {}", k, summary);
        }
    } else {
        println!("uns: {}", uns.raw);
    }

    Ok(())
}
