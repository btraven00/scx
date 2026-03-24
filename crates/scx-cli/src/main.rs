use std::path::Path;

use clap::Parser;
use futures::StreamExt;
use owo_colors::{OwoColorize, Stream::Stdout};
use scx_core::{
    detect,
    detect::Format,
    dtype::DataType,
    ir::ColumnData,
    h5::ScxH5Reader,
    h5ad::{H5AdReader, H5AdWriter},
    h5seurat::{H5SeuratWriter, open_h5seurat},
    npy::{NpyIrReader, NpyIrWriter, SlotFilter},
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

    /// Dump a materialised IR snapshot to a directory of NPY files
    ///
    /// Reads the input file and writes raw-binary NPY arrays plus a
    /// meta.json manifest to OUTPUT_DIR.  The snapshot can then be fed
    /// back into `scx convert` for benchmarking isolated from I/O.
    ///
    /// Examples:
    ///   scx snapshot pbmc.h5seurat ir/          # everything
    ///   scx snapshot pbmc.h5seurat ir/ --only X,obs_index
    ///   scx snapshot pbmc.h5seurat ir/ --exclude layers,obsp
    ///
    /// Slot specifiers for --only / --exclude:
    ///   X, obs_index, var_index, uns
    ///   obs, obs:col_name, var, var:col_name
    ///   obsm, obsm:key, varm, varm:key
    ///   layers, layers:key, obsp, obsp:key, varp, varp:key
    Snapshot {
        /// Input file (.h5seurat, .h5ad, …)
        input: String,

        /// Output directory (created if absent)
        output_dir: String,

        /// Include only these comma-separated slot specifiers
        #[arg(long, conflicts_with = "exclude")]
        only: Option<String>,

        /// Exclude these comma-separated slot specifiers
        #[arg(long, conflicts_with = "only")]
        exclude: Option<String>,

        /// Cells per streaming chunk (for reading the input)
        #[arg(long, default_value = "5000")]
        chunk_size: usize,

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
            let fmt = detect::sniff_dir(input_path)
                .or_else(|| detect::sniff(input_path))
                .or_else(|| {
                    match input_path.extension().and_then(|e| e.to_str()) {
                        Some("h5seurat") => Some(Format::H5Seurat),
                        Some("h5ad")     => Some(Format::H5Ad),
                        _                => Some(Format::ScxH5),
                    }
                });

            let chunk = 1000;
            match fmt {
                Some(Format::NpyDir) => {
                    let mut r = NpyIrReader::open(input_path, chunk)?;
                    inspect(&mut r, &input, "NPY snapshot").await?;
                }
                Some(Format::H5Seurat) => {
                    let mut r = open_h5seurat(input_path, chunk, Some(&assay), Some(&layer))?;
                    inspect(&mut *r, &input, "H5Seurat").await?;
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

            // NPY snapshot directory takes priority.
            let fmt = detect::sniff_dir(input_path)
                .or_else(|| detect::sniff(input_path))
                .or_else(|| {
                    match input_path.extension().and_then(|e| e.to_str()) {
                        Some("h5seurat") => Some(Format::H5Seurat),
                        Some("h5ad")     => Some(Format::H5Ad),
                        _                => Some(Format::ScxH5),
                    }
                });

            match fmt {
                Some(Format::NpyDir) => {
                    tracing::info!(path = %input, "detected format: NPY snapshot directory");
                    let mut reader = NpyIrReader::open(input_path, chunk_size)?;
                    convert_with_reader(&mut reader, Path::new(&output), out_dtype, &assay, &layer).await?;
                }
                Some(Format::H5Seurat) => {
                    tracing::info!(path = %input, "detected format: H5Seurat");
                    let mut reader = open_h5seurat(input_path, chunk_size, Some(&assay), Some(&layer))?;
                    convert_with_reader(&mut *reader, Path::new(&output), out_dtype, &assay, &layer).await?;
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

        Cli::Snapshot { input, output_dir, only, exclude, chunk_size, assay, layer } => {
            let input_path = Path::new(&input);
            let output_path = Path::new(&output_dir);

            let filter = match (only.as_deref(), exclude.as_deref()) {
                (Some(o), _) => SlotFilter::from_only(o),
                (_, Some(e)) => SlotFilter::from_exclude(e),
                _            => SlotFilter::all(),
            };

            tracing::info!(
                input = %input,
                output = %output_dir,
                "materialising IR snapshot"
            );

            // Read the full dataset from any supported input format.
            let fmt = detect::sniff_dir(input_path)
                .or_else(|| detect::sniff(input_path))
                .or_else(|| {
                    match input_path.extension().and_then(|e| e.to_str()) {
                        Some("h5seurat") => Some(Format::H5Seurat),
                        Some("h5ad")     => Some(Format::H5Ad),
                        _                => Some(Format::ScxH5),
                    }
                });

            let dataset = match fmt {
                Some(Format::NpyDir) => {
                    NpyIrReader::open(input_path, chunk_size)?.into_dataset()
                }
                Some(Format::H5Seurat) => {
                    let mut r = open_h5seurat(input_path, chunk_size, Some(&assay), Some(&layer))?;
                    materialise_dataset(&mut *r).await?
                }
                Some(Format::H5Ad) => {
                    materialise_dataset(&mut H5AdReader::open(input_path, chunk_size)?).await?
                }
                Some(Format::ScxH5) | None => {
                    materialise_dataset(&mut ScxH5Reader::open(input_path, chunk_size)?).await?
                }
            };

            NpyIrWriter::write(output_path, &dataset, &filter)?;

            tracing::info!(
                output = %output_dir,
                n_obs  = dataset.x.shape.0,
                n_vars = dataset.x.shape.1,
                "snapshot written"
            );
        }
    }

    Ok(())
}

/// Fully materialise a streaming reader into a [`SingleCellDataset`].
async fn materialise_dataset(
    reader: &mut dyn DatasetReader,
) -> anyhow::Result<scx_core::ir::SingleCellDataset> {
    use futures::StreamExt;
    use scx_core::ir::{SparseMatrixCSR, SingleCellDataset};
    use scx_core::dtype::TypedVec;

    let (n_obs, n_vars) = reader.shape();
    let x_dtype = reader.dtype();

    let obs    = reader.obs().await?;
    let var    = reader.var().await?;
    let obsm   = reader.obsm().await?;
    let uns    = reader.uns().await?;
    let layers = reader.layers().await?;
    let obsp   = reader.obsp().await?;
    let varp   = reader.varp().await?;
    let varm   = reader.varm().await?;

    // Accumulate X chunks into a full CSR.
    let mut x_indptr: Vec<u64> = Vec::with_capacity(n_obs + 1);
    x_indptr.push(0);
    let mut x_indices: Vec<u32> = Vec::new();
    let mut x_data_f32: Vec<f32> = Vec::new();
    let mut x_data_f64: Vec<f64> = Vec::new();
    let mut x_data_i32: Vec<i32> = Vec::new();
    let mut x_data_u32: Vec<u32> = Vec::new();

    let mut stream = reader.x_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        x_indices.extend_from_slice(&chunk.data.indices);
        match &chunk.data.data {
            TypedVec::F32(v) => x_data_f32.extend_from_slice(v),
            TypedVec::F64(v) => x_data_f64.extend_from_slice(v),
            TypedVec::I32(v) => x_data_i32.extend_from_slice(v),
            TypedVec::U32(v) => x_data_u32.extend_from_slice(v),
        }
        // Extend indptr (skip the leading 0 of each chunk's indptr).
        let base = *x_indptr.last().unwrap();
        for &p in &chunk.data.indptr[1..] {
            x_indptr.push(base + p);
        }
    }

    let x_data = match x_dtype {
        DataType::F32 => TypedVec::F32(x_data_f32),
        DataType::F64 => TypedVec::F64(x_data_f64),
        DataType::I32 => TypedVec::I32(x_data_i32),
        DataType::U32 => TypedVec::U32(x_data_u32),
    };

    let x = SparseMatrixCSR {
        shape: (n_obs, n_vars),
        indptr: x_indptr,
        indices: x_indices,
        data: x_data,
    };

    Ok(SingleCellDataset { x, x_dtype, obs, var, obsm, uns, layers, obsp, varp, varm })
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

    // Colour helpers -- all checks against Stdout so piped output stays plain.
    macro_rules! bold  { ($x:expr) => { $x.if_supports_color(Stdout, |t| t.bold()) } }
    macro_rules! cyan  { ($x:expr) => { $x.if_supports_color(Stdout, |t| t.bright_cyan()) } }
    macro_rules! green { ($x:expr) => { $x.if_supports_color(Stdout, |t| t.bright_green()) } }
    macro_rules! dim   { ($x:expr) => { $x.if_supports_color(Stdout, |t| t.dimmed()) } }
    // yellow! and bold_cyan! use Style to avoid borrow-of-temporary when chaining.
    macro_rules! yellow { ($x:expr) => {{
        use owo_colors::Style;
        $x.if_supports_color(Stdout, |t| t.style(Style::new().bright_yellow()))
    }} }
    macro_rules! bold_cyan { ($x:expr) => {{
        use owo_colors::Style;
        $x.if_supports_color(Stdout, |t| t.style(Style::new().bold().bright_cyan()))
    }} }

    println!("{} {}",  bold!("File   :"), green!(path));
    println!("{} {}",  bold!("Format :"), cyan!(format_name));
    println!("{} {} × {}  {}",
        bold!("Shape  :"),
        yellow!(n_obs),
        yellow!(n_vars),
        dim!("obs × vars"),
    );
    let dtype_str = reader.dtype().to_string();
    println!("{} {}", bold!("X dtype:"), cyan!(&dtype_str));
    println!();

    // section header helper
    let section = |name: &str, count: usize, unit: &str| {
        let label = format!(" {unit}):");
        println!("{} {}{}{}",
            bold_cyan!(name),
            bold!("("),
            yellow!(count),
            bold!(&label),
        );
    };

    // ── obs ──────────────────────────────────────────────────────────────────
    let obs = reader.obs().await?;
    section("obs", obs.columns.len(), "columns");
    if obs.columns.is_empty() { println!("  {}", dim!("(none)")); }
    for col in &obs.columns {
        let extra = cat_levels_preview(&col.data);
        let type_str = col_type_str(&col.data);
        if extra.is_empty() {
            println!("  {:<30} {}", col.name, dim!(&type_str));
        } else {
            println!("  {:<30} {}  {} {}", col.name, dim!(&type_str), dim!("—"), dim!(&extra));
        }
    }
    println!();

    // ── var ──────────────────────────────────────────────────────────────────
    let var = reader.var().await?;
    section("var", var.columns.len(), "columns");
    if var.columns.is_empty() { println!("  {}", dim!("(none)")); }
    for col in &var.columns {
        let extra = cat_levels_preview(&col.data);
        let type_str = col_type_str(&col.data);
        if extra.is_empty() {
            println!("  {:<30} {}", col.name, dim!(&type_str));
        } else {
            println!("  {:<30} {}  {} {}", col.name, dim!(&type_str), dim!("—"), dim!(&extra));
        }
    }
    println!();

    // ── obsm ─────────────────────────────────────────────────────────────────
    let obsm = reader.obsm().await?;
    section("obsm", obsm.map.len(), "keys");
    let mut keys: Vec<_> = obsm.map.keys().collect();
    keys.sort();
    for k in keys {
        let m = &obsm.map[k];
        let shape = format!("({}, {})", m.shape.0, m.shape.1);
        println!("  {:<30} {}", k, dim!(&shape));
    }
    if obsm.map.is_empty() { println!("  {}", dim!("(none)")); }
    println!();

    // ── varm ─────────────────────────────────────────────────────────────────
    let varm = reader.varm().await?;
    section("varm", varm.map.len(), "keys");
    let mut keys: Vec<_> = varm.map.keys().collect();
    keys.sort();
    for k in keys {
        let m = &varm.map[k];
        let shape = format!("({}, {})", m.shape.0, m.shape.1);
        println!("  {:<30} {}", k, dim!(&shape));
    }
    if varm.map.is_empty() { println!("  {}", dim!("(none)")); }
    println!();

    // ── layers ───────────────────────────────────────────────────────────────
    let layers = reader.layers().await?;
    section("layers", layers.map.len(), "keys");
    let mut keys: Vec<_> = layers.map.keys().collect();
    keys.sort();
    for k in keys {
        let m = &layers.map[k];
        println!("  {:<30} {} × {}  {}{}",
            k, yellow!(m.shape.0), yellow!(m.shape.1),
            dim!("nnz="), yellow!(m.indices.len()));
    }
    if layers.map.is_empty() { println!("  {}", dim!("(none)")); }
    println!();

    // ── obsp ─────────────────────────────────────────────────────────────────
    let obsp = reader.obsp().await?;
    section("obsp", obsp.map.len(), "keys");
    let mut keys: Vec<_> = obsp.map.keys().collect();
    keys.sort();
    for k in keys {
        let m = &obsp.map[k];
        println!("  {:<30} {} × {}  {}{}",
            k, yellow!(m.shape.0), yellow!(m.shape.1),
            dim!("nnz="), yellow!(m.indices.len()));
    }
    if obsp.map.is_empty() { println!("  {}", dim!("(none)")); }
    println!();

    // ── varp ─────────────────────────────────────────────────────────────────
    let varp = reader.varp().await?;
    section("varp", varp.map.len(), "keys");
    let mut keys: Vec<_> = varp.map.keys().collect();
    keys.sort();
    for k in keys {
        let m = &varp.map[k];
        println!("  {:<30} {} × {}  {}{}",
            k, yellow!(m.shape.0), yellow!(m.shape.1),
            dim!("nnz="), yellow!(m.indices.len()));
    }
    if varp.map.is_empty() { println!("  {}", dim!("(none)")); }
    println!();

    // ── uns ──────────────────────────────────────────────────────────────────
    let uns = reader.uns().await?;
    if uns.raw.is_null() {
        section("uns", 0, "keys");
        println!("  {}", dim!("(none)"));
    } else if let Some(obj) = uns.raw.as_object() {
        section("uns", obj.len(), "keys");
        let mut keys: Vec<_> = obj.keys().collect();
        keys.sort();
        for k in keys {
            let v = &obj[k];
            let summary = match v {
                serde_json::Value::Array(a)  => format!("array [{}]", a.len()),
                serde_json::Value::Object(o) => format!("dict  ({} keys)", o.len()),
                serde_json::Value::String(s) => {
                    if s.len() > 60 { format!("\"{}...\"", &s[..57]) }
                    else { format!("\"{s}\"") }
                }
                other => format!("{other}"),
            };
            println!("  {:<30} {}", k, dim!(&summary));
        }
    } else {
        println!("{} {}", bold_cyan!("uns"), uns.raw);
    }

    Ok(())
}
