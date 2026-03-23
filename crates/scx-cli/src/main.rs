use std::path::Path;

use clap::Parser;
use futures::StreamExt;
use scx_core::{
    detect,
    detect::Format,
    dtype::DataType,
    h5::ScxH5Reader,
    h5ad::{H5AdReader, H5AdWriter},
    h5seurat::H5SeuratReader,
    stream::{DatasetReader, DatasetWriter},
};

#[derive(Parser)]
#[command(name = "scx", about = "Single-cell streaming interop engine")]
enum Cli {
    /// Convert a single-cell file to AnnData .h5ad
    ///
    /// Auto-detects format by extension:
    ///   .h5seurat  — SeuratDisk H5Seurat (Seurat v3/v4)
    ///   .h5ad      — AnnData H5AD (CSR X only)
    ///   .h5        — SCX internal HDF5 schema
    Convert {
        /// Input file (.h5seurat or .h5)
        input: String,

        /// Output file (.h5ad)
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
                    convert_with_reader(&mut reader, Path::new(&output), out_dtype).await?;
                }
                Some(Format::H5Ad) => {
                    tracing::info!(path = %input, "detected format: H5AD");
                    let mut reader = H5AdReader::open(input_path, chunk_size)?;
                    convert_with_reader(&mut reader, Path::new(&output), out_dtype).await?;
                }
                Some(Format::ScxH5) | None => {
                    tracing::info!(path = %input, "detected format: SCX H5 (internal)");
                    let mut reader = ScxH5Reader::open(input_path, chunk_size)?;
                    convert_with_reader(&mut reader, Path::new(&output), out_dtype).await?;
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
) -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let (n_obs, n_vars) = reader.shape();

    tracing::info!(
        output = %output.display(),
        n_obs, n_vars,
        dtype = %out_dtype,
        "starting conversion"
    );

    let obs  = reader.obs().await?;
    let var  = reader.var().await?;
    let obsm = reader.obsm().await?;
    let uns  = reader.uns().await?;

    tracing::info!(
        obs_cols = obs.columns.len(),
        var_cols = var.columns.len(),
        embeddings = obsm.map.len(),
        "metadata loaded in {:.2?}", t0.elapsed()
    );

    let mut writer = H5AdWriter::create(output, n_obs, n_vars, out_dtype)?;
    writer.write_obs(&obs).await?;
    writer.write_var(&var).await?;
    writer.write_obsm(&obsm).await?;
    writer.write_uns(&uns).await?;

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
