mod cmd_export;
mod cmd_merge;

use std::path::Path;

use clap::Parser;
use futures::StreamExt;
use owo_colors::{OwoColorize, Stream::Stdout};
use scx_core::{
    bpcells::BpcellsDatasetReader,
    detect,
    detect::Format,
    dtype::DataType,
    h5::ScxH5Reader,
    h5ad::{H5AdReader, H5AdWriter},
    h5bpcells::BpcellsH5Writer,
    h5seurat::{open_h5seurat, H5SeuratWriter},
    ir::{ColumnData, Embeddings, UnsTable, Varm},
    npy::{NpyIrReader, NpyIrWriter, SlotFilter},
    provenance::{self, OutputInfo, ProvenanceRecord, SourceInfo},
    stream::{DatasetReader, DatasetWriter},
    tenx::{read_tenx_summary, walk_h5, H5Node, H5NodeKind, TenxH5Reader},
    validate::{run_validation, ValidationSchema},
};

#[derive(Parser)]
#[command(name = "scx", about = "Single-cell streaming interop engine")]
enum Cli {
    /// Convert between single-cell formats
    ///
    /// Input auto-detected by content:
    ///   .h5seurat  — SeuratDisk H5Seurat (Seurat v3/v4)
    ///   .h5ad      — AnnData H5AD (CSR X only)
    ///   .h5        — SCX internal HDF5 schema, or 10x HDF5 (Cell Ranger output)
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

        /// Number of genes (var axis) for Parquet input. Required for the
        /// per-cell-list layout (e.g. Tahoe-100M: the gene_metadata row count,
        /// ~62710); derived automatically for dense (one column per gene) or
        /// when --genes is given.
        #[arg(long)]
        n_vars: Option<usize>,

        /// Gene dictionary for per-cell-list Parquet input: a path or URL to the
        /// gene_metadata Parquet (columns token_id, ensembl_id, gene_symbol).
        /// Remaps `genes` token ids to real columns, drops the marker token,
        /// fills var, and derives n_vars. For Tahoe-100M this is the
        /// gene_metadata subset alongside expression_data.
        #[arg(long)]
        genes: Option<String>,

        /// Output data type [f32, f64, i32, u32]
        #[arg(long, default_value = "f32")]
        dtype: String,

        /// Seurat assay to convert (H5Seurat only)
        #[arg(long, default_value = "RNA")]
        assay: String,

        /// Seurat layer to convert (H5Seurat only)
        #[arg(long, default_value = "counts")]
        layer: String,

        /// Write legacy `.h5seurat` output using dgCMatrix storage instead of BPCells
        #[arg(long)]
        dgcmatrix: bool,

        /// H5Seurat slot to write X into: counts, data, or auto (default).
        ///
        /// auto: if the source has a layer named "counts", X is assumed to be
        /// normalised and goes into the "data" slot; otherwise X goes into "counts".
        #[arg(long, default_value = "auto")]
        x_slot: String,

        /// Seurat project name written into the H5Seurat root (H5Seurat output only)
        #[arg(long, default_value = "SeuratProject")]
        project: String,

        /// Write SeuratDisk-compatible H5Seurat (adds version/project attrs and
        /// empty scaffold groups required by LoadH5Seurat).
        /// Default: lean output for Seurat v5 + BPCells direct loading.
        #[arg(long)]
        seuratdisk_compat: bool,

        /// Canonical URL of the source file (baked into uns["scx_provenance"])
        #[arg(long)]
        source_url: Option<String>,

        /// Pre-computed SHA-256 of the source file (64 lowercase hex chars).
        /// Skips rehashing — intended for pipelines like hapiq that already
        /// compute the hash on download.
        #[arg(long)]
        source_sha256: Option<String>,

        /// gzip-compress `.h5ad` output. Pass `--compress` for the default
        /// level 6, or `--compress N` for level N (0..=9). Off by default.
        /// Ignored for `.h5seurat` output.
        #[arg(long, num_args = 0..=1, default_missing_value = "6")]
        compress: Option<u8>,

        /// Write only these comma-separated auxiliary slots (e.g.
        /// `--only obsp` or `--only obsm:X_pca,layers`). Applies to
        /// obsm/varm/uns/layers/obsp; X/obs/var are always written.
        /// Excluding obsm also skips reading it (lowers peak RAM).
        /// Mutually exclusive with --exclude.
        #[arg(long, conflicts_with = "exclude")]
        only: Option<String>,

        /// Drop these comma-separated auxiliary slots (e.g.
        /// `--exclude obsp,obsm,uns`). Applies to obsm/varm/uns/layers/obsp;
        /// X/obs/var are always written. Mutually exclusive with --only.
        #[arg(long, conflicts_with = "only")]
        exclude: Option<String>,
    },

    /// Inspect a single-cell file
    ///
    /// Prints format, shape, and a summary of every slot (obs, var, obsm,
    /// layers, obsp, varp, varm, uns) without converting.
    ///
    /// For 10x HDF5 files (Cell Ranger output), shows barcode/feature counts
    /// and feature-type breakdown instead of obs/var slots.
    ///
    /// For any other valid HDF5 file (plain/unrecognized), prints a depth-2
    /// tree of groups and datasets with shapes and dtypes — useful for
    /// exploring intermediate files saved in HDF5.
    Inspect {
        /// Input file
        input: String,

        /// Seurat assay (H5Seurat only)
        #[arg(long, default_value = "RNA")]
        assay: String,

        /// Matrix layer to read: the Seurat layer (H5Seurat), or an H5AD
        /// `/layers/<name>` used in place of `/X`. Defaults to `/X` for H5AD
        /// (auto-falling-back to a sole layer when `/X` is absent) and to
        /// `counts` for H5Seurat.
        #[arg(long)]
        layer: Option<String>,
    },

    /// Validate a single-cell file against a YAML schema
    ///
    /// Checks shape, slot presence, dtypes, and index uniqueness without
    /// loading any matrix data. Exits 0 on pass, 1 on validation failure,
    /// 2 on file or schema parse error.
    ///
    /// Example schema (schemas/after_normalization.yaml):
    ///   obs: ">= 1000"
    ///   vars: ">= 500"
    ///   layers: [normalized]
    ///   obsm: [X_pca]
    ///   x_dtype: f32
    ///   obs_index_unique: true
    Validate {
        /// Input file (.h5ad)
        input: String,

        /// Path to YAML schema file
        #[arg(long)]
        schema: String,

        /// Emit results as JSON instead of the default human-readable report
        #[arg(long)]
        json: bool,
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

    /// Export a tabular slot to CSV or Parquet
    ///
    /// Reads the specified slot from the input file and writes it as a flat
    /// table.  The output format is determined by the file extension:
    ///   .csv      — comma-separated values (default)
    ///   .parquet  — Apache Parquet
    ///
    /// Supported slots:
    ///   obs           — cell metadata (index + all columns)
    ///   var           — gene/feature metadata (index + all columns)
    ///   obsm/<name>   — embedding matrix (index + dim_0, dim_1, …)
    ///
    /// Examples:
    ///   scx export merged.h5ad --slot obs --output cells.csv
    ///   scx export merged.h5ad --slot var --output genes.parquet
    ///   scx export merged.h5ad --slot obsm/X_pca --output pca.csv
    Export {
        /// Input file (h5ad or h5seurat)
        input: String,

        /// Slot to export: obs, var, or obsm/<name>
        #[arg(long, default_value = "obs")]
        slot: String,

        /// Output path (.csv or .parquet)
        #[arg(long, short = 'o')]
        output: String,

        /// Seurat assay (H5Seurat only)
        #[arg(long, default_value = "RNA")]
        assay: String,

        /// Seurat layer (H5Seurat only)
        #[arg(long, default_value = "counts")]
        layer: String,
    },

    /// Concatenate datasets along the obs (cell) axis
    ///
    /// Follows `anndata.concat()` semantics: cells are stacked, genes are
    /// aligned by name (inner join by default), and obs columns missing from an
    /// input are NA-filled (float NaN, int 0, bool false, string "",
    /// categorical "NA").  X and layers stream chunk-by-chunk; obs/var/obsm are
    /// materialised.  Output is always .h5ad.
    ///
    /// Not carried over: obsp/varp (anndata's `pairwise`, off by default),
    /// varm, and uns — the output's uns holds only the concat provenance record.
    ///
    /// Examples:
    ///   scx concat a.h5ad b.h5ad -o all.h5ad --label sample
    ///   scx concat s*.h5ad -o all.h5ad --join outer --index-unique -
    Concat {
        /// Input files (two or more)
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<String>,

        /// Output .h5ad path
        #[arg(long, short = 'o')]
        output: String,

        /// Gene-axis join: inner (shared genes) or outer (union, zero-filled)
        #[arg(long, default_value = "inner")]
        join: String,

        /// Add an obs column with this name recording each cell's source
        #[arg(long)]
        label: Option<String>,

        /// Comma-separated source names, one per input (default: file stems).
        /// Used as --label values and as the --index-unique suffix.
        #[arg(long)]
        keys: Option<String>,

        /// Separator appended to obs_names to keep them unique (e.g. `-`)
        #[arg(long)]
        index_unique: Option<String>,

        /// How to carry var columns over [none, same, unique, first, only]
        #[arg(long, default_value = "none")]
        merge: String,

        /// Cells per streaming chunk
        #[arg(long, default_value = "5000")]
        chunk_size: usize,

        /// Output data type [f32, f64, i32, u32]
        #[arg(long, default_value = "f32")]
        dtype: String,

        /// gzip-compress the output (`--compress` for level 6, `--compress N` for 0..=9)
        #[arg(long, num_args = 0..=1, default_missing_value = "6")]
        compress: Option<u8>,
    },

    /// Assemble slots from multiple files onto a base h5ad (slot-patch merge)
    ///
    /// Two modes:
    ///   Create: --base source.h5ad --patch ... --output merged.h5ad
    ///   Append: --into merged.h5ad --patch ...
    ///
    /// Patch format: file.h5ad:slot/name[,slot/name...]
    ///   Supported slot groups: layers, obs, var, obsm, varm, obsp, uns
    ///
    /// Examples:
    ///   scx merge --base raw.h5ad --patch norm.h5ad:layers/normalized --output merged.h5ad
    ///   scx merge --config merge.yaml
    Merge {
        /// Base file to copy (create mode; mutually exclusive with --into)
        #[arg(long, conflicts_with = "into")]
        base: Option<String>,

        /// Output path for the new merged file (create mode)
        #[arg(long, conflicts_with = "into")]
        output: Option<String>,

        /// Existing merged file to append into (append mode)
        #[arg(long, conflicts_with = "base")]
        into: Option<String>,

        /// Patch spec: file.h5ad:slot/name[,slot/name...]
        #[arg(long = "patch", value_name = "SPEC")]
        patches: Vec<String>,

        /// How to handle a slot that already exists [error, skip, overwrite]
        #[arg(long, default_value = "error")]
        on_conflict: String,

        /// Embed a tag in the provenance record (key=value)
        #[arg(long = "tag", value_name = "KEY=VALUE")]
        tags: Vec<String>,

        /// Path to a YAML config file (supersedes all other flags)
        #[arg(long)]
        config: Option<String>,

        /// Cells per streaming chunk
        #[arg(long, default_value = "5000")]
        chunk_size: usize,
    },
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    if let Err(e) = drive(run()) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Drive the async CLI to completion.
///
/// With `net`, object-store backends (S3/GCS/HTTP) need a live tokio reactor, so
/// the whole CLI runs inside a multi-thread tokio runtime — the single entry
/// point where the runtime is constructed. Multi-thread (not current-thread) is
/// deliberate: the merge/export paths nest `futures::executor::block_on`, which
/// would deadlock a single-threaded runtime but merely parks one worker here.
#[cfg(feature = "net")]
fn drive<F: std::future::Future<Output = anyhow::Result<()>>>(fut: F) -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime")
        .block_on(fut)
}

/// Without `net` there is no tokio dependency and all I/O is local/synchronous,
/// so the lightweight futures executor is sufficient.
#[cfg(not(feature = "net"))]
fn drive<F: std::future::Future<Output = anyhow::Result<()>>>(fut: F) -> anyhow::Result<()> {
    futures::executor::block_on(fut)
}

async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli {
        Cli::Validate {
            input,
            schema,
            json,
        } => {
            let schema_src = std::fs::read_to_string(&schema)
                .map_err(|e| anyhow::anyhow!("cannot read schema '{schema}': {e}"))?;
            let schema_parsed: ValidationSchema = serde_yaml::from_str(&schema_src)
                .map_err(|e| anyhow::anyhow!("invalid schema '{schema}': {e}"))?;

            let opts = scx_core::OpenOptions {
                metadata_only: true,
                ..scx_core::OpenOptions::new(1000)
            };
            let mut reader = scx_core::open(&input, &opts)
                .await
                .map_err(|e| anyhow::anyhow!("cannot open '{input}': {e}"))?;
            let report = run_validation(&mut *reader, &schema_parsed, &input, &schema).await?;

            if json {
                print_report_json(&report);
            } else {
                print_report_human(&report);
            }

            if !report.passed() {
                std::process::exit(1);
            }
        }

        Cli::Inspect {
            input,
            assay,
            layer,
        } => {
            let input_path = Path::new(&input);
            let fmt = detect::detect(input_path);

            let chunk = 1000;
            match fmt {
                Some(Format::NpyDir) => {
                    let mut r = NpyIrReader::open(input_path, chunk)?;
                    inspect(&mut r, &input, "NPY snapshot").await?;
                }
                Some(Format::BPCells) => {
                    let mut r = BpcellsDatasetReader::open(input_path, chunk)?;
                    inspect(&mut r, &input, "BPCells").await?;
                }
                Some(Format::H5Seurat) => {
                    let seurat_layer = layer.as_deref().unwrap_or("counts");
                    let mut r = open_h5seurat(input_path, chunk, Some(&assay), Some(seurat_layer))?;
                    let fmt_name = if r.x_indptr().is_empty() {
                        "H5Seurat (BPCells)"
                    } else {
                        "H5Seurat"
                    };
                    inspect(&mut *r, &input, fmt_name).await?;
                }
                Some(Format::H5Ad) => {
                    let mut r = H5AdReader::open_layer(input_path, chunk, layer.as_deref())?;
                    if r.x_source() != "X" {
                        println!("matrix source: /{} (no /X)", r.x_source());
                    }
                    inspect(&mut r, &input, "H5AD").await?;
                }
                Some(Format::ScxH5) | None => {
                    let mut r = ScxH5Reader::open(input_path, chunk)?;
                    inspect(&mut r, &input, "SCX H5").await?;
                }
                Some(Format::TenxH5) => {
                    let mut r = TenxH5Reader::open(input_path, chunk)?;
                    inspect(&mut r, &input, "10x HDF5").await?;
                    let summary = read_tenx_summary(input_path)?;
                    print_tenx_summary(&summary);
                }
                Some(Format::Mtx) => {
                    let mut r = scx_core::mtx::MtxReader::open(input_path, chunk)?;
                    inspect(&mut r, &input, "MatrixMarket").await?;
                }
                Some(Format::PlainH5) => {
                    let nodes = walk_h5(input_path, 2)?;
                    inspect_plain_h5(&nodes, &input);
                }
                Some(Format::Parquet) => {
                    anyhow::bail!(
                        "inspecting Parquet input is not supported yet — use 'scx convert'"
                    )
                }
            }
        }

        Cli::Convert {
            input,
            output,
            chunk_size,
            n_vars: n_vars_arg,
            genes: genes_arg,
            dtype,
            assay,
            layer,
            dgcmatrix,
            x_slot,
            project,
            seuratdisk_compat,
            source_url,
            source_sha256: source_sha256_arg,
            compress,
            only,
            exclude,
        } => {
            let filter = match (only.as_deref(), exclude.as_deref()) {
                (Some(o), _) => SlotFilter::from_only(o),
                (_, Some(e)) => SlotFilter::from_exclude(e),
                _ => SlotFilter::all(),
            };

            let out_dtype = parse_dtype(&dtype)?;

            let input_path = Path::new(&input);
            let output_path = Path::new(&output);

            let source_sha256 = if let Some(hex) = source_sha256_arg {
                if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
                    anyhow::bail!(
                        "--source-sha256 must be 64 hex characters, got {} chars",
                        hex.len()
                    );
                }
                Some(hex.to_ascii_lowercase())
            } else if input_path.is_file() {
                Some(
                    provenance::sha256_file(input_path)
                        .map_err(|e| anyhow::anyhow!("hashing source '{input}': {e}"))?,
                )
            } else {
                None
            };

            // Detect the format (or route a URL to Parquet) and open the reader
            // in one call. --assay/--layer apply to H5Seurat, --n-vars/--genes to
            // Parquet; options irrelevant to the detected format are ignored.
            let opts = scx_core::OpenOptions {
                chunk_size,
                assay: Some(assay.clone()),
                layer: Some(layer.clone()),
                metadata_only: false,
                n_vars: n_vars_arg,
                genes: genes_arg.clone(),
            };
            let mut reader = scx_core::open(&input, &opts)
                .await
                .map_err(|e| anyhow::anyhow!("cannot open '{input}': {e}"))?;
            let (n_obs, n_vars) = convert_with_reader(
                &mut *reader,
                output_path,
                out_dtype,
                &assay,
                &layer,
                &x_slot,
                &project,
                chunk_size,
                dgcmatrix,
                seuratdisk_compat,
                &input,
                source_url.as_deref(),
                source_sha256.clone(),
                compress,
                &filter,
            )
            .await?;

            let output_sha256 = provenance::sha256_file(output_path)
                .map_err(|e| anyhow::anyhow!("hashing output '{output}': {e}"))?;

            let record = ProvenanceRecord {
                scx_version: env!("CARGO_PKG_VERSION").to_string(),
                converted_at: provenance::utc_now_rfc3339(),
                source: SourceInfo {
                    path: input,
                    url: source_url,
                    sha256: source_sha256,
                },
                output: OutputInfo {
                    path: output.clone(),
                    sha256: output_sha256,
                    n_obs,
                    n_vars,
                },
            };
            provenance::write_sidecar(&record, output_path)
                .map_err(|e| anyhow::anyhow!("writing provenance sidecar: {e}"))?;
        }

        Cli::Snapshot {
            input,
            output_dir,
            only,
            exclude,
            chunk_size,
            assay,
            layer,
        } => {
            let output_path = Path::new(&output_dir);

            let filter = match (only.as_deref(), exclude.as_deref()) {
                (Some(o), _) => SlotFilter::from_only(o),
                (_, Some(e)) => SlotFilter::from_exclude(e),
                _ => SlotFilter::all(),
            };

            tracing::info!(
                input = %input,
                output = %output_dir,
                "streaming IR snapshot"
            );

            let opts = scx_core::OpenOptions {
                assay: Some(assay.clone()),
                layer: Some(layer.clone()),
                ..scx_core::OpenOptions::new(chunk_size)
            };
            let mut reader = scx_core::open(&input, &opts)
                .await
                .map_err(|e| anyhow::anyhow!("cannot open '{input}': {e}"))?;
            let (n_obs, n_vars) = reader.shape();
            NpyIrWriter::stream(output_path, &mut *reader, &filter, chunk_size).await?;

            tracing::info!(
                output = %output_dir,
                n_obs,
                n_vars,
                "snapshot written"
            );
        }

        Cli::Export {
            input,
            slot,
            output,
            assay,
            layer,
        } => {
            cmd_export::run_export(cmd_export::ExportArgs {
                input,
                slot,
                output,
                assay,
                layer,
            })
            .await?;
        }

        Cli::Concat {
            inputs,
            output,
            join,
            label,
            keys,
            index_unique,
            merge,
            chunk_size,
            dtype,
            compress,
        } => {
            let opts = scx_core::concat::ConcatOptions {
                join: join.parse()?,
                label,
                keys: keys
                    .map(|k| k.split(',').map(|s| s.trim().to_string()).collect())
                    .unwrap_or_default(),
                index_unique,
                merge: merge.parse()?,
                chunk_size,
                dtype: parse_dtype(&dtype)?,
                compress,
            };
            let (n_obs, n_vars) =
                scx_core::concat::concat(&inputs, Path::new(&output), &opts).await?;
            println!(
                "concat: {} files -> {} ({n_obs} x {n_vars})",
                inputs.len(),
                output
            );
        }

        Cli::Merge {
            base,
            output,
            into,
            patches,
            on_conflict,
            tags,
            config,
            chunk_size,
        } => {
            cmd_merge::run_merge(cmd_merge::MergeArgs {
                base,
                output,
                into,
                patches,
                on_conflict,
                tags,
                config,
                chunk_size,
            })
            .await?;
        }
    }

    Ok(())
}

fn parse_dtype(s: &str) -> anyhow::Result<DataType> {
    match s {
        "f32" => Ok(DataType::F32),
        "f64" => Ok(DataType::F64),
        "i32" => Ok(DataType::I32),
        "u32" => Ok(DataType::U32),
        other => anyhow::bail!("unknown dtype '{other}': use f32, f64, i32, u32"),
    }
}

#[allow(clippy::too_many_arguments)]
async fn convert_with_reader(
    reader: &mut dyn DatasetReader,
    output: &Path,
    out_dtype: DataType,
    out_assay: &str,
    out_layer: &str,
    x_slot: &str,
    project: &str,
    chunk_size: usize,
    use_dgcmatrix: bool,
    seuratdisk_compat: bool,
    source_path: &str,
    source_url: Option<&str>,
    source_sha256: Option<String>,
    compress: Option<u8>,
    filter: &SlotFilter,
) -> anyhow::Result<(usize, usize)> {
    let t0 = std::time::Instant::now();
    let (n_obs, n_vars) = reader.shape();

    let is_h5seurat = output.extension().and_then(|e| e.to_str()) == Some("h5seurat");

    if compress.is_some() && is_h5seurat {
        tracing::warn!("--compress only applies to .h5ad output; ignored for .h5seurat");
    }

    // X, obs and var are always written (the matrix plus the mandatory frame
    // indices); the slot filter only governs the auxiliary slots. Excluding a
    // whole slot skips its read too, so `--exclude obsm` avoids materialising
    // the (potentially large) embeddings.
    let write_uns = filter.includes("uns");
    let obs = reader.obs().await?;
    let var = reader.var().await?;
    let mut obsm = if filter.includes("obsm") {
        reader.obsm().await?
    } else {
        Embeddings::default()
    };
    obsm.map
        .retain(|k, _| filter.includes(&format!("obsm:{k}")));
    let mut varm = if filter.includes("varm") {
        reader.varm().await?
    } else {
        Varm::default()
    };
    varm.map
        .retain(|k, _| filter.includes(&format!("varm:{k}")));
    let mut uns = if write_uns {
        reader.uns().await?
    } else {
        UnsTable::default()
    };

    if write_uns {
        let prov = scx_core::provenance::det_record(
            source_path,
            source_url,
            source_sha256.as_deref(),
            n_obs,
            n_vars,
        );
        match uns.raw.as_object_mut() {
            Some(obj) => {
                obj.insert("scx_provenance".to_string(), prov);
            }
            None => {
                uns.raw = serde_json::json!({ "scx_provenance": prov });
            }
        }
    }

    let layer_metas = reader.layer_metas().await?;
    let obsp_metas = reader.obsp_metas().await?;

    // Resolve the effective X slot for H5Seurat output.
    // auto: if source has a "counts" layer, X is assumed normalised → goes to "data".
    // Explicit "counts" or "data" overrides the heuristic.
    let effective_x_slot: &str = if is_h5seurat {
        match x_slot {
            "auto" => {
                if layer_metas.iter().any(|m| m.name == "counts") {
                    tracing::info!(
                        "source has a 'counts' layer — writing X to 'data' slot, \
                         'counts' layer to 'counts' slot (use --x-slot counts to override)"
                    );
                    "data"
                } else {
                    out_layer
                }
            }
            other => other,
        }
    } else {
        out_layer
    };

    tracing::info!(
        output = %output.display(),
        n_obs, n_vars,
        dtype = %out_dtype,
        format = if is_h5seurat { "h5seurat" } else { "h5ad" },
        bpcells = is_h5seurat && !use_dgcmatrix,
        dgcmatrix = use_dgcmatrix && is_h5seurat,
        x_slot = effective_x_slot,
        "starting conversion"
    );

    tracing::info!(
        obs_cols = obs.columns.len(),
        var_cols = var.columns.len(),
        embeddings = obsm.map.len(),
        layers = layer_metas.len(),
        obsp = obsp_metas.len(),
        "metadata loaded in {:.2?}",
        t0.elapsed()
    );

    let mut writer: Box<dyn DatasetWriter> = if is_h5seurat {
        if use_dgcmatrix {
            Box::new(H5SeuratWriter::create(
                output,
                n_obs,
                n_vars,
                out_dtype,
                Some(out_assay),
                Some(effective_x_slot),
                Some(project),
                seuratdisk_compat,
            )?)
        } else {
            Box::new(BpcellsH5Writer::create(
                output,
                n_obs,
                n_vars,
                out_dtype,
                Some(out_assay),
                Some(effective_x_slot),
                Some(project),
                seuratdisk_compat,
            )?)
        }
    } else {
        Box::new(H5AdWriter::create_compressed(
            output, n_obs, n_vars, out_dtype, compress,
        )?)
    };

    writer.write_obs(&obs).await?;
    writer.write_var(&var).await?;
    if !obsm.map.is_empty() {
        writer.write_obsm(&obsm).await?;
    }
    if write_uns {
        writer.write_uns(&uns).await?;
    }
    if !varm.map.is_empty() {
        writer.write_varm(&varm).await?;
    }

    // Stream each layer — skip any whose name would collide with the X slot,
    // or that the slot filter drops.
    for meta in &layer_metas {
        if !filter.includes(&format!("layers:{}", meta.name)) {
            tracing::info!(name = %meta.name, "skipping layer (slot filter)");
            continue;
        }
        if is_h5seurat && meta.name == effective_x_slot {
            tracing::warn!(
                name = %meta.name,
                "skipping layer '{}': same name as X slot — \
                 use --x-slot to change slot assignment", meta.name
            );
            continue;
        }
        tracing::info!(name = %meta.name, shape = ?meta.shape, "streaming layer");
        writer.begin_sparse("layers", &meta.name, meta).await?;
        let mut stream = reader.layer_stream(meta, chunk_size);
        while let Some(chunk) = stream.next().await {
            writer.write_sparse_chunk(&chunk?).await?;
        }
        writer.end_sparse().await?;
    }

    // Stream each obsp matrix.
    for meta in &obsp_metas {
        if !filter.includes(&format!("obsp:{}", meta.name)) {
            tracing::info!(name = %meta.name, "skipping obsp (slot filter)");
            continue;
        }
        tracing::info!(name = %meta.name, shape = ?meta.shape, "streaming obsp");
        writer.begin_sparse("obsp", &meta.name, meta).await?;
        let mut stream = reader.obsp_stream(meta, chunk_size);
        while let Some(chunk) = stream.next().await {
            writer.write_sparse_chunk(&chunk?).await?;
        }
        writer.end_sparse().await?;
    }

    let t_x = std::time::Instant::now();
    let mut stream = reader.x_stream();
    let mut total_nnz = 0usize;
    let mut n_chunks = 0usize;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        total_nnz += chunk.data.indices.len();
        n_chunks += 1;
        writer.write_x_chunk(&chunk).await?;
    }

    tracing::info!(
        n_chunks,
        total_nnz,
        throughput_cells_s = (n_obs as f64 / t_x.elapsed().as_secs_f64()) as u64,
        "matrix streamed in {:.2?}",
        t_x.elapsed()
    );

    writer.finalize().await?;

    tracing::info!(
        total = ?t0.elapsed(),
        output = %output.display(),
        "conversion complete"
    );

    Ok((n_obs, n_vars))
}

// ---------------------------------------------------------------------------
// Inspect
// ---------------------------------------------------------------------------

fn col_type_str(data: &ColumnData) -> &'static str {
    data.dtype_str()
}

fn fmt_stat(v: f64) -> String {
    let abs = v.abs();
    if v == 0.0 {
        "0".to_string()
    } else if (0.001..100_000.0).contains(&abs) {
        let s = format!("{:.4}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        format!("{:.3e}", v)
    }
}

fn indptr_row_stats(indptr: &[u64], n_rows: usize, n_cols: usize) -> String {
    if indptr.len() < 2 || n_rows == 0 || n_cols == 0 {
        return String::new();
    }
    let nnz = *indptr.last().unwrap_or(&0) as usize;
    let sparsity = 100.0 * (1.0 - nnz as f64 / (n_rows as f64 * n_cols as f64));
    let mut per_row: Vec<u64> = indptr.windows(2).map(|w| w[1] - w[0]).collect();
    per_row.sort_unstable();
    let n = per_row.len();
    let q = |p: f64| per_row[(p * (n - 1) as f64).round() as usize];
    format!(
        "nnz={}  sparse={:.1}%  nnz/cell Q1={}  med={}  Q3={}  max={}",
        nnz,
        sparsity,
        q(0.25),
        q(0.5),
        q(0.75),
        per_row[n - 1]
    )
}

fn numeric_stats(data: &ColumnData) -> String {
    let mut vals: Vec<f64> = match data {
        ColumnData::Float(v) => v.clone(),
        ColumnData::Int(v) => v.iter().map(|&x| x as f64).collect(),
        _ => return String::new(),
    };
    if vals.is_empty() {
        return String::new();
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();

    // Binary {0, 1} column — show counts instead of quartiles
    if vals[0] >= 0.0 && vals[n - 1] <= 1.0 {
        let ones = vals.iter().filter(|&&v| v == 1.0).count();
        let zeros = n - ones;
        if ones + zeros == n {
            return format!(
                "bool-like  0: {} ({:.1}%)  1: {} ({:.1}%)",
                zeros,
                100.0 * zeros as f64 / n as f64,
                ones,
                100.0 * ones as f64 / n as f64,
            );
        }
    }

    let q = |p: f64| vals[(p * (n - 1) as f64).round() as usize];
    format!(
        "min={}  Q1={}  med={}  Q3={}  max={}",
        fmt_stat(vals[0]),
        fmt_stat(q(0.25)),
        fmt_stat(q(0.5)),
        fmt_stat(q(0.75)),
        fmt_stat(vals[n - 1])
    )
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

async fn inspect(
    reader: &mut dyn DatasetReader,
    path: &str,
    format_name: &str,
) -> anyhow::Result<()> {
    let (n_obs, n_vars) = reader.shape();

    // Colour helpers -- all checks against Stdout so piped output stays plain.
    macro_rules! bold {
        ($x:expr) => {
            $x.if_supports_color(Stdout, |t| t.bold())
        };
    }
    macro_rules! cyan {
        ($x:expr) => {
            $x.if_supports_color(Stdout, |t| t.bright_cyan())
        };
    }
    macro_rules! green {
        ($x:expr) => {
            $x.if_supports_color(Stdout, |t| t.bright_green())
        };
    }
    macro_rules! dim {
        ($x:expr) => {
            $x.if_supports_color(Stdout, |t| t.dimmed())
        };
    }
    // yellow! and bold_cyan! use Style to avoid borrow-of-temporary when chaining.
    macro_rules! yellow {
        ($x:expr) => {{
            use owo_colors::Style;
            $x.if_supports_color(Stdout, |t| t.style(Style::new().bright_yellow()))
        }};
    }
    macro_rules! bold_cyan {
        ($x:expr) => {{
            use owo_colors::Style;
            $x.if_supports_color(Stdout, |t| t.style(Style::new().bold().bright_cyan()))
        }};
    }

    println!("{} {}", bold!("File   :"), green!(path));
    println!("{} {}", bold!("Format :"), cyan!(format_name));
    println!(
        "{} {} {} × {} {}",
        bold!("Shape  :"),
        yellow!(n_obs),
        dim!("obs"),
        yellow!(n_vars),
        dim!("vars"),
    );
    let dtype_str = reader.dtype().to_string();
    println!("{} {}", bold!("X dtype:"), cyan!(&dtype_str));
    let x_stats = indptr_row_stats(reader.x_indptr(), n_obs, n_vars);
    if !x_stats.is_empty() {
        println!("{} {}", bold!("X      :"), dim!(&x_stats));
    }
    println!();

    // section header helper
    let section = |name: &str, count: usize, unit: &str| {
        let label = format!(" {unit}):");
        println!(
            "{} {}{}{}",
            bold_cyan!(name),
            bold!("("),
            yellow!(count),
            bold!(&label),
        );
    };

    // ── obs ──────────────────────────────────────────────────────────────────
    let obs = reader.obs().await?;
    section("obs", obs.columns.len(), "columns");
    if obs.columns.is_empty() {
        println!("  {}", dim!("(none)"));
    }
    for col in &obs.columns {
        let extra = cat_levels_preview(&col.data);
        let stats = numeric_stats(&col.data);
        let type_str = col_type_str(&col.data);
        let annotation = if !extra.is_empty() {
            format!("  {} {}", dim!("—"), dim!(&extra))
        } else if !stats.is_empty() {
            format!("  {} {}", dim!("—"), dim!(&stats))
        } else {
            String::new()
        };
        println!("  {:<30} {}{}", col.name, dim!(&type_str), annotation);
    }
    println!();

    // ── var ──────────────────────────────────────────────────────────────────
    let var = reader.var().await?;
    section("var", var.columns.len(), "columns");
    if var.columns.is_empty() {
        println!("  {}", dim!("(none)"));
    }
    for col in &var.columns {
        let extra = cat_levels_preview(&col.data);
        let type_str = col_type_str(&col.data);
        if extra.is_empty() {
            println!("  {:<30} {}", col.name, dim!(&type_str));
        } else {
            println!(
                "  {:<30} {}  {} {}",
                col.name,
                dim!(&type_str),
                dim!("—"),
                dim!(&extra)
            );
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
    if obsm.map.is_empty() {
        println!("  {}", dim!("(none)"));
    }
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
    if varm.map.is_empty() {
        println!("  {}", dim!("(none)"));
    }
    println!();

    // ── layers ───────────────────────────────────────────────────────────────
    let layer_metas = reader.layer_metas().await?;
    section("layers", layer_metas.len(), "keys");
    let mut sorted_layers = layer_metas.clone();
    sorted_layers.sort_by(|a, b| a.name.cmp(&b.name));
    for m in &sorted_layers {
        let stats = if m.indptr.is_empty() {
            "(dense)".to_string()
        } else {
            indptr_row_stats(&m.indptr, m.shape.0, m.shape.1)
        };
        println!(
            "  {:<30} {} × {}  {}",
            m.name,
            yellow!(m.shape.0),
            yellow!(m.shape.1),
            dim!(&stats),
        );
    }
    if layer_metas.is_empty() {
        println!("  {}", dim!("(none)"));
    }
    println!();

    // ── obsp ─────────────────────────────────────────────────────────────────
    let obsp_metas = reader.obsp_metas().await?;
    section("obsp", obsp_metas.len(), "keys");
    let mut sorted_obsp = obsp_metas.clone();
    sorted_obsp.sort_by(|a, b| a.name.cmp(&b.name));
    for m in &sorted_obsp {
        let stats = if m.indptr.is_empty() {
            "(dense)".to_string()
        } else {
            indptr_row_stats(&m.indptr, m.shape.0, m.shape.1)
        };
        println!(
            "  {:<30} {} × {}  {}",
            m.name,
            yellow!(m.shape.0),
            yellow!(m.shape.1),
            dim!(&stats),
        );
    }
    if obsp_metas.is_empty() {
        println!("  {}", dim!("(none)"));
    }
    println!();

    // ── provenance + uns ─────────────────────────────────────────────────────
    let uns = reader.uns().await?;
    let prov = uns.raw.as_object().and_then(|o| o.get("scx_provenance"));

    println!("{}", bold_cyan!("provenance"));
    match prov {
        Some(serde_json::Value::Object(p)) => {
            let scx_version = p.get("scx_version").and_then(|v| v.as_str()).unwrap_or("?");
            println!("  {:<16} {}", "scx_version", dim!(scx_version));

            // Convert provenance: "source" key
            if let Some(src) = p.get("source").and_then(|v| v.as_object()) {
                if let Some(url) = src.get("url").and_then(|v| v.as_str()) {
                    println!("  {:<16} {}", "source.url", dim!(url));
                }
                if let Some(path) = src.get("path").and_then(|v| v.as_str()) {
                    println!("  {:<16} {}", "source.path", dim!(path));
                }
                if let Some(sha) = src.get("sha256").and_then(|v| v.as_str()) {
                    println!("  {:<16} {}", "source.sha256", dim!(sha));
                }
            }

            // Merge provenance: "base" anchor
            if let Some(base) = p.get("base").and_then(|v| v.as_object()) {
                if let Some(path) = base.get("path").and_then(|v| v.as_str()) {
                    println!("  {:<16} {}", "base.path", dim!(path));
                }
                if let Some(sha) = base.get("sha256").and_then(|v| v.as_str()) {
                    let short = &sha[..12.min(sha.len())];
                    println!("  {:<16} {}…", "base.sha256", dim!(short));
                }
                if let Some(n) = base.get("n_obs").and_then(|v| v.as_u64()) {
                    println!("  {:<16} {}", "base.n_obs", dim!(n));
                }
            }

            // Tags
            if let Some(tags) = p.get("tags").and_then(|v| v.as_object()) {
                for (k, v) in tags {
                    if let Some(s) = v.as_str() {
                        println!("  {:<16} {}", format!("tag.{k}"), dim!(s));
                    }
                }
            }

            // Slots map (merge provenance)
            if let Some(slots) = p.get("slots").and_then(|v| v.as_object()) {
                if !slots.is_empty() {
                    println!();
                    let hdr = format!("provenance slots ({} patched)", slots.len());
                    println!("{}", bold_cyan!(&hdr));
                    let mut keys: Vec<&String> = slots.keys().collect();
                    keys.sort();
                    for key in keys {
                        if let Some(entry) = slots[key].as_object() {
                            let src = entry
                                .get("source_path")
                                .and_then(|v| v.as_str())
                                .unwrap_or("?");
                            let sha = entry.get("sha256").and_then(|v| v.as_str()).unwrap_or("");
                            let sha_short = &sha[..12.min(sha.len())];
                            let at = entry.get("added_at").and_then(|v| v.as_str()).unwrap_or("");
                            println!(
                                "  {:<38} {}  {}  {}",
                                key,
                                dim!(src),
                                dim!(&format!("[{sha_short}…]")),
                                dim!(at),
                            );
                        }
                    }
                }
            }
        }
        _ => println!("  {}", dim!("(none)")),
    }
    println!();

    let uns_obj_filtered: Option<Vec<(&String, &serde_json::Value)>> =
        uns.raw.as_object().map(|o| {
            o.iter()
                .filter(|(k, _)| k.as_str() != "scx_provenance")
                .collect()
        });

    if uns.raw.is_null() {
        section("uns", 0, "keys");
        println!("  {}", dim!("(none)"));
    } else if let Some(mut entries) = uns_obj_filtered {
        section("uns", entries.len(), "keys");
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (k, v) in entries {
            let summary = match v {
                serde_json::Value::Array(a) => format!("array [{}]", a.len()),
                serde_json::Value::Object(o) => format!("dict  ({} keys)", o.len()),
                serde_json::Value::String(s) => {
                    if s.len() > 60 {
                        format!("\"{}...\"", &s[..57])
                    } else {
                        format!("\"{s}\"")
                    }
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

// ---------------------------------------------------------------------------
// 10x HDF5 supplementary summary
// ---------------------------------------------------------------------------

fn print_tenx_summary(s: &scx_core::tenx::TenxSummary) {
    use owo_colors::{OwoColorize, Stream::Stdout, Style};

    if s.feature_types.is_empty() && s.genomes.is_empty() {
        return;
    }

    let header = "10x".if_supports_color(Stdout, |t| t.style(Style::new().bold().bright_cyan()));
    println!("{header}");

    if !s.genomes.is_empty() {
        let label = "  genome".if_supports_color(Stdout, |t| t.bold());
        let val = s
            .genomes
            .join(", ")
            .if_supports_color(Stdout, |t| t.dimmed())
            .to_string();
        println!("{label:<10} {val}");
    }

    if !s.feature_types.is_empty() {
        let label = "  feature_types".if_supports_color(Stdout, |t| t.bold());
        println!("{label}");
        for (ft, count) in &s.feature_types {
            let count_s = count
                .to_string()
                .if_supports_color(Stdout, |t| t.style(Style::new().bright_yellow()))
                .to_string();
            println!("    {ft:<40} {count_s}");
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// Plain HDF5 inspect
// ---------------------------------------------------------------------------

fn inspect_plain_h5(nodes: &[H5Node], path: &str) {
    use owo_colors::OwoColorize;
    use owo_colors::Stream::Stdout;

    macro_rules! bold {
        ($x:expr) => {
            $x.if_supports_color(Stdout, |t| t.bold())
        };
    }
    macro_rules! cyan {
        ($x:expr) => {
            $x.if_supports_color(Stdout, |t| t.bright_cyan())
        };
    }
    macro_rules! green {
        ($x:expr) => {
            $x.if_supports_color(Stdout, |t| t.bright_green())
        };
    }
    macro_rules! dim {
        ($x:expr) => {
            $x.if_supports_color(Stdout, |t| t.dimmed())
        };
    }

    println!("{} {}", bold!("File   :"), green!(path));
    println!("{} {}", bold!("Format :"), cyan!("HDF5 (unrecognized)"));
    println!();

    fn print_nodes(nodes: &[H5Node], prefix: &str) {
        use owo_colors::{OwoColorize, Stream::Stdout};
        macro_rules! dim {
            ($x:expr) => {
                $x.if_supports_color(Stdout, |t| t.dimmed())
            };
        }

        for (i, node) in nodes.iter().enumerate() {
            let is_last = i == nodes.len() - 1;
            let connector = if is_last { "└─" } else { "├─" };
            let child_prefix = format!("{}{}  ", prefix, if is_last { "  " } else { "│ " });

            match &node.kind {
                H5NodeKind::Dataset { shape, dtype } => {
                    let shape_str = if shape.is_empty() {
                        "scalar".to_string()
                    } else {
                        format!(
                            "({})",
                            shape
                                .iter()
                                .map(|d| d.to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    };
                    println!(
                        "{}{}  {}  {}",
                        prefix,
                        connector,
                        node.name,
                        dim!(format!("{shape_str}  {dtype}").as_str()),
                    );
                }
                H5NodeKind::Group {
                    children,
                    truncated,
                } => {
                    println!("{}{} {}/", prefix, connector, node.name);
                    print_nodes(children, &child_prefix);
                    if *truncated > 0 {
                        println!(
                            "{}  {}",
                            child_prefix,
                            dim!(format!("… {truncated} more (depth limit)").as_str()),
                        );
                    }
                }
            }
        }
    }

    if nodes.is_empty() {
        println!("{}", dim!("(empty file)"));
    } else {
        print_nodes(nodes, "");
    }
}

// ---------------------------------------------------------------------------
// Validation report printers
// ---------------------------------------------------------------------------

fn print_report_human(report: &scx_core::validate::ValidationReport) {
    use owo_colors::OwoColorize;

    for c in &report.checks {
        let status = if c.passed {
            "PASS".if_supports_color(Stdout, |t| t.green()).to_string()
        } else {
            "FAIL".if_supports_color(Stdout, |t| t.red()).to_string()
        };
        println!("{status}  {:<20} {}", c.name, c.detail);
    }

    println!();
    if report.passed() {
        println!("All {} checks passed.", report.n_passed());
    } else {
        println!(
            "{} check{} failed.",
            report.n_failed(),
            if report.n_failed() == 1 { "" } else { "s" }
        );
    }
}

fn print_report_json(report: &scx_core::validate::ValidationReport) {
    let checks: Vec<serde_json::Value> = report
        .checks
        .iter()
        .map(|c| {
            serde_json::json!({
                "name":   c.name,
                "status": if c.passed { "PASS" } else { "FAIL" },
                "detail": c.detail,
            })
        })
        .collect();

    let out = serde_json::json!({
        "file":   report.file,
        "schema": report.schema,
        "passed": report.n_passed(),
        "failed": report.n_failed(),
        "checks": checks,
    });

    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
