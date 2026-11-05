mod cmd_export;
mod cmd_merge;

use std::path::Path;

use clap::Parser;
use futures::StreamExt;
use owo_colors::{OwoColorize, Stream::Stdout};
use scx_core::{
    bpcells::{BpcellsDatasetReader, BpcellsDirWriter},
    detect,
    detect::Format,
    dtype::DataType,
    h5::{walk_h5, H5Node, H5NodeKind},
    h5ad::{H5AdReader, H5AdWriter},
    h5bpcells::BpcellsH5Writer,
    h5seurat::{open_h5seurat, H5SeuratWriter},
    ir::{ColumnData, Embeddings, UnsTable, Varm},
    npy::{NpyIrReader, NpyIrWriter, SlotFilter},
    provenance::{self, OutputInfo, ProvenanceRecord, SourceInfo},
    stream::{DatasetReader, DatasetWriter},
    tenx::{read_tenx_summary, TenxH5Reader},
    validate::{run_validation, ValidationSchema},
};

// `scx --version` names the optional features this binary was built with:
// a URL input or a Zarr store fails without them, so that is the first
// thing to check.
#[cfg(feature = "net")]
macro_rules! net_flag {
    () => {
        "+net"
    };
}
#[cfg(not(feature = "net"))]
macro_rules! net_flag {
    () => {
        "-net"
    };
}
#[cfg(feature = "zarr")]
macro_rules! zarr_flag {
    () => {
        "+zarr"
    };
}
#[cfg(not(feature = "zarr"))]
macro_rules! zarr_flag {
    () => {
        "-zarr"
    };
}
const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (",
    net_flag!(),
    " ",
    zarr_flag!(),
    ")"
);

#[derive(Clone, Copy, clap::ValueEnum)]
enum OutputHash {
    Blake3,
    Sha256,
    None,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum SourceHashAlgo {
    Blake3,
    Sha256,
    None,
}

impl SourceHashAlgo {
    /// The algorithm for a source hashed inside scx-core (downloads, stdin);
    /// `None` hashes with the default there and the result is dropped.
    fn algo(self) -> provenance::HashAlgo {
        match self {
            Self::Sha256 => provenance::HashAlgo::Sha256,
            Self::Blake3 | Self::None => provenance::HashAlgo::Blake3,
        }
    }
}

/// blake3 of a file's bytes, hashed across threads. Parallel `pread`s fill
/// one 16 MiB buffer at a time, so peak RSS stays at the buffer: an mmap of
/// the whole file (`update_mmap_rayon`) gives the same digest at the same
/// speed, but every mapped page counts toward the process's peak RSS, which
/// put a 3 GB output's size into convert's measured memory.
#[cfg(unix)]
fn blake3_file(path: &Path) -> std::io::Result<blake3::Hash> {
    use rayon::prelude::*;
    use std::os::unix::fs::FileExt;
    const BUF: usize = 16 << 20;
    const PIECE: usize = 1 << 20;
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    let mut buf = vec![0u8; BUF.min(len as usize)];
    let mut hasher = blake3::Hasher::new();
    let mut off = 0u64;
    while off < len {
        let n = BUF.min((len - off) as usize);
        buf[..n]
            .par_chunks_mut(PIECE)
            .enumerate()
            .try_for_each(|(i, c)| file.read_exact_at(c, off + (i * PIECE) as u64))?;
        hasher.update_rayon(&buf[..n]);
        off += n as u64;
    }
    Ok(hasher.finalize())
}

// No positional read in std off unix; Windows keeps the mmap path.
#[cfg(not(unix))]
fn blake3_file(path: &Path) -> std::io::Result<blake3::Hash> {
    Ok(blake3::Hasher::new().update_mmap_rayon(path)?.finalize())
}

#[derive(Parser)]
#[command(name = "scx", version = VERSION, about = "Single-cell streaming interop engine")]
enum Cli {
    /// Convert between single-cell formats
    ///
    /// Input auto-detected by content:
    ///   .h5seurat  — SeuratDisk H5Seurat (Seurat v3/v4)
    ///   .h5ad      — AnnData H5AD (CSR X only)
    ///   .h5        — SCX internal HDF5 schema, or 10x HDF5 (Cell Ranger output)
    ///   .zarr      — AnnData Zarr store, v2 or v3 (directory)
    ///
    /// Output format selected by extension:
    ///   .h5ad      — AnnData H5AD  (default)
    ///   .h5seurat  — SeuratDisk H5Seurat
    ///   .bpcells   — BPCells matrix directory (X + dimnames only)
    ///   .zarr      — AnnData Zarr v3 store (directory)
    #[command(verbatim_doc_comment)]
    Convert {
        /// Input file, a URL (with the `net` feature), or `-` to read from
        /// stdin (copied to a temporary file first, as a URL is)
        input: String,

        /// Output file (.h5ad or .h5seurat), or a directory: `.zarr` (AnnData Zarr v3) or `.bpcells` (X only)
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
        /// gene_metadata Parquet (columns token_id, ensembl_id, gene_symbol;
        /// Orion's gene_token_id/gene_name are accepted too).
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

        /// Where to write the matrix. H5Seurat: counts, data, or auto
        /// (default). H5AD: X (= auto) or layers/<name>, which writes the
        /// matrix there and leaves /X absent.
        ///
        /// auto (H5Seurat): if the source has a layer named "counts", or an
        /// AnnData raw.X with X's cells and genes, X is assumed to be
        /// normalised and goes into the "data" slot, and that layer or raw.X
        /// goes into "counts"; otherwise X goes into "counts". A source layer
        /// with the target's name is skipped.
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

        /// Pre-computed SHA-256 of the source file (64 hex chars, optionally
        /// prefixed `sha256:`).
        /// Skips rehashing — intended for pipelines like hapiq that already
        /// compute the hash on download.
        #[arg(long)]
        source_sha256: Option<String>,

        /// Hash of the source recorded in the provenance records
        /// (`source.hash`), computed while the conversion runs: `blake3`
        /// (default, multithreaded: ~0.15 s for 3 GB), `sha256` (~1.6 s on
        /// one core; what repositories publish) or `none`. `--source-sha256`
        /// records a hash you already have instead.
        #[arg(long, value_enum, default_value_t = SourceHashAlgo::Blake3)]
        source_hash: SourceHashAlgo,

        /// Hash of the output recorded in the provenance sidecar
        /// (`output.hash`). Hashing reads the whole output once more after
        /// conversion: for 3.5 GB, ~0.1 s with blake3 (multithreaded) vs ~1.7 s
        /// with sha256. `none` leaves it out.
        #[arg(long, value_enum, default_value_t = OutputHash::Blake3)]
        output_hash: OutputHash,

        /// gzip-compress `.h5ad` output. Pass `--compress` for the default
        /// level 6, or `--compress N` for level N (0..=9). Off by default.
        /// Ignored for `.h5seurat` output.
        #[arg(long, num_args = 0..=1, default_missing_value = "6")]
        compress: Option<u8>,

        /// Write X gene-major (each gene's cells contiguous, stored as
        /// anndata `csc_matrix`; `.h5ad` output only), for per-gene
        /// consumers. X is read once to count, then once per block of genes
        /// that fits --transpose-mem.
        #[arg(long)]
        gene_major: bool,

        /// Memory budget in GB for the transposed nonzeros --gene-major holds
        /// at a time. Less memory means more reads of X.
        #[arg(long, value_name = "GB", default_value = "4", requires = "gene_major")]
        transpose_mem: f64,

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

        /// Keep only the cells whose obs column COL equals VALUE. Repeat it:
        /// values given for the same column OR together, different columns
        /// AND (`--where cell_type=B --where cell_type=T --where donor=d1`
        /// keeps B and T cells of donor d1). Values compare as text
        /// (booleans as True/False); NA never matches. The text up to the
        /// first `=` is the column, so a value may contain `=`; quote it for
        /// the shell when it has spaces (`--where "cell_type=CD4 T"`).
        ///
        /// X, layers, raw.X, obs and obsm keep the matching rows, and obsp
        /// those rows and columns; var, varm and uns are unchanged. h5ad,
        /// Zarr, dgCMatrix H5Seurat and 10x HDF5 read only the matching rows
        /// of X; other inputs read all of it. No match writes a valid file
        /// with 0 cells.
        #[arg(long = "where", value_name = "COL=VALUE", value_parser = parse_where_arg)]
        where_: Vec<(String, String)>,
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

        /// Download a URL input without asking, however large. Without it,
        /// inspect asks before downloading more than 1 GB, and stops when it
        /// can't ask (stdin is not a terminal).
        #[arg(long, short)]
        yes: bool,
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
    #[command(verbatim_doc_comment)]
    Validate {
        /// Input file or directory, in any format scx reads
        input: String,

        /// Path to YAML schema file
        #[arg(long)]
        schema: String,

        /// Emit results as JSON instead of the default human-readable report
        #[arg(long)]
        json: bool,
    },

    /// Compare two datasets slot by slot
    ///
    /// Reads A and B (any formats scx reads) and reports, per slot, whether
    /// they are the same, the same within tolerance, different, or present
    /// on one side only. Matrices are streamed. What conversion legitimately
    /// changes (storage dtype, categorical level order) is a NOTE, not a
    /// difference; empty and absent slots compare equal.
    ///
    /// Exits 0 when everything compared is the same, 1 when something
    /// differs or couldn't be compared, 2 when the comparison couldn't run.
    /// See docs/diff.md.
    #[command(verbatim_doc_comment)]
    Diff {
        /// First dataset (A)
        a: String,

        /// Second dataset (B)
        b: String,

        /// Relative tolerance for floats: |a - b| <= atol + rtol * |b|
        #[arg(long, default_value_t = 1e-6)]
        rtol: f64,

        /// Absolute tolerance for floats
        #[arg(long, default_value_t = 0.0)]
        atol: f64,

        /// Require floats to match exactly (rtol = atol = 0)
        #[arg(long)]
        exact: bool,

        /// Compare only these slots (same syntax as convert --only)
        #[arg(long)]
        only: Option<String>,

        /// Skip these slots (same syntax as convert --exclude)
        #[arg(long)]
        exclude: Option<String>,

        /// Pair matrix slots whose names differ, as A=B (repeatable), e.g.
        /// --map X=layers/data --map raw=X. Names: X, raw, layers/<n>, obsp/<n>
        #[arg(long = "map", value_name = "A=B")]
        map: Vec<String>,

        /// Match cells and genes by index name or by position
        #[arg(long, default_value = "name", value_parser = ["name", "position"])]
        by: String,

        /// Mismatching entries listed per slot
        #[arg(long, default_value_t = 5)]
        max_examples: usize,

        /// Rows per streamed chunk
        #[arg(long, default_value_t = 5000)]
        chunk_size: usize,

        /// Emit the report as JSON
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
    #[command(verbatim_doc_comment)]
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
    #[command(verbatim_doc_comment)]
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
    #[command(verbatim_doc_comment)]
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
    /// Provenance: uns["scx_provenance"] records the base and each patched
    /// slot's source and SHA-256. When each slot was added goes in the sidecar
    /// <output>.prov.json (extended by --into), so the same merge gives the
    /// same bytes.
    ///
    /// Files written by h5py/anndata carry HDF5 group timestamps that merge
    /// would update, so merge refuses them as a base or --into target. Rewrite
    /// the file first with `scx convert in.h5ad out.h5ad`, or pass
    /// --allow-nonreproducible.
    ///
    /// Examples:
    ///   scx merge --base raw.h5ad --patch norm.h5ad:layers/normalized --output merged.h5ad
    ///   scx merge --config merge.yaml
    #[command(verbatim_doc_comment)]
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

        /// Merge into an h5py/anndata-written file even though its HDF5 group
        /// timestamps make the output differ on every run
        #[arg(long)]
        allow_nonreproducible: bool,
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

/// Check `input` against the YAML schema at `schema`.
async fn validate_file(
    input: &str,
    schema: &str,
) -> anyhow::Result<scx_core::validate::ValidationReport> {
    let schema_src = std::fs::read_to_string(schema)
        .map_err(|e| anyhow::anyhow!("cannot read schema '{schema}': {e}"))?;
    let schema_parsed: ValidationSchema = serde_yaml::from_str(&schema_src)
        .map_err(|e| anyhow::anyhow!("invalid schema '{schema}': {e}"))?;
    let opts = scx_core::OpenOptions {
        source_hash: Default::default(),
        metadata_only: true,
        ..scx_core::OpenOptions::new(1000)
    };
    let mut reader = scx_core::open(input, &opts)
        .await
        .map_err(|e| anyhow::anyhow!("cannot open '{input}': {e}"))?;
    let report = run_validation(&mut *reader, &schema_parsed, input, schema).await?;
    if report.checks.is_empty() {
        anyhow::bail!("schema '{schema}' has no checks");
    }
    Ok(report)
}

/// Open A and B and compare them.
async fn run_diff(
    a: &str,
    b: &str,
    map: &[String],
    mut opts: scx_core::diff::DiffOptions,
) -> anyhow::Result<scx_core::diff::DiffReport> {
    for m in map {
        let (x, y) = m
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--map '{m}': expected A=B, e.g. X=layers/data"))?;
        opts.map.push((x.trim().to_string(), y.trim().to_string()));
    }
    let open_opts = scx_core::OpenOptions::new(opts.chunk_size);
    let mut ra = scx_core::open(a, &open_opts)
        .await
        .map_err(|e| anyhow::anyhow!("cannot open '{a}': {e}"))?;
    let mut rb = scx_core::open(b, &open_opts)
        .await
        .map_err(|e| anyhow::anyhow!("cannot open '{b}': {e}"))?;
    Ok(scx_core::diff::diff(&mut *ra, &mut *rb, &opts).await?)
}

fn print_diff_human(report: &scx_core::diff::DiffReport) {
    use scx_core::diff::Status;
    for s in &report.slots {
        let label = format!("{:<7}", s.status.label());
        let label = match s.status {
            Status::Same => label.if_supports_color(Stdout, |t| t.green()).to_string(),
            _ => label.if_supports_color(Stdout, |t| t.red()).to_string(),
        };
        println!("{label} {:<20} {}", s.slot, s.summary);
        for d in &s.details {
            println!("{:8} {d}", "");
        }
        for n in &s.notes {
            let note = format!("{:<7}", "NOTE");
            println!(
                "{} {:<20} {n}",
                note.if_supports_color(Stdout, |t| t.yellow()),
                s.slot
            );
        }
    }
    println!();
    let n = report.n_differing();
    if n == 0 {
        println!("All {} slots the same.", report.slots.len());
    } else {
        println!(
            "{n} slot{} differ{}.",
            if n == 1 { "" } else { "s" },
            if n == 1 { "s" } else { "" }
        );
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
            // Exit 1: the file failed a check. Exit 2: validation couldn't
            // run (unreadable file, bad or empty schema), so a pipeline can
            // tell the two apart.
            let report = match validate_file(&input, &schema).await {
                Ok(report) => report,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(2);
                }
            };
            if json {
                print_report_json(&report);
            } else {
                print_report_human(&report);
            }
            if !report.passed() {
                std::process::exit(1);
            }
        }

        Cli::Diff {
            a,
            b,
            rtol,
            atol,
            exact,
            only,
            exclude,
            map,
            by,
            max_examples,
            chunk_size,
            json,
        } => {
            let filter = match (only.as_deref(), exclude.as_deref()) {
                (Some(o), _) => SlotFilter::from_only(o),
                (_, Some(e)) => SlotFilter::from_exclude(e),
                _ => SlotFilter::all(),
            };
            let opts = scx_core::diff::DiffOptions {
                rtol: if exact { 0.0 } else { rtol },
                atol: if exact { 0.0 } else { atol },
                by: if by == "position" {
                    scx_core::diff::MatchBy::Position
                } else {
                    scx_core::diff::MatchBy::Name
                },
                max_examples,
                chunk_size,
                filter,
                map: Vec::new(),
            };
            // 2: the comparison couldn't run; 1: the datasets differ.
            let report = match run_diff(&a, &b, &map, opts).await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(2);
                }
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_diff_human(&report);
            }
            if !report.equal() {
                std::process::exit(1);
            }
        }

        Cli::Inspect {
            input,
            assay,
            layer,
            yes,
        } => {
            // A URL other than Parquet is downloaded first (kept until inspect
            // returns), as `convert` does through the reader factory.
            #[cfg(feature = "net")]
            let spool = if scx_core::net::is_network_url(&input)
                && !scx_core::net::reads_by_range(&input)
            {
                confirm_download(&input, yes).await?;
                Some(scx_core::spool::spool(&input, Default::default()).await?)
            } else {
                None
            };
            #[cfg(feature = "net")]
            let input_path = spool.as_ref().map_or(Path::new(&input), |s| s.path());
            #[cfg(not(feature = "net"))]
            let (input_path, _) = (Path::new(&input), yes);
            if !input_path.exists() && input.contains("://") {
                anyhow::bail!(
                    "'{input}': inspect reads URLs only in a build with the `net` feature, \
                     and not Parquet URLs"
                );
            }
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
                None => anyhow::bail!("could not detect the format of '{input}'"),
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
                Some(Format::Rds) => {
                    match scx_core::rds::RdsSeuratReader::open(
                        input_path,
                        chunk,
                        None,
                        layer.as_deref(),
                    ) {
                        Ok(mut r) => inspect(&mut r, &input, "RDS (Seurat)").await?,
                        // Not something scx converts: show what the file holds.
                        Err(e) => {
                            println!("File   : {input}\nFormat : RDS\n{e}\n");
                            print!("{}", scx_core::rds::tree(input_path, 4)?);
                        }
                    }
                }
                #[cfg(feature = "bixverse")]
                Some(Format::Bixverse) => {
                    let mut r = scx_core::bixverse::BixverseReader::open(
                        input_path,
                        chunk,
                        scx_core::bixverse::BixverseLayer::Raw,
                    )?;
                    inspect(&mut r, &input, "bixverse store").await?;
                }
                #[cfg(not(feature = "bixverse"))]
                Some(Format::Bixverse) => {
                    anyhow::bail!(
                        "bixverse input requires building scx with the `bixverse` feature"
                    )
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
                #[cfg(feature = "zarr")]
                Some(Format::ZarrAd) => {
                    let mut r = scx_core::zarr_ad::ZarrAdReader::open_layer(
                        input_path,
                        chunk,
                        layer.as_deref(),
                    )?;
                    if r.x_source() != "X" {
                        println!("matrix source: {} (no X)", r.x_source());
                    }
                    inspect(&mut r, &input, "Zarr (AnnData)").await?;
                }
                #[cfg(not(feature = "zarr"))]
                Some(Format::ZarrAd) => {
                    anyhow::bail!("Zarr input requires building scx with the `zarr` feature")
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
            source_hash,
            output_hash,
            compress,
            gene_major,
            transpose_mem,
            only,
            exclude,
            where_,
        } => {
            let filters = scx_core::subset::group_filters(where_);
            if transpose_mem.is_nan() || transpose_mem <= 0.0 {
                anyhow::bail!("--transpose-mem must be a positive GB budget");
            }
            let csc_mem = gene_major.then_some((transpose_mem * 1e9) as usize);
            let filter = match (only.as_deref(), exclude.as_deref()) {
                (Some(o), _) => SlotFilter::from_only(o),
                (_, Some(e)) => SlotFilter::from_exclude(e),
                _ => SlotFilter::all(),
            };

            let out_dtype = dtype.parse::<DataType>()?;

            let input_path = Path::new(&input);
            let output_path = Path::new(&output);

            // The source hash is only needed for the provenance records, so it
            // is computed on its own thread while the conversion runs.
            let source_sha256: SourceHash = if let Some(arg) = source_sha256_arg {
                let hex = arg.strip_prefix("sha256:").unwrap_or(&arg).to_string();
                if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
                    anyhow::bail!(
                        "--source-sha256 must be 64 hex characters, got {} chars",
                        hex.len()
                    );
                }
                std::thread::spawn(move || Ok(Some(format!("sha256:{}", hex.to_ascii_lowercase()))))
            } else if input != "-" && input_path.is_file() && source_hash != SourceHashAlgo::None {
                let (path, name) = (input_path.to_path_buf(), input.clone());
                // Sequential, on this thread alone: the parallel `blake3_file`
                // would take the pool from the conversion running beside it.
                std::thread::spawn(move || {
                    provenance::hash_file_with(source_hash.algo(), &path)
                        .map(Some)
                        .map_err(|e| anyhow::anyhow!("hashing source '{name}': {e}"))
                })
            } else {
                std::thread::spawn(|| Ok(None))
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
                source_hash: source_hash.algo(),
            };
            let mut reader = if input == "-" {
                // Formats need random access, so stdin goes to a temporary
                // file (hashed on the way) and is detected by content.
                let spool = scx_core::spool::Spool::from_reader(
                    "stdin",
                    std::io::stdin().lock(),
                    source_hash.algo(),
                )
                .map_err(|e| anyhow::anyhow!("reading stdin: {e}"))?;
                let path = spool.path().to_string_lossy().into_owned();
                let inner = scx_core::open(&path, &opts).await.map_err(|e| {
                    anyhow::anyhow!(
                        "cannot open stdin: {e} (MatrixMarket is detected by its .mtx name, \
                         so it can't be piped)"
                    )
                })?;
                Box::new(scx_core::spool::Spooled::new(inner, spool))
            } else {
                scx_core::open(&input, &opts)
                    .await
                    .map_err(|e| anyhow::anyhow!("cannot open '{input}': {e}"))?
            };
            if !filters.is_empty() {
                let sub = scx_core::subset::SubsetReader::new(reader, &filters, chunk_size)
                    .await
                    .map_err(|e| anyhow::anyhow!("--where: {e}"))?;
                tracing::info!(
                    n_obs = sub.shape().0,
                    "--where keeps {} cells",
                    sub.shape().0
                );
                reader = Box::new(sub);
            }
            // Convert into a sibling path and rename on success, so a failed
            // run leaves nothing behind and doesn't clobber an existing output.
            // The name keeps the extension, which picks the writer.
            if output_path.is_dir() {
                anyhow::bail!("output '{output}' already exists");
            }
            let partial = output_path.with_file_name(format!(
                ".partial-{}",
                output_path
                    .file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default()
            ));
            remove_path(&partial)?; // left over from a killed run
            let converted = convert_with_reader(
                &mut *reader,
                &partial,
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
                source_sha256,
                source_hash != SourceHashAlgo::None,
                compress,
                csc_mem,
                &filter,
                &filters,
            )
            .await;
            let (n_obs, n_vars, source_sha256) = match converted {
                Ok(converted) => {
                    std::fs::rename(&partial, output_path)?;
                    converted
                }
                Err(e) => {
                    remove_path(&partial)?;
                    return Err(e);
                }
            };

            // Directory outputs (BPCells, Zarr) are not hashed.
            let output_hash = match output_hash {
                _ if !output_path.is_file() => None,
                OutputHash::None => None,
                OutputHash::Sha256 => {
                    Some(provenance::sha256_file(output_path).map(|h| format!("sha256:{h}")))
                }
                OutputHash::Blake3 => {
                    Some(blake3_file(output_path).map(|h| format!("blake3:{}", h.to_hex())))
                }
            }
            .transpose()
            .map_err(|e| anyhow::anyhow!("hashing output '{output}': {e}"))?;

            let record = ProvenanceRecord {
                scx_version: env!("CARGO_PKG_VERSION").to_string(),
                converted_at: provenance::utc_now_rfc3339(),
                source: SourceInfo {
                    path: input,
                    url: source_url,
                    hash: source_sha256,
                },
                output: OutputInfo {
                    path: output.clone(),
                    hash: output_hash,
                    n_obs,
                    n_vars,
                },
            };
            // `where` goes in only when set, so unfiltered sidecars keep
            // their shape.
            let mut record = serde_json::to_value(&record)?;
            if !filters.is_empty() {
                record["where"] = where_json(&filters);
            }
            std::fs::write(
                provenance::sidecar_path(output_path),
                serde_json::to_string_pretty(&record)?,
            )
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
                source_hash: Default::default(),
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
                dtype: dtype.parse::<DataType>()?,
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
            allow_nonreproducible,
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
                allow_nonreproducible,
            })
            .await?;
        }
    }

    Ok(())
}

/// Remove a file or directory if it exists.
fn remove_path(p: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => std::fs::remove_dir_all(p),
        Ok(_) => std::fs::remove_file(p),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// Hand freed heap memory back to the OS. glibc keeps what a large frame
/// (obs strings, embeddings) freed, so the next phase's allocations add to the
/// peak instead of reusing it. No-op off glibc.
fn release_freed_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        // SAFETY: malloc_trim only returns free heap pages to the kernel.
        unsafe { malloc_trim(0) };
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
    source_sha256: SourceHash,
    // Record the hash a download or stdin spool computed on the way in.
    reader_hash: bool,
    compress: Option<u8>,
    csc_mem: Option<usize>,
    filter: &SlotFilter,
    filters: &[(String, Vec<String>)],
) -> anyhow::Result<(usize, usize, Option<String>)> {
    let t0 = std::time::Instant::now();
    let (n_obs, n_vars) = reader.shape();

    let is_h5seurat = output.extension().and_then(|e| e.to_str()) == Some("h5seurat");
    if csc_mem.is_some()
        && matches!(
            output.extension().and_then(|e| e.to_str()),
            Some("h5seurat" | "zarr" | "bpcells")
        )
    {
        anyhow::bail!("--gene-major only applies to .h5ad output");
    }
    scx_core::stream::require_cell_major(reader)?;

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
    let mut uns = if write_uns {
        reader.uns().await?
    } else {
        UnsTable::default()
    };

    // An excluded slot is not read at all, so a matrix scx cannot read there
    // never blocks the conversion.
    let layer_metas = if filter.includes_any("layers") {
        reader.layer_metas().await?
    } else {
        Vec::new()
    };
    let obsp_metas = if filter.includes_any("obsp") {
        reader.obsp_metas().await?
    } else {
        Vec::new()
    };

    // Resolve the effective X slot.
    // H5Seurat auto: if the source has a "counts" layer or a matching raw.X, X
    // is assumed normalised → goes to "data" (scx_core::h5seurat::auto_counts_source).
    // H5AD auto: "/X". Anything explicit is passed to the writer as-is.
    let counts_source = if is_h5seurat && x_slot == "auto" {
        scx_core::h5seurat::auto_counts_source(reader, &layer_metas).await?
    } else {
        None
    };
    let effective_x_slot: &str = match (is_h5seurat, x_slot) {
        (true, "auto") => match &counts_source {
            Some(src) => {
                let from = match src {
                    scx_core::h5seurat::CountsSource::Layer => "'counts' layer",
                    scx_core::h5seurat::CountsSource::Raw(_) => "raw.X",
                };
                tracing::info!(
                    "writing X to 'data' slot, {from} to 'counts' slot \
                     (use --x-slot counts to override)"
                );
                "data"
            }
            None => out_layer,
        },
        (false, "auto") => "X",
        (_, other) => other,
    };
    // Source layer that would collide with the matrix's destination.
    let x_layer: Option<&str> = if is_h5seurat {
        Some(effective_x_slot)
    } else {
        effective_x_slot.strip_prefix("layers/")
    };

    tracing::info!(
        output = %output.display(),
        n_obs, n_vars,
        dtype = %out_dtype,
        format = match output.extension().and_then(|e| e.to_str()) {
            Some("zarr") => "zarr",
            Some("bpcells") => "bpcells",
            _ if is_h5seurat => "h5seurat",
            _ => "h5ad",
        },
        bpcells = is_h5seurat && !use_dgcmatrix,
        dgcmatrix = use_dgcmatrix && is_h5seurat,
        x_slot = effective_x_slot,
        "starting conversion"
    );

    tracing::info!(
        obs_cols = obs.columns.len(),
        var_cols = var.columns.len(),
        layers = layer_metas.len(),
        obsp = obsp_metas.len(),
        "metadata loaded in {:.2?}",
        t0.elapsed()
    );

    let out_ext = output.extension().and_then(|e| e.to_str());
    let mut writer: Box<dyn DatasetWriter> = if out_ext == Some("bpcells") {
        // BPCells matrix directory: X + dimnames only.
        Box::new(BpcellsDirWriter::create(output, n_obs, n_vars)?)
    } else if out_ext == Some("zarr") {
        #[cfg(feature = "zarr")]
        {
            Box::new(scx_core::zarr_ad::ZarrAdWriter::create_at(
                output,
                n_obs,
                n_vars,
                out_dtype,
                effective_x_slot,
            )?)
        }
        #[cfg(not(feature = "zarr"))]
        {
            anyhow::bail!("Zarr output requires building scx with the `zarr` feature")
        }
    } else if is_h5seurat {
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
        Box::new(H5AdWriter::create_at(
            output,
            n_obs,
            n_vars,
            out_dtype,
            compress,
            effective_x_slot,
        )?)
    };

    writer.write_obs(&obs).await?;
    writer.write_var(&var).await?;
    // Each frame is read, written and freed in turn, so that at most one of
    // them is in memory, and none while the matrices are copied or streamed.
    drop((obs, var));
    release_freed_memory();
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
    tracing::info!(
        embeddings = obsm.map.len(),
        varm = varm.map.len(),
        "read obsm, varm"
    );
    if !obsm.map.is_empty() {
        writer.write_obsm(&obsm).await?;
    }
    if !varm.map.is_empty() {
        writer.write_varm(&varm).await?;
    }
    drop((obsm, varm));
    release_freed_memory();

    // Stream each layer — skip any whose name would collide with the X slot,
    // or that the slot filter drops.
    for meta in &layer_metas {
        if !filter.includes(&format!("layers:{}", meta.name)) {
            tracing::info!(name = %meta.name, "skipping layer (slot filter)");
            continue;
        }
        if Some(meta.name.as_str()) == x_layer {
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

    // A local h5ad's sparse X and raw.X can go into a dgCMatrix h5seurat as
    // stored, compressed chunks and all, when that is the matrix streaming
    // would write (H5SeuratWriter::copy_csr_from_h5ad checks); else they stream.
    let src_h5 = Path::new(source_path);
    // A --where subset is never the matrix as stored.
    let can_copy = filters.is_empty()
        && source_url.is_none()
        && src_h5.extension().and_then(|e| e.to_str()) == Some("h5ad")
        && src_h5.is_file();
    let copy = |writer: &mut Box<dyn DatasetWriter>, group: &str, slot: &str| {
        let t = std::time::Instant::now();
        let copied = can_copy && writer.copy_csr_from_h5ad(src_h5, group, slot)?;
        if copied {
            tracing::info!(
                "copied {group} into '{slot}' as stored in {:.2?}",
                t.elapsed()
            );
        }
        anyhow::Ok(copied)
    };

    // AnnData's raw.X is the counts of a normalised X: Seurat's `counts` slot.
    if let Some(scx_core::h5seurat::CountsSource::Raw(meta)) = &counts_source {
        if copy(&mut writer, "raw/X", "counts")? {
            // done
        } else {
            tracing::info!(shape = ?meta.shape, "streaming raw.X into 'counts'");
            writer.begin_sparse("layers", "counts", meta).await?;
            let mut stream = reader.raw_stream(meta, chunk_size);
            while let Some(chunk) = stream.next().await {
                writer.write_sparse_chunk(&chunk?).await?;
            }
            writer.end_sparse().await?;
        }
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
    if let Some(mem) = csc_mem {
        let blocks = scx_core::transpose::write_x_csc(reader, &mut *writer, mem).await?;
        tracing::info!(
            blocks,
            reads = blocks + 1,
            "matrix transposed to CSC in {:.2?}",
            t_x.elapsed()
        );
    } else if !copy(&mut writer, "X", effective_x_slot)? {
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
    }

    // uns goes last: its provenance record needs the source hash, which has
    // been computing alongside everything above.
    let source_sha256 = source_sha256
        .join()
        .map_err(|_| anyhow::anyhow!("source hashing thread panicked"))??
        // A remote or piped source was hashed by the reader on the way in.
        .or_else(|| reader.source_hash().filter(|_| reader_hash));
    if write_uns {
        let mut prov = scx_core::provenance::det_record(
            source_path,
            source_url,
            source_sha256.as_deref(),
            n_obs,
            n_vars,
        );
        if !filters.is_empty() {
            prov["where"] = where_json(filters);
        }
        match uns.raw.as_object_mut() {
            Some(obj) => {
                obj.insert("scx_provenance".to_string(), prov);
            }
            None => {
                uns.raw = serde_json::json!({ "scx_provenance": prov });
            }
        }
        writer.write_uns(&uns).await?;
    }

    writer.finalize().await?;
    drop(writer);
    release_freed_memory();

    tracing::info!(
        total = ?t0.elapsed(),
        output = %output.display(),
        "conversion complete"
    );

    Ok((n_obs, n_vars, source_sha256))
}

/// The source file's SHA-256, being computed on its own thread.
/// One `--where COL=VALUE`, split at the first `=`.
fn parse_where_arg(s: &str) -> Result<(String, String), String> {
    match s.split_once('=') {
        Some((col, value)) if !col.is_empty() => Ok((col.to_string(), value.to_string())),
        _ => Err(format!("expected COL=VALUE, got '{s}'")),
    }
}

/// The filter as provenance records it: `{"col": ["v1", "v2"], ...}`.
fn where_json(filters: &[(String, Vec<String>)]) -> serde_json::Value {
    filters
        .iter()
        .map(|(c, v)| (c.clone(), serde_json::json!(v)))
        .collect::<serde_json::Map<_, _>>()
        .into()
}

type SourceHash = std::thread::JoinHandle<anyhow::Result<Option<String>>>;

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

/// nnz and sparsity, plus quartiles of nnz per `unit` (each `indptr` entry).
fn indptr_row_stats(indptr: &[u64], n_rows: usize, n_cols: usize, unit: &str) -> String {
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
        "nnz={}  sparse={:.1}%  nnz/{unit} Q1={}  med={}  Q3={}  max={}",
        nnz,
        sparsity,
        q(0.25),
        q(0.5),
        q(0.75),
        per_row[n - 1]
    )
}

fn numeric_stats(data: &ColumnData) -> String {
    // NaN is dropped rather than sorted around. `partial_cmp().unwrap_or(Equal)`
    // is not a total order once NaN is present — NaN compares Equal to
    // everything, which breaks transitivity — and Rust's sort detects that and
    // panics. Real files hit this: an obs column with missing values (anndata
    // writes them as NaN) crashed `scx inspect` outright. Quartiles over NaN
    // would be meaningless anyway, so the count of missing values is reported
    // instead.
    let raw: Vec<f64> = match data {
        ColumnData::Float(v) => v.clone(),
        ColumnData::Int(v) => v.iter().map(|&x| x as f64).collect(),
        ColumnData::Int64(v) => v.iter().map(|&x| x as f64).collect(),
        _ => return String::new(),
    };
    let total = raw.len();
    let mut vals: Vec<f64> = raw.into_iter().filter(|x| x.is_finite()).collect();
    let missing = total - vals.len();
    if vals.is_empty() {
        return if total > 0 {
            format!("all {total} values missing")
        } else {
            String::new()
        };
    }
    vals.sort_by(f64::total_cmp);
    let n = vals.len();
    let na_note = if missing > 0 {
        format!("  ({missing} missing)")
    } else {
        String::new()
    };

    // Binary {0, 1} column — show counts instead of quartiles
    if vals[0] >= 0.0 && vals[n - 1] <= 1.0 {
        let ones = vals.iter().filter(|&&v| v == 1.0).count();
        let zeros = n - ones;
        if ones + zeros == n {
            return format!(
                "bool-like  0: {} ({:.1}%)  1: {} ({:.1}%){}",
                zeros,
                100.0 * zeros as f64 / n as f64,
                ones,
                100.0 * ones as f64 / n as f64,
                na_note,
            );
        }
    }

    let q = |p: f64| vals[(p * (n - 1) as f64).round() as usize];
    format!(
        "min={}  Q1={}  med={}  Q3={}  max={}{}",
        fmt_stat(vals[0]),
        fmt_stat(q(0.25)),
        fmt_stat(q(0.5)),
        fmt_stat(q(0.75)),
        fmt_stat(vals[n - 1]),
        na_note,
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

/// Above this size inspect asks before downloading a URL in full.
// TODO: fixed threshold; make it a flag if 1 GB is wrong for someone.
#[cfg(feature = "net")]
const ASK_ABOVE_BYTES: u64 = 1_000_000_000;

/// Ask before inspect downloads a large URL in full, as it must for anything
/// but Parquet. The size comes from a HEAD request; if the server won't say,
/// the download just goes ahead.
#[cfg(feature = "net")]
async fn confirm_download(url: &str, yes: bool) -> anyhow::Result<()> {
    use std::io::IsTerminal;
    let Some(size) = scx_core::net::size(url).await else {
        return Ok(());
    };
    if yes || size <= ASK_ABOVE_BYTES {
        return Ok(());
    }
    let gb = size as f64 / 1e9;
    if !std::io::stdin().is_terminal() {
        anyhow::bail!(
            "'{url}' is {gb:.1} GB, and inspect would download all of it first; \
             pass --yes to go ahead"
        );
    }
    eprint!("'{url}' is {gb:.1} GB, and inspect downloads all of it first. Continue? [y/N] ");
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    if !matches!(answer.trim(), "y" | "Y" | "yes") {
        anyhow::bail!("cancelled");
    }
    Ok(())
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
    let x_stats = if reader.x_col_ptr().is_empty() {
        indptr_row_stats(reader.x_indptr(), n_obs, n_vars, "cell")
    } else {
        println!(
            "{} {}",
            bold!("X layout:"),
            cyan!("gene-major (CSC): stream by gene, open_stream(axis=\"var\")")
        );
        indptr_row_stats(reader.x_col_ptr(), n_vars, n_obs, "gene")
    };
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
            indptr_row_stats(&m.indptr, m.shape.0, m.shape.1, "cell")
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
            indptr_row_stats(&m.indptr, m.shape.0, m.shape.1, "cell")
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
                // `sha256` (bare hex) is the pre-0.6 key.
                if let Some(h) = src
                    .get("hash")
                    .or_else(|| src.get("sha256"))
                    .and_then(|v| v.as_str())
                {
                    println!("  {:<16} {}", "source.hash", dim!(h));
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
                    // added_at lives in the sidecar; older merges kept it in uns.
                    let sidecar: serde_json::Value =
                        std::fs::read(provenance::sidecar_path(Path::new(path)))
                            .ok()
                            .and_then(|b| serde_json::from_slice(&b).ok())
                            .unwrap_or_default();
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
                            let at = entry
                                .get("added_at")
                                .or_else(|| sidecar["slots"][key.as_str()].get("added_at"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
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

#[cfg(test)]
mod blake3_file_tests {
    /// The recorded output hash must not change: `blake3_file` gives the same
    /// digest as the whole-file mmap it replaced, for an empty file, a file
    /// inside one buffer, and one spanning several buffers with a ragged tail.
    #[test]
    fn matches_whole_file_mmap_hash() {
        let dir = tempfile::tempdir().unwrap();
        for len in [0usize, 1, (1 << 20) + 1, (16 << 20) * 2 + (3 << 20) + 7] {
            let bytes: Vec<u8> = (0..len).map(|i| (i * 31 + i / 4099) as u8).collect();
            let p = dir.path().join(format!("f{len}"));
            std::fs::write(&p, &bytes).unwrap();
            let old = blake3::Hasher::new()
                .update_mmap_rayon(&p)
                .unwrap()
                .finalize();
            assert_eq!(super::blake3_file(&p).unwrap(), old, "len {len}");
            assert_eq!(old, blake3::hash(&bytes), "len {len}");
        }
    }
}

#[cfg(test)]
mod stats_tests {
    use super::numeric_stats;
    use scx_core::ir::ColumnData;

    /// Regression: `scx inspect` panicked outright on any numeric obs column
    /// holding a missing value. The quartile sort used
    /// `partial_cmp().unwrap_or(Equal)`, which is not a total order once NaN is
    /// present — NaN compares Equal to everything, breaking transitivity — and
    /// Rust's sort detects that and aborts with "user-provided comparison
    /// function does not correctly implement a total order". anndata writes
    /// missing numbers as NaN, so real atlas files tripped it.
    #[test]
    fn numeric_stats_survives_nan() {
        let with_nan = ColumnData::Float(vec![3.0, f64::NAN, 1.0, 2.0, f64::NAN]);
        let out = numeric_stats(&with_nan);
        assert!(
            out.contains("min=1"),
            "stats computed over the finite values: {out}"
        );
        assert!(out.contains("2 missing"), "missing count reported: {out}");

        // Enough values to make a non-total order actually trip the sort.
        let mut many: Vec<f64> = (0..64).map(|i| i as f64).collect();
        for i in (0..64).step_by(3) {
            many[i] = f64::NAN;
        }
        let out = numeric_stats(&ColumnData::Float(many));
        assert!(out.contains("missing"), "{out}");

        // All-missing is reported, not silently blank.
        let all_nan = ColumnData::Float(vec![f64::NAN; 4]);
        assert_eq!(numeric_stats(&all_nan), "all 4 values missing");

        // No NaN: unchanged output, no note.
        let clean = ColumnData::Float(vec![1.0, 2.0, 3.0, 40.0]);
        assert!(!numeric_stats(&clean).contains("missing"));
    }
}

#[cfg(test)]
mod where_tests {
    use super::*;

    #[test]
    fn where_args_parse_and_group() {
        let cli = Cli::try_parse_from([
            "scx",
            "convert",
            "in.h5ad",
            "out.h5ad",
            "--where",
            "cell_type=B",
            "--where",
            "donor=d1",
            "--where",
            "cell_type=CD4 T",
            "--where",
            "note=a=b",
            "--where",
            "empty=",
        ])
        .unwrap();
        let Cli::Convert { where_, .. } = cli else {
            panic!("not convert")
        };
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(
            scx_core::subset::group_filters(where_),
            vec![
                ("cell_type".to_string(), s(&["B", "CD4 T"])),
                ("donor".to_string(), s(&["d1"])),
                ("note".to_string(), s(&["a=b"])),
                ("empty".to_string(), s(&[""])),
            ]
        );
        for bad in ["cell_type", "=B"] {
            assert!(
                Cli::try_parse_from(["scx", "convert", "a.h5ad", "b.h5ad", "--where", bad])
                    .is_err(),
                "{bad}"
            );
        }
        let Cli::Convert { where_, .. } =
            Cli::try_parse_from(["scx", "convert", "a.h5ad", "b.h5ad"]).unwrap()
        else {
            panic!("not convert")
        };
        assert!(where_.is_empty());
        assert_eq!(
            where_json(&scx_core::subset::group_filters(vec![
                ("a".into(), "1".into()),
                ("a".into(), "1".into())
            ])),
            serde_json::json!({ "a": ["1"] })
        );
    }
}
