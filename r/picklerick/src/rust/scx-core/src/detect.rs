//! Format detection by file content, not extension.
//!
//! Opens the file as HDF5 and checks for well-known fingerprints:
//!
//! | Format   | Fingerprint |
//! |----------|-------------|
//! | H5AD     | Root attr `encoding-type = "anndata"` |
//! | H5Seurat | Root dataset `cell.names` + root attr `active.assay` |
//! | ScxH5    | Root dataset `X/shape` (SCX internal golden fixture schema) |
//! | TenxH5   | `/matrix` group with `/matrix/barcodes` + `/matrix/features` |
//! | PlainH5  | Any valid HDF5 file not matching the above |

use std::path::Path;

use hdf5::types::VarLenUnicode;
use hdf5::File;

/// The detected on-disk format of an HDF5 file or NPY directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    H5Ad,
    H5Seurat,
    /// SCX internal HDF5 schema (golden test fixtures).
    ScxH5,
    /// NPY snapshot directory (contains `meta.json`).
    NpyDir,
    /// BPCells directory-format matrix (contains `version` + `storage_order`).
    BPCells,
    /// 10x Genomics HDF5 feature-barcode matrix (Cell Ranger output).
    TenxH5,
    /// Valid HDF5 file with no recognized single-cell format fingerprint.
    PlainH5,
}

impl Format {
    pub fn display_name(self) -> &'static str {
        match self {
            Format::H5Ad => "H5AD",
            Format::H5Seurat => "H5Seurat",
            Format::ScxH5 => "ScxH5",
            Format::NpyDir => "NpyDir",
            Format::BPCells => "BPCells",
            Format::TenxH5 => "10x HDF5",
            Format::PlainH5 => "HDF5 (unrecognized)",
        }
    }
}

/// Sniff the format of a directory.
///
/// - `Format::NpyDir`  — directory contains `meta.json`
/// - `Format::BPCells` — directory contains `version` + `storage_order`
pub fn sniff_dir(path: &Path) -> Option<Format> {
    if !path.is_dir() {
        return None;
    }
    if path.join("meta.json").exists() {
        return Some(Format::NpyDir);
    }
    if path.join("version").exists() && path.join("storage_order").exists() {
        return Some(Format::BPCells);
    }
    None
}

/// Sniff the format of `path` by inspecting HDF5 structure.
///
/// Returns `None` if the file cannot be opened as HDF5 or matches no
/// known fingerprint. Callers should fall back to extension-based routing
/// in that case.
pub fn sniff(path: &Path) -> Option<Format> {
    let file = File::open(path).ok()?;

    // --- H5AD ---
    // Root group carries encoding-type = "anndata" written by AnnData ≥ 0.8.
    if let Ok(root) = file.group("/") {
        if let Ok(attr) = root.attr("encoding-type") {
            if let Ok(enc) = attr.read_scalar::<VarLenUnicode>() {
                if enc.as_str() == "anndata" {
                    return Some(Format::H5Ad);
                }
            }
        }
    }

    // --- H5AD (legacy, pre-0.8) ---
    // Older AnnData did not write the root `encoding-type = "anndata"` attr,
    // but still lays out the canonical /X, /obs, /var structure and tags the
    // /obs and /var groups with `encoding-type = "dataframe"`. Recognise that
    // structural fingerprint so these files (e.g. the GEARS perturb datasets)
    // are read as H5AD rather than misclassified as generic PlainH5. This is
    // strictly more specific than the 10x check below (which keys on /matrix),
    // so there is no ambiguity.
    let group_encoding = |name: &str| -> Option<String> {
        let attr = file.group(name).ok()?.attr("encoding-type").ok()?;
        attr.read_scalar::<VarLenUnicode>()
            .ok()
            .map(|s| s.as_str().to_owned())
    };
    let obs_is_df = group_encoding("obs").as_deref() == Some("dataframe");
    let var_is_df = group_encoding("var").as_deref() == Some("dataframe");
    let has_x = file.group("X").is_ok() || file.dataset("X").is_ok();
    if has_x && obs_is_df && var_is_df {
        return Some(Format::H5Ad);
    }

    // --- H5Seurat ---
    // The /cell.names root dataset + /assays group are the structural
    // fingerprint. SeuratDisk also writes an active.assay root attr, but
    // our lean BPCells-mode output skips that unless --seuratdisk-compat
    // is passed, so we accept either signal as confirmation.
    let has_cell_names = file.dataset("cell.names").is_ok();
    let has_assays = file.group("assays").is_ok();
    let has_active_assay = file
        .group("/")
        .ok()
        .map(|g| g.attr("active.assay").is_ok())
        .unwrap_or(false);
    if has_cell_names && (has_assays || has_active_assay) {
        return Some(Format::H5Seurat);
    }

    // --- SCX internal ---
    // Our golden fixture schema stores /X/shape as a dataset (distinct from
    // H5AD which stores shape as an *attribute* on the /X group).
    if file.dataset("X/shape").is_ok() {
        return Some(Format::ScxH5);
    }

    // --- 10x Genomics HDF5 ---
    // Cell Ranger writes a /matrix group containing /matrix/barcodes and
    // /matrix/features (both present in v2+ multi-modal outputs).
    let has_barcodes = file.dataset("matrix/barcodes").is_ok();
    let has_features = file.group("matrix/features").is_ok();
    if has_barcodes && has_features {
        return Some(Format::TenxH5);
    }

    // Valid HDF5 but no recognised single-cell fingerprint — generic fallback.
    Some(Format::PlainH5)
}

#[cfg(test)]
mod tests {
    use super::*;

    const H5SEURAT: &str = "../../tests/golden/pbmc3k.h5seurat";
    const SCX_H5: &str = "../../tests/golden/pbmc3k.h5";
    const H5AD_REF: &str = "../../tests/golden/pbmc3k_reference.h5ad";

    #[test]
    fn test_sniff_h5seurat() {
        if !Path::new(H5SEURAT).exists() {
            return;
        }
        assert_eq!(sniff(Path::new(H5SEURAT)), Some(Format::H5Seurat));
    }

    #[test]
    fn test_sniff_scx_h5() {
        if !Path::new(SCX_H5).exists() {
            return;
        }
        assert_eq!(sniff(Path::new(SCX_H5)), Some(Format::ScxH5));
    }

    #[test]
    fn test_sniff_h5ad() {
        if !Path::new(H5AD_REF).exists() {
            return;
        }
        assert_eq!(sniff(Path::new(H5AD_REF)), Some(Format::H5Ad));
    }
}
