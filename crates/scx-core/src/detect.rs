//! Format detection by file content, not extension.
//!
//! Opens the file as HDF5 and checks for well-known fingerprints:
//!
//! | Format   | Fingerprint |
//! |----------|-------------|
//! | H5AD     | Root attr `encoding-type = "anndata"` |
//! | H5Seurat | Root dataset `cell.names` + root attr `active.assay` |
//! | ScxH5    | Root dataset `X/shape` (SCX internal golden fixture schema) |

use std::path::Path;

use hdf5::File;
use hdf5::types::VarLenUnicode;

/// The detected on-disk format of an HDF5 file or NPY directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    H5Ad,
    H5Seurat,
    /// SCX internal HDF5 schema (golden test fixtures).
    ScxH5,
    /// NPY snapshot directory (contains `meta.json`).
    NpyDir,
}

/// Sniff the format of a directory by checking for `meta.json`.
///
/// Returns `Some(Format::NpyDir)` if `path` is a directory containing
/// `meta.json`, `None` otherwise.
pub fn sniff_dir(path: &Path) -> Option<Format> {
    if path.is_dir() && path.join("meta.json").exists() {
        Some(Format::NpyDir)
    } else {
        None
    }
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

    // --- H5Seurat ---
    // SeuratDisk always writes /cell.names (string dataset) and sets
    // active.assay as a root-level HDF5 attribute.
    let has_cell_names   = file.dataset("cell.names").is_ok();
    let has_active_assay = file.group("/")
        .ok()
        .map(|g| g.attr("active.assay").is_ok())
        .unwrap_or(false);
    if has_cell_names && has_active_assay {
        return Some(Format::H5Seurat);
    }

    // --- SCX internal ---
    // Our golden fixture schema stores /X/shape as a dataset (distinct from
    // H5AD which stores shape as an *attribute* on the /X group).
    if file.dataset("X/shape").is_ok() {
        return Some(Format::ScxH5);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const H5SEURAT: &str = "../../tests/golden/pbmc3k.h5seurat";
    const SCX_H5:   &str = "../../tests/golden/pbmc3k.h5";
    const H5AD_REF: &str = "../../tests/golden/pbmc3k_reference.h5ad";

    #[test]
    fn test_sniff_h5seurat() {
        if !Path::new(H5SEURAT).exists() { return; }
        assert_eq!(sniff(Path::new(H5SEURAT)), Some(Format::H5Seurat));
    }

    #[test]
    fn test_sniff_scx_h5() {
        if !Path::new(SCX_H5).exists() { return; }
        assert_eq!(sniff(Path::new(SCX_H5)), Some(Format::ScxH5));
    }

    #[test]
    fn test_sniff_h5ad() {
        if !Path::new(H5AD_REF).exists() { return; }
        assert_eq!(sniff(Path::new(H5AD_REF)), Some(Format::H5Ad));
    }
}
