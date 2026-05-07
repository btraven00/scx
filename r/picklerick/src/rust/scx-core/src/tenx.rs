use std::collections::HashMap;
use std::path::Path;

use hdf5::types::{TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::File;

use crate::error::{Result, ScxError};

// ---------------------------------------------------------------------------
// 10x HDF5
// ---------------------------------------------------------------------------

/// Summary of a 10x Genomics HDF5 feature-barcode matrix.
pub struct TenxH5Info {
    pub n_barcodes: usize,
    pub n_features: usize,
    /// Counts per feature type, e.g. `[("Gene Expression", 33538), ("Antibody Capture", 17)]`.
    pub feature_types: Vec<(String, usize)>,
    /// Genome name(s) from `/matrix/features/genome`, `None` if the dataset is absent.
    pub genome: Option<String>,
}

/// Read metadata from a 10x HDF5 file without loading the count matrix.
pub fn read_tenx_h5(path: &Path) -> Result<TenxH5Info> {
    let file = File::open(path)?;

    let barcodes_ds = file
        .dataset("matrix/barcodes")
        .map_err(|_| ScxError::InvalidFormat("missing /matrix/barcodes".into()))?;
    let n_barcodes = barcodes_ds.shape().first().copied().unwrap_or(0);

    let feat_id_ds = file
        .dataset("matrix/features/id")
        .map_err(|_| ScxError::InvalidFormat("missing /matrix/features/id".into()))?;
    let n_features = feat_id_ds.shape().first().copied().unwrap_or(0);

    // Count occurrences of each feature type.
    let feature_types = if let Ok(ft_ds) = file.dataset("matrix/features/feature_type") {
        let strings = read_str_dataset_raw(&ft_ds)?;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for s in strings {
            *counts.entry(s).or_insert(0) += 1;
        }
        let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        pairs
    } else {
        Vec::new()
    };

    // Genome: collect unique values and join with ", ".
    let genome = if let Ok(g_ds) = file.dataset("matrix/features/genome") {
        let strings = read_str_dataset_raw(&g_ds)?;
        let mut seen: Vec<String> = Vec::new();
        for s in strings {
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
        if seen.is_empty() {
            None
        } else {
            Some(seen.join(", "))
        }
    } else {
        None
    };

    Ok(TenxH5Info {
        n_barcodes,
        n_features,
        feature_types,
        genome,
    })
}

fn read_str_dataset_raw(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    Ok(match ds.dtype()?.to_descriptor()? {
        TypeDescriptor::VarLenUnicode => ds
            .read_1d::<VarLenUnicode>()?
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        TypeDescriptor::VarLenAscii => ds
            .read_1d::<VarLenAscii>()?
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        _ => Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Plain / unrecognized HDF5
// ---------------------------------------------------------------------------

/// A node in the HDF5 file tree.
pub struct H5Node {
    pub name: String,
    pub kind: H5NodeKind,
}

pub enum H5NodeKind {
    Dataset {
        shape: Vec<usize>,
        dtype: String,
    },
    Group {
        children: Vec<H5Node>,
        /// Number of children that were omitted due to depth limit.
        truncated: usize,
    },
}

/// Walk the root of an HDF5 file up to `max_depth` levels deep.
pub fn walk_h5(path: &Path, max_depth: usize) -> Result<Vec<H5Node>> {
    let file = File::open(path)?;
    let root = file
        .group("/")
        .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
    walk_group(&file, &root, max_depth)
}

fn walk_group(file: &File, grp: &hdf5::Group, depth: usize) -> Result<Vec<H5Node>> {
    let names = grp.member_names().unwrap_or_default();
    let mut nodes = Vec::with_capacity(names.len());

    for name in &names {
        let full_path = {
            let grp_name = grp.name();
            if grp_name == "/" {
                format!("/{name}")
            } else {
                format!("{grp_name}/{name}")
            }
        };

        let is_group = file.group(&full_path).is_ok() && file.dataset(&full_path).is_err();

        let kind = if is_group {
            if depth == 0 {
                H5NodeKind::Group {
                    children: Vec::new(),
                    truncated: file
                        .group(&full_path)
                        .ok()
                        .and_then(|g| g.member_names().ok())
                        .map(|v| v.len())
                        .unwrap_or(0),
                }
            } else {
                let child_grp = file
                    .group(&full_path)
                    .map_err(|e| ScxError::InvalidFormat(e.to_string()))?;
                let children = walk_group(file, &child_grp, depth - 1)?;
                H5NodeKind::Group {
                    children,
                    truncated: 0,
                }
            }
        } else {
            match file.dataset(&full_path) {
                Ok(ds) => {
                    let shape = ds.shape();
                    let dtype = dtype_str(&ds);
                    H5NodeKind::Dataset { shape, dtype }
                }
                Err(_) => continue,
            }
        };

        nodes.push(H5Node {
            name: name.clone(),
            kind,
        });
    }

    nodes.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(nodes)
}

fn dtype_str(ds: &hdf5::Dataset) -> String {
    match ds.dtype().and_then(|d| d.to_descriptor()) {
        Ok(TypeDescriptor::Float(s)) => format!("f{}", (s as usize) * 8),
        Ok(TypeDescriptor::Integer(s)) => format!("i{}", (s as usize) * 8),
        Ok(TypeDescriptor::Unsigned(s)) => format!("u{}", (s as usize) * 8),
        Ok(TypeDescriptor::Boolean) => "bool".into(),
        Ok(TypeDescriptor::VarLenUnicode) => "str".into(),
        Ok(TypeDescriptor::VarLenAscii) => "str".into(),
        Ok(TypeDescriptor::FixedAscii(n)) => format!("str[{n}]"),
        Ok(TypeDescriptor::FixedUnicode(n)) => format!("str[{n}]"),
        _ => "?".into(),
    }
}
