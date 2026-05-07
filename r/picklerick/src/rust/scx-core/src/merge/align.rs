use std::collections::HashMap;

use crate::{error::Result, ir::ColumnData};

/// Build a row-reindex map from a patch index into a base index.
///
/// Returns a `Vec<Option<usize>>` of length `base_index.len()`:
/// - `Some(i)` → base row j is found at patch row i
/// - `None`    → base row j is absent from the patch (NA-fill on application)
///
/// Patch rows not present in base are ignored silently — the caller validates
/// that the patch covers the required alignment (all-present for dense slots).
pub fn build_obs_reindex(
    base_index: &[String],
    patch_index: &[String],
) -> Result<Vec<Option<usize>>> {
    let patch_pos: HashMap<&str, usize> = patch_index
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    Ok(base_index
        .iter()
        .map(|b| patch_pos.get(b.as_str()).copied())
        .collect())
}

/// Same as `build_obs_reindex` but for the variable (gene) axis.
pub fn build_var_reindex(
    base_index: &[String],
    patch_index: &[String],
) -> Result<Vec<Option<usize>>> {
    build_obs_reindex(base_index, patch_index)
}

/// Reindex a `ColumnData` according to a row-reindex map.
///
/// NA-fill values per variant:
/// - Float   → `f64::NAN`
/// - Int     → `0`
/// - Bool    → `false`
/// - String  → `""`
/// - Categorical → code `0` (first level; caller should ensure level 0 is a
///   sentinel like `"NA"` when this matters)
pub fn reindex_column(col: &ColumnData, reindex: &[Option<usize>]) -> ColumnData {
    match col {
        ColumnData::Float(v) => ColumnData::Float(
            reindex
                .iter()
                .map(|r| r.map(|i| v[i]).unwrap_or(f64::NAN))
                .collect(),
        ),
        ColumnData::Int(v) => ColumnData::Int(
            reindex
                .iter()
                .map(|r| r.map(|i| v[i]).unwrap_or(0))
                .collect(),
        ),
        ColumnData::Bool(v) => ColumnData::Bool(
            reindex
                .iter()
                .map(|r| r.map(|i| v[i]).unwrap_or(false))
                .collect(),
        ),
        ColumnData::String(v) => ColumnData::String(
            reindex
                .iter()
                .map(|r| r.map(|i| v[i].clone()).unwrap_or_default())
                .collect(),
        ),
        ColumnData::Categorical { codes, levels } => {
            let new_codes = reindex
                .iter()
                .map(|r| r.map(|i| codes[i]).unwrap_or(0))
                .collect();
            ColumnData::Categorical {
                codes: new_codes,
                levels: levels.clone(),
            }
        }
    }
}

/// Merge two categorical level lists into a single unified list.
///
/// Returns `(unified_levels, patch_remap)` where `patch_remap[old_code]`
/// gives the new code in `unified_levels`.  Levels that already appear in
/// `base_levels` keep their original position; new levels are appended.
pub fn unify_categorical_levels(
    base_levels: &[String],
    patch_levels: &[String],
) -> (Vec<String>, Vec<u32>) {
    let mut unified: Vec<String> = base_levels.to_vec();
    let mut remap: Vec<u32> = Vec::with_capacity(patch_levels.len());
    for level in patch_levels {
        if let Some(pos) = unified.iter().position(|l| l == level) {
            remap.push(pos as u32);
        } else {
            remap.push(unified.len() as u32);
            unified.push(level.clone());
        }
    }
    (unified, remap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn reindex_subset() {
        let base = sv(&["c1", "c2", "c3", "c4"]);
        let patch = sv(&["c1", "c3"]);
        let ri = build_obs_reindex(&base, &patch).unwrap();
        assert_eq!(ri, vec![Some(0), None, Some(1), None]);
    }

    #[test]
    fn reindex_reordered() {
        let base = sv(&["c1", "c2", "c3"]);
        let patch = sv(&["c3", "c1", "c2"]);
        let ri = build_obs_reindex(&base, &patch).unwrap();
        assert_eq!(ri, vec![Some(1), Some(2), Some(0)]);
    }

    #[test]
    fn reindex_full_match() {
        let base = sv(&["a", "b", "c"]);
        let ri = build_obs_reindex(&base, &base).unwrap();
        assert_eq!(ri, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn reindex_column_float_na_fill() {
        let col = ColumnData::Float(vec![1.0, 2.0]);
        let ri = vec![Some(0), None, Some(1)];
        match reindex_column(&col, &ri) {
            ColumnData::Float(v) => {
                assert_eq!(v[0], 1.0);
                assert!(v[1].is_nan());
                assert_eq!(v[2], 2.0);
            }
            _ => panic!("expected Float"),
        }
    }

    #[test]
    fn reindex_column_categorical() {
        let col = ColumnData::Categorical {
            codes: vec![0, 1, 0],
            levels: vec!["A".to_string(), "B".to_string()],
        };
        let ri = vec![Some(2), None, Some(0)];
        match reindex_column(&col, &ri) {
            ColumnData::Categorical { codes, levels } => {
                assert_eq!(codes, vec![0, 0, 0]); // patch[2]=0, NA→0, patch[0]=0
                assert_eq!(levels, vec!["A", "B"]);
            }
            _ => panic!("expected Categorical"),
        }
    }

    #[test]
    fn unify_levels_overlap() {
        let (unified, remap) = unify_categorical_levels(&sv(&["A", "B"]), &sv(&["B", "C"]));
        assert_eq!(unified, sv(&["A", "B", "C"]));
        assert_eq!(remap, vec![1, 2]);
    }

    #[test]
    fn unify_levels_disjoint() {
        let (unified, remap) = unify_categorical_levels(&sv(&["A"]), &sv(&["B", "C"]));
        assert_eq!(unified, sv(&["A", "B", "C"]));
        assert_eq!(remap, vec![1, 2]);
    }

    #[test]
    fn unify_levels_identical() {
        let lvls = sv(&["X", "Y", "Z"]);
        let (unified, remap) = unify_categorical_levels(&lvls, &lvls);
        assert_eq!(unified, lvls);
        assert_eq!(remap, vec![0, 1, 2]);
    }
}
