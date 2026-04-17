use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Deserializer};

use crate::{dtype::DataType, error::Result, stream::DatasetReader};

// ---------------------------------------------------------------------------
// Qualifier
// ---------------------------------------------------------------------------

/// Numeric comparison operator for shape constraints.
/// Deserializes from either a plain integer (`1000`) or a string (`">= 1000"`).
#[derive(Debug, Clone)]
pub enum Qualifier {
    Eq(usize),
    Ge(usize),
    Gt(usize),
    Le(usize),
    Lt(usize),
}

impl Qualifier {
    pub fn check(&self, value: usize) -> bool {
        match self {
            Qualifier::Eq(n) => value == *n,
            Qualifier::Ge(n) => value >= *n,
            Qualifier::Gt(n) => value > *n,
            Qualifier::Le(n) => value <= *n,
            Qualifier::Lt(n) => value < *n,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Qualifier::Eq(n) => format!("== {n}"),
            Qualifier::Ge(n) => format!(">= {n}"),
            Qualifier::Gt(n) => format!("> {n}"),
            Qualifier::Le(n) => format!("<= {n}"),
            Qualifier::Lt(n) => format!("< {n}"),
        }
    }
}

impl<'de> Deserialize<'de> for Qualifier {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Visitor;

        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Qualifier;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "an integer or a qualifier string like \">= 1000\"")
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> std::result::Result<Qualifier, E> {
                Ok(Qualifier::Eq(v as usize))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> std::result::Result<Qualifier, E> {
                Ok(Qualifier::Eq(v.max(0) as usize))
            }

            fn visit_str<E: serde::de::Error>(self, s: &str) -> std::result::Result<Qualifier, E> {
                let s = s.trim();
                let (op, rest) = if let Some(r) = s.strip_prefix(">=") {
                    (">=", r.trim())
                } else if let Some(r) = s.strip_prefix("<=") {
                    ("<=", r.trim())
                } else if let Some(r) = s.strip_prefix("==") {
                    ("==", r.trim())
                } else if let Some(r) = s.strip_prefix('>') {
                    (">", r.trim())
                } else if let Some(r) = s.strip_prefix('<') {
                    ("<", r.trim())
                } else {
                    let n = s.parse::<usize>().map_err(E::custom)?;
                    return Ok(Qualifier::Eq(n));
                };
                let n = rest.parse::<usize>().map_err(E::custom)?;
                Ok(match op {
                    ">=" => Qualifier::Ge(n),
                    "<=" => Qualifier::Le(n),
                    "==" => Qualifier::Eq(n),
                    ">"  => Qualifier::Gt(n),
                    "<"  => Qualifier::Lt(n),
                    _    => unreachable!(),
                })
            }
        }

        d.deserialize_any(Visitor)
    }
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ValidationSchema {
    /// Minimum/exact obs count, e.g. `">= 1000"` or `8000`.
    pub obs: Option<Qualifier>,
    /// Minimum/exact var count, e.g. `">= 500"`.
    pub vars: Option<Qualifier>,

    /// Required layer names.
    pub layers: Option<Vec<String>>,
    /// Required obsm keys.
    pub obsm: Option<Vec<String>>,
    /// Required obsp keys.
    pub obsp: Option<Vec<String>>,
    /// Required var column names.
    pub var_columns: Option<Vec<String>>,
    /// Required obs column names.
    pub obs_columns: Option<Vec<String>>,

    /// Expected X dtype: f32 | f64 | i32 | u32.
    pub x_dtype: Option<String>,

    /// Assert obs index has no duplicates.
    pub obs_index_unique: Option<bool>,
    /// Assert var index has no duplicates.
    pub var_index_unique: Option<bool>,
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Default)]
pub struct ValidationReport {
    pub file: String,
    pub schema: String,
    pub checks: Vec<CheckResult>,
}

impl ValidationReport {
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    pub fn n_passed(&self) -> usize {
        self.checks.iter().filter(|c| c.passed).count()
    }

    pub fn n_failed(&self) -> usize {
        self.checks.iter().filter(|c| !c.passed).count()
    }
}

// ---------------------------------------------------------------------------
// run_validation
// ---------------------------------------------------------------------------

pub async fn run_validation(
    reader: &mut dyn DatasetReader,
    schema: &ValidationSchema,
    file: &str,
    schema_path: &str,
) -> Result<ValidationReport> {
    let mut report = ValidationReport {
        file: file.to_string(),
        schema: schema_path.to_string(),
        ..Default::default()
    };

    let (n_obs, n_vars) = reader.shape();
    let dtype = reader.dtype();

    // --- Shape ---
    if let Some(q) = &schema.obs {
        report.checks.push(check(
            "obs",
            q.check(n_obs),
            format!("{n_obs}  ({})", q.describe()),
            format!("{n_obs}  FAILED {}", q.describe()),
        ));
    }

    if let Some(q) = &schema.vars {
        report.checks.push(check(
            "vars",
            q.check(n_vars),
            format!("{n_vars}  ({})", q.describe()),
            format!("{n_vars}  FAILED {}", q.describe()),
        ));
    }

    // --- dtype ---
    if let Some(expected) = &schema.x_dtype {
        let actual = dtype_str(dtype);
        let ok = actual == expected.as_str();
        report.checks.push(check(
            "x_dtype",
            ok,
            actual.to_string(),
            format!("expected {expected}, got {actual}"),
        ));
    }

    // --- Lazily read only what is needed ---

    let need_layers = schema.layers.is_some();
    let need_obsm   = schema.obsm.is_some();
    let need_obsp   = schema.obsp.is_some();
    let need_var    = schema.var_columns.is_some() || schema.var_index_unique == Some(true);
    let need_obs    = schema.obs_columns.is_some() || schema.obs_index_unique == Some(true);

    let layer_metas = if need_layers { Some(reader.layer_metas().await?) } else { None };
    let obsm_data   = if need_obsm   { Some(reader.obsm().await?)        } else { None };
    let obsp_metas  = if need_obsp   { Some(reader.obsp_metas().await?)  } else { None };
    let var_data    = if need_var    { Some(reader.var().await?)          } else { None };
    let obs_data    = if need_obs    { Some(reader.obs().await?)          } else { None };

    // --- Layers ---
    if let (Some(required), Some(metas)) = (&schema.layers, &layer_metas) {
        let available: HashSet<&str> = metas.iter().map(|m| m.name.as_str()).collect();
        let missing: Vec<&str> = required.iter().map(String::as_str)
            .filter(|s| !available.contains(s)).collect();
        let ok = missing.is_empty();
        report.checks.push(check(
            "layers",
            ok,
            required.join(", "),
            format!("missing: {}", missing.join(", ")),
        ));
    }

    // --- obsm ---
    if let (Some(required), Some(obsm)) = (&schema.obsm, &obsm_data) {
        let missing: Vec<&str> = required.iter().map(String::as_str)
            .filter(|s| !obsm.map.contains_key(*s)).collect();
        let ok = missing.is_empty();
        report.checks.push(check(
            "obsm",
            ok,
            required.join(", "),
            format!("missing: {}", missing.join(", ")),
        ));
    }

    // --- obsp ---
    if let (Some(required), Some(metas)) = (&schema.obsp, &obsp_metas) {
        let available: HashSet<&str> = metas.iter().map(|m| m.name.as_str()).collect();
        let missing: Vec<&str> = required.iter().map(String::as_str)
            .filter(|s| !available.contains(s)).collect();
        let ok = missing.is_empty();
        report.checks.push(check(
            "obsp",
            ok,
            required.join(", "),
            format!("missing: {}", missing.join(", ")),
        ));
    }

    // --- var columns ---
    if let (Some(required), Some(var)) = (&schema.var_columns, &var_data) {
        let available: HashSet<&str> = var.columns.iter().map(|c| c.name.as_str()).collect();
        let missing: Vec<&str> = required.iter().map(String::as_str)
            .filter(|s| !available.contains(s)).collect();
        let ok = missing.is_empty();
        report.checks.push(check(
            "var_columns",
            ok,
            required.join(", "),
            format!("missing: {}", missing.join(", ")),
        ));
    }

    // --- obs columns ---
    if let (Some(required), Some(obs)) = (&schema.obs_columns, &obs_data) {
        let available: HashSet<&str> = obs.columns.iter().map(|c| c.name.as_str()).collect();
        let missing: Vec<&str> = required.iter().map(String::as_str)
            .filter(|s| !available.contains(s)).collect();
        let ok = missing.is_empty();
        report.checks.push(check(
            "obs_columns",
            ok,
            required.join(", "),
            format!("missing: {}", missing.join(", ")),
        ));
    }

    // --- Index uniqueness ---
    if schema.obs_index_unique == Some(true) {
        if let Some(obs) = &obs_data {
            let unique = is_unique(&obs.index);
            report.checks.push(check(
                "obs_index_unique",
                unique,
                "unique".into(),
                "duplicates found".into(),
            ));
        }
    }

    if schema.var_index_unique == Some(true) {
        if let Some(var) = &var_data {
            let unique = is_unique(&var.index);
            report.checks.push(check(
                "var_index_unique",
                unique,
                "unique".into(),
                "duplicates found".into(),
            ));
        }
    }

    Ok(report)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check(name: &str, passed: bool, pass_detail: String, fail_detail: String) -> CheckResult {
    CheckResult {
        name: name.to_string(),
        passed,
        detail: if passed { pass_detail } else { fail_detail },
    }
}

fn dtype_str(dtype: DataType) -> &'static str {
    match dtype {
        DataType::F32 => "f32",
        DataType::F64 => "f64",
        DataType::I32 => "i32",
        DataType::U32 => "u32",
    }
}

fn is_unique(v: &[String]) -> bool {
    let mut seen = HashSet::new();
    v.iter().all(|s| seen.insert(s.as_str()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_qualifier(s: &str) -> Qualifier {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn test_qualifier_from_int() {
        let q = parse_qualifier("500");
        assert!(q.check(500));
        assert!(!q.check(499));
        assert!(!q.check(501));
        assert_eq!(q.describe(), "== 500");
    }

    #[test]
    fn test_qualifier_ge() {
        let q = parse_qualifier("\">= 1000\"");
        assert!(q.check(1000));
        assert!(q.check(9999));
        assert!(!q.check(999));
        assert_eq!(q.describe(), ">= 1000");
    }

    #[test]
    fn test_qualifier_gt() {
        let q = parse_qualifier("\"> 0\"");
        assert!(q.check(1));
        assert!(!q.check(0));
    }

    #[test]
    fn test_qualifier_le() {
        let q = parse_qualifier("\"<= 50000\"");
        assert!(q.check(50000));
        assert!(q.check(0));
        assert!(!q.check(50001));
    }

    #[test]
    fn test_qualifier_lt() {
        let q = parse_qualifier("\"< 100\"");
        assert!(q.check(99));
        assert!(!q.check(100));
    }

    #[test]
    fn test_qualifier_eq_string() {
        let q = parse_qualifier("\"== 8000\"");
        assert!(q.check(8000));
        assert!(!q.check(7999));
    }

    #[test]
    fn test_is_unique() {
        let v: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        assert!(is_unique(&v));
        let v2: Vec<String> = vec!["a".into(), "b".into(), "a".into()];
        assert!(!is_unique(&v2));
    }

    #[test]
    fn test_schema_deserialize_empty() {
        let schema: ValidationSchema = serde_json::from_str("{}").unwrap();
        assert!(schema.obs.is_none());
        assert!(schema.layers.is_none());
    }

    // --- Integration tests against real fixtures ---

    const NORMAN_SUBSET: &str = "../../tests/fixtures/norman_subset.h5ad";

    fn norman_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("NORMAN_H5AD") {
            let pb = std::path::PathBuf::from(&p);
            if pb.exists() { return Some(pb); }
        }
        let pb = std::path::PathBuf::from(NORMAN_SUBSET);
        if pb.exists() { Some(pb) } else { None }
    }

    #[tokio::test]
    async fn test_run_validation_shape_pass() {
        let Some(path) = norman_path() else { return; };
        let (n_obs, n_vars) = crate::h5ad::H5AdReader::open(&path, 500).unwrap().shape();

        let mut reader = crate::h5ad::H5AdReader::open(&path, 500).unwrap();
        let schema: ValidationSchema = serde_yaml::from_str(&format!(
            "obs: \">= {n_obs}\"\nvars: \">= {n_vars}\""
        )).unwrap();
        let report = run_validation(&mut reader, &schema, "test", "test").await.unwrap();
        assert!(report.passed(), "shape checks should pass with exact counts");
        assert_eq!(report.n_failed(), 0);
    }

    #[tokio::test]
    async fn test_run_validation_shape_fail() {
        let Some(path) = norman_path() else { return; };
        let (n_obs, _) = crate::h5ad::H5AdReader::open(&path, 500).unwrap().shape();

        let mut reader = crate::h5ad::H5AdReader::open(&path, 500).unwrap();
        // Require more obs than the file has
        let schema: ValidationSchema = serde_yaml::from_str(&format!(
            "obs: \"> {}\"\n", n_obs
        )).unwrap();
        let report = run_validation(&mut reader, &schema, "test", "test").await.unwrap();
        assert!(!report.passed());
        assert_eq!(report.n_failed(), 1);
        assert_eq!(report.checks[0].name, "obs");
    }

    #[tokio::test]
    async fn test_run_validation_missing_layer() {
        let Some(path) = norman_path() else { return; };
        let mut reader = crate::h5ad::H5AdReader::open(&path, 500).unwrap();
        let schema: ValidationSchema = serde_yaml::from_str(
            "layers:\n  - does_not_exist\n"
        ).unwrap();
        let report = run_validation(&mut reader, &schema, "test", "test").await.unwrap();
        assert!(!report.passed());
        let layer_check = report.checks.iter().find(|c| c.name == "layers").unwrap();
        assert!(layer_check.detail.contains("does_not_exist"));
    }

    #[tokio::test]
    async fn test_run_validation_index_unique() {
        let Some(path) = norman_path() else { return; };
        let mut reader = crate::h5ad::H5AdReader::open(&path, 500).unwrap();
        let schema: ValidationSchema = serde_yaml::from_str(
            "obs_index_unique: true\nvar_index_unique: true\n"
        ).unwrap();
        let report = run_validation(&mut reader, &schema, "test", "test").await.unwrap();
        // Norman subset has unique indices
        assert!(report.passed(), "Norman indices should be unique");
    }

    #[test]
    fn test_schema_deserialize_full() {
        let yaml = r#"
obs: ">= 1000"
vars: 500
layers:
  - normalized
obsm:
  - X_pca
x_dtype: f32
obs_index_unique: true
"#;
        let schema: ValidationSchema = serde_yaml::from_str(yaml).unwrap();
        let obs_q = schema.obs.unwrap();
        assert!(obs_q.check(1000));
        assert!(!obs_q.check(999));
        let vars_q = schema.vars.unwrap();
        assert!(vars_q.check(500));
        assert!(!vars_q.check(499));
        assert_eq!(schema.layers.unwrap(), vec!["normalized"]);
        assert_eq!(schema.x_dtype.unwrap(), "f32");
        assert_eq!(schema.obs_index_unique, Some(true));
    }
}
