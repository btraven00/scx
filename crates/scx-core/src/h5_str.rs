//! One 1-D string-dataset reader, shared by every HDF5 format scx reads.
//!
//! Four readers previously each dispatched on `TypeDescriptor` themselves and each
//! handled only the two variable-length descriptors. Fixed-length strings — what
//! `rhdf5` / `HDF5Array` write by default, so every R-produced file — fell off the
//! end of all four, most damagingly in `tenx::read_str_dataset_raw`, which returned
//! an empty `Vec` and `Ok(())` for a dataset with 156 881 rows. The caller only
//! learned about it by panicking somewhere unrelated.
//!
//! ## Why the const-generic ladder
//!
//! `ds.read_raw::<u8>()` looks like the obvious way to get the bytes and is not
//! available — HDF5 refuses the conversion outright:
//!
//! ```text
//! no conversion paths found from '<HDF5 datatype: string (len 32)>'
//!                             to '<HDF5 datatype: uint8>'
//! ```
//!
//! `FixedAscii<N>` / `FixedUnicode<N>` carry their width as a const generic, so a
//! runtime width has to be dispatched to compile-time sizes. HDF5 *will* widen — a
//! `len 32` dataset reads fine into a `FixedAscii<64>` — so a short ladder of
//! powers of two covers every real width without a match arm per byte count.

use hdf5::types::{FixedAscii, FixedUnicode, TypeDescriptor, VarLenAscii, VarLenUnicode};

use crate::error::{Result, ScxError};

/// Read a 1-D string dataset, whatever flavour of HDF5 string it holds.
///
/// Errors on non-string datasets rather than returning an empty vector: a caller
/// asking for strings from an integer dataset has a bug, and silence is how the
/// original defect stayed invisible.
pub fn read_str_1d(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    if ds.size() == 0 {
        return Ok(Vec::new());
    }
    let desc = ds.dtype()?.to_descriptor()?;
    Ok(match desc {
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
        TypeDescriptor::FixedAscii(n) => match n {
            0..=32 => fixed_ascii::<32>(ds)?,
            33..=64 => fixed_ascii::<64>(ds)?,
            65..=128 => fixed_ascii::<128>(ds)?,
            129..=256 => fixed_ascii::<256>(ds)?,
            257..=1024 => fixed_ascii::<1024>(ds)?,
            _ => return Err(too_wide("FixedAscii", n)),
        },
        TypeDescriptor::FixedUnicode(n) => match n {
            0..=32 => fixed_unicode::<32>(ds)?,
            33..=64 => fixed_unicode::<64>(ds)?,
            65..=128 => fixed_unicode::<128>(ds)?,
            129..=256 => fixed_unicode::<256>(ds)?,
            257..=1024 => fixed_unicode::<1024>(ds)?,
            _ => return Err(too_wide("FixedUnicode", n)),
        },
        other => {
            return Err(ScxError::InvalidFormat(format!(
                "expected a string dataset at '{}', found {other:?}",
                ds.name()
            )))
        }
    })
}

fn fixed_ascii<const N: usize>(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    Ok(ds
        .read_1d::<FixedAscii<N>>()?
        .into_iter()
        .map(|s| s.trim_end_matches('\0').to_string())
        .collect())
}

fn fixed_unicode<const N: usize>(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    Ok(ds
        .read_1d::<FixedUnicode<N>>()?
        .into_iter()
        .map(|s| s.trim_end_matches('\0').to_string())
        .collect())
}

fn too_wide(kind: &str, n: usize) -> ScxError {
    ScxError::InvalidFormat(format!(
        "{kind} string width {n} exceeds the widest supported ladder rung (1024); \
         add a rung in h5_str.rs if this is a real file"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::File;
    use std::str::FromStr;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("scx_h5str_{name}_{}.h5", std::process::id()));
        p
    }

    /// The reported bug: `|S32` barcodes read back as an empty vector.
    #[test]
    fn reads_fixed_ascii() {
        let path = tmp("fixed_ascii");
        let want = ["AAACCTGAGAAACCAT-1", "AAACCTGAGAAACCGC-1", "TTTGTCATCTTTAGTC-1"];
        {
            let f = File::create(&path).unwrap();
            let vals: Vec<FixedAscii<32>> =
                want.iter().map(|s| FixedAscii::from_ascii(s).unwrap()).collect();
            f.new_dataset::<FixedAscii<32>>()
                .shape([vals.len()])
                .create("barcodes")
                .unwrap()
                .write(&vals)
                .unwrap();
        }
        let f = File::open(&path).unwrap();
        let got = read_str_1d(&f.dataset("barcodes").unwrap()).unwrap();
        assert_eq!(got, want, "fixed-length ASCII must round-trip, not vanish");
        let _ = std::fs::remove_file(&path);
    }

    /// A width that is not a ladder rung must still read — HDF5 widens into the
    /// next rung up.  `|S13` is what Cell Ranger v2 `matrix/genes` uses.
    #[test]
    fn reads_fixed_ascii_off_rung_width() {
        let path = tmp("off_rung");
        let want = ["ENSG00000243485", "ENSG00000237613"];
        {
            let f = File::create(&path).unwrap();
            let vals: Vec<FixedAscii<15>> =
                want.iter().map(|s| FixedAscii::from_ascii(s).unwrap()).collect();
            f.new_dataset::<FixedAscii<15>>()
                .shape([vals.len()])
                .create("genes")
                .unwrap()
                .write(&vals)
                .unwrap();
        }
        let f = File::open(&path).unwrap();
        let got = read_str_1d(&f.dataset("genes").unwrap()).unwrap();
        assert_eq!(got, want);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reads_var_len_unicode() {
        let path = tmp("varlen");
        let want = ["cell_a", "cell_b"];
        {
            let f = File::create(&path).unwrap();
            let vals: Vec<VarLenUnicode> = want
                .iter()
                .map(|s| VarLenUnicode::from_str(s).unwrap())
                .collect();
            f.new_dataset::<VarLenUnicode>()
                .shape([vals.len()])
                .create("index")
                .unwrap()
                .write(&vals)
                .unwrap();
        }
        let f = File::open(&path).unwrap();
        assert_eq!(read_str_1d(&f.dataset("index").unwrap()).unwrap(), want);
        let _ = std::fs::remove_file(&path);
    }

    /// The `_` arm must fail loudly.  Returning an empty vector here is the exact
    /// shape of the original defect.
    #[test]
    fn non_string_dataset_errors() {
        let path = tmp("ints");
        {
            let f = File::create(&path).unwrap();
            f.new_dataset::<i32>()
                .shape([3])
                .create("counts")
                .unwrap()
                .write(&[1i32, 2, 3])
                .unwrap();
        }
        let f = File::open(&path).unwrap();
        let err = read_str_1d(&f.dataset("counts").unwrap()).unwrap_err();
        assert!(
            matches!(err, ScxError::InvalidFormat(_)),
            "non-string dataset must error, got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }
}
