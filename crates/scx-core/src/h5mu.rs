//! MuData `.h5mu` input: each modality is an AnnData group under `/mod`, read
//! by [`H5AdReader::open_modality`](crate::h5ad::H5AdReader::open_modality).
//!
//! Modality discovery (`/mod` `mod-order`, else the groups under `/mod`) and
//! the Seurat assay aliases follow scConvert's `sc_h5mu.c` and
//! `sc_modality.c` (MIT, Copyright (c) 2026 Ziyu Zeng,
//! <https://github.com/mianaz/scConvert>).

use std::path::Path;

use hdf5::{File, Group};

use crate::error::{Result, ScxError};
use crate::h5_str::read_str_array_attr;

/// Seurat assay names and the modality each stands for.
const ASSAY_ALIASES: &[(&str, &str)] = &[
    ("RNA", "rna"),
    ("ADT", "prot"),
    ("Protein", "prot"),
    ("ATAC", "atac"),
    ("Peaks", "atac"),
];

/// The modalities of the `.h5mu` at `path`, in its `mod-order` when it has
/// one, else in storage order.
pub fn modalities(path: &Path) -> Result<Vec<String>> {
    list(&File::open(path)?.group("/")?)
}

fn list(file: &Group) -> Result<Vec<String>> {
    let grp = file
        .group("mod")
        .map_err(|_| ScxError::InvalidFormat("not MuData: no /mod group".into()))?;
    let groups: Vec<String> = grp
        .member_names()?
        .into_iter()
        .filter(|n| grp.group(n).is_ok())
        .collect();
    match read_str_array_attr(&grp, "mod-order") {
        Ok(order) if !order.is_empty() => {
            Ok(order.into_iter().filter(|n| groups.contains(n)).collect())
        }
        _ => Ok(groups),
    }
}

/// The modality to read: `want` matched exactly, then case-insensitively,
/// then as a Seurat assay name (`ADT` is `prot`). With no `want`, the sole
/// modality, else `rna`; otherwise an error naming the choices.
pub fn resolve_modality(file: &Group, want: Option<&str>) -> Result<String> {
    let mods = list(file)?;
    let find = |w: &str| {
        mods.iter()
            .find(|m| *m == w)
            .or_else(|| mods.iter().find(|m| m.eq_ignore_ascii_case(w)))
            .cloned()
    };
    let found = match want {
        Some(w) => find(w).or_else(|| {
            ASSAY_ALIASES
                .iter()
                .filter(|(a, _)| a.eq_ignore_ascii_case(w))
                .find_map(|(_, m)| find(m))
        }),
        None if mods.len() == 1 => Some(mods[0].clone()),
        None => find("rna"),
    };
    found.ok_or_else(|| {
        let what = want.map_or("several modalities".into(), |w| {
            format!("no modality '{w}'")
        });
        ScxError::InvalidFormat(format!(
            "h5mu: {what} in /mod ({}); pick one with --assay <name>",
            mods.join(", ")
        ))
    })
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use crate::detect::{detect, Format};
    use crate::h5_str::{write_str_array_attr, write_str_attr, write_strings};
    use crate::OpenOptions;

    /// A dataframe group `name` under `g` with index `ids` and no columns.
    fn df(g: &hdf5::Group, name: &str, ids: &[&str]) {
        let d = g.create_group(name).unwrap();
        write_str_attr(&d, "encoding-type", "dataframe").unwrap();
        write_str_attr(&d, "_index", "_index").unwrap();
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        write_strings(&d, "_index", &ids).unwrap();
    }

    /// An `.h5mu` with a 2×3 CSR `rna` and a 2×2 dense `prot`, listed in
    /// `mod-order` as prot, rna. The root carries MuData obs/var too, which
    /// must not be what a modality reads.
    fn write_h5mu(path: &std::path::Path) {
        let f = hdf5::File::create(path).unwrap();
        write_str_attr(&f, "encoding-type", "MuData").unwrap();
        df(&f, "obs", &["c1", "c2", "c3"]);
        df(&f, "var", &["g1", "g2", "g3", "p1", "p2"]);
        let m = f.create_group("mod").unwrap();
        write_str_array_attr(&m, "mod-order", &["prot", "rna"]).unwrap();

        let rna = m.create_group("rna").unwrap();
        write_str_attr(&rna, "encoding-type", "anndata").unwrap();
        let x = rna.create_group("X").unwrap();
        write_str_attr(&x, "encoding-type", "csr_matrix").unwrap();
        x.new_attr_builder()
            .with_data(&[2i64, 3])
            .create("shape")
            .unwrap();
        x.new_dataset_builder()
            .with_data(&[0i32, 1, 3])
            .create("indptr")
            .unwrap();
        x.new_dataset_builder()
            .with_data(&[2i32, 0, 1])
            .create("indices")
            .unwrap();
        x.new_dataset_builder()
            .with_data(&[5f32, 6., 7.])
            .create("data")
            .unwrap();
        df(&rna, "obs", &["c1", "c2"]);
        df(&rna, "var", &["g1", "g2", "g3"]);

        let prot = m.create_group("prot").unwrap();
        prot.new_dataset_builder()
            .with_data(&ndarray::array![[1f32, 0.], [0., 2.]])
            .create("X")
            .unwrap();
        df(&prot, "obs", &["c2", "c3"]);
        df(&prot, "var", &["p1", "p2"]);
    }

    #[tokio::test]
    async fn h5mu_streams_one_modality() {
        let tmp = tempfile::NamedTempFile::with_suffix(".h5mu").unwrap();
        write_h5mu(tmp.path());
        assert_eq!(detect(tmp.path()), Some(Format::H5Mu));
        assert_eq!(super::modalities(tmp.path()).unwrap(), ["prot", "rna"]);

        // Default: rna. `layer` is the CLI's H5Seurat "counts" default and is
        // not used, as for h5ad.
        let input = tmp.path().to_str().unwrap();
        let mut opts = OpenOptions::new(1);
        opts.layer = Some("counts".into());
        let mut r = crate::open(input, &opts).await.unwrap();
        assert_eq!(r.shape(), (2, 3));
        assert_eq!(r.obs().await.unwrap().index, ["c1", "c2"]);
        assert_eq!(r.var().await.unwrap().index, ["g1", "g2", "g3"]);
        let chunks: Vec<_> = r.x_stream().map(|c| c.unwrap()).collect().await;
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].row_offset, 1);
        assert_eq!(chunks[1].data.indices, [0, 1]);
        assert_eq!(chunks[1].data.data.to_f64(), [6., 7.]);

        // A Seurat assay name picks its modality.
        opts.assay = Some("ADT".into());
        let mut r = crate::open(input, &opts).await.unwrap();
        assert_eq!(r.shape(), (2, 2));
        assert_eq!(r.obs().await.unwrap().index, ["c2", "c3"]);

        opts.assay = Some("atac".into());
        let err = crate::open(input, &opts).await.err().unwrap().to_string();
        assert!(err.contains("prot, rna"), "{err}");
    }

    /// Without `mod-order` every modality is listed, in storage order; a modality
    /// group that declares another encoding than AnnData is refused.
    #[test]
    fn storage_order_and_non_anndata_modality() {
        let tmp = tempfile::NamedTempFile::with_suffix(".h5mu").unwrap();
        write_h5mu(tmp.path());
        {
            let f = hdf5::File::open_rw(tmp.path()).unwrap();
            f.group("mod").unwrap().delete_attr("mod-order").unwrap();
            let rna = f.group("mod/rna").unwrap();
            rna.delete_attr("encoding-type").unwrap();
            write_str_attr(&rna, "encoding-type", "MuData").unwrap();
        }
        // Storage order depends on the HDF5 build (name or creation order).
        let mut mods = super::modalities(tmp.path()).unwrap();
        mods.sort();
        assert_eq!(mods, ["prot", "rna"]);
        let err = crate::h5ad::H5AdReader::open_modality(tmp.path(), 1, Some("rna"), None)
            .err()
            .unwrap()
            .to_string();
        assert!(
            err.contains("not AnnData: /mod/rna encoding-type = 'MuData'"),
            "{err}"
        );
    }
}
