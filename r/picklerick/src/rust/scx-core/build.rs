//! Detect the HDF5 ABI version we actually link against and expose it as a cfg.
//!
//! `h5_chunk.rs` calls the raw chunk-read symbol, which is ABI-versioned: HDF5
//! < 2.0 exports the 5-arg `H5Dread_chunk1` while HDF5 >= 2.0 drops it and
//! exports only the 6-arg `H5Dread_chunk2`. Our own `vendored-hdf5` feature is
//! NOT a reliable discriminator: it selects static-vs-system *linking*, but a
//! distro/conda build can disable it and still link an HDF5 2.x, or keep it on
//! while hdf5-metno-sys picks up an older system library via `HDF5_DIR`.
//!
//! hdf5-metno-sys (`links = "hdf5"`) detects the real library version and emits
//! `cargo::metadata=version_M_m_p=1` for every known version <= the one it built
//! or found. Cargo forwards those to direct dependents as `DEP_HDF5_VERSION_*`,
//! so the presence of `DEP_HDF5_VERSION_2_0_0` tells us the linked HDF5 is >= 2.0
//! and exports `H5Dread_chunk2`. Gate the chunk read on that, not on a feature.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(hdf5_2_0)");
    println!("cargo::rerun-if-env-changed=DEP_HDF5_VERSION_2_0_0");
    if std::env::var_os("DEP_HDF5_VERSION_2_0_0").is_some() {
        println!("cargo::rustc-cfg=hdf5_2_0");
    }
}
