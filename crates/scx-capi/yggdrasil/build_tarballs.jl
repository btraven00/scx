# Draft for Yggdrasil: S/scx_capi/build_tarballs.jl -> scx_capi_jll (used by ScxStream.jl).
# Fill in the commit of the scx release tag before opening the PR.
using BinaryBuilder, Pkg

name = "scx_capi"
version = v"0.1.0"

sources = [
    GitSource("https://github.com/btraven00/scx.git", "COMMIT_SHA_OF_RELEASE_TAG"),
]

# HDF5 comes from HDF5_jll (no vendored HDF5 build); hdf5-metno-sys finds it via HDF5_DIR.
script = raw"""
cd ${WORKSPACE}/srcdir/scx
export HDF5_DIR=${prefix}
cargo build --release -p scx-capi --no-default-features
install -Dvm 755 target/${rust_target}/release/*scx_capi.${dlext} -t ${libdir}
install_license LICENSE*
"""

# Rust + HDF5_jll intersection; prune what fails on the Yggdrasil CI.
platforms = supported_platforms(; exclude=p -> Sys.isfreebsd(p) || arch(p) in ("i686", "armv6l", "armv7l", "powerpc64le", "riscv64"))

products = [
    LibraryProduct(["libscx_capi", "scx_capi"], :libscx_capi),
]

dependencies = [
    Dependency("HDF5_jll"; compat="~1.14.6"),
]

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               julia_compat="1.6", compilers=[:c, :rust], preferred_rust_version=v"1.91")
