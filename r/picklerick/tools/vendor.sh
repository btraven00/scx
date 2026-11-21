#!/bin/sh
# Vendor the Rust dependencies into src/rust/vendor.tar.xz.NN, so the package
# builds offline (CRAN/Bioconductor policy). Run before `R CMD build`:
#
#   sh tools/vendor.sh && R CMD build .
#
# src/Makevars uses them when present, crates.io otherwise. The archive is
# split in 4 MB parts: Bioconductor caps every file at 5 MB.
# To keep the tarball small it:
#   - stubs crates that no R platform compiles (wasm, other OSes): manifest
#     plus an empty lib.rs, which is all cargo needs to resolve the lockfile;
#   - drops tests and benches of the crates kept;
#   - drops winapi's bundled MinGW import libraries (~100 MB). A Windows build
#     must set WINAPI_NO_BUNDLED_LIBRARIES=1 to link Rtools' own instead.
# Checksums of individual files are blanked (cargo then checks only the
# package checksum from Cargo.lock).
set -eu

cd "$(dirname "$0")/../src/rust"
out=$(pwd)
# Targets R ships binaries for: CRAN/Bioconductor Linux, macOS, Rtools.
PLATFORMS="x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu
x86_64-apple-darwin aarch64-apple-darwin
x86_64-pc-windows-gnu aarch64-pc-windows-gnullvm"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cargo vendor --quiet --locked --versioned-dirs "$tmp/vendor" >/dev/null

for p in $PLATFORMS; do
    cargo tree --locked --target "$p" -e normal,build --prefix none --format '{p}' |
        awk '{sub(/^v/, "", $2); print $1 "-" $2}'
done | sort -u >"$tmp/keep"
# windows-sys API modules (src/Windows/Win32/Storage/FileSystem/...) of the
# features enabled, e.g. Win32_Storage_FileSystem.
for p in $PLATFORMS; do
    cargo tree --locked --target "$p" -e features --prefix none -i windows-sys 2>/dev/null || true
done | sed -n 's/^windows-sys feature "\([A-Z][^"]*\)".*/\1/p' | tr _ / | sort -u >"$tmp/winfeat"

cd "$tmp/vendor"
for d in */; do
    d=${d%/}
    if grep -qxF "$d" "$tmp/keep"; then
        # Not docs or examples: crates include_str! them.
        rm -rf "$d/tests" "$d/benches" "$d/fuzz" "$d/.github"
    else
        find "$d" -mindepth 1 -maxdepth 1 ! -name Cargo.toml \
            ! -name .cargo-checksum.json -exec rm -rf {} +
        mkdir "$d/src" && : >"$d/src/lib.rs"
        # Point the manifest's lib and build script (cargo rejects `links`
        # without one) at the stubs.
        echo 'fn main() {}' >"$d/build.rs"
        sed -i.bak -e 's|^build *=.*|build = "build.rs"|' \
            -e 's|^path *= *"[^"]*"|path = "src/lib.rs"|' "$d/Cargo.toml"
        rm -f "$d/Cargo.toml.bak"
    fi
    sed -i.bak 's/"files":{[^}]*}/"files":{}/' "$d/.cargo-checksum.json"
    rm -f "$d/.cargo-checksum.json.bak"
done
rm -f winapi-*-pc-windows-gnu-*/lib/*.a
# windows-sys is ~250 API modules, each behind its own feature; drop the
# modules of disabled ones (1.9 MB of the xz otherwise).
for ws in windows-sys-*; do
    [ -d "$ws/src/Windows" ] || continue
    (cd "$ws/src/Windows" && find . -mindepth 2 -name '*.rs' | while read -r f; do
        dir=$(dirname "${f#./}")
        grep -qxF "$dir" "$tmp/winfeat" || rm "$f"
    done && find . -type d -empty -delete)
done
rm -f "$out"/vendor.tar.xz.*
XZ_OPT=-9e tar -cJf - -C "$tmp" vendor | split -b 4m -d -a 2 - "$out/vendor.tar.xz."
echo "src/rust/vendor.tar.xz.*: $(cat "$out"/vendor.tar.xz.* | wc -c | awk '{printf "%.1f MB", $1 / 1e6}') in $(ls "$out"/vendor.tar.xz.* | wc -l) parts, $(grep -c . "$tmp/keep") crates kept"
