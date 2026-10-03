//! Locate the large, locally generated fixtures under `tests/golden/`.
//!
//! They are gitignored (`pixi run -e test fixtures` builds them), so a test
//! that needs one does `let Some(path) = golden("x.h5ad") else { return };`.
//! The skip is printed even though libtest captures test output, and with
//! `SCX_REQUIRE_GOLDEN=1` a missing file fails the test instead, so a
//! pre-release run can't pass by skipping. `SCX_GOLDEN` overrides the
//! directory.
//!
//! Shared by scx-core's unit tests, its integration tests and scx-cli's tests
//! via `#[path]`; `CARGO_MANIFEST_DIR` is `crates/<crate>` for all of them.

use std::io::Write;
use std::path::PathBuf;

pub fn golden(name: &str) -> Option<PathBuf> {
    let dir = std::env::var_os("SCX_GOLDEN").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden"),
        PathBuf::from,
    );
    let path = dir.join(name);
    if path.exists() {
        return Some(path);
    }
    assert!(
        std::env::var_os("SCX_REQUIRE_GOLDEN").is_none(),
        "golden fixture missing: {} (SCX_REQUIRE_GOLDEN is set)",
        path.display()
    );
    // Write to the handle directly: libtest captures eprintln!, not this.
    let _ = writeln!(
        std::io::stderr(),
        "SKIP: golden fixture missing: {}",
        path.display()
    );
    None
}
