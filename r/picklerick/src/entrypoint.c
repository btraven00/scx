/* entrypoint.c
 *
 * R calls R_init_picklerick() when loading the shared library.
 * We bridge to the extendr-generated init, which registers all
 * Rust functions exposed via extendr_module!{ mod picklerick; ... }.
 *
 * Note: the H5Literate shim that previously lived here (for HDF5 1.10.x
 * dynamic linking) is no longer needed: we now statically link HDF5 via
 * hdf5-sys/features=["static"], which bundles HDF5 1.14.x with H5Literate
 * as a proper exported symbol.
 */
#include <R_ext/Rdynload.h>

void R_init_picklerick_extendr(DllInfo *dll);

void R_init_picklerick(DllInfo *dll) {
    R_init_picklerick_extendr(dll);
}
