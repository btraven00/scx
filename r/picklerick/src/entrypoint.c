/* entrypoint.c
 *
 * R calls R_init_picklerick() when loading the shared library.
 * We bridge to the extendr-generated init, which registers all
 * Rust functions exposed via extendr_module!{ mod picklerick; ... }.
 */
#include <R_ext/Rdynload.h>

void R_init_picklerick_extendr(DllInfo *dll);

void R_init_picklerick(DllInfo *dll) {
    R_init_picklerick_extendr(dll);
}

/* -----------------------------------------------------------------------
 * HDF5 API version shim
 *
 * hdf5-sys (via bindgen) generates an extern reference to `H5Literate`,
 * but HDF5 1.10.x only exports `H5Literate1` — the unversioned name is a
 * preprocessor macro, not a real symbol.  Provide a thin wrapper so the
 * linker can resolve the reference.
 * ----------------------------------------------------------------------- */
#include <hdf5.h>

/* Remove the macro so we can define a real function with this name. */
#undef H5Literate

herr_t H5Literate(hid_t grp_id, H5_index_t idx_type, H5_iter_order_t order,
                   hsize_t *idx, H5L_iterate1_t op, void *op_data) {
    return H5Literate1(grp_id, idx_type, order, idx, op, op_data);
}
