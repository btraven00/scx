/* Build-time stand-in, never linked into picklerick: hdf5-metno-sys only
 * accepts an HDF5_DIR once it has dlopen-ed a shared libhdf5 and checked that
 * its version matches the headers, and Rhdf5lib ships only the static
 * libhdf5.a. This reports the headers' version. HDF5 itself is linked from
 * Rhdf5lib through PKG_LIBS (see Makevars). */
#include "H5public.h"

herr_t H5open(void) { return 0; }

herr_t H5get_libversion(unsigned *majnum, unsigned *minnum, unsigned *relnum) {
    *majnum = H5_VERS_MAJOR;
    *minnum = H5_VERS_MINOR;
    *relnum = H5_VERS_RELEASE;
    return 0;
}
