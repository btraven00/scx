//! Minimal C ABI over scx-core's row-chunk stream, for Julia `ccall` (ScxStream.jl).
//!
//! Same shape as picklerick's `open_stream`: a background thread drives the
//! reader and hands chunks over a bounded channel. Each chunk is copied into
//! caller-owned buffers (1-based Int32 indices, Float32 values) so Julia can
//! wrap it directly as a `SparseMatrixCSC` (CSR cells×genes == CSC genes×cells).

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::sync::mpsc::{sync_channel, Receiver};

use futures::executor::block_on;
use futures::StreamExt;
use scx_core::dtype::TypedVec;
use scx_core::ir::MatrixChunk;

pub struct ScxStream {
    rx: Receiver<Result<MatrixChunk, String>>,
    n_obs: usize,
    n_vars: usize,
    cur: Option<MatrixChunk>,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: impl ToString) {
    let msg = CString::new(msg.to_string().replace('\0', " ")).unwrap();
    LAST_ERROR.with(|e| *e.borrow_mut() = msg);
}

/// Message of the last failed call on this thread (valid until the next failure).
#[no_mangle]
pub extern "C" fn scx_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Open `path` (any format scx detects) and start streaming X in chunks of
/// `chunk_size` cells. Returns null on error; see `scx_last_error`.
///
/// # Safety
/// `path` must be a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn scx_open(path: *const c_char, chunk_size: usize) -> *mut ScxStream {
    let path = match CStr::from_ptr(path).to_str() {
        Ok(p) => p,
        Err(e) => {
            set_error(e);
            return std::ptr::null_mut();
        }
    };
    let opts = scx_core::OpenOptions::new(chunk_size);
    let mut reader = match block_on(scx_core::open(path, &opts)) {
        Ok(r) => r,
        Err(e) => {
            set_error(e);
            return std::ptr::null_mut();
        }
    };
    let (n_obs, n_vars) = reader.shape();
    // One chunk queued + one being read: enough to overlap reading with the
    // consumer's compute while keeping memory flat. Depth is in chunks, not
    // bytes, and an atlas chunk can be ~100 MB, so deeper read-ahead costs.
    let (tx, rx) = sync_channel(1);
    std::thread::spawn(move || {
        block_on(async move {
            let mut stream = reader.x_stream();
            while let Some(chunk) = stream.next().await {
                if tx.send(chunk.map_err(|e| e.to_string())).is_err() {
                    break; // consumer closed the stream
                }
            }
        });
    });
    Box::into_raw(Box::new(ScxStream {
        rx,
        n_obs,
        n_vars,
        cur: None,
    }))
}

/// # Safety
/// `h` must come from `scx_open`.
#[no_mangle]
pub unsafe extern "C" fn scx_shape(h: *const ScxStream, n_obs: *mut usize, n_vars: *mut usize) {
    *n_obs = (*h).n_obs;
    *n_vars = (*h).n_vars;
}

/// Advance to the next chunk. Returns 1 and sets its row offset, row count and
/// nnz; 0 at end of stream; -1 on error (see `scx_last_error`), including a
/// chunk too large for Int32 indices.
///
/// # Safety
/// `h` must come from `scx_open`; out-pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn scx_next(
    h: *mut ScxStream,
    row_offset: *mut usize,
    nrows: *mut usize,
    nnz: *mut usize,
) -> i32 {
    let h = &mut *h;
    h.cur = None;
    match h.rx.recv() {
        Ok(Ok(chunk)) => {
            *row_offset = chunk.row_offset;
            *nrows = chunk.nrows;
            *nnz = chunk.data.indices.len();
            if *nnz >= i32::MAX as usize {
                set_error(format!(
                    "chunk has {} nonzeros, more than Int32 indices hold; use a smaller chunk size",
                    *nnz
                ));
                return -1;
            }
            h.cur = Some(chunk);
            1
        }
        Ok(Err(msg)) => {
            set_error(msg);
            -1
        }
        Err(_) => 0, // reader thread finished
    }
}

/// Copy the current chunk into caller buffers of length nrows+1, nnz, nnz.
/// Indptr and indices are shifted to 1-based.
///
/// # Safety
/// `h` must have a current chunk from `scx_next`; buffers must have the sizes above.
#[no_mangle]
pub unsafe extern "C" fn scx_copy(
    h: *const ScxStream,
    indptr: *mut i32,
    indices: *mut i32,
    data: *mut f32,
) {
    let csr = &(*h)
        .cur
        .as_ref()
        .expect("scx_copy without a current chunk")
        .data;
    let nnz = csr.indices.len();
    let indptr = std::slice::from_raw_parts_mut(indptr, csr.indptr.len());
    let base = csr.indptr[0];
    for (o, &p) in indptr.iter_mut().zip(&csr.indptr) {
        *o = (p - base) as i32 + 1;
    }
    let indices = std::slice::from_raw_parts_mut(indices, nnz);
    for (o, &i) in indices.iter_mut().zip(&csr.indices) {
        *o = i as i32 + 1;
    }
    // ponytail: always Float32, which is what OnlinePCA computes in; counts are
    // exact below 2^24. Add a dtype tag if f64 X ever needs to survive.
    let data = std::slice::from_raw_parts_mut(data, nnz);
    match &csr.data {
        TypedVec::F32(v) => data.copy_from_slice(v),
        TypedVec::F64(v) => data.iter_mut().zip(v).for_each(|(o, &x)| *o = x as f32),
        TypedVec::I32(v) => data.iter_mut().zip(v).for_each(|(o, &x)| *o = x as f32),
        TypedVec::U32(v) => data.iter_mut().zip(v).for_each(|(o, &x)| *o = x as f32),
    }
}

/// Stop streaming and free the handle. Safe to call before the stream ends.
///
/// # Safety
/// `h` must come from `scx_open` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn scx_close(h: *mut ScxStream) {
    if !h.is_null() {
        drop(Box::from_raw(h)); // drops rx → reader thread's send fails → it exits
    }
}

/// Cell (`which == 0`) or gene (`which == 1`) names of `path`, joined by '\n'.
/// Returns null on error (see `scx_last_error`); free the result with
/// `scx_free_string`.
///
/// # Safety
/// `path` must be a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn scx_names(path: *const c_char, which: i32) -> *mut c_char {
    let run = || -> Result<CString, String> {
        let path = CStr::from_ptr(path).to_str().map_err(|e| e.to_string())?;
        let mut reader = block_on(scx_core::open(path, &scx_core::OpenOptions::new(1)))
            .map_err(|e| e.to_string())?;
        // ponytail: obs()/var() read the whole metadata table to get its index;
        // add an index-only reader to scx-core if a wide obs ever makes this slow.
        let names = match which {
            0 => block_on(reader.obs()).map_err(|e| e.to_string())?.index,
            1 => block_on(reader.var()).map_err(|e| e.to_string())?.index,
            _ => {
                return Err(format!(
                    "scx_names: which must be 0 (obs) or 1 (var), got {which}"
                ))
            }
        };
        if let Some(n) = names.iter().find(|n| n.contains('\n') || n.contains('\0')) {
            return Err(format!("name {n:?} contains a newline or NUL"));
        }
        CString::new(names.join("\n")).map_err(|e| e.to_string())
    };
    match run() {
        Ok(s) => s.into_raw(),
        Err(e) => {
            set_error(e);
            std::ptr::null_mut()
        }
    }
}

/// Free a string returned by `scx_names`.
///
/// # Safety
/// `s` must come from `scx_names` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn scx_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
