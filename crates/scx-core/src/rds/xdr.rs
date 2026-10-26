//! Skeleton parser for R's XDR serialization stream (`saveRDS`, `save`).
//!
//! The byte layout follows R's documentation of the format (the "R Internals"
//! manual, *Serialization Formats*) and the behaviour of `ReadItem`/`ReadBC`
//! in R's `src/main/serialize.c`. It is written from that description; no R
//! (GPL) source is copied. The Python `rdata` package (MIT,
//! <https://github.com/vnmabus/rdata>) was read as a second implementation to
//! check the layout against; none of its code is used here.
//!
//! One forward pass builds a [`Node`] tree. Numeric vectors longer than
//! [`LAZY_MIN`] are not read: the node keeps their offset in the decompressed
//! stream so a later pass can fetch only what it needs. Closures,
//! environments, bytecode and other language objects are parsed (the stream
//! has no length prefixes, so they cannot be skipped) and then dropped.

use std::io::Read;

use crate::error::{Result, ScxError};

/// Numeric vectors above this many elements are recorded by offset, not read.
pub(crate) const LAZY_MIN: u64 = 4096;

// SEXP type codes (R Internals, "SEXPTYPEs"; serialization-only codes from
// "Serialization Formats").
const NILSXP: u8 = 0;
const SYMSXP: u8 = 1;
const LISTSXP: u8 = 2;
const CLOSXP: u8 = 3;
const ENVSXP: u8 = 4;
const PROMSXP: u8 = 5;
const LANGSXP: u8 = 6;
const SPECIALSXP: u8 = 7;
const BUILTINSXP: u8 = 8;
const CHARSXP: u8 = 9;
const LGLSXP: u8 = 10;
const INTSXP: u8 = 13;
const REALSXP: u8 = 14;
const CPLXSXP: u8 = 15;
const STRSXP: u8 = 16;
const DOTSXP: u8 = 17;
const VECSXP: u8 = 19;
const EXPRSXP: u8 = 20;
const BCODESXP: u8 = 21;
const EXTPTRSXP: u8 = 22;
const WEAKREFSXP: u8 = 23;
const RAWSXP: u8 = 24;
const S4SXP: u8 = 25;
const ALTREP_SXP: u8 = 238;
const ATTRLISTSXP: u8 = 239;
const ATTRLANGSXP: u8 = 240;
const BASEENV_SXP: u8 = 241;
const EMPTYENV_SXP: u8 = 242;
const BCREPREF: u8 = 243;
const BCREPDEF: u8 = 244;
const GENERICREFSXP: u8 = 245;
const CLASSREFSXP: u8 = 246;
const PERSISTSXP: u8 = 247;
const PACKAGESXP: u8 = 248;
const NAMESPACESXP: u8 = 249;
const BASENAMESPACE_SXP: u8 = 250;
const MISSINGARG_SXP: u8 = 251;
const UNBOUNDVALUE_SXP: u8 = 252;
const GLOBALENV_SXP: u8 = 253;
const NILVALUE_SXP: u8 = 254;
const REFSXP: u8 = 255;

const HAS_ATTR: i32 = 1 << 9;
const HAS_TAG: i32 = 1 << 10;

/// R's NA_integer_ / NA for logicals.
pub(crate) const NA_INT: i32 = i32::MIN;

/// Element type of a numeric vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Num {
    Lgl,
    Int,
    Real,
}

impl Num {
    pub(crate) fn width(self) -> u64 {
        match self {
            Num::Lgl | Num::Int => 4,
            Num::Real => 8,
        }
    }
}

/// A parsed R object, payload-light.
#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub kind: Kind,
    /// Attributes (for S4 objects: the slots), in stream order.
    pub attr: Vec<(String, Node)>,
}

#[derive(Debug, Clone)]
pub(crate) enum Kind {
    Null,
    Sym(String),
    Str(Vec<Option<String>>),
    /// Integer vector (NA = [`NA_INT`]).
    Int(Vec<i32>),
    /// Logical vector: 0, 1, or NA = [`NA_INT`].
    Lgl(Vec<i32>),
    Real(Vec<f64>),
    /// A numeric vector left in the stream: `len` elements at `offset`.
    Lazy {
        num: Num,
        offset: u64,
        len: u64,
    },
    /// Generic vector (`list()`) or pairlist; pairlist entries carry their tag.
    List(Vec<(Option<String>, Node)>),
    S4,
    /// Anything else (closures, environments, bytecode, ...), contents dropped.
    Other,
}

impl Node {
    fn new(kind: Kind) -> Self {
        Self {
            kind,
            attr: Vec::new(),
        }
    }

    pub(crate) fn attr(&self, name: &str) -> Option<&Node> {
        self.attr.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    /// The `class` attribute, empty when there is none.
    pub(crate) fn class(&self) -> Vec<&str> {
        match self.attr("class").map(|c| &c.kind) {
            Some(Kind::Str(v)) => v.iter().flatten().map(String::as_str).collect(),
            _ => Vec::new(),
        }
    }

    pub(crate) fn has_class(&self, name: &str) -> bool {
        self.class().contains(&name)
    }

    /// Elements of a list, or nothing.
    pub(crate) fn items(&self) -> &[(Option<String>, Node)] {
        match &self.kind {
            Kind::List(v) => v,
            _ => &[],
        }
    }

    /// The `names` attribute of a vector, as strings.
    pub(crate) fn names(&self) -> Vec<String> {
        match self.attr("names").map(|n| &n.kind) {
            Some(Kind::Str(v)) => v.iter().map(|s| s.clone().unwrap_or_default()).collect(),
            _ => Vec::new(),
        }
    }

    /// Element `name` of a named list.
    pub(crate) fn get(&self, name: &str) -> Option<&Node> {
        let i = self.names().iter().position(|n| n == name)?;
        self.items().get(i).map(|(_, v)| v)
    }

    /// A NULL value, or the pseudo-NULL symbol R stores in empty S4 slots.
    pub(crate) fn is_null(&self) -> bool {
        match &self.kind {
            Kind::Null => true,
            Kind::Sym(s) => s == "\u{1}NULL\u{1}",
            _ => false,
        }
    }

    /// Element count of a vector node (lazy or not).
    pub(crate) fn len(&self) -> usize {
        match &self.kind {
            Kind::Str(v) => v.len(),
            Kind::Int(v) | Kind::Lgl(v) => v.len(),
            Kind::Real(v) => v.len(),
            Kind::Lazy { len, .. } => *len as usize,
            Kind::List(v) => v.len(),
            _ => 0,
        }
    }
}

/// A forward-only reader over the decompressed stream that tracks its offset.
pub(crate) struct Xdr<R> {
    r: R,
    pub pos: u64,
}

impl<R: Read> Xdr<R> {
    pub(crate) fn new(r: R) -> Self {
        Self { r, pos: 0 }
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.r.read_exact(&mut buf).map_err(truncated)?;
        self.pos += n as u64;
        Ok(buf)
    }

    fn int(&mut self) -> Result<i32> {
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b).map_err(truncated)?;
        self.pos += 4;
        Ok(i32::from_be_bytes(b))
    }

    /// Vector length; `-1` announces a long vector as two 32-bit halves.
    fn length(&mut self) -> Result<u64> {
        let n = self.int()?;
        if n == -1 {
            let hi = self.int()? as u32 as u64;
            let lo = self.int()? as u32 as u64;
            return Ok((hi << 32) | lo);
        }
        u64::try_from(n).map_err(|_| bad(format!("negative vector length {n}")))
    }

    /// Skip forward to `offset` (must not be behind the current position).
    pub(crate) fn seek_to(&mut self, offset: u64) -> Result<()> {
        if offset < self.pos {
            return Err(bad("rds reader asked to move backwards".into()));
        }
        let n = offset - self.pos;
        let copied =
            std::io::copy(&mut (&mut self.r).take(n), &mut std::io::sink()).map_err(truncated)?;
        if copied != n {
            return Err(truncated(std::io::ErrorKind::UnexpectedEof.into()));
        }
        self.pos = offset;
        Ok(())
    }

    /// Read `len` big-endian 32-bit integers.
    pub(crate) fn ints(&mut self, len: usize) -> Result<Vec<i32>> {
        Ok(be_i32s(&self.bytes(len * 4)?))
    }

    /// Read `len` big-endian doubles.
    pub(crate) fn reals(&mut self, len: usize) -> Result<Vec<f64>> {
        Ok(be_f64s(&self.bytes(len * 8)?))
    }

    /// The `X\n` marker and version header. A `.rda` (`save()`) file carries an
    /// extra `RDX2\n`/`RDX3\n` line first; its object is a pairlist of the
    /// saved variables.
    pub(crate) fn header(&mut self) -> Result<()> {
        let mut magic = self.bytes(2)?;
        if magic == b"RD" {
            let rest = self.bytes(3)?;
            if rest != b"X2\n" && rest != b"X3\n" {
                return Err(bad("only XDR .rda files are supported".into()));
            }
            magic = self.bytes(2)?;
        }
        if magic != b"X\n" {
            return Err(bad(
                "not an XDR R serialization stream (ASCII and native-binary saves are unsupported)"
                    .into(),
            ));
        }
        let version = self.int()?;
        let _writer_r_version = self.int()?;
        let _min_r_version = self.int()?;
        match version {
            2 => {}
            3 => {
                let n = self.int()?;
                self.bytes(n.max(0) as usize)?; // native encoding name
            }
            v => return Err(bad(format!("unsupported serialization format version {v}"))),
        }
        Ok(())
    }
}

/// Parser state: the reference table of objects the stream can point back to.
pub(crate) struct Parser<'a, R> {
    pub x: &'a mut Xdr<R>,
    refs: Vec<Node>,
}

impl<'a, R: Read> Parser<'a, R> {
    pub(crate) fn new(x: &'a mut Xdr<R>) -> Self {
        Self {
            x,
            refs: Vec::new(),
        }
    }

    pub(crate) fn item(&mut self) -> Result<Node> {
        let flags = self.x.int()?;
        self.item_with(flags)
    }

    fn item_with(&mut self, flags: i32) -> Result<Node> {
        let ty = (flags & 0xFF) as u8;
        let has_attr = flags & HAS_ATTR != 0;
        let mut node = match ty {
            NILVALUE_SXP => return Ok(Node::new(Kind::Null)),
            EMPTYENV_SXP | BASEENV_SXP | GLOBALENV_SXP | UNBOUNDVALUE_SXP | MISSINGARG_SXP
            | BASENAMESPACE_SXP => return Ok(Node::new(Kind::Other)),
            REFSXP => {
                let packed = (flags >> 8) as u32 as usize;
                let idx = if packed == 0 {
                    self.x.int()? as usize
                } else {
                    packed
                };
                return self
                    .refs
                    .get(idx.wrapping_sub(1))
                    .cloned()
                    .ok_or_else(|| bad(format!("reference {idx} out of range")));
            }
            PERSISTSXP | PACKAGESXP | NAMESPACESXP => {
                if self.x.int()? != 0 {
                    return Err(bad("malformed persistent-name string vector".into()));
                }
                let n = self.x.int()?;
                for _ in 0..n {
                    self.item()?;
                }
                let node = Node::new(Kind::Other);
                self.refs.push(node.clone());
                return Ok(node);
            }
            SYMSXP => {
                let name = match self.item()?.kind {
                    Kind::Str(mut v) => v.pop().flatten().unwrap_or_default(),
                    _ => String::new(),
                };
                let node = Node::new(Kind::Sym(name));
                self.refs.push(node.clone());
                return Ok(node);
            }
            ENVSXP => {
                // Registered before its contents, which may refer back to it.
                self.refs.push(Node::new(Kind::Other));
                self.x.int()?; // locked
                for _ in 0..4 {
                    self.item()?; // enclosure, frame, hash table, attributes
                }
                return Ok(Node::new(Kind::Other));
            }
            LISTSXP | LANGSXP | CLOSXP | PROMSXP | DOTSXP | ATTRLISTSXP | ATTRLANGSXP => {
                return self.pairlist(flags);
            }
            EXTPTRSXP => {
                self.refs.push(Node::new(Kind::Other));
                self.item()?; // protected value
                self.item()?; // tag
                Node::new(Kind::Other)
            }
            WEAKREFSXP => {
                self.refs.push(Node::new(Kind::Other));
                Node::new(Kind::Other)
            }
            SPECIALSXP | BUILTINSXP => {
                let n = self.x.int()?;
                self.x.bytes(n.max(0) as usize)?;
                Node::new(Kind::Other)
            }
            CHARSXP => {
                // One-element Str so SYMSXP and STRSXP share the decode. Encoding
                // flags (UTF-8 / latin1 / bytes) live in the gp bits; latin1 is
                // decoded lossily, which only matters for non-ASCII legacy names.
                let n = self.x.int()?;
                let s =
                    if n == -1 {
                        None
                    } else {
                        let raw = self.x.bytes(n.max(0) as usize)?;
                        Some(String::from_utf8(raw).unwrap_or_else(|e| {
                            e.into_bytes().iter().map(|&b| b as char).collect()
                        }))
                    };
                Node::new(Kind::Str(vec![s]))
            }
            LGLSXP | INTSXP | REALSXP => {
                let num = match ty {
                    LGLSXP => Num::Lgl,
                    INTSXP => Num::Int,
                    _ => Num::Real,
                };
                let len = self.x.length()?;
                if len > LAZY_MIN {
                    let offset = self.x.pos;
                    self.x.seek_to(offset + len * num.width())?;
                    Node::new(Kind::Lazy { num, offset, len })
                } else {
                    let n = len as usize;
                    Node::new(match num {
                        Num::Real => Kind::Real(self.x.reals(n)?),
                        Num::Int => Kind::Int(self.x.ints(n)?),
                        Num::Lgl => Kind::Lgl(self.x.ints(n)?),
                    })
                }
            }
            CPLXSXP | RAWSXP => {
                let len = self.x.length()?;
                let width = if ty == CPLXSXP { 16 } else { 1 };
                let to = self.x.pos + len * width;
                self.x.seek_to(to)?;
                Node::new(Kind::Other)
            }
            STRSXP => {
                let len = self.x.length()?;
                let mut v = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    match self.item()?.kind {
                        Kind::Str(mut s) => v.push(s.pop().flatten()),
                        _ => return Err(bad("character vector element is not a string".into())),
                    }
                }
                Node::new(Kind::Str(v))
            }
            VECSXP | EXPRSXP => {
                let len = self.x.length()?;
                let mut v = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    v.push((None, self.item()?));
                }
                Node::new(Kind::List(v))
            }
            BCODESXP => {
                let n_reps = self.x.int()?;
                let mut reps = vec![false; n_reps.max(0) as usize];
                self.bytecode(&mut reps)?;
                Node::new(Kind::Other)
            }
            S4SXP => Node::new(Kind::S4),
            ALTREP_SXP => {
                // info (class pairlist), state, attributes. The compact and
                // wrapper classes base R serializes this way are expanded only
                // as far as the reader needs: compact_intseq (e.g. 1:n).
                let info = self.item()?;
                let state = self.item()?;
                let attr = self.item()?;
                let class = info.items().first().and_then(|(_, n)| match &n.kind {
                    Kind::Sym(s) => Some(s.clone()),
                    _ => None,
                });
                let mut node = expand_altrep(class.as_deref(), state)?;
                node.attr = attr.into_pairs();
                return Ok(node);
            }
            GENERICREFSXP | CLASSREFSXP => {
                return Err(bad("obsolete reference type in stream".into()));
            }
            NILSXP => Node::new(Kind::Null),
            t => return Err(bad(format!("unknown SEXP type {t}"))),
        };
        if has_attr {
            node.attr = self.item()?.into_pairs();
        }
        Ok(node)
    }

    /// Pairlist-like cells: [attr] [tag] car cdr, with the cdr chain read
    /// iteratively so long pairlists don't recurse.
    fn pairlist(&mut self, mut flags: i32) -> Result<Node> {
        let mut cells = Vec::new();
        let mut first_attr = None;
        loop {
            let ty = (flags & 0xFF) as u8;
            let has_attr = flags & HAS_ATTR != 0 || ty == ATTRLISTSXP || ty == ATTRLANGSXP;
            if has_attr {
                let a = self.item()?;
                first_attr.get_or_insert(a);
            }
            let tag = if flags & HAS_TAG != 0 {
                match self.item()?.kind {
                    Kind::Sym(s) => Some(s),
                    _ => None,
                }
            } else {
                None
            };
            cells.push((tag, self.item()?));
            flags = self.x.int()?;
            match (flags & 0xFF) as u8 {
                LISTSXP | LANGSXP | CLOSXP | PROMSXP | DOTSXP | ATTRLISTSXP | ATTRLANGSXP => {}
                _ => {
                    let tail = self.item_with(flags)?;
                    if !matches!(tail.kind, Kind::Null) {
                        cells.push((None, tail));
                    }
                    break;
                }
            }
        }
        let mut node = Node::new(Kind::List(cells));
        if let Some(a) = first_attr {
            node.attr = a.into_pairs();
        }
        Ok(node)
    }

    /// Bytecode: code vector + constant pool (`ReadBC1` in serialize.c).
    fn bytecode(&mut self, reps: &mut [bool]) -> Result<()> {
        self.item()?; // code
        let n = self.x.int()?;
        for _ in 0..n {
            let ty = self.x.int()? as u8;
            match ty {
                BCODESXP => self.bytecode(reps)?,
                LANGSXP | LISTSXP | BCREPDEF | BCREPREF | ATTRLANGSXP | ATTRLISTSXP => {
                    self.bc_lang(ty, reps)?
                }
                _ => {
                    self.item()?;
                }
            }
        }
        Ok(())
    }

    /// Language objects inside a constant pool (`ReadBCLang`): cells may be
    /// defined once (BCREPDEF) and referenced later (BCREPREF).
    fn bc_lang(&mut self, ty: u8, reps: &mut [bool]) -> Result<()> {
        match ty {
            BCREPREF => {
                let i = self.x.int()? as usize;
                if !reps.get(i).copied().unwrap_or(false) {
                    return Err(bad("bytecode back-reference before its definition".into()));
                }
            }
            BCREPDEF | LANGSXP | LISTSXP | ATTRLANGSXP | ATTRLISTSXP => {
                let mut ty = ty;
                let mut pos = None;
                if ty == BCREPDEF {
                    pos = Some(self.x.int()? as usize);
                    ty = self.x.int()? as u8;
                }
                if ty == ATTRLANGSXP || ty == ATTRLISTSXP {
                    self.item()?;
                }
                self.item()?; // tag
                let car = self.x.int()? as u8;
                self.bc_lang(car, reps)?;
                let cdr = self.x.int()? as u8;
                self.bc_lang(cdr, reps)?;
                if let Some(p) = pos {
                    *reps
                        .get_mut(p)
                        .ok_or_else(|| bad("bytecode repeat index out of range".into()))? = true;
                }
            }
            _ => {
                // `WriteBCLang` pads a non-language constant with a 0 type word,
                // then writes it as an ordinary item.
                self.item()?;
            }
        }
        Ok(())
    }
}

impl Node {
    /// Attribute pairlist -> (name, value) pairs.
    fn into_pairs(self) -> Vec<(String, Node)> {
        match self.kind {
            Kind::List(cells) => cells
                .into_iter()
                .filter_map(|(t, v)| t.map(|t| (t, v)))
                .collect(),
            _ => Vec::new(),
        }
    }
}

fn expand_altrep(class: Option<&str>, state: Node) -> Result<Node> {
    match class {
        // state: REALSXP (n, start, step)
        Some("compact_intseq") => match state.kind {
            Kind::Real(v) if v.len() == 3 => {
                let (n, start, step) = (v[0] as i64, v[1] as i64, v[2] as i64);
                Ok(Node::new(Kind::Int(
                    (0..n).map(|k| (start + k * step) as i32).collect(),
                )))
            }
            _ => Err(bad("malformed compact_intseq".into())),
        },
        Some("compact_realseq") => match state.kind {
            Kind::Real(v) if v.len() == 3 => {
                let (n, start, step) = (v[0] as i64, v[1], v[2]);
                Ok(Node::new(Kind::Real(
                    (0..n).map(|k| start + k as f64 * step).collect(),
                )))
            }
            _ => Err(bad("malformed compact_realseq".into())),
        },
        // state: list(payload, metadata); the payload is the plain vector.
        Some(c) if c.starts_with("wrap_") => match state.kind {
            Kind::List(mut v) if !v.is_empty() => Ok(v.swap_remove(0).1),
            _ => Err(bad(format!("malformed {c}"))),
        },
        // state: pairlist(original numeric vector, scipen); the strings were
        // never materialized. Readers here never need them as text.
        Some("deferred_string") => Ok(Node::new(Kind::Other)),
        Some(c) => Err(bad(format!(
            "unsupported ALTREP class '{c}' (re-save the object after materializing it)"
        ))),
        None => Err(bad("ALTREP object without a class".into())),
    }
}

/// Big-endian 32-bit integers.
pub(crate) fn be_i32s(raw: &[u8]) -> Vec<i32> {
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_be_bytes(*c))
        .collect()
}

/// Big-endian doubles.
pub(crate) fn be_f64s(raw: &[u8]) -> Vec<f64> {
    raw.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_be_bytes(*c))
        .collect()
}

fn bad(msg: String) -> ScxError {
    ScxError::InvalidFormat(format!("rds: {msg}"))
}

fn truncated(e: std::io::Error) -> ScxError {
    ScxError::InvalidFormat(format!("rds: truncated or unreadable stream ({e})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny XDR stream writer for building fixtures in tests.
    #[derive(Default)]
    pub(crate) struct W(pub Vec<u8>);

    impl W {
        pub fn int(&mut self, v: i32) -> &mut Self {
            self.0.extend_from_slice(&v.to_be_bytes());
            self
        }
        pub fn head(&mut self) -> &mut Self {
            self.0.extend_from_slice(b"X\n");
            self.int(3).int(0x040601).int(0x030500).int(5);
            self.0.extend_from_slice(b"UTF-8");
            self
        }
        pub fn chars(&mut self, s: &str) -> &mut Self {
            self.int(CHARSXP as i32 | (1 << 15)).int(s.len() as i32);
            self.0.extend_from_slice(s.as_bytes());
            self
        }
        pub fn sym(&mut self, s: &str) -> &mut Self {
            self.int(SYMSXP as i32).chars(s)
        }
    }

    fn parse(bytes: &[u8]) -> Node {
        let mut x = Xdr::new(bytes);
        x.header().unwrap();
        let node = Parser::new(&mut x).item().unwrap();
        assert_eq!(x.pos as usize, bytes.len(), "stream not fully consumed");
        node
    }

    #[test]
    fn small_and_lazy_vectors() {
        let mut w = W::default();
        w.head().int(VECSXP as i32).int(2);
        w.int(INTSXP as i32).int(3).int(1).int(NA_INT).int(3);
        let n = LAZY_MIN as i32 + 1;
        w.int(REALSXP as i32).int(n);
        let payload_at = w.0.len();
        for k in 0..n {
            w.0.extend_from_slice(&(k as f64).to_be_bytes());
        }
        let node = parse(&w.0);
        let items = node.items();
        assert!(matches!(&items[0].1.kind, Kind::Int(v) if v == &[1, NA_INT, 3]));
        match items[1].1.kind {
            Kind::Lazy { num, offset, len } => {
                assert_eq!(
                    (num, offset as usize, len),
                    (Num::Real, payload_at, n as u64)
                );
            }
            ref k => panic!("expected lazy, got {k:?}"),
        }
    }

    #[test]
    fn attributes_refs_and_long_length() {
        // c(a = 1L) with names attribute; then a symbol and a back-reference to it.
        let mut w = W::default();
        w.head().int(VECSXP as i32).int(3);
        w.int(INTSXP as i32 | HAS_ATTR).int(1).int(7);
        w.int(LISTSXP as i32 | HAS_TAG).sym("names");
        w.int(STRSXP as i32).int(1).chars("a");
        w.int(NILVALUE_SXP as i32);
        w.sym("foo");
        w.int(REFSXP as i32 | (2 << 8)); // ref 1 = `names`, ref 2 = `foo`

        let node = parse(&w.0);
        let items = node.items();
        assert_eq!(items[0].1.names(), vec!["a"]);
        assert!(matches!(&items[2].1.kind, Kind::Sym(s) if s == "foo"));

        // Long-vector length: -1, then the high and low 32-bit halves.
        let mut w = W::default();
        w.head()
            .int(INTSXP as i32)
            .int(-1)
            .int(0)
            .int(2)
            .int(4)
            .int(5);
        assert!(matches!(parse(&w.0).kind, Kind::Int(v) if v == [4, 5]));
    }

    #[test]
    fn environments_are_parsed_and_dropped() {
        // new.env() holding x = 1L, then a reference back to that environment.
        let mut w = W::default();
        w.head().int(VECSXP as i32).int(2);
        w.int(ENVSXP as i32).int(0); // locked = FALSE
        w.int(EMPTYENV_SXP as i32); // enclosure
        w.int(LISTSXP as i32 | HAS_TAG)
            .sym("x")
            .int(INTSXP as i32)
            .int(1)
            .int(1);
        w.int(NILVALUE_SXP as i32); // end of frame
        w.int(NILVALUE_SXP as i32); // hash table
        w.int(NILVALUE_SXP as i32); // attributes
        w.int(REFSXP as i32 | (1 << 8));
        let node = parse(&w.0);
        assert!(matches!(node.items()[1].1.kind, Kind::Other));
    }

    #[test]
    fn compact_intseq_expands() {
        // 1:3 as ALTREP: info = pairlist(sym compact_intseq, sym base, 13L)
        let mut w = W::default();
        w.head().int(ALTREP_SXP as i32);
        w.int(LISTSXP as i32).sym("compact_intseq");
        w.int(LISTSXP as i32).sym("base");
        w.int(LISTSXP as i32).int(INTSXP as i32).int(1).int(13);
        w.int(NILVALUE_SXP as i32);
        w.int(REALSXP as i32).int(3);
        for v in [3.0f64, 1.0, 1.0] {
            w.0.extend_from_slice(&v.to_be_bytes());
        }
        w.int(NILVALUE_SXP as i32);
        assert!(matches!(parse(&w.0).kind, Kind::Int(v) if v == [1, 2, 3]));
    }

    #[test]
    fn rejects_non_xdr() {
        let mut x = Xdr::new(&b"A\n"[..]);
        assert!(x.header().is_err());
    }
}
