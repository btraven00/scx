//! Phase anchors in the obkit-events format (spec 0.1,
//! <https://github.com/omnibenchmark/obkit>, SPEC.md), so a benchmark can
//! split a conversion's time and memory by phase.
//!
//! Off unless `OBKIT_DIR` is set; then each [`span`] appends a `start` event
//! to `$OBKIT_DIR/obkit-events.jsonl` and its guard an `end` event when
//! dropped (also on an early `?` return). As the spec requires, writing can
//! never fail the conversion: an I/O error prints one warning and further
//! events are dropped.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// An open phase; its `end` event is written when it is dropped.
#[must_use = "the phase ends when this guard is dropped"]
pub struct Span(&'static str);

impl Drop for Span {
    fn drop(&mut self) {
        emit(self.0, "end");
    }
}

/// Start phase `event`. `None` (and nothing written) when `OBKIT_DIR` is unset.
pub fn span(event: &'static str) -> Option<Span> {
    dir()?;
    emit(event, "start");
    Some(Span(event))
}

fn dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        std::env::var_os("OBKIT_DIR")
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
    })
    .as_ref()
}

fn host() -> &'static str {
    static HOST: OnceLock<String> = OnceLock::new();
    HOST.get_or_init(|| {
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .or_else(|_| std::env::var("HOSTNAME"))
            .map(|h| h.trim().to_string())
            .unwrap_or_default()
    })
}

/// One event line, as the spec writes it.
fn line(event: &str, phase: &str, ts: chrono::DateTime<chrono::Utc>) -> String {
    let mut v = serde_json::json!({
        "ts": ts.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
        "event": event,
        "phase": phase,
        "pid": std::process::id(),
    });
    if !host().is_empty() {
        v["host"] = host().into();
    }
    v.to_string() + "\n"
}

fn emit(event: &str, phase: &str) {
    static FAILED: AtomicBool = AtomicBool::new(false);
    let Some(dir) = dir() else { return };
    if FAILED.load(Ordering::Relaxed) {
        return;
    }
    let text = line(event, phase, chrono::Utc::now());
    let wrote = std::fs::create_dir_all(dir).and_then(|_| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("obkit-events.jsonl"))?
            .write_all(text.as_bytes())
    });
    if let Err(e) = wrote {
        if !FAILED.swap(true, Ordering::Relaxed) {
            eprintln!("warning: OBKIT_DIR: cannot write phase events ({e}); not writing any more");
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    #[test]
    fn event_line_follows_the_spec() {
        let ts = chrono::Utc
            .with_ymd_and_hms(2026, 4, 20, 17, 10, 0)
            .unwrap()
            + chrono::Duration::milliseconds(123);
        let v: serde_json::Value = serde_json::from_str(&super::line("x", "start", ts)).unwrap();
        assert_eq!(v["ts"], "2026-04-20T17:10:00.123Z");
        assert_eq!(v["event"], "x");
        assert_eq!(v["phase"], "start");
        assert_eq!(v["pid"], std::process::id());
    }
}
