//! Sidebar follow (local patch): keep an external file-tree sidebar
//! (treelix) pointed at the focused document, and told about diagnostics.
//!
//! When the launcher exports TREELIX_SOCKET_PATH (see the dotfiles'
//! launch-editor.sh and the ad-hoc `hx()` shell wrapper), any change of the
//! focused view's document pushes a `reveal-follow <abspath>` line to the
//! sidebar's reveal socket (newline-delimited protocol, treelix src/ipc.rs).
//! The `-follow` tag lets the sidebar treat these automatic pushes more
//! gently than explicit user reveals (plain `reveal`, sent by A-r/space-f).
//!
//! The same socket carries diagnostics: whenever any file's error/warning
//! counts change, a complete snapshot (`diagnostics-begin <seq>`, one
//! `diagnostics <errors> <warnings> <abspath>` line per file that has any,
//! `diagnostics-end <seq>`) covering every file the language servers report
//! on, open in a buffer or not. Snapshots, not deltas: the sidebar applies
//! one atomically and ignores any older than the last applied, so nothing
//! depends on every write arriving, or arriving in order. The latest
//! snapshot is also sent with every reveal, so a sidebar restarted later
//! catches up at the next buffer switch.
//!
//! Why a render() poll and not a DocumentFocusLost/DocumentDidOpen hook:
//! event coverage is incomplete — `Editor::close` and
//! `Editor::close_document` change the focused document without dispatching
//! any focus event, and the `Action::Load` startup path (`hx a b`) never
//! focuses the opened docs at dispatch time. The poll is the one chokepoint
//! every buffer switch funnels through (pickers, `:open`, goto commands,
//! socket commands, closes), and it self-heals: any missed transition is
//! corrected at the next frame. The same holds for diagnostics, which arrive
//! by push (`publishDiagnostics`, for any file) and by pull (open documents):
//! fingerprinting `editor.diagnostics` per frame (no allocation) covers both.
//! All socket writes happen on one long-lived sender thread, in order, and
//! failures are ignored — a missing, stale, or wedged sidebar must never
//! affect the editor.

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use helix_core::diagnostic::DiagnosticProvider;
use helix_core::Uri;
use helix_lsp::lsp;
// `doc!` comes from the crate-wide `#[macro_use] extern crate helix_view`.
use helix_view::Editor;

/// Re-send cadence while no send has succeeded yet.
const RETRY_INTERVAL: Duration = Duration::from_secs(1);
/// Give up retrying this long after the first attempt: the retry exists
/// only for the startup race where helix renders before the sidebar has
/// bound its socket. A sidebar that hasn't bound within this window is not
/// coming; later sends still fire on every buffer change as usual.
const RETRY_DEADLINE: Duration = Duration::from_secs(30);

/// Error and warning counts per file, for the files that have any.
type DiagnosticSummary = BTreeMap<PathBuf, (u32, u32)>;

/// What the sender thread writes to the sidebar
enum Outgoing {
    Reveal(PathBuf),
    Snapshot {
        seq: u64,
        files: Arc<DiagnosticSummary>,
    },
}

pub struct SidebarFollow {
    /// Channel to the sender thread; None disables the feature entirely (no
    /// TREELIX_SOCKET_PATH, or the thread could not be started).
    tx: Option<Sender<Outgoing>>,
    /// Last pushed path — send only on change. Updated even when a send
    /// fails or is skipped: once the sidebar has been reachable, a dead
    /// sidebar must not turn every redraw into a reconnect attempt.
    revealed: Option<PathBuf>,
    /// Latches true after the first successful write (by the sender
    /// thread). Until then, `first/last_attempt` drive the startup retry.
    ready: Arc<AtomicBool>,
    first_attempt: Option<Instant>,
    last_attempt: Option<Instant>,
    /// Fingerprint of `editor.diagnostics` as last summarized; the summary is
    /// only rebuilt when it changes.
    fingerprint: u64,
    /// The latest snapshot and its sequence number. Sequences are seeded from
    /// the process start time so an editor started later always outranks an
    /// earlier one in the sidebar's ordering: a restarted helix must not have
    /// its snapshots ignored until it catches up with the old one's count.
    snapshot: Arc<DiagnosticSummary>,
    seq: u64,
    /// Whether a snapshot has been produced yet.
    has_snapshot: bool,
}

impl SidebarFollow {
    pub fn from_env() -> Self {
        let ready = Arc::new(AtomicBool::new(false));
        let tx = std::env::var_os("TREELIX_SOCKET_PATH")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .and_then(|socket| {
                let (tx, rx) = mpsc::channel();
                let ready = Arc::clone(&ready);
                // `Builder::spawn` reports failure instead of panicking, so a
                // system that cannot give us a thread just loses the feature.
                std::thread::Builder::new()
                    .name("sidebar-follow".into())
                    .spawn(move || sender_loop(socket, rx, ready))
                    .ok()
                    .map(|_| tx)
            });
        SidebarFollow {
            tx,
            revealed: None,
            ready,
            first_attempt: None,
            last_attempt: None,
            fingerprint: 0,
            snapshot: Arc::new(DiagnosticSummary::new()),
            seq: initial_sequence(),
            has_snapshot: false,
        }
    }

    /// Called once per render.
    pub fn sync(&mut self, editor: &Editor) {
        if self.tx.is_none() {
            return;
        }
        self.sync_diagnostics(editor);
        self.sync_reveal(editor);
    }

    /// Push a reveal if the focused document's path changed (or re-send the
    /// last one while the startup retry is live), followed by the latest
    /// diagnostics snapshot. Cheap when nothing changed: one borrowed path
    /// compare, no allocation.
    fn sync_reveal(&mut self, editor: &Editor) {
        let path = doc!(editor).path();
        let changed = path != self.revealed.as_deref();
        if !changed && (path.is_none() || !self.retry_due()) {
            return;
        }
        self.revealed = path.map(Path::to_path_buf);
        let Some(path) = self.revealed.clone() else {
            // Scratch buffer: reset the cache but send nothing.
            return;
        };
        if has_newline(&path) {
            return;
        }
        let now = Instant::now();
        self.first_attempt.get_or_insert(now);
        self.last_attempt = Some(now);
        let Some(tx) = &self.tx else {
            return;
        };
        let _ = tx.send(Outgoing::Reveal(path));
        if self.has_snapshot {
            let _ = tx.send(Outgoing::Snapshot {
                seq: self.seq,
                files: Arc::clone(&self.snapshot),
            });
        }
    }

    /// Push a new snapshot when the per-file error/warning counts changed.
    fn sync_diagnostics(&mut self, editor: &Editor) {
        let fingerprint = fingerprint(&editor.diagnostics);
        if fingerprint == self.fingerprint {
            return;
        }
        self.fingerprint = fingerprint;
        let summary = summarize(&editor.diagnostics);
        if summary == *self.snapshot {
            return;
        }
        self.seq += 1;
        self.has_snapshot = true;
        self.snapshot = Arc::new(summary);
        if let Some(tx) = &self.tx {
            let _ = tx.send(Outgoing::Snapshot {
                seq: self.seq,
                files: Arc::clone(&self.snapshot),
            });
        }
    }

    /// True while the startup retry should re-send: no send has ever
    /// succeeded, the last attempt is stale, and the deadline hasn't passed.
    fn retry_due(&self) -> bool {
        if self.ready.load(Ordering::Relaxed) {
            return false;
        }
        let (Some(first), Some(last)) = (self.first_attempt, self.last_attempt) else {
            return false;
        };
        first.elapsed() < RETRY_DEADLINE && last.elapsed() >= RETRY_INTERVAL
    }
}

/// The sender thread: writes go out one at a time, in order. Whatever queued
/// up while a write was in progress is coalesced first, since only the latest
/// reveal and the latest snapshot matter.
fn sender_loop(socket: PathBuf, rx: Receiver<Outgoing>, ready: Arc<AtomicBool>) {
    while let Ok(first) = rx.recv() {
        let mut reveal = None;
        let mut snapshot = None;
        let mut take = |msg| match msg {
            Outgoing::Reveal(path) => reveal = Some(path),
            Outgoing::Snapshot { seq, files } => snapshot = Some((seq, files)),
        };
        take(first);
        while let Ok(msg) = rx.try_recv() {
            take(msg);
        }
        if let Some(path) = reveal {
            write(
                &socket,
                &format!("reveal-follow {}\n", path.display()),
                &ready,
            );
        }
        if let Some((seq, files)) = snapshot {
            write(&socket, &snapshot_payload(seq, &files), &ready);
        }
    }
}

/// One connection, one payload; `ready` latches when it went through.
fn write(socket: &Path, payload: &str, ready: &AtomicBool) {
    use std::io::Write as _;
    let Ok(mut stream) = std::os::unix::net::UnixStream::connect(socket) else {
        return;
    };
    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
    if stream.write_all(payload.as_bytes()).is_ok() {
        ready.store(true, Ordering::Relaxed);
    }
}

/// Where this process's snapshot sequence starts: microseconds since the
/// epoch, so it exceeds anything an editor started earlier could have sent
/// (one per render at most, far fewer than one per microsecond).
fn initial_sequence() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|since| since.as_micros() as u64)
        .unwrap_or(0)
}

/// A newline in a file name would smuggle a second line into the
/// newline-delimited protocol; better a missed update than a corrupted
/// command stream.
fn has_newline(path: &Path) -> bool {
    std::os::unix::ffi::OsStrExt::as_bytes(path.as_os_str()).contains(&b'\n')
}

/// Visit every file with errors or warnings, in map order, with its counts.
fn for_each_counted(
    diagnostics: &BTreeMap<Uri, Vec<(lsp::Diagnostic, DiagnosticProvider)>>,
    mut visit: impl FnMut(&Path, u32, u32),
) {
    for (uri, diags) in diagnostics {
        let Some(path) = uri.as_path() else {
            continue;
        };
        if has_newline(path) {
            continue;
        }
        let (errors, warnings) = diags
            .iter()
            .fold((0, 0), |(e, w), (d, _)| match d.severity {
                Some(lsp::DiagnosticSeverity::ERROR) => (e + 1, w),
                Some(lsp::DiagnosticSeverity::WARNING) => (e, w + 1),
                _ => (e, w),
            });
        if errors + warnings > 0 {
            visit(path, errors, warnings);
        }
    }
}

/// A hash of what `summarize` would produce, without building it.
fn fingerprint(diagnostics: &BTreeMap<Uri, Vec<(lsp::Diagnostic, DiagnosticProvider)>>) -> u64 {
    let mut hasher = DefaultHasher::new();
    for_each_counted(diagnostics, |path, errors, warnings| {
        path.hash(&mut hasher);
        errors.hash(&mut hasher);
        warnings.hash(&mut hasher);
    });
    hasher.finish()
}

/// Error and warning counts per file, for the files that have any.
fn summarize(
    diagnostics: &BTreeMap<Uri, Vec<(lsp::Diagnostic, DiagnosticProvider)>>,
) -> DiagnosticSummary {
    let mut summary = DiagnosticSummary::new();
    for_each_counted(diagnostics, |path, errors, warnings| {
        summary.insert(path.to_path_buf(), (errors, warnings));
    });
    summary
}

/// The wire form of one snapshot.
fn snapshot_payload(seq: u64, files: &DiagnosticSummary) -> String {
    let mut payload = format!("diagnostics-begin {seq}\n");
    for (path, (errors, warnings)) in files {
        payload.push_str(&format!(
            "diagnostics {errors} {warnings} {}\n",
            path.display()
        ));
    }
    payload.push_str(&format!("diagnostics-end {seq}\n"));
    payload
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag(severity: lsp::DiagnosticSeverity) -> (lsp::Diagnostic, DiagnosticProvider) {
        (
            lsp::Diagnostic {
                severity: Some(severity),
                ..Default::default()
            },
            DiagnosticProvider::Lsp {
                server_id: helix_lsp::LanguageServerId::default(),
                identifier: None,
            },
        )
    }

    fn diagnostics() -> BTreeMap<Uri, Vec<(lsp::Diagnostic, DiagnosticProvider)>> {
        let mut diagnostics = BTreeMap::new();
        diagnostics.insert(
            Uri::from(PathBuf::from("/r/a.rs")),
            vec![
                diag(lsp::DiagnosticSeverity::ERROR),
                diag(lsp::DiagnosticSeverity::ERROR),
                diag(lsp::DiagnosticSeverity::WARNING),
                diag(lsp::DiagnosticSeverity::HINT),
                diag(lsp::DiagnosticSeverity::INFORMATION),
            ],
        );
        diagnostics.insert(
            Uri::from(PathBuf::from("/r/hints-only.rs")),
            vec![diag(lsp::DiagnosticSeverity::HINT)],
        );
        diagnostics.insert(Uri::from(PathBuf::from("/r/empty.rs")), vec![]);
        diagnostics
    }

    #[test]
    fn summarize_counts_errors_and_warnings_per_file_and_drops_the_rest() {
        let summary = summarize(&diagnostics());
        assert_eq!(summary.len(), 1, "{summary:?}");
        assert_eq!(summary[Path::new("/r/a.rs")], (2, 1));
    }

    #[test]
    fn fingerprint_tracks_the_summary() {
        let mut diagnostics = diagnostics();
        let before = fingerprint(&diagnostics);
        assert_eq!(before, fingerprint(&diagnostics), "stable");
        // Hint/info changes do not count.
        diagnostics
            .get_mut(&Uri::from(PathBuf::from("/r/hints-only.rs")))
            .unwrap()
            .push(diag(lsp::DiagnosticSeverity::HINT));
        assert_eq!(before, fingerprint(&diagnostics));
        // A warning does.
        diagnostics
            .get_mut(&Uri::from(PathBuf::from("/r/hints-only.rs")))
            .unwrap()
            .push(diag(lsp::DiagnosticSeverity::WARNING));
        assert_ne!(before, fingerprint(&diagnostics));
        assert_eq!(summarize(&diagnostics).len(), 2);
    }

    #[test]
    fn a_later_process_outranks_an_earlier_one() {
        let earlier = initial_sequence();
        std::thread::sleep(Duration::from_millis(2));
        assert!(initial_sequence() > earlier + 1000);
    }

    #[test]
    fn snapshot_payload_is_a_complete_ordered_batch() {
        let mut files = DiagnosticSummary::new();
        files.insert(PathBuf::from("/r/new dir/new.rs"), (0, 1));
        files.insert(PathBuf::from("/r/a.rs"), (3, 1));
        assert_eq!(
            snapshot_payload(7, &files),
            "diagnostics-begin 7\n\
             diagnostics 3 1 /r/a.rs\n\
             diagnostics 0 1 /r/new dir/new.rs\n\
             diagnostics-end 7\n"
        );
        assert_eq!(
            snapshot_payload(8, &DiagnosticSummary::new()),
            "diagnostics-begin 8\ndiagnostics-end 8\n",
            "an empty snapshot clears everything"
        );
    }
}
