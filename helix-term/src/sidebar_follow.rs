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
//! The same socket carries `diagnostics <errors> <warnings> <abspath>` lines:
//! one per file whose error/warning counts changed since the last push, for
//! every file the language servers report on, open in a buffer or not. The
//! sidebar colors the file (and its folders) by the worst severity.
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
//! summarizing `editor.diagnostics` per frame covers both. Failures are
//! ignored and writes happen on detached threads — a missing, stale, or
//! wedged sidebar must never affect the editor.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

pub struct SidebarFollow {
    /// Reveal socket path from TREELIX_SOCKET_PATH; None disables the
    /// feature entirely.
    socket: Option<PathBuf>,
    /// Last pushed path — send only on change. Updated even when a send
    /// fails or is skipped: once the sidebar has been reachable, a dead
    /// sidebar must not turn every redraw into a reconnect attempt.
    revealed: Option<PathBuf>,
    /// Latches true after the first successful send (written by the sender
    /// thread). Until then, `first/last_attempt` drive the startup retry.
    ready: Arc<AtomicBool>,
    first_attempt: Option<Instant>,
    last_attempt: Option<Instant>,
    /// What the sidebar was last told about diagnostics. Emptied when the
    /// sidebar first becomes reachable, so everything is (re)sent to it.
    pushed_diagnostics: DiagnosticSummary,
    /// Whether `ready` had already been observed true by the diagnostics
    /// sync; the false -> true edge triggers the full resend.
    sidebar_seen: bool,
}

impl SidebarFollow {
    pub fn from_env() -> Self {
        SidebarFollow {
            socket: std::env::var_os("TREELIX_SOCKET_PATH")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
            revealed: None,
            ready: Arc::new(AtomicBool::new(false)),
            first_attempt: None,
            last_attempt: None,
            pushed_diagnostics: DiagnosticSummary::new(),
            sidebar_seen: false,
        }
    }

    /// Called once per render.
    pub fn sync(&mut self, editor: &Editor) {
        let Some(socket) = self.socket.clone() else {
            return;
        };
        self.sync_reveal(editor, &socket);
        self.sync_diagnostics(editor, &socket);
    }

    /// Push a reveal if the focused document's path changed (or re-send the
    /// last one while the startup retry is live). Cheap when nothing
    /// changed: one borrowed path compare, no allocation.
    fn sync_reveal(&mut self, editor: &Editor, socket: &Path) {
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
        let ready = Arc::clone(&self.ready);
        send_lines(
            socket.to_path_buf(),
            vec![format!("reveal-follow {}", path.display())],
            Some(ready),
        );
    }

    /// Push the diagnostics that changed since the last push. Until the
    /// sidebar has answered once nothing is sent (it would be lost); when it
    /// first does, everything is sent.
    fn sync_diagnostics(&mut self, editor: &Editor, socket: &Path) {
        if !self.ready.load(Ordering::Relaxed) {
            return;
        }
        if !self.sidebar_seen {
            self.sidebar_seen = true;
            self.pushed_diagnostics.clear();
        }
        let current = summarize(&editor.diagnostics);
        if current == self.pushed_diagnostics {
            return;
        }
        let lines = delta(&self.pushed_diagnostics, &current);
        self.pushed_diagnostics = current;
        if !lines.is_empty() {
            send_lines(socket.to_path_buf(), lines, None);
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

/// A newline in a file name would smuggle a second line into the
/// newline-delimited protocol; better a missed update than a corrupted
/// command stream.
fn has_newline(path: &Path) -> bool {
    std::os::unix::ffi::OsStrExt::as_bytes(path.as_os_str()).contains(&b'\n')
}

/// Write `lines` to the sidebar on a detached thread; `ready` (if given)
/// latches when the write succeeds.
fn send_lines(socket: PathBuf, lines: Vec<String>, ready: Option<Arc<AtomicBool>>) {
    std::thread::spawn(move || {
        use std::io::Write as _;
        let Ok(mut stream) = std::os::unix::net::UnixStream::connect(&socket) else {
            return;
        };
        let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
        let mut payload = String::with_capacity(lines.iter().map(|l| l.len() + 1).sum());
        for line in &lines {
            payload.push_str(line);
            payload.push('\n');
        }
        if stream.write_all(payload.as_bytes()).is_ok() {
            if let Some(ready) = ready {
                ready.store(true, Ordering::Relaxed);
            }
        }
    });
}

/// Error and warning counts per file, keeping only files that have some and
/// whose URI is a path the sidebar can show.
fn summarize(
    diagnostics: &BTreeMap<Uri, Vec<(lsp::Diagnostic, DiagnosticProvider)>>,
) -> DiagnosticSummary {
    let mut summary = DiagnosticSummary::new();
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
            summary.insert(path.to_path_buf(), (errors, warnings));
        }
    }
    summary
}

/// The wire lines that take the sidebar from `previous` to `current`: a
/// zero line for every file that dropped out, counts for every file whose
/// counts differ.
fn delta(previous: &DiagnosticSummary, current: &DiagnosticSummary) -> Vec<String> {
    let mut lines = Vec::new();
    for path in previous.keys() {
        if !current.contains_key(path) {
            lines.push(format!("diagnostics 0 0 {}", path.display()));
        }
    }
    for (path, counts) in current {
        if previous.get(path) != Some(counts) {
            lines.push(format!(
                "diagnostics {} {} {}",
                counts.0,
                counts.1,
                path.display()
            ));
        }
    }
    lines
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

    #[test]
    fn summarize_counts_errors_and_warnings_per_file_and_drops_the_rest() {
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
        let summary = summarize(&diagnostics);
        assert_eq!(summary.len(), 1, "{summary:?}");
        assert_eq!(summary[Path::new("/r/a.rs")], (2, 1));
    }

    #[test]
    fn delta_sends_only_what_changed_and_clears_what_went_away() {
        let mut previous = DiagnosticSummary::new();
        previous.insert(PathBuf::from("/r/gone.rs"), (1, 0));
        previous.insert(PathBuf::from("/r/same.rs"), (0, 2));
        previous.insert(PathBuf::from("/r/changed.rs"), (1, 1));
        let mut current = DiagnosticSummary::new();
        current.insert(PathBuf::from("/r/same.rs"), (0, 2));
        current.insert(PathBuf::from("/r/changed.rs"), (3, 1));
        current.insert(PathBuf::from("/r/new dir/new.rs"), (0, 1));
        assert_eq!(
            delta(&previous, &current),
            vec![
                "diagnostics 0 0 /r/gone.rs",
                "diagnostics 3 1 /r/changed.rs",
                "diagnostics 0 1 /r/new dir/new.rs",
            ]
        );
        assert!(delta(&current, &current).is_empty());
        assert_eq!(
            delta(&DiagnosticSummary::new(), &current).len(),
            3,
            "an empty previous (first contact) resends everything"
        );
    }
}
