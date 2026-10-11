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
//! counts change, a complete snapshot (`diagnostics-begin <sender> <seq>`,
//! one `diagnostics <errors> <warnings> <abspath>` line per file that has
//! any, `diagnostics-end <seq>`) covering every file the language servers
//! report on, open in a buffer or not. Snapshots, not deltas: the sidebar
//! applies one atomically and, from the same sender, ignores any older than
//! the last applied, so nothing depends on every write arriving, or arriving
//! in order; `sender` names this process, so a restarted editor is taken as
//! is rather than having to outrun the old one's count. The first snapshot
//! is sent even when empty, which registers this editor as a sender. The
//! sidebar (treelix 0.1.25 and later) shows the union of every live sender's
//! latest snapshot and prunes senders whose process is gone, so editors that
//! share a sidebar never overwrite each other's markers. The latest snapshot
//! is also sent with every reveal, so a sidebar restarted later catches up
//! at the next buffer switch. On exit the editor flushes any snapshot still
//! queued and appends a farewell (`diagnostics-bye <sender> <seq>`) to it on
//! the same connection. The farewell retracts only this editor's markers, so
//! a file whose fix the language server never got to report does not stay
//! marked after the editor is gone, and other editors' markers stay.
//!
//! Why a render() poll and not a DocumentFocusLost/DocumentDidOpen hook:
//! event coverage is incomplete. `Editor::close` and
//! `Editor::close_document` change the focused document without dispatching
//! any focus event, and the `Action::Load` startup path (`hx a b`) never
//! focuses the opened docs at dispatch time. The poll is the one chokepoint
//! every buffer switch funnels through (pickers, `:open`, goto commands,
//! socket commands, closes), and it self-heals: any missed transition is
//! corrected at the next frame. The same holds for diagnostics, which arrive
//! by push (`publishDiagnostics`, for any file) and by pull (open documents):
//! fingerprinting `editor.diagnostics` per frame (no allocation) covers both.
//! All socket writes happen on one long-lived sender thread, in order, fed
//! through a latest-wins slot (a stalled sidebar cannot make anything queue
//! up), and failures are ignored: a missing, stale, or wedged sidebar must
//! never affect the editor. That includes quitting: the wait for the farewell
//! is bounded (`SHUTDOWN_FLUSH`), and it runs after the terminal is restored.

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
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
/// How long `Drop` waits for the farewell to go out. One connect and one
/// short write to a local socket take well under a millisecond; this is only
/// ever spent in full when the sidebar is wedged, and by then the terminal is
/// already restored, so the worst case is a late shell prompt.
const SHUTDOWN_FLUSH: Duration = Duration::from_millis(250);

/// Error and warning counts per file, for the files that have any.
type DiagnosticSummary = BTreeMap<PathBuf, (u32, u32)>;

/// What is waiting to be written, latest of each kind only: a reveal that has
/// not gone out yet is superseded by a newer one, and so is a snapshot.
#[derive(Default)]
struct Pending {
    reveal: Option<PathBuf>,
    snapshot: Option<(u64, Arc<DiagnosticSummary>)>,
    /// Farewell to write on the way out: the sequence to retract at.
    bye: Option<u64>,
    /// The editor is shutting down: let the sender thread finish.
    closed: bool,
    /// Set by the sender thread on every exit path, so the bounded wait in
    /// `Drop` ends as soon as the flush is done (or the thread is gone).
    finished: bool,
}

/// Hand-off between the render loop and the sender thread.
struct Outbox {
    pending: Mutex<Pending>,
    /// There is work, or the editor is closing (the sender thread waits).
    wake: Condvar,
    /// The sender thread has finished (`Drop` waits). Separate from `wake`
    /// so `put`'s `notify_one` can never be delivered to `Drop` instead.
    done: Condvar,
}

impl Outbox {
    fn new() -> Self {
        Outbox {
            pending: Mutex::new(Pending::default()),
            wake: Condvar::new(),
            done: Condvar::new(),
        }
    }
}

impl Outbox {
    fn put(&self, update: impl FnOnce(&mut Pending)) {
        if let Ok(mut pending) = self.pending.lock() {
            update(&mut pending);
            self.wake.notify_one();
        }
    }
}

pub struct SidebarFollow {
    /// Hand-off to the sender thread; None disables the feature entirely (no
    /// TREELIX_SOCKET_PATH, or the thread could not be started).
    outbox: Option<Arc<Outbox>>,
    /// Last pushed path — send only on change. Updated even when a send
    /// fails or is skipped: once the sidebar has been reachable, a dead
    /// sidebar must not turn every redraw into a reconnect attempt.
    revealed: Option<PathBuf>,
    /// Latches true after the first reveal that went through (set by the
    /// sender thread). Until then, `first/last_attempt` drive the startup
    /// retry.
    ready: Arc<AtomicBool>,
    first_attempt: Option<Instant>,
    last_attempt: Option<Instant>,
    /// Fingerprint of `editor.diagnostics` as last summarized; the summary is
    /// only rebuilt when it changes.
    fingerprint: u64,
    /// The latest snapshot and its sequence number, counted from 1 in this
    /// process (0: none yet); the sender id tells the sidebar whose count
    /// it is.
    snapshot: Arc<DiagnosticSummary>,
    seq: u64,
    /// Whether a snapshot was already queued in the current frame (then the
    /// reveal path need not queue the same one again).
    queued_this_frame: bool,
}

impl SidebarFollow {
    pub fn from_env() -> Self {
        Self::new(
            std::env::var_os("TREELIX_SOCKET_PATH")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
        )
    }

    /// `None` disables the feature entirely.
    fn new(socket: Option<PathBuf>) -> Self {
        let ready = Arc::new(AtomicBool::new(false));
        let outbox = socket.and_then(|socket| {
            let outbox = Arc::new(Outbox::new());
            let ready = Arc::clone(&ready);
            let inbox = Arc::clone(&outbox);
            // `Builder::spawn` reports failure instead of panicking, so a
            // system that cannot give us a thread just loses the feature.
            std::thread::Builder::new()
                .name("sidebar-follow".into())
                .spawn(move || sender_loop(socket, inbox, ready))
                .ok()
                .map(|_| outbox)
        });
        SidebarFollow {
            outbox,
            revealed: None,
            ready,
            first_attempt: None,
            last_attempt: None,
            fingerprint: 0,
            snapshot: Arc::new(DiagnosticSummary::new()),
            seq: 0,
            queued_this_frame: false,
        }
    }

    /// Called once per render.
    pub fn sync(&mut self, editor: &Editor) {
        if self.outbox.is_none() {
            return;
        }
        self.queued_this_frame = false;
        self.sync_diagnostics(editor);
        self.sync_reveal(editor);
    }

    /// Push a reveal if the focused document's path changed (or re-send the
    /// last one while the startup retry is live), with the latest diagnostics
    /// snapshot alongside. Cheap when nothing changed: one borrowed path
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
        let Some(outbox) = &self.outbox else {
            return;
        };
        // Resend the latest snapshot alongside, unless this frame already
        // queued it: a sidebar restarted since then catches up here.
        let snapshot = (self.seq > 0 && !self.queued_this_frame)
            .then(|| (self.seq, Arc::clone(&self.snapshot)));
        outbox.put(|pending| {
            pending.reveal = Some(path);
            if let Some(snapshot) = snapshot {
                pending.snapshot = Some(snapshot);
            }
        });
    }

    /// Push a new snapshot when the per-file error/warning counts changed.
    /// The first one is sent even when empty: it registers this editor with
    /// the sidebar, which shows the union of all live editors' snapshots.
    fn sync_diagnostics(&mut self, editor: &Editor) {
        let fingerprint = fingerprint(&editor.diagnostics);
        if self.seq > 0 && fingerprint == self.fingerprint {
            return;
        }
        self.fingerprint = fingerprint;
        self.queue_snapshot(summarize(&editor.diagnostics));
    }

    /// Bump the sequence and queue `summary`, unless it is what was last sent.
    fn queue_snapshot(&mut self, summary: DiagnosticSummary) {
        if self.seq > 0 && summary == *self.snapshot {
            return;
        }
        self.seq += 1;
        self.snapshot = Arc::new(summary);
        self.queued_this_frame = true;
        if let Some(outbox) = &self.outbox {
            let snapshot = (self.seq, Arc::clone(&self.snapshot));
            outbox.put(|pending| pending.snapshot = Some(snapshot));
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

impl Drop for SidebarFollow {
    fn drop(&mut self) {
        let Some(outbox) = &self.outbox else {
            return;
        };
        // Nothing to retract if this editor never claimed the sidebar.
        let bye = (self.seq > 0).then(|| self.seq + 1);
        let Ok(mut pending) = outbox.pending.lock() else {
            return;
        };
        // A reveal from a dying editor would move the sidebar's highlight to
        // a file nobody is editing, and cost a connect out of the budget.
        pending.reveal = None;
        pending.bye = bye;
        pending.closed = true;
        outbox.wake.notify_one();
        // Bounded, so a wedged sidebar cannot delay quitting, and never
        // unwrapped, since a panic here would abort during unwinding.
        // `wait_timeout_while` re-waits with the remaining time on spurious
        // wakeups instead of restarting the budget.
        let _ = outbox
            .done
            .wait_timeout_while(pending, SHUTDOWN_FLUSH, |pending| !pending.finished);
    }
}

/// The sender thread: takes whatever is pending (latest reveal, latest
/// snapshot), writes them one at a time, reveal first, and waits for more.
/// When the editor closes, the snapshot and farewell still pending go out
/// together, and `Drop` is told the thread is done however it ends.
fn sender_loop(socket: PathBuf, outbox: Arc<Outbox>, ready: Arc<AtomicBool>) {
    send_until_closed(&socket, &outbox, &ready);
    if let Ok(mut pending) = outbox.pending.lock() {
        pending.finished = true;
    }
    outbox.done.notify_all();
}

fn send_until_closed(socket: &Path, outbox: &Outbox, ready: &AtomicBool) {
    let sender = sender_id();
    loop {
        let (reveal, snapshot, bye, closing) = {
            let Ok(mut pending) = outbox.pending.lock() else {
                return;
            };
            while pending.reveal.is_none() && pending.snapshot.is_none() && !pending.closed {
                pending = match outbox.wake.wait(pending) {
                    Ok(pending) => pending,
                    Err(_) => return,
                };
            }
            if pending.closed {
                // Closing: the latest snapshot, then the farewell, never a
                // reveal (`Drop` already cleared it).
                (None, pending.snapshot.take(), pending.bye.take(), true)
            } else {
                (pending.reveal.take(), pending.snapshot.take(), None, false)
            }
        };
        if let Some(path) = reveal {
            // Only a reveal that went through proves the sidebar is up and
            // pointed at us; the startup retry keys off this.
            if write(socket, &format!("reveal-follow {}\n", path.display())) {
                ready.store(true, Ordering::Relaxed);
            }
        }
        // One connection for both: the sidebar reads a connection on one
        // thread, in order, so the farewell can never overtake the snapshot
        // it is meant to follow.
        let mut payload = String::new();
        if let Some((seq, files)) = snapshot {
            payload.push_str(&snapshot_payload(&sender, seq, &files));
        }
        if let Some(seq) = bye {
            payload.push_str(&bye_payload(&sender, seq));
        }
        if !payload.is_empty() {
            write(socket, &payload);
        }
        if closing {
            return;
        }
    }
}

/// One connection, one payload; true when it went through. The connect has
/// no timeout (std's unix sockets offer none; `socket2::Socket::connect_timeout`
/// would be the way if it ever matters): it fails at once when nothing
/// listens, and only blocks when a live listener's backlog is full, which the
/// bounded wait in `Drop` covers.
fn write(socket: &Path, payload: &str) -> bool {
    use std::io::Write as _;
    let Ok(mut stream) = std::os::unix::net::UnixStream::connect(socket) else {
        return false;
    };
    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
    stream.write_all(payload.as_bytes()).is_ok()
}

/// Names this editor process to the sidebar: pid plus start time, so a new
/// process never shares an id with one whose pid it inherited.
fn sender_id() -> String {
    static START: std::sync::OnceLock<u128> = std::sync::OnceLock::new();
    let start = START.get_or_init(|| {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|since| since.as_micros())
            .unwrap_or(0)
    });
    format!("hx-{}-{start}", std::process::id())
}

/// The wire form of one farewell: an empty snapshot at `seq` that only its
/// own sender's display may apply.
fn bye_payload(sender: &str, seq: u64) -> String {
    format!("diagnostics-bye {sender} {seq}\n")
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
fn snapshot_payload(sender: &str, seq: u64, files: &DiagnosticSummary) -> String {
    let mut payload = format!("diagnostics-begin {sender} {seq}\n");
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
    fn the_sender_id_is_stable_within_the_process_and_names_it() {
        let id = sender_id();
        assert_eq!(id, sender_id());
        assert!(
            id.starts_with(&format!("hx-{}-", std::process::id())),
            "{id}"
        );
        assert!(!id.contains(' '), "the wire form is space separated");
    }

    #[test]
    fn snapshot_payload_is_a_complete_ordered_batch() {
        let mut files = DiagnosticSummary::new();
        files.insert(PathBuf::from("/r/new dir/new.rs"), (0, 1));
        files.insert(PathBuf::from("/r/a.rs"), (3, 1));
        assert_eq!(
            snapshot_payload("hx-1-2", 7, &files),
            "diagnostics-begin hx-1-2 7\n\
             diagnostics 3 1 /r/a.rs\n\
             diagnostics 0 1 /r/new dir/new.rs\n\
             diagnostics-end 7\n"
        );
        assert_eq!(
            snapshot_payload("hx-1-2", 8, &DiagnosticSummary::new()),
            "diagnostics-begin hx-1-2 8\ndiagnostics-end 8\n",
            "an empty snapshot clears everything"
        );
    }

    #[test]
    fn bye_payload_is_one_line() {
        assert_eq!(bye_payload("hx-1-2", 8), "diagnostics-bye hx-1-2 8\n");
    }

    /// A listener on a short socket path (macOS caps `sun_path` at 104
    /// bytes, which `$TMPDIR` alone nearly fills) collecting every line with
    /// the index of the connection it arrived on.
    struct Sidebar {
        socket: PathBuf,
        lines: Arc<Mutex<Vec<(usize, String)>>>,
    }

    impl Sidebar {
        fn listen(tag: &str) -> Self {
            let socket = PathBuf::from(format!("/tmp/hx-sf-{}-{tag}.sock", std::process::id()));
            let _ = std::fs::remove_file(&socket);
            let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
            let lines = Arc::new(Mutex::new(Vec::new()));
            let sink = Arc::clone(&lines);
            std::thread::spawn(move || {
                use std::io::BufRead as _;
                for (index, stream) in listener.incoming().enumerate() {
                    let Ok(stream) = stream else { break };
                    for line in std::io::BufReader::new(stream).lines() {
                        let Ok(line) = line else { break };
                        sink.lock().unwrap().push((index, line));
                    }
                }
            });
            Sidebar { socket, lines }
        }

        fn wait_for(&self, count: usize) -> Vec<(usize, String)> {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let lines = self.lines.lock().unwrap().clone();
                if lines.len() >= count || Instant::now() > deadline {
                    return lines;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    impl Drop for Sidebar {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.socket);
        }
    }

    fn one_file() -> DiagnosticSummary {
        let mut files = DiagnosticSummary::new();
        files.insert(PathBuf::from("/r/a.rs"), (1, 0));
        files
    }

    #[test]
    fn exit_flushes_the_pending_snapshot_and_the_farewell_together() {
        let sidebar = Sidebar::listen("flush");
        let mut follow = SidebarFollow::new(Some(sidebar.socket.clone()));
        follow.queue_snapshot(one_file());
        drop(follow);
        let lines = sidebar.wait_for(4);
        let sender = sender_id();
        assert_eq!(
            lines,
            vec![
                (0, format!("diagnostics-begin {sender} 1")),
                (0, "diagnostics 1 0 /r/a.rs".to_string()),
                (0, "diagnostics-end 1".to_string()),
                (0, format!("diagnostics-bye {sender} 2")),
            ],
            "in order, on one connection"
        );
    }

    #[test]
    fn exit_sends_nothing_when_no_snapshot_was_ever_claimed() {
        let sidebar = Sidebar::listen("silent");
        drop(SidebarFollow::new(Some(sidebar.socket.clone())));
        std::thread::sleep(Duration::from_millis(100));
        assert!(sidebar.lines.lock().unwrap().is_empty());
    }

    #[test]
    fn closing_sends_the_snapshot_and_farewell_but_never_a_reveal() {
        let sidebar = Sidebar::listen("reveal");
        let follow = SidebarFollow::new(Some(sidebar.socket.clone()));
        let outbox = Arc::clone(follow.outbox.as_ref().unwrap());
        // Everything at once, so the sender thread sees the close and the
        // pending items together (Drop clears the reveal itself; this is the
        // sender side's own guarantee).
        outbox.put(|pending| {
            pending.reveal = Some(PathBuf::from("/r/a.rs"));
            pending.snapshot = Some((1, Arc::new(one_file())));
            pending.bye = Some(2);
            pending.closed = true;
        });
        // Not dropped: Drop would race the sender thread for the same slots.
        std::mem::forget(follow);
        let lines = sidebar.wait_for(4);
        let sender = sender_id();
        assert_eq!(
            lines,
            vec![
                (0, format!("diagnostics-begin {sender} 1")),
                (0, "diagnostics 1 0 /r/a.rs".to_string()),
                (0, "diagnostics-end 1".to_string()),
                (0, format!("diagnostics-bye {sender} 2")),
            ]
        );
        let pending = outbox.pending.lock().unwrap();
        assert!(pending.finished, "the sender thread reports it is done");
    }

    #[test]
    fn exit_does_not_wait_for_a_sidebar_that_is_not_there() {
        let socket = PathBuf::from(format!("/tmp/hx-sf-{}-absent.sock", std::process::id()));
        let _ = std::fs::remove_file(&socket);
        let mut follow = SidebarFollow::new(Some(socket));
        follow.queue_snapshot(one_file());
        let outbox = Arc::clone(follow.outbox.as_ref().unwrap());
        let started = Instant::now();
        drop(follow);
        assert!(
            outbox.pending.lock().unwrap().finished,
            "the wait ended because the sender thread reported done, not by timing out"
        );
        assert!(started.elapsed() < Duration::from_secs(5), "no hang");
    }

    /// The outbox keeps only the latest of each kind, so a stalled sidebar
    /// can never make writes pile up.
    #[test]
    fn the_outbox_is_latest_wins() {
        let outbox = Outbox::new();
        let files = Arc::new(DiagnosticSummary::new());
        for seq in 1..=3 {
            let files = Arc::clone(&files);
            outbox.put(|pending| pending.snapshot = Some((seq, files)));
        }
        outbox.put(|pending| pending.reveal = Some(PathBuf::from("/r/a.rs")));
        outbox.put(|pending| pending.reveal = Some(PathBuf::from("/r/b.rs")));
        let pending = outbox.pending.lock().unwrap();
        assert_eq!(pending.snapshot.as_ref().map(|(seq, _)| *seq), Some(3));
        assert_eq!(pending.reveal.as_deref(), Some(Path::new("/r/b.rs")));
    }
}
