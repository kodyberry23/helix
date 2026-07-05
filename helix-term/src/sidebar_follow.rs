//! Sidebar follow (local patch): keep an external file-tree sidebar
//! (treelix) pointed at the focused document.
//!
//! When the launcher exports TREELIX_SOCKET_PATH (see the dotfiles'
//! launch-editor.sh and the ad-hoc `hx()` shell wrapper), any change of the
//! focused view's document pushes a `reveal-follow <abspath>` line to the
//! sidebar's reveal socket (newline-delimited protocol, treelix src/ipc.rs).
//! The `-follow` tag lets the sidebar treat these automatic pushes more
//! gently than explicit user reveals (plain `reveal`, sent by A-r/space-f).
//!
//! Why a render() poll and not a DocumentFocusLost/DocumentDidOpen hook:
//! event coverage is incomplete — `Editor::close` and
//! `Editor::close_document` change the focused document without dispatching
//! any focus event, and the `Action::Load` startup path (`hx a b`) never
//! focuses the opened docs at dispatch time. The poll is the one chokepoint
//! every buffer switch funnels through (pickers, `:open`, goto commands,
//! socket commands, closes), and it self-heals: any missed transition is
//! corrected at the next frame. Failures are ignored and the write happens
//! on a detached thread — a missing, stale, or wedged sidebar must never
//! affect the editor.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// `doc!` comes from the crate-wide `#[macro_use] extern crate helix_view`.
use helix_view::Editor;

/// Re-send cadence while no send has succeeded yet.
const RETRY_INTERVAL: Duration = Duration::from_secs(1);
/// Give up retrying this long after the first attempt: the retry exists
/// only for the startup race where helix renders before the sidebar has
/// bound its socket. A sidebar that hasn't bound within this window is not
/// coming; later sends still fire on every buffer change as usual.
const RETRY_DEADLINE: Duration = Duration::from_secs(30);

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
        }
    }

    /// Called once per render: push a reveal if the focused document's path
    /// changed (or re-send the last one while the startup retry is live).
    /// Cheap when nothing changed: one borrowed path compare, no allocation.
    pub fn sync(&mut self, editor: &Editor) {
        let Some(socket) = &self.socket else {
            return;
        };
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
        // A newline in the file name would smuggle a second line into the
        // newline-delimited reveal protocol; better a missed reveal than a
        // corrupted command stream.
        if std::os::unix::ffi::OsStrExt::as_bytes(path.as_os_str()).contains(&b'\n') {
            return;
        }
        let now = Instant::now();
        self.first_attempt.get_or_insert(now);
        self.last_attempt = Some(now);
        let socket = socket.clone();
        let ready = Arc::clone(&self.ready);
        std::thread::spawn(move || {
            use std::io::Write as _;
            let Ok(mut stream) = std::os::unix::net::UnixStream::connect(&socket) else {
                return;
            };
            let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
            if writeln!(stream, "reveal-follow {}", path.display()).is_ok() {
                ready.store(true, Ordering::Relaxed);
            }
        });
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
