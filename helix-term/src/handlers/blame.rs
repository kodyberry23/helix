use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{Arc, LazyLock},
    time::Duration,
};

use helix_event::{register_hook, AsyncHook};
use helix_loader::workspace_trust::{TrustQuery, WorkspaceTrust};
use helix_vcs::FileBlame;
use helix_view::{
    events::{ConfigDidChange, DocumentDidOpen},
    handlers::{BlameEvent, Handlers},
    Document, DocumentId,
};
use parking_lot::Mutex;
use tokio::{sync::Semaphore, time::Instant};

use crate::job;

/// Concurrent full-file blames allowed at once. gix blame is CPU-bound, so a burst
/// (every open document after enabling `auto-fetch`, or a refs update with many
/// buffers open) must not take over every core the editor shares with the user.
static BLAME_PERMITS: LazyLock<Semaphore> = LazyLock::new(|| {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    Semaphore::new((cores / 2).max(1))
});

/// Whether git may be run with full trust for `doc`'s workspace
pub fn git_trust_full(workspace_trust: &WorkspaceTrust, doc: &Document) -> bool {
    workspace_trust
        .query(doc.workspace_root(), TrustQuery::Git)
        .is_trusted()
}

/// Why [`request_blame`] sent nothing
#[derive(Debug, PartialEq, Eq)]
pub enum NotRequested {
    /// The document has no path, so there is nothing to blame
    NoPath,
    /// The blame the document holds was computed from the current history
    Current,
}

/// Ask for `doc`'s blame.
///
/// Every source of blame requests goes through here so the decision to skip work is
/// made in one place: a blame computed from the current history (`HEAD` and the
/// object replacements in effect) is kept unless `force` is set by a caller that has
/// already decided. `line` asks for that line's blame to be shown in the statusline
/// when the result lands, which is how `blame_line` works with auto-fetch off.
pub fn request_blame(
    handlers: &Handlers,
    doc: &Document,
    trust_full: bool,
    line: Option<u32>,
    force: bool,
) -> Result<(), NotRequested> {
    let Some(path) = doc.path() else {
        return Err(NotRequested::NoPath);
    };
    if !force && doc.blame_is_current() {
        return Err(NotRequested::Current);
    }
    helix_event::send_blocking(
        &handlers.blame,
        BlameEvent {
            path: path.to_path_buf(),
            doc_id: doc.id(),
            line,
            trust_full,
        },
    );
    Ok(())
}

/// `newer` supersedes `older` for the same document. The only field that carries
/// user intent is `line` (show this line once the blame lands), so a later
/// request without one must not drop it.
fn merge_requests(older: &BlameEvent, mut newer: BlameEvent) -> BlameEvent {
    newer.line = newer.line.or(older.line);
    newer
}

/// Per-document sequencing of blame computations, shared between the hook and
/// the tasks it spawns.
///
/// One blame runs at a time per document and results land in the order they were
/// requested, so a result computed from older history can never overwrite a newer
/// one. A request that arrives while a blame for its document is running waits in
/// a single slot and replaces whatever was waiting there: only the newest request
/// matters, and no work in progress is thrown away.
#[derive(Default)]
struct DocQueue {
    /// Documents with a blame running right now, each with at most one request
    /// to run afterwards.
    running: HashMap<DocumentId, Option<BlameEvent>>,
}

impl DocQueue {
    /// Admit a request. Returns it when it can start now; otherwise it waits for
    /// the blame already running for that document.
    fn submit(&mut self, event: BlameEvent) -> Option<BlameEvent> {
        match self.running.entry(event.doc_id) {
            Entry::Occupied(mut slot) => {
                let waiting = slot.get_mut();
                *waiting = Some(match waiting.take() {
                    Some(older) => merge_requests(&older, event),
                    None => event,
                });
                None
            }
            Entry::Vacant(slot) => {
                slot.insert(None);
                Some(event)
            }
        }
    }

    /// The blame for `doc_id` finished. Returns the request to start next, if one
    /// arrived meanwhile; otherwise the document is idle again.
    fn finished(&mut self, doc_id: DocumentId) -> Option<BlameEvent> {
        let next = self.running.get_mut(&doc_id)?.take();
        if next.is_none() {
            self.running.remove(&doc_id);
        }
        next
    }
}

/// Computes Git Blame for documents
#[derive(Default)]
pub struct BlameHandler {
    /// Requests received during the current debounce window, one per document.
    /// A newer request for a document replaces the older one, so a burst of
    /// events costs one blame per document rather than one per event, and no
    /// document is lost.
    pending: HashMap<DocumentId, BlameEvent>,
    queue: Arc<Mutex<DocQueue>>,
}

impl AsyncHook for BlameHandler {
    type Event = BlameEvent;

    fn handle_event(
        &mut self,
        event: Self::Event,
        _timeout: Option<tokio::time::Instant>,
    ) -> Option<tokio::time::Instant> {
        let event = match self.pending.get(&event.doc_id) {
            Some(older) => merge_requests(older, event),
            None => event,
        };
        self.pending.insert(event.doc_id, event);
        Some(Instant::now() + Duration::from_millis(50))
    }

    fn finish_debounce(&mut self) {
        for (_, event) in self.pending.drain() {
            // Bound before the `if`, so the lock is not held while spawning.
            let started = self.queue.lock().submit(event);
            if let Some(event) = started {
                spawn_blame(Arc::clone(&self.queue), event);
            }
        }
    }
}

/// Compute one blame off the main thread, hand the result to the editor, then start
/// the request that arrived for the same document while this one ran, if any.
fn spawn_blame(queue: Arc<Mutex<DocQueue>>, event: BlameEvent) {
    tokio::spawn(async move {
        let BlameEvent {
            path,
            doc_id,
            line,
            trust_full,
        } = event;
        let result = match BLAME_PERMITS.acquire().await {
            // gix blame walks the history of the whole file and can take seconds on
            // large or old files. Run it on the blocking pool so it never stalls the
            // async runtime or this hook's event loop.
            Ok(_permit) => {
                tokio::task::spawn_blocking(move || FileBlame::try_new(path, trust_full))
                    .await
                    .unwrap_or_else(|err| Err(anyhow::anyhow!("blame task failed: {err}")))
            }
            Err(_) => Err(anyhow::anyhow!("blame queue closed")),
        };
        job::dispatch(move |editor, _| {
            let auto_fetch = editor.config().inline_blame.auto_fetch;
            let Some(doc) = editor.document_mut(doc_id) else {
                return;
            };
            // History moved on while this blame ran (a rebase or checkout, possibly
            // undone since). A blame that is still current beats one that never was;
            // otherwise keep this one for now and compute a fresh one.
            let stale = matches!(&result, Ok(blame) if !blame.is_current());
            if stale && doc.blame_is_current() {
                return;
            }
            doc.file_blame = Some(result);
            if stale {
                if let Some(doc) = editor.document(doc_id) {
                    let trust_full = git_trust_full(&editor.workspace_trust, doc);
                    let _ = request_blame(&editor.handlers, doc, trust_full, line, true);
                }
                return;
            }
            if !auto_fetch {
                if let Some(line) = line {
                    crate::commands::blame_line_impl(editor, doc_id, line);
                } else {
                    editor.set_status("Blame for this file is now available")
                }
            }
        })
        .await;
        // Queued after this result's dispatch, so the next result lands after it.
        let next = queue.lock().finished(doc_id);
        if let Some(next) = next {
            spawn_blame(queue, next);
        }
    });
}

pub(super) fn register_hooks(_handlers: &Handlers) {
    register_hook!(move |event: &mut DocumentDidOpen<'_>| {
        if event.editor.config().inline_blame.auto_fetch {
            if let Some(doc) = event.editor.document(event.doc) {
                let trust_full = git_trust_full(&event.editor.workspace_trust, doc);
                let _ = request_blame(&event.editor.handlers, doc, trust_full, None, false);
            }
        }
        Ok(())
    });
    register_hook!(move |event: &mut ConfigDidChange<'_>| {
        if !event.old.inline_blame.auto_fetch && event.new.inline_blame.auto_fetch {
            // Fetch for every open document; those whose blame is already
            // current are skipped by `request_blame`.
            for doc in event.editor.documents() {
                let trust_full = git_trust_full(&event.editor.workspace_trust, doc);
                let _ = request_blame(&event.editor.handlers, doc, trust_full, None, false);
            }
        }
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(line: Option<u32>) -> BlameEvent {
        BlameEvent {
            path: "/repo/file.rs".into(),
            doc_id: DocumentId::default(),
            line,
            trust_full: true,
        }
    }

    #[tokio::test]
    async fn later_request_for_a_document_replaces_the_earlier_one() {
        let mut handler = BlameHandler::default();
        let first = handler.handle_event(request(Some(5)), None);
        assert!(first.is_some(), "a request must open a debounce window");
        handler.handle_event(request(Some(9)), first);
        assert_eq!(handler.pending.len(), 1, "one pending blame per document");
        assert_eq!(
            handler.pending[&DocumentId::default()].line,
            Some(9),
            "the newest request wins"
        );
        handler.handle_event(request(None), first);
        assert_eq!(
            handler.pending[&DocumentId::default()].line,
            Some(9),
            "a later request without a line keeps the line the user asked for"
        );
    }

    #[test]
    fn one_blame_at_a_time_per_document_and_only_the_newest_waits() {
        let doc = DocumentId::default();
        let mut queue = DocQueue::default();
        assert!(queue.submit(request(Some(1))).is_some(), "idle: starts now");
        assert!(queue.submit(request(Some(2))).is_none(), "busy: waits");
        assert!(queue.submit(request(Some(3))).is_none(), "busy: replaces 2");
        assert!(queue.submit(request(None)).is_none(), "busy: replaces 3");
        let next = queue
            .finished(doc)
            .expect("the waiting request starts next");
        assert_eq!(
            next.line,
            Some(3),
            "only the newest waiting request survives, keeping the requested line"
        );
        assert!(
            queue.finished(doc).is_none(),
            "nothing was waiting behind it"
        );
        assert!(queue.submit(request(Some(4))).is_some(), "idle again");
        assert!(queue.finished(doc).is_none());
        assert!(queue.running.is_empty(), "idle documents are not tracked");
    }
}
