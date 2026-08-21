use std::{collections::HashMap, sync::LazyLock, time::Duration};

use helix_event::{register_hook, AsyncHook, TaskController};
use helix_loader::workspace_trust::TrustQuery;
use helix_vcs::FileBlame;
use helix_view::{
    events::{ConfigDidChange, DocumentDidOpen},
    handlers::{BlameEvent, Handlers},
    DocumentId,
};
use tokio::{sync::Semaphore, time::Instant};

use crate::job;

/// Concurrent full-file blames allowed at once. gix blame is CPU-bound, so a burst
/// (every open document after enabling `auto-fetch`, or a refs update with many
/// buffers open) must not take over every core the editor shares with the user.
static BLAME_PERMITS: LazyLock<Semaphore> = LazyLock::new(|| {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    Semaphore::new((cores / 2).max(1))
});

/// Computes Git Blame for documents
#[derive(Default)]
pub struct BlameHandler {
    /// Requests received during the current debounce window, one per document.
    /// A newer request for a document replaces the older one, so a burst of
    /// events costs one blame per document rather than one per event, and no
    /// document is lost.
    pending: HashMap<DocumentId, BlameEvent>,
    /// One controller per document. Starting a new blame cancels the one still
    /// in flight for that document, so a result computed against an older
    /// `HEAD` can never land after (and overwrite) a newer one. Entries are a
    /// few words each and live for the session.
    in_flight: HashMap<DocumentId, TaskController>,
}

impl AsyncHook for BlameHandler {
    type Event = BlameEvent;

    fn handle_event(
        &mut self,
        event: Self::Event,
        _timeout: Option<tokio::time::Instant>,
    ) -> Option<tokio::time::Instant> {
        self.pending.insert(event.doc_id, event);
        Some(Instant::now() + Duration::from_millis(50))
    }

    fn finish_debounce(&mut self) {
        for (doc_id, request) in self.pending.drain() {
            let cancel = self.in_flight.entry(doc_id).or_default().restart();
            tokio::spawn(async move {
                let BlameEvent {
                    path,
                    trust_full,
                    line,
                    ..
                } = request;
                let result = {
                    let Ok(_permit) = BLAME_PERMITS.acquire().await else {
                        return;
                    };
                    // Superseded while waiting for a permit: skip the work entirely.
                    if cancel.is_canceled() {
                        return;
                    }
                    // gix blame walks the history of the whole file and can take
                    // seconds on large or old files. Run it on the blocking pool so
                    // it never stalls the async runtime or this hook's event loop.
                    tokio::task::spawn_blocking(move || FileBlame::try_new(path, trust_full))
                        .await
                        .unwrap_or_else(|err| Err(anyhow::anyhow!("blame task failed: {err}")))
                };
                if cancel.is_canceled() {
                    return;
                }
                job::dispatch(move |editor, _| {
                    // Re-checked on the main thread: a newer request may have been
                    // started while this result was queued for dispatch.
                    if cancel.is_canceled() {
                        return;
                    }
                    let Some(doc) = editor.document_mut(doc_id) else {
                        return;
                    };
                    doc.file_blame = Some(result);
                    if !editor.config().inline_blame.auto_fetch {
                        if let Some(line) = line {
                            crate::commands::blame_line_impl(editor, doc_id, line);
                        } else {
                            editor.set_status("Blame for this file is now available")
                        }
                    }
                })
                .await;
            });
        }
    }
}

pub(super) fn register_hooks(handlers: &Handlers) {
    let tx = handlers.blame.clone();
    register_hook!(move |event: &mut DocumentDidOpen<'_>| {
        if event.editor.config().inline_blame.auto_fetch {
            let trust_full = event
                .editor
                .document(event.doc)
                .map(|doc| {
                    event
                        .editor
                        .workspace_trust
                        .query(doc.workspace_root(), TrustQuery::Git)
                        .is_trusted()
                })
                .unwrap_or(false);
            helix_event::send_blocking(
                &tx,
                BlameEvent {
                    path: event.path.to_path_buf(),
                    doc_id: event.doc,
                    line: None,
                    trust_full,
                },
            );
        }
        Ok(())
    });
    let tx = handlers.blame.clone();
    register_hook!(move |event: &mut ConfigDidChange<'_>| {
        let has_enabled_inline_blame =
            !event.old.inline_blame.auto_fetch && event.new.inline_blame.auto_fetch;

        if has_enabled_inline_blame {
            // request blame for all documents, since any of them could have
            // outdated blame
            let workspace_trust = &event.editor.workspace_trust;
            let doc_paths: Vec<_> = event
                .editor
                .documents()
                .filter_map(|doc| {
                    doc.path().map(|path| {
                        let trust_full = workspace_trust
                            .query(doc.workspace_root(), TrustQuery::Git)
                            .is_trusted();
                        (path.to_path_buf(), doc.id(), trust_full)
                    })
                })
                .collect();
            for (path, doc_id, trust_full) in doc_paths {
                helix_event::send_blocking(
                    &tx,
                    BlameEvent {
                        path,
                        doc_id,
                        line: None,
                        trust_full,
                    },
                );
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
    }
}
