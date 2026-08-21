use std::{collections::HashMap, path::PathBuf, time::Duration};

use helix_event::register_hook;
use helix_loader::workspace_trust::TrustQuery;
use helix_vcs::FileBlame;
use helix_view::{
    events::{ConfigDidChange, DocumentDidOpen},
    handlers::{BlameEvent, Handlers},
    DocumentId,
};
use tokio::time::Instant;

use crate::job;

/// A blame request waiting for the debounce window to close
struct PendingBlame {
    path: PathBuf,
    trust_full: bool,
    /// If `Some`, show the blame for this line in the statusline once it is available
    line: Option<u32>,
}

/// Computes Git Blame for documents
#[derive(Default)]
pub struct BlameHandler {
    /// Requests received during the current debounce window, one per document.
    /// A newer request for a document replaces the older one, so a burst of
    /// events (e.g. every open document after enabling `auto-fetch`) costs one
    /// blame per document rather than one per event, and no document is lost.
    pending: HashMap<DocumentId, PendingBlame>,
}

impl helix_event::AsyncHook for BlameHandler {
    type Event = BlameEvent;

    fn handle_event(
        &mut self,
        event: Self::Event,
        _timeout: Option<tokio::time::Instant>,
    ) -> Option<tokio::time::Instant> {
        self.pending.insert(
            event.doc_id,
            PendingBlame {
                path: event.path,
                trust_full: event.trust_full,
                line: event.line,
            },
        );
        Some(Instant::now() + Duration::from_millis(50))
    }

    fn finish_debounce(&mut self) {
        for (doc_id, request) in self.pending.drain() {
            let PendingBlame {
                path,
                trust_full,
                line,
            } = request;
            tokio::spawn(async move {
                // gix blame walks the history of the whole file and can take
                // seconds on large or old files. Run it on the blocking pool so
                // it never stalls the async runtime or this hook's event loop.
                let result =
                    tokio::task::spawn_blocking(move || FileBlame::try_new(path, trust_full))
                        .await
                        .unwrap_or_else(|err| Err(anyhow::anyhow!("blame task failed: {err}")));
                job::dispatch(move |editor, _| {
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
