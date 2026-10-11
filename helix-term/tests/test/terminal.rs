use super::*;

/// A terminal that can no longer be read (its pane or window closed) ends the
/// event loop like SIGHUP does, instead of panicking on the error, so the
/// usual shutdown (language servers, sidebar farewell) still runs.
#[tokio::test(flavor = "multi_thread")]
async fn a_terminal_read_error_ends_the_event_loop() -> anyhow::Result<()> {
    let mut app = AppBuilder::new().build()?;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut rx_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

    tx.send(Err(std::io::Error::new(
        std::io::ErrorKind::BrokenPipe,
        "pty closed",
    )))?;
    let keep_running = app.event_loop_until_idle(&mut rx_stream).await;
    assert!(!keep_running, "the loop must end once the terminal is gone");

    // Nothing was asked to quit, so this is the terminal-lost path, not a
    // regular :quit.
    assert!(!app.editor.should_close());
    Ok(())
}
