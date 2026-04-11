use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, Ordering};

pub const MAILBOX_EMPTY: u8 = 0;
pub const MAILBOX_PROCESSING: u8 = 1;
pub const MAILBOX_READY: u8 = 2;

/// A lock-free, single-shot mailbox to route a single `EvaluationResponse` from the GPU back to the MCTS CPU worker.
pub struct AtomicMailbox<T> {
    state: AtomicU8,
    payload: UnsafeCell<Option<T>>,
}

// Mailbox must be able to be shared across threads.
unsafe impl<T: Send> Send for AtomicMailbox<T> {}
unsafe impl<T: Send> Sync for AtomicMailbox<T> {}

impl<T> Default for AtomicMailbox<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AtomicMailbox<T> {
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(MAILBOX_EMPTY),
            payload: UnsafeCell::new(None),
        }
    }

    /// The Inference Thread writes the response and immediately signals it as READY.
    #[inline(always)]
    pub fn write_and_notify(&self, response: T) {
        unsafe {
            *self.payload.get() = Some(response);
        }
        self.state.store(MAILBOX_READY, Ordering::Release);
    }

    /// Checks if the payload is ready without blocking or polling.
    #[inline(always)]
    pub fn is_ready(&self) -> bool {
        self.state.load(Ordering::Acquire) == MAILBOX_READY
    }

    /// Extracts the payload. Panics if called before state is READY.
    #[inline(always)]
    pub fn extract(&self) -> T {
        unsafe {
            let data = (*self.payload.get())
                .take()
                .expect("Mailbox Ready but payload was None");
            self.state.store(MAILBOX_EMPTY, Ordering::Release);
            data
        }
    }
}

// Bounded spin loop for awaiting a specific mailbox if needed.
// However, in optimal design, we poll an array of mailboxes.
pub fn spin_wait<T>(
    mailbox: &AtomicMailbox<T>,
    active_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<T, String> {
    let mut spins = 0;
    loop {
        if !active_flag.load(Ordering::Relaxed) {
            return Err("Training stopped".to_string());
        }

        if mailbox.is_ready() {
            return Ok(mailbox.extract());
        }

        spins += 1;
        if spins < 1000 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}
