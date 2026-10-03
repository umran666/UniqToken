//! Opt-in, untimed Rust allocation diagnostics. Absent from default builds.

use crate::error::{core_error, CoreResult};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{
    AtomicBool, AtomicUsize,
    Ordering::{Relaxed, SeqCst},
};

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static RESERVED: AtomicBool = AtomicBool::new(false);

fn allocated(size: usize) {
    let live = LIVE.fetch_add(size, Relaxed) + size;
    if ACTIVE.load(Relaxed) {
        CALLS.fetch_add(1, Relaxed);
        REQUESTED.fetch_add(size, Relaxed);
        PEAK.fetch_max(live, Relaxed);
    }
}

// Delegate layouts and pointers unchanged to System; diagnostics only count.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc_zeroed(layout);
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout);
        LIVE.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let resized = System.realloc(pointer, layout, size);
        if !resized.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        resized
    }
}

pub(crate) type Counts = (usize, usize, usize, usize, usize);

pub(crate) fn measure<T>(call: impl FnOnce() -> T) -> CoreResult<(T, Counts)> {
    if RESERVED
        .compare_exchange(false, true, SeqCst, Relaxed)
        .is_err()
    {
        return core_error("allocation profiling is already active; use an isolated process");
    }
    let before = LIVE.load(Relaxed);
    CALLS.store(0, Relaxed);
    REQUESTED.store(0, Relaxed);
    PEAK.store(before, Relaxed);
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ACTIVE.store(false, Relaxed);
            RESERVED.store(false, SeqCst);
        }
    }
    let reset = Reset;
    ACTIVE.store(true, Relaxed);
    let result = call();
    ACTIVE.store(false, Relaxed);
    let counts = (
        CALLS.load(Relaxed),
        REQUESTED.load(Relaxed),
        PEAK.load(Relaxed),
        before,
        LIVE.load(Relaxed),
    );
    drop(reset);
    Ok((result, counts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_and_live_heap_are_observed_without_changing_output() {
        let (buffer, counts) = measure(|| std::hint::black_box(vec![7_u8; 8192])).unwrap();
        assert_eq!(buffer.len(), 8192);
        assert!(counts.0 >= 1);
        assert!(counts.1 >= 8192);
        assert!(counts.2 >= counts.3 + 8192);
        assert!(counts.4 >= counts.3 + 8192);
    }

    #[test]
    fn nested_measurement_rejects_instead_of_blocking() {
        let (inner, _) = measure(|| measure(|| 42)).unwrap();
        assert!(inner.is_err());
        assert_eq!(measure(|| 43).unwrap().0, 43);
    }

    #[test]
    fn panic_releases_measurement_reservation() {
        let failed = std::panic::catch_unwind(|| measure(|| panic!("injected profiler panic")));
        assert!(failed.is_err());
        assert_eq!(measure(|| 44).unwrap().0, 44);
    }
}
