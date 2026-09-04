use core::{cell::UnsafeCell, fmt, mem::ManuallyDrop};

use crate::lock::{rank, RankData, RwLock, RwLockReadGuard, RwLockWriteGuard};
use crate::resource::DestructibleResourceState;

/// A guard that provides read access to snatchable data.
pub struct SnatchGuard<'a>(RwLockReadGuard<'a, ()>);
/// A guard that allows snatching the snatchable data.
pub struct ExclusiveSnatchGuard<'a>(#[expect(dead_code)] RwLockWriteGuard<'a, ()>);

/// A value that is mostly immutable but can be "snatched" if we need to destroy
/// it early.
///
/// In order to safely access the underlying data, the device's global snatchable
/// lock must be taken. To guarantee it, methods take a read or write guard of that
/// special lock.
pub struct SnatchableInner<T> {
    value: UnsafeCell<T>,
}

pub type Snatchable<T> = SnatchableInner<Option<T>>;

impl<T> Snatchable<T> {
    pub fn new(val: T) -> Self {
        SnatchableInner {
            value: UnsafeCell::new(Some(val)),
        }
    }

    #[allow(dead_code)]
    pub fn empty() -> Self {
        SnatchableInner {
            value: UnsafeCell::new(None),
        }
    }

    /// Get read access to the value. Requires a the snatchable lock's read guard.
    pub fn get<'a>(&'a self, _guard: &'a SnatchGuard) -> Option<&'a T> {
        unsafe { (*self.value.get()).as_ref() }
    }

    /// Take the value. Requires a the snatchable lock's write guard.
    pub fn snatch(&self, _guard: &mut ExclusiveSnatchGuard) -> Option<T> {
        unsafe { (*self.value.get()).take() }
    }

    /// Take the value without a guard. This can only be used with exclusive access
    /// to self, so it does not require locking.
    ///
    /// Typically useful in a drop implementation.
    pub fn take(&mut self) -> Option<T> {
        self.value.get_mut().take()
    }
}

// Can't safely print the contents of a snatchable object without holding
// the lock.
impl<T> fmt::Debug for SnatchableInner<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<snatchable>")
    }
}

unsafe impl<T> Sync for SnatchableInner<T> {}

/// A value that is mostly immutable but can be "snatched" if we need to destroy
/// it early.
///
/// In order to safely access the underlying data, the device's global snatchable
/// lock must be taken. To guarantee it, methods take a read or write guard of that
/// special lock.
pub type Snatchable2<T> = SnatchableInner<DestructibleResourceState<T>>;

impl<T> Snatchable2<T> {
    pub fn new(val: T) -> Self {
        SnatchableInner {
            value: UnsafeCell::new(DestructibleResourceState::Valid(val)),
        }
    }

    pub fn invalid() -> Self {
        SnatchableInner {
            value: UnsafeCell::new(DestructibleResourceState::Invalid),
        }
    }

    /// Get read access to the value. Requires a the snatchable lock's read guard.
    pub fn get<'a>(&'a self, _guard: &'a SnatchGuard) -> DestructibleResourceState<&'a T> {
        unsafe { (*self.value.get()).as_ref() }
    }

    /// Take the value. Requires a the snatchable lock's write guard.
    pub fn snatch(&self, _guard: &mut ExclusiveSnatchGuard) -> DestructibleResourceState<T> {
        unsafe { (*self.value.get()).take() }
    }

    /// Take the value without a guard. This can only be used with exclusive access
    /// to self, so it does not require locking.
    ///
    /// Typically useful in a drop implementation.
    pub fn take(&mut self) -> DestructibleResourceState<T> {
        self.value.get_mut().take()
    }
}

use trace::LockTrace;
#[cfg(all(debug_assertions, feature = "std"))]
mod trace {
    use core::{cell::Cell, fmt, panic::Location};
    use std::{backtrace::Backtrace, thread};

    pub(super) struct LockTrace {
        purpose: &'static str,
        caller: &'static Location<'static>,
        backtrace: Backtrace,
    }

    impl fmt::Display for LockTrace {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "a {} lock at {}\n{}",
                self.purpose, self.caller, self.backtrace
            )
        }
    }

    impl LockTrace {
        // ⛔ `try_with`, NOT `take`/`set`: THIS RUNS FROM DESTRUCTORS AND A PANIC THERE ABORTS.
        // `LocalKey::take`/`set` panic once the thread-local has been destroyed. Dropping any wgpu
        // resource reaches `Buffer::unmap` -> `SnatchLock::read` -> here, and resources routinely
        // outlive this TLS: Ferric caches `wgpu::Buffer` in its own `thread_local!` (INFOBUFS,
        // UNIBUFS), and the destruction ORDER between two crates' thread-locals is unspecified. When
        // this one goes first, the panic lands inside a `Drop`, which is a `fatal runtime error:
        // thread local panicked on drop` and an ABORT — after the program has printed its complete,
        // correct output. `examples/bandwidth` did exactly that in CI: exit 134 with every number
        // right. It is latent in ANY program exiting with a cached wgpu resource still alive, and
        // it is debug-only, because this whole module is `cfg(debug_assertions)` — so it fires in CI
        // and never in a release build anyone benchmarks.
        //
        // The recursion check is a debug aid. Skipping it when the TLS is already gone costs a
        // diagnostic that could not have fired anyway; panicking costs the entire run.
        #[track_caller]
        pub(super) fn enter(purpose: &'static str) {
            let caller = Location::caller();
            let _ = SNATCH_LOCK_TRACE.try_with(|slot| {
                let new = LockTrace {
                    purpose,
                    caller,
                    backtrace: Backtrace::capture(),
                };

                if let Some(prev) = slot.take() {
                    let current = thread::current();
                    let name = current.name().unwrap_or("<unnamed>");
                    panic!(
                        "thread '{name}' attempted to acquire a snatch lock recursively.\n\
                     - Currently trying to acquire {new}\n\
                     - Previously acquired {prev}",
                    );
                } else {
                    slot.set(Some(new));
                }
            });
        }

        pub(super) fn exit() {
            // Same reason as `enter`: reached from `Drop`, so it must not panic on a dead TLS.
            let _ = SNATCH_LOCK_TRACE.try_with(|slot| slot.take());
        }
    }

    std::thread_local! {
        static SNATCH_LOCK_TRACE: Cell<Option<LockTrace>> = const { Cell::new(None) };
    }
}
#[cfg(not(all(debug_assertions, feature = "std")))]
mod trace {
    pub(super) struct LockTrace {
        _private: (),
    }

    impl LockTrace {
        pub(super) fn enter(_purpose: &'static str) {}
        pub(super) fn exit() {}
    }
}

/// A Device-global lock for all snatchable data.
pub struct SnatchLock {
    lock: RwLock<()>,
}

impl SnatchLock {
    /// The safety of `Snatchable::get` and `Snatchable::snatch` rely on their using of the
    /// right SnatchLock (the one associated to the same device). This method is unsafe
    /// to force force sers to think twice about creating a SnatchLock. The only place this
    /// method should be called is when creating the device.
    pub unsafe fn new(rank: rank::LockRank) -> Self {
        SnatchLock {
            lock: RwLock::new(rank, ()),
        }
    }

    /// Request read access to snatchable resources.
    #[track_caller]
    pub fn read(&self) -> SnatchGuard<'_> {
        LockTrace::enter("read");
        SnatchGuard(self.lock.read())
    }

    /// Request write access to snatchable resources.
    ///
    /// This should only be called when a resource needs to be snatched. This has
    /// a high risk of causing lock contention if called concurrently with other
    /// wgpu work.
    #[track_caller]
    pub fn write(&self) -> ExclusiveSnatchGuard<'_> {
        LockTrace::enter("write");
        ExclusiveSnatchGuard(self.lock.write())
    }

    #[track_caller]
    pub unsafe fn force_unlock_read(&self, data: RankData) {
        // This is unsafe because it can cause deadlocks if the lock is held.
        // It should only be used in very specific cases, like when a resource
        // needs to be snatched in a panic handler.
        LockTrace::exit();
        unsafe { self.lock.force_unlock_read(data) };
    }
}

impl SnatchGuard<'_> {
    /// Forget the guard, leaving the lock in a locked state with no guard.
    ///
    /// This is equivalent to `std::mem::forget`, but preserves the information about the lock
    /// rank.
    pub fn forget(this: Self) -> RankData {
        // Cancel the drop implementation of the current guard.
        let manually_drop = ManuallyDrop::new(this);

        // As we are unable to destructure out of this guard due to the drop implementation,
        // so we manually read the inner value.
        // SAFETY: This is safe because we never access the original guard again.
        let inner_guard = unsafe { core::ptr::read(&manually_drop.0) };

        RwLockReadGuard::forget(inner_guard)
    }
}

impl Drop for SnatchGuard<'_> {
    fn drop(&mut self) {
        LockTrace::exit();
    }
}

impl Drop for ExclusiveSnatchGuard<'_> {
    fn drop(&mut self) {
        LockTrace::exit();
    }
}
