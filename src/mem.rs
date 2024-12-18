use core::alloc::{GlobalAlloc, Layout};
use core::ptr;

pub type AlignedAllocFn = unsafe extern "C" fn(alignment: usize, size: usize) -> *mut u8;
pub type AlignedFreeFn = unsafe extern "C" fn(ptr: *mut u8);
pub type ReallocFn = unsafe extern "C" fn(ptr: *mut u8, new_size: usize) -> *mut u8;
pub type CallocFn = unsafe extern "C" fn(num_elements: usize, element_size: usize) -> *mut u8;

#[cfg(not(test))]
#[global_allocator]
static mut GLOBAL_ALLOC: GlobalLibcAllocator = GlobalLibcAllocator::uninit();

#[cfg(not(test))]
pub fn set_allocator(vtable: LibcAllocVtable) {
    unsafe {
        GLOBAL_ALLOC = vtable.into();
    }
}

#[cfg(test)]
pub fn set_allocator(_: LibcAllocVtable) {
    // should not be called when testing
    unreachable!();
}

pub struct LibcAllocVtable {
    pub aligned_alloc_fn_ptr: AlignedAllocFn,
    pub aligned_free_fn_ptr: AlignedFreeFn,
    pub realloc_fn_ptr: ReallocFn,
    pub calloc_fn_ptr: CallocFn,
}

pub struct GlobalLibcAllocator {
    vtable: Option<LibcAllocVtable>,
}

impl GlobalLibcAllocator {
    pub const fn uninit() -> Self {
        GlobalLibcAllocator { vtable: None }
    }

    fn vtable(&self) -> &LibcAllocVtable {
        self.vtable
            .as_ref()
            .expect("Allocator functions not initialized")
    }
}

impl From<LibcAllocVtable> for GlobalLibcAllocator {
    fn from(value: LibcAllocVtable) -> Self {
        GlobalLibcAllocator {
            vtable: Some(value),
        }
    }
}

unsafe impl GlobalAlloc for GlobalLibcAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        (self.vtable().aligned_alloc_fn_ptr)(layout.align(), layout.size())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _: Layout) {
        (self.vtable().aligned_free_fn_ptr)(ptr)
    }

    // Mirrors the unix libc impl for GlobalAlloc

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
            (self.vtable().calloc_fn_ptr)(layout.size(), 1)
        } else {
            let ptr = self.alloc(layout);
            if !ptr.is_null() {
                ptr::write_bytes(ptr, 0, layout.size());
            }
            ptr
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
            (self.vtable().realloc_fn_ptr)(ptr, new_size)
        } else {
            // Docs for GlobalAlloc::realloc require this to be valid:
            let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());

            let new_ptr = self.alloc(new_layout);
            if !new_ptr.is_null() {
                let size = new_size.min(layout.size());
                ptr::copy_nonoverlapping(ptr, new_ptr, size);
                self.dealloc(ptr, layout);
            }
            new_ptr
        }
    }
}

// The minimum alignment guaranteed by the architecture. This value is used to
// add fast paths for low alignment values.
#[allow(dead_code)]
const MIN_ALIGN: usize = if cfg!(any(
    all(
        target_arch = "riscv32",
        any(target_os = "espidf", target_os = "zkvm")
    ),
    all(target_arch = "xtensa", target_os = "espidf"),
)) {
    // The allocator on the esp-idf and zkvm platforms guarantees 4 byte alignment.
    4
} else if cfg!(any(
    target_arch = "x86",
    target_arch = "arm",
    target_arch = "m68k",
    target_arch = "csky",
    target_arch = "mips",
    target_arch = "mips32r6",
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "sparc",
    target_arch = "wasm32",
    target_arch = "hexagon",
    target_arch = "riscv32",
    target_arch = "xtensa",
)) {
    8
} else if cfg!(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm64ec",
    target_arch = "loongarch64",
    target_arch = "mips64",
    target_arch = "mips64r6",
    target_arch = "s390x",
    target_arch = "sparc64",
    target_arch = "riscv64",
    target_arch = "wasm64",
)) {
    16
} else {
    panic!("add a value for MIN_ALIGN")
};
