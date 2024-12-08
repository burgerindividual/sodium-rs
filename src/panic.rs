use std::io::{Cursor, Write};
use std::panic::{self, PanicHookInfo};
use std::sync::OnceLock;

use crate::jni::JPtr;

pub type PanicHandlerFn = extern "C" fn(data: JPtr<u8>, len: i32) -> !;

static EXTERNAL_PANIC_HANDLER: OnceLock<PanicHandlerFn> = OnceLock::new();

pub fn set_panic_handler(panic_handler_fn_ptr: PanicHandlerFn) {
    let _ = EXTERNAL_PANIC_HANDLER.set(panic_handler_fn_ptr);
    panic::set_hook(Box::new(panic_hook));
}

fn panic_hook(info: &PanicHookInfo) {
    let mut buffer = [0_u8; 1000];
    let mut cursor = Cursor::new(buffer.as_mut());
    let _ = writeln!(cursor, "{info}"); // we can't panic if this fails
    let length = cursor.position();

    // we can't really do anything if the panic handler function isn't populated
    if let Some(panic_handler_fn) = EXTERNAL_PANIC_HANDLER.get() {
        panic_handler_fn(buffer.as_ptr().into(), length as i32);
    }
}
