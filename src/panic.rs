use core::fmt::Write;
use std::panic::PanicHookInfo;
use std::string::String;

pub type PanicHandlerFn = extern "C" fn(data: *const u8, len: i32) -> !;

static mut PANIC_HANDLER: Option<PanicHandlerFn> = None;

pub fn set_panic_handler(panic_handler_fn_ptr: PanicHandlerFn) {
    unsafe {
        PANIC_HANDLER = Some(panic_handler_fn_ptr);
        std::panic::set_hook(Box::new(panic_hook));
    }
}

fn panic_hook(info: &PanicHookInfo) {
    let mut message = String::new();
    let _ = write!(&mut message, "{}", info);
    // we can't panic if this fails

    unsafe {
        if let Some(panic_handler_fn) = PANIC_HANDLER {
            panic_handler_fn(message.as_ptr(), message.len() as i32)
        }
        // can't really do anything if it's not populated
    }
}
