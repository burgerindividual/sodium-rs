use std::io::{Cursor, Write};
use std::panic::{self, PanicHookInfo};
use std::sync::OnceLock;

use crash_handler::{make_crash_event, CrashContext, CrashEventResult, CrashHandler};

pub type PanicHandlerFn = extern "C" fn(data: *const u8, len: i32) -> !;

static PANIC_HANDLER: OnceLock<PanicHandlerFn> = OnceLock::new();
static mut CRASH_HANDLER: OnceLock<CrashHandler> = OnceLock::new();

pub fn set_panic_handler(panic_handler_fn_ptr: PanicHandlerFn) {
    let _ = PANIC_HANDLER.set(panic_handler_fn_ptr);
    panic::set_hook(Box::new(panic_hook));
    unsafe {
        if let Ok(crash_handler) = CrashHandler::attach(make_crash_event(crash_hook)) {
            let _ = CRASH_HANDLER.set(crash_handler);
        }
    }
}

const CRASH_STRING: [u8; 59] = *b"HARD CRASH ENCOUNTERED IN NATIVE CODE. Crash context dump:\n";
const STRING_BUFFER_LEN: usize = 4000;
static mut STRING_BUFFER: [u8; STRING_BUFFER_LEN] = {
    let mut buffer = [0; STRING_BUFFER_LEN];

    let mut idx = 0;
    while idx < CRASH_STRING.len() {
        buffer[idx] = CRASH_STRING[idx];
        idx += 1;
    }

    buffer
};

fn panic_hook(info: &PanicHookInfo) {
    unsafe {
        let mut cursor = Cursor::new(&mut STRING_BUFFER[..]);
        let _ = writeln!(cursor, "{info}");
        // we can't panic if this fails

        // if the crash handler exists, this will detach it
        drop(CRASH_HANDLER.take());

        if let Some(panic_handler_fn) = PANIC_HANDLER.get() {
            panic_handler_fn(
                (&raw const STRING_BUFFER).cast::<u8>(),
                cursor.position() as i32,
            );
        }
        // we can't really do anything if it's not populated
    }
}

fn crash_hook(context: &CrashContext) -> CrashEventResult {
    const BYTE_HEX_LUT: [u8; 512] = *b"000102030405060708090A0B0C0D0E0F\
                            101112131415161718191A1B1C1D1E1F\
                            202122232425262728292A2B2C2D2E2F\
                            303132333435363738393A3B3C3D3E3F\
                            404142434445464748494A4B4C4D4E4F\
                            505152535455565758595A5B5C5D5E5F\
                            606162636465666768696A6B6C6D6E6F\
                            707172737475767778797A7B7C7D7E7F\
                            808182838485868788898A8B8C8D8E8F\
                            909192939495969798999A9B9C9D9E9F\
                            A0A1A2A3A4A5A6A7A8A9AAABACADAEAF\
                            B0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF\
                            C0C1C2C3C4C5C6C7C8C9CACBCCCDCECF\
                            D0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF\
                            E0E1E2E3E4E5E6E7E8E9EAEBECEDEEEF\
                            F0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF";

    unsafe {
        let mut buf_idx = CRASH_STRING.len();
        for &byte in context.as_bytes() {
            let lut_idx = byte as usize * 2;
            STRING_BUFFER[buf_idx] = BYTE_HEX_LUT[lut_idx];
            STRING_BUFFER[buf_idx + 1] = BYTE_HEX_LUT[lut_idx + 1];
            buf_idx += 2;
        }
        STRING_BUFFER[buf_idx] = b'\n';
        buf_idx += 1;

        if let Some(panic_handler_fn) = PANIC_HANDLER.get() {
            panic_handler_fn((&raw const STRING_BUFFER).cast::<u8>(), buf_idx as i32);
        } else {
            CrashEventResult::Handled(false)
        }
    }
}
