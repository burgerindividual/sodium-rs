#![feature(portable_simd)]
// In most cases, we want to have functions and structs laying around
// that may come in handy in the future
#![allow(dead_code)]

mod bitset;
mod ffi;
mod graph;
mod math;
mod mem;
#[macro_use]
mod panic;
mod tests;
