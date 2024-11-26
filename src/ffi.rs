#![allow(non_snake_case)]

use std::boxed::Box;
use std::ptr;

use context::GraphSearchContext;
use core_simd::simd::Simd;

use crate::results::SectionBitArray;
use crate::graph::*;
use crate::jni::*;
use crate::math::*;
use crate::mem::LibcAllocVtable;
use crate::panic::PanicHandlerFn;

#[repr(C)]
struct FFISlice<T> {
    count: u32,
    data_ptr: JPtr<T>,
}

impl<T> From<&[T]> for FFISlice<T> {
    fn from(value: &[T]) -> Self {
        Self {
            count: value.len().try_into().expect("len is not a valid u32"),
            data_ptr: if value.len() == 0 {
                ptr::null::<T>().into()
            } else {
                value.as_ptr().into()
            },
        }
    }
}

#[repr(C)]
struct FFIFrustum {
    planes: [[f32; 6]; 4],
    pos_int: [i32; 3],
    pos_frac: [f32; 3],
}

#[repr(C)]
struct FFISectionOpaqueBlocks([u8; 512]);

#[repr(C)]
struct FFISectionBitArray {
    size: u32,
    data: JPtr<u64>,
}

impl From<SectionBitArray> for FFISectionBitArray {
    fn from(value: SectionBitArray) -> Self {
        Self {
            size: value.size,
            data: value.data.as_ptr().into(),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn Java_me_jellysquid_mods_sodium_ffi_core_CoreLib_setAllocator(
    _: *mut JEnv,
    _: *mut JClass,
    vtable: JPtr<LibcAllocVtable>,
) -> bool {
    if let Some(vtable) = vtable.as_ptr().as_ref() {
        crate::mem::set_allocator(vtable)
    } else {
        true
    }
}

#[no_mangle]
pub unsafe extern "C" fn Java_me_jellysquid_mods_sodium_ffi_core_CoreLib_setPanicHandler(
    _: *mut JEnv,
    _: *mut JClass,
    panic_handler_fn_ptr: JFnPtr<PanicHandlerFn>,
) -> bool {
    if let Some(panic_handler_fn_ptr) = panic_handler_fn_ptr.as_fn_ptr() {
        crate::panic::set_panic_handler(panic_handler_fn_ptr);
        false
    } else {
        true
    }
}

#[no_mangle]
pub unsafe extern "C" fn Java_me_jellysquid_mods_sodium_ffi_core_CoreLib_graphCreate(
    _: *mut JEnv,
    _: *mut JClass,
) -> JPtrMut<Graph> {
    let graph = Box::new(Graph::new());

    Box::leak(graph).into()
}

#[no_mangle]
pub unsafe extern "C" fn Java_me_jellysquid_mods_sodium_ffi_core_CoreLib_graphSetSection(
    _: *mut JEnv,
    _: *mut JClass,
    graph: JPtrMut<Graph>,
    x: Jint,
    y: Jint,
    z: Jint,
    opaque_blocks: JPtr<FFISectionOpaqueBlocks>,
) {
    let graph = graph.into_mut_ref();

    graph.set_section(i32x3::from_xyz(x, y, z), &opaque_blocks.as_ref().0);
}

#[no_mangle]
pub unsafe extern "C" fn Java_me_jellysquid_mods_sodium_ffi_core_CoreLib_graphRemoveSection(
    _: *mut JEnv,
    _: *mut JClass,
    graph: JPtrMut<Graph>,
    x: Jint,
    y: Jint,
    z: Jint,
) {
    let graph = graph.into_mut_ref();
    graph.remove_section(i32x3::from_xyz(x, y, z));
}

#[no_mangle]
pub unsafe extern "C" fn Java_me_jellysquid_mods_sodium_ffi_core_CoreLib_graphSearch(
    _: *mut JEnv,
    _: *mut JClass,
    graph: JPtrMut<Graph>,
    frustum: JPtr<FFIFrustum>,
    search_distance: Jfloat,
    world_bottom_section_y: Jbyte,
    world_top_section_y: Jbyte,
    use_occlusion_culling: Jboolean,
    return_value: JPtrMut<FFISectionBitArray>,
) {
    let graph = graph.into_mut_ref();
    let frustum = frustum.as_ref();

    let simd_camera_pos_int = Simd::from_array(frustum.pos_int);
    let simd_camera_pos_frac = Simd::from_array(frustum.pos_frac);
    let simd_frustum_planes = [
        Simd::from_array(frustum.planes[0]),
        Simd::from_array(frustum.planes[1]),
        Simd::from_array(frustum.planes[2]),
        Simd::from_array(frustum.planes[3]),
    ];

    let coord_context = GraphSearchContext::new(
        simd_frustum_planes,
        simd_camera_world_pos,
        search_distance,
        world_bottom_section_y,
        world_top_section_y,
    );

    graph.cull(&coord_context);

    *return_value.into_mut_ref() = graph.results.into();
}

#[no_mangle]
pub unsafe extern "C" fn Java_me_jellysquid_mods_sodium_ffi_core_CoreLib_graphDelete(
    _: *mut JEnv,
    _: *mut JClass,
    graph: JPtrMut<Graph>,
) {
    let graph_box = Box::from_raw(graph.into_mut_ref());
    drop(graph_box);
}
