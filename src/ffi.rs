#![allow(non_snake_case)]

use std::boxed::Box;

use context::GraphSearchContext;
use core_simd::simd::Simd;
use tile::NodeStorage;

use crate::graph::*;
use crate::math::*;
use crate::mem::*;
use crate::panic::PanicHandlerFn;
use crate::{mem, panic};

type JEnv = core::ffi::c_void;
type JClass = core::ffi::c_void;

#[repr(C)]
pub struct FFISlice<T> {
    count: u32,
    data_ptr: *const T,
}

impl<T> From<&[T]> for FFISlice<T> {
    fn from(value: &[T]) -> Self {
        Self {
            count: value.len().try_into().expect("len is not a valid u32"),
            data_ptr: value.as_ptr(),
        }
    }
}

#[repr(C)]
pub struct FFICamera {
    frustum_planes: [[f32; 6]; 4],
    pos: [f64; 3],
}

#[repr(C)]
pub struct FFISectionTraversableBlocks([u8; 512]);

#[repr(C)]
pub struct FFIVisibleSectionsTile {
    visible_sections_ptr: *const [u64; 8],
    origin_section_coords: [i32; 3],
}

impl FFIVisibleSectionsTile {
    pub fn new(visible_sections: *const NodeStorage, origin_section_coords: i32x3) -> Self {
        Self {
            visible_sections_ptr: visible_sections.cast::<[u64; 8]>(),
            origin_section_coords: origin_section_coords.to_array(),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn Java_net_caffeinemc_mods_sodium_ffi_NativeCull_setAllocator(
    _: *mut JEnv,
    _: *mut JClass,
    aligned_alloc_fn_ptr: AlignedAllocFn,
    aligned_free_fn_ptr: AlignedFreeFn,
    realloc_fn_ptr: ReallocFn,
    calloc_fn_ptr: CallocFn,
) {
    mem::set_allocator(LibcAllocVtable {
        aligned_alloc_fn_ptr,
        aligned_free_fn_ptr,
        realloc_fn_ptr,
        calloc_fn_ptr,
    });
}

#[no_mangle]
pub unsafe extern "C" fn Java_net_caffeinemc_mods_sodium_ffi_NativeCull_setPanicHandler(
    _: *mut JEnv,
    _: *mut JClass,
    panic_handler_fn_ptr: PanicHandlerFn,
) {
    if cfg!(feature = "panic_handler") {
        panic::set_panic_handler(panic_handler_fn_ptr);
    }
}

#[no_mangle]
pub extern "C" fn Java_net_caffeinemc_mods_sodium_ffi_NativeCull_graphCreate(
    _: *mut JEnv,
    _: *mut JClass,
    render_distance: u8,
    world_bottom_section_y: i8,
    world_top_section_y: i8,
) -> *mut Graph {
    let graph = Box::new(Graph::new(
        render_distance,
        world_bottom_section_y,
        world_top_section_y,
    ));

    Box::leak(graph)
}

#[no_mangle]
pub unsafe extern "C" fn Java_net_caffeinemc_mods_sodium_ffi_NativeCull_graphSetSection(
    _: *mut JEnv,
    _: *mut JClass,
    graph_ptr: *mut Graph,
    x: i32,
    y: i32,
    z: i32,
    traversable_blocks_ptr: *const FFISectionTraversableBlocks,
) {
    let graph = graph_ptr
        .as_mut()
        .expect("expected pointer to graph to be valid");

    let traversable_blocks = traversable_blocks_ptr
        .as_ref()
        .expect("expected pointer to traversable blocks to be valid");

    graph.set_section(i32x3::from_xyz(x, y, z), &traversable_blocks.0);
}

#[no_mangle]
pub unsafe extern "C" fn Java_net_caffeinemc_mods_sodium_ffi_NativeCull_graphRemoveSection(
    _: *mut JEnv,
    _: *mut JClass,
    graph_ptr: *mut Graph,
    x: i32,
    y: i32,
    z: i32,
) {
    let graph = graph_ptr
        .as_mut()
        .expect("expected pointer to graph to be valid");

    graph.remove_section(i32x3::from_xyz(x, y, z));
}

#[no_mangle]
pub unsafe extern "C" fn Java_net_caffeinemc_mods_sodium_ffi_NativeCull_graphSearch(
    _: *mut JEnv,
    _: *mut JClass,
    return_value_ptr: *mut FFISlice<FFIVisibleSectionsTile>,
    graph_ptr: *mut Graph,
    camera_ptr: *const FFICamera,
    search_distance: f32,
    use_occlusion_culling: bool,
) {
    println!("start search --------------------------");

    let graph = graph_ptr
        .as_mut()
        .expect("expected pointer to graph to be valid");

    let camera = camera_ptr
        .as_ref()
        .expect("expected pointer to camera to be valid");

    let simd_camera_pos = Simd::from_array(camera.pos);
    let simd_frustum_planes = [
        Simd::from_array(camera.frustum_planes[0]),
        Simd::from_array(camera.frustum_planes[1]),
        Simd::from_array(camera.frustum_planes[2]),
        Simd::from_array(camera.frustum_planes[3]),
    ];

    let context = GraphSearchContext::new(
        &graph.coord_space,
        simd_frustum_planes,
        simd_camera_pos,
        search_distance,
        use_occlusion_culling,
    );

    graph.cull(&context);

    let mut sum: u32 = 0;
    for tile in &graph.visible_tiles {
        for part in *tile.visible_sections_ptr {
            sum += part.count_ones();
        }
    }

    println!("Visible Sections: {}", sum);

    *return_value_ptr = graph.visible_tiles.as_slice().into();
}

#[no_mangle]
pub unsafe extern "C" fn Java_net_caffeinemc_mods_sodium_ffi_NativeCull_graphDelete(
    _: *mut JEnv,
    _: *mut JClass,
    graph_ptr: *mut Graph,
) {
    let graph = graph_ptr
        .as_mut()
        .expect("expected pointer to graph to be valid");

    let graph_box = Box::from_raw(graph);
    drop(graph_box);
}
