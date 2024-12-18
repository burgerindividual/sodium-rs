use std::boxed::Box;

use context::GraphSearchContext;
use coords::LocalTileCoords;
use core_simd::simd::Simd;
use tile::NodeStorage;

use crate::graph::*;
use crate::math::*;
use crate::mem::LibcAllocVtable;
use crate::panic::PanicHandlerFn;

pub type Jbyte = i8;
pub type Jshort = i16;
pub type Jint = i32;
pub type Jlong = i64;

pub type Jfloat = f32;
pub type Jdouble = f64;

pub type Jchar = u16;

pub type Jboolean = bool;

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
pub struct FFISectionOpaqueBlocks([u8; 512]);

#[repr(C)]
pub struct FFIVisibleSectionsTile {
    visible_sections_ptr: *const [u64; 8],
    tile_coords: [u16; 3],
}

impl FFIVisibleSectionsTile {
    pub fn new(visible_sections: *const NodeStorage, coords: LocalTileCoords) -> Self {
        Self {
            visible_sections_ptr: visible_sections.cast::<[u64; 8]>(),
            tile_coords: coords.0.to_array(),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn set_allocator(vtable: *const LibcAllocVtable) -> bool {
    if let Some(&vtable) = vtable.as_ref() {
        crate::mem::set_allocator(vtable)
    } else {
        true
    }
}

#[no_mangle]
pub unsafe extern "C" fn set_panic_handler(panic_handler_fn_ptr: PanicHandlerFn) {
    if cfg!(feature = "panic_handler") {
        crate::panic::set_panic_handler(panic_handler_fn_ptr);
    }
}

#[no_mangle]
pub extern "C" fn graph_create(
    render_distance: Jbyte,
    world_bottom_section_y: Jbyte,
    world_top_section_y: Jbyte,
) -> *mut Graph {
    let graph = Box::new(Graph::new(
        render_distance as u8,
        world_bottom_section_y,
        world_top_section_y,
    ));

    Box::leak(graph)
}

#[no_mangle]
pub unsafe extern "C" fn graph_set_section(
    graph: *mut Graph,
    x: Jint,
    y: Jint,
    z: Jint,
    opaque_blocks: *const FFISectionOpaqueBlocks,
) {
    let graph = graph
        .as_mut()
        .expect("expected pointer to graph to be valid");

    let opaque_blocks = opaque_blocks
        .as_ref()
        .expect("expected pointer to opaque blocks to be valid");

    graph.set_section(i32x3::from_xyz(x, y, z), &opaque_blocks.0);
}

#[no_mangle]
pub unsafe extern "C" fn graph_remove_section(graph: *mut Graph, x: Jint, y: Jint, z: Jint) {
    let graph = graph
        .as_mut()
        .expect("expected pointer to graph to be valid");

    graph.remove_section(i32x3::from_xyz(x, y, z));
}

#[no_mangle]
pub unsafe extern "C" fn graph_search(
    return_value: *mut FFISlice<FFIVisibleSectionsTile>,
    graph: *mut Graph,
    camera: *const FFICamera,
    search_distance: Jfloat,
    use_occlusion_culling: Jboolean,
) {
    let graph = graph
        .as_mut()
        .expect("expected pointer to graph to be valid");

    let camera = camera
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

    *return_value = graph.visible_tiles.as_slice().into();
}

#[no_mangle]
pub unsafe extern "C" fn graph_delete(graph: *mut Graph) {
    let graph = graph
        .as_mut()
        .expect("expected pointer to graph to be valid");
    
    let graph_box = Box::from_raw(graph);
    drop(graph_box);
}
