use std::boxed::Box;
use std::ptr;

use context::GraphSearchContext;
use coords::LocalTileCoords;
use core_simd::simd::Simd;
use tile::NodeStorage;

use crate::graph::*;
use crate::java::*;
use crate::math::*;
use crate::mem::LibcAllocVtable;
use crate::panic::PanicHandlerFn;

#[repr(C)]
pub struct FFISlice<T> {
    count: u32,
    data_ptr: JPtr<T>,
}

impl<T> From<&[T]> for FFISlice<T> {
    fn from(value: &[T]) -> Self {
        Self {
            count: value.len().try_into().expect("len is not a valid u32"),
            data_ptr: if value.is_empty() {
                ptr::null::<T>().into()
            } else {
                value.as_ptr().into()
            },
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
    visible_sections_ptr: JPtr<[u64; 8]>,
    tile_coords: [u16; 3],
}

impl FFIVisibleSectionsTile {
    pub fn new(visible_sections: *const NodeStorage, coords: LocalTileCoords) -> Self {
        Self {
            visible_sections_ptr: visible_sections.cast::<[u64; 8]>().into(),
            tile_coords: coords.0.to_array(),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn set_allocator(vtable: JPtr<LibcAllocVtable>) -> bool {
    if let Some(&vtable) = vtable.as_ptr().as_ref() {
        crate::mem::set_allocator(vtable)
    } else {
        true
    }
}

#[no_mangle]
pub unsafe extern "C" fn set_panic_handler(panic_handler_fn_ptr: JFnPtr<PanicHandlerFn>) -> bool {
    if cfg!(feature = "panic_handler") {
        if let Some(panic_handler_fn_ptr) = panic_handler_fn_ptr.as_fn_ptr() {
            crate::panic::set_panic_handler(panic_handler_fn_ptr);
            false
        } else {
            true
        }
    } else {
        true
    }
}

#[no_mangle]
pub extern "C" fn graph_create(
    render_distance: Jbyte,
    world_bottom_section_y: Jbyte,
    world_top_section_y: Jbyte,
) -> JPtrMut<Graph> {
    let graph = Box::new(Graph::new(
        render_distance as u8,
        world_bottom_section_y,
        world_top_section_y,
    ));

    Box::leak(graph).into()
}

#[no_mangle]
pub unsafe extern "C" fn graph_set_section(
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
pub unsafe extern "C" fn graph_remove_section(graph: JPtrMut<Graph>, x: Jint, y: Jint, z: Jint) {
    let graph = graph.into_mut_ref();
    graph.remove_section(i32x3::from_xyz(x, y, z));
}

#[no_mangle]
pub unsafe extern "C" fn graph_search(
    return_value: JPtrMut<FFISlice<FFIVisibleSectionsTile>>,
    graph: JPtrMut<Graph>,
    camera: JPtr<FFICamera>,
    search_distance: Jfloat,
    use_occlusion_culling: Jboolean,
) {
    let graph = graph.into_mut_ref();
    let camera = camera.as_ref();

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

    *return_value.into_mut_ref() = (graph.visible_tiles.as_slice()).into();
}

#[no_mangle]
pub unsafe extern "C" fn graph_delete(graph: JPtrMut<Graph>) {
    let graph_box = Box::from_raw(graph.into_mut_ref());
    drop(graph_box);
}
