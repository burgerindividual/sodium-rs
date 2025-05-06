use core_simd::simd::prelude::*;
use std_float::StdFloat;

use super::coords::RelativeBoundingBox;
use super::tile::frustum::Frustum;
use crate::graph::*;

pub struct GraphSearchContext {
    pub frustum: Frustum,

    pub global_section_offset: i32x3,

    pub fog_distance: f32,

    // the camera coords (in blocks) relative to the local origin, which is the (0, 0, 0) point of
    // the graph. the representation here is slightly different than the representation in
    // CameraTransform.java, as camera_pos_frac can never be negative in our representation.
    pub camera_pos_int: u16x3,
    pub camera_pos_frac: f32x3,

    pub camera_section_in_tile: u8x3,

    pub camera_tile_coords: LocalTileCoords,
    pub direction_step_counts: Simd<u8, DIRECTION_COUNT>,

    pub use_occlusion_culling: bool,

    pub outward_direction_masks: [u8x64; DIRECTION_COUNT],
}

impl GraphSearchContext {
    pub fn new(
        coord_space: &GraphCoordSpace,
        frustum_planes: [f32x4; 6],
        global_camera_pos: f64x3,
        search_distance: f32,
        use_occlusion_culling: bool,
    ) -> Self {
        // TODO: check against graph size
        // TODO: assert search distance size isn't too big
        // TODO: assert camera pos isn't ridiculous
        // TODO: deal with camera above and below world

        let frustum = Frustum::new(frustum_planes);

        let global_camera_pos_floor = global_camera_pos.floor();
        // see the comment in CameraTransform.java for why we reduce the precision
        const PRECISION_MODIFIER: f32x3 = Simd::splat(128.0);
        let camera_pos_frac = ((global_camera_pos - global_camera_pos_floor).cast::<f32>()
            + PRECISION_MODIFIER)
            - PRECISION_MODIFIER;

        let global_camera_pos_int = unsafe { global_camera_pos_floor.to_int_unchecked::<i32>() };

        let camera_pos_int = coord_space.block_to_local_coords(global_camera_pos_int);
        let global_section_offset = (global_camera_pos_int - camera_pos_int.cast::<i32>()) >> 4;
        let camera_tile_coords = (camera_pos_int >> 7).cast::<u8>();

        let camera_pos = camera_pos_int.cast::<f32>() + camera_pos_frac;

        // TODO: is the -1 necessary?
        let local_top_block_y = (((coord_space.world_top_section_y as i16
            - coord_space.world_bottom_section_y as i16
            + 1) as u16)
            << 4)
            - 1;

        let positive_step_counts = unsafe {
            ((camera_pos + Simd::splat(search_distance))
                .to_int_unchecked::<u16>()
                .simd_min(Simd::from_xyz(u16::MAX, local_top_block_y, u16::MAX))
                >> 7)
                .cast::<u8>()
                - camera_tile_coords
        };
        // we cast from f32 to i16 to u8 here. this is to allow underflowing, as we
        // want an underflow to wrap around on the X and Z axis
        let negative_step_counts = unsafe {
            camera_tile_coords
                - ((camera_pos - Simd::splat(search_distance))
                    .to_int_unchecked::<i16>()
                    .simd_max(Simd::from_xyz(i16::MIN, 0, i16::MIN))
                    >> 7)
                    .cast::<u8>()
        };

        let direction_step_counts = simd_swizzle!(
            negative_step_counts.cast::<u8>(),
            positive_step_counts.cast::<u8>(),
            [0, 1, 2, 3, 4, 5,],
        );

        let camera_section_in_tile = (camera_pos_int >> 4).cast::<u8>() & Simd::splat(0b111);

        Self {
            frustum,
            global_section_offset,
            fog_distance: search_distance,
            camera_pos_int,
            camera_pos_frac,
            camera_section_in_tile,
            camera_tile_coords: LocalTileCoords(camera_tile_coords.cast::<i8>()),
            direction_step_counts,
            use_occlusion_culling,
            outward_direction_masks: traversal::gen_outward_direction_masks(
                camera_section_in_tile,
            ),
        }
    }

    pub fn test_tile(
        &self,
        coord_space: &GraphCoordSpace,
        coords: LocalTileCoords,
        relative_pos: f32x3,
        do_height_checks: bool,
    ) -> CombinedTestResults {
        let mut results = CombinedTestResults::ALL_INSIDE;

        let relative_bounds = RelativeBoundingBox::new(
            relative_pos,
            relative_pos + Simd::splat(LocalTileCoords::LENGTH_IN_BLOCKS as f32),
        );

        self.frustum.test_box(relative_bounds, &mut results);

        if results == CombinedTestResults::OUTSIDE {
            // early exit
            return results;
        }
        fog::test_box(relative_bounds, self.fog_distance, &mut results);

        if results == CombinedTestResults::OUTSIDE {
            // early exit
            return results;
        }

        if do_height_checks {
            height::test_coords(coord_space, coords, &mut results);
        }

        results
    }

    pub fn relative_tile_pos(&self, coords: LocalTileCoords) -> f32x3 {
        let pos_int = coords.to_local_block_coords() - self.camera_pos_int.cast::<i16>();
        pos_int.cast::<f32>() - self.camera_pos_frac
    }

    #[inline(never)]
    pub fn voxelize_fog_cylinder(&self, relative_tile_pos: f32x3, visible_sections: &mut u8x64) {
        *visible_sections &= fog::voxelize_cylinder(relative_tile_pos, self.fog_distance);
    }
}

// If the value of this is not OUTSIDE, the following applies:
// Each test is represented by a single bit in this bit set. For each test:
// 1-bit = Partially inside, partially outside
// 0-bit = Inside
#[derive(PartialEq, Copy, Clone)]
pub struct CombinedTestResults(u16);

impl CombinedTestResults {
    pub const ALL_INSIDE: Self = Self(0b000);
    pub const OUTSIDE: Self = Self(!0);

    const FRUSTUM_PLANE_BITS: u16 = 0b00111111;
    pub const FOG_BIT: u16 = 0b01000000;
    pub const HEIGHT_BIT: u16 = 0b10000000;

    pub fn is_partial<const BIT: u16>(self) -> bool {
        bitset::contains_u16(self.0, BIT)
    }

    pub fn set_partial<const BIT: u16>(&mut self, value: bool) {
        self.0 |= (value as u16) << BIT.trailing_zeros();
    }

    pub fn set_intersecting_planes(&mut self, value: u8) {
        self.0 |= value as u16;
    }

    pub fn get_intersecting_planes(self) -> u8 {
        (self.0 & Self::FRUSTUM_PLANE_BITS) as u8
    }
}
