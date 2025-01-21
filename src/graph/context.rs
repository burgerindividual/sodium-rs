use core_simd::simd::prelude::*;
use std_float::StdFloat;

use crate::graph::*;

pub struct GraphSearchContext {
    frustum: LocalFrustum,

    pub global_tile_offset: i32x3,

    fog_distance: f32,

    // the camera coords (in blocks) relative to the local origin, which is the (0, 0, 0) point of
    // the graph. the representation here is slightly different than the representation in
    // CameraTransform.java, as camera_pos_frac can never be negative in our representation.
    pub camera_pos_int: u16x3,
    pub camera_pos_frac: f32x3,

    pub camera_section_in_tile: u8x3,

    pub camera_tile_coords: LocalTileCoords,
    pub direction_step_counts: u8x6,

    // TODO: actually use this
    pub use_occlusion_culling: bool,

    pub camera_direction_masks: [u8x64; DIRECTION_COUNT],
}

impl GraphSearchContext {
    pub fn new(
        coord_space: &GraphCoordSpace,
        frustum_planes: [f32x6; 4],
        global_camera_pos: f64x3,
        search_distance: f32,
        use_occlusion_culling: bool,
    ) -> Self {
        // TODO: check against graph size
        // TODO: assert search distance size isn't too big
        // TODO: assert camera pos isn't ridiculous

        let frustum = LocalFrustum::new(frustum_planes);

        let global_camera_pos_floor = global_camera_pos.floor();
        // see the comment in CameraTransform.java for why we reduce the precision
        const PRECISION_MODIFIER: f32x3 = Simd::splat(128.0);
        let camera_pos_frac = ((global_camera_pos - global_camera_pos_floor).cast::<f32>()
            + PRECISION_MODIFIER)
            - PRECISION_MODIFIER;

        let global_camera_pos_int = unsafe { global_camera_pos_floor.to_int_unchecked::<i32>() };

        let camera_pos_int = coord_space.block_to_local_coords(global_camera_pos_int);
        let global_tile_offset =
            (global_camera_pos_int - camera_pos_int.cast::<i32>()) >> Simd::splat(7);
        let camera_tile_coords = (camera_pos_int >> Simd::splat(7)).cast::<u8>();

        let camera_pos = camera_pos_int.cast::<f32>() + camera_pos_frac;
        let positive_step_counts = unsafe {
            ((camera_pos + Simd::splat(search_distance)).to_int_unchecked::<u16>()
                >> Simd::splat(7))
            .cast::<u8>()
                - camera_tile_coords
        };
        // we cast from f32 to i16 to u16 here. this is to allow underflowing, as we
        // want an underflow to
        let negative_step_counts = unsafe {
            camera_tile_coords
                - ((camera_pos - Simd::splat(search_distance)).to_int_unchecked::<i16>()
                    >> Simd::splat(7))
                .cast::<u8>()
        };

        let direction_step_counts = simd_swizzle!(
            negative_step_counts.cast::<u8>(),
            positive_step_counts.cast::<u8>(),
            [0, 1, 2, 3, 4, 5,],
        );

        let camera_section_in_tile =
            (camera_pos_int >> Simd::splat(4)).cast::<u8>() & Simd::splat(0b111);

        Self {
            frustum,
            global_tile_offset,
            fog_distance: search_distance,
            camera_pos_int,
            camera_pos_frac,
            camera_section_in_tile,
            camera_tile_coords: LocalTileCoords(camera_tile_coords.cast::<i8>()),
            direction_step_counts,
            use_occlusion_culling,
            camera_direction_masks: tile::create_camera_direction_masks(camera_section_in_tile),
        }
    }

    pub fn test_tile(
        &self,
        coord_space: &GraphCoordSpace,
        coords: LocalTileCoords,
        do_height_checks: bool,
    ) -> CombinedTestResults {
        let mut results = CombinedTestResults::ALL_INSIDE;

        let relative_bounds = self.tile_get_relative_bounds(coords);

        self.frustum.test_box(relative_bounds, &mut results);

        if results == CombinedTestResults::OUTSIDE {
            // early exit
            return results;
        }
        self.bounds_inside_fog(relative_bounds, &mut results);

        if results == CombinedTestResults::OUTSIDE {
            // early exit
            return results;
        }

        if do_height_checks {
            self.bounds_inside_world_height(coord_space, coords, &mut results);
        }

        results
    }

    fn bounds_inside_world_height(
        &self,
        coord_space: &GraphCoordSpace,
        coords: LocalTileCoords,
        results: &mut CombinedTestResults,
    ) {
        let tile_min_y = coords.y();
        let tile_max_y = tile_min_y + LocalTileCoords::LENGTH_IN_SECTIONS as i8 - 1;
        let world_max_y = coord_space.world_top_section_y;

        let min_out_of_bounds = tile_min_y > world_max_y;

        if min_out_of_bounds {
            // early exit
            *results = CombinedTestResults::OUTSIDE;
            return;
        }

        let max_out_of_bounds = tile_max_y > world_max_y;

        results.set_partial::<{ CombinedTestResults::HEIGHT_BIT }>(max_out_of_bounds);
    }

    // based on this algo
    // https://github.com/CaffeineMC/sodium-fabric/blob/dd25399c139004e863beb8a2195b9d80b847d95c/common/src/main/java/net/caffeinemc/mods/sodium/client/render/chunk/occlusion/OcclusionCuller.java#L153
    pub fn bounds_inside_fog(
        &self,
        relative_bounds: RelativeBoundingBox,
        results: &mut CombinedTestResults,
    ) {
        // find closest to (0,0) because the bounding box coordinates are relative to
        // the camera
        let closest_in_chunk = f32x3::splat(0.0)
            .simd_max(relative_bounds.min)
            .simd_min(relative_bounds.max);

        let furthest_in_chunk = relative_bounds
            .min
            .abs()
            .simd_gt(relative_bounds.max.abs())
            .select(relative_bounds.min, relative_bounds.max);

        // combine operations and single out the XZ lanes on both extrema from here.
        // also, we don't have to subtract from the camera pos because the bounds are
        // already relative to it
        let xz_distances = simd_swizzle!(closest_in_chunk, furthest_in_chunk, [X, Z, X + 3, Z + 3]);
        let xz_distances_squared = xz_distances * xz_distances;

        // add Xs and Zs
        let combined_distances_squared = simd_swizzle!(xz_distances_squared, [0, 2])
            + simd_swizzle!(xz_distances_squared, [1, 3]);

        let y_distances = simd_swizzle!(closest_in_chunk, furthest_in_chunk, [Y, Y + 3]);
        let y_distances_abs = y_distances.abs();

        let outside_fog_mask = combined_distances_squared
            .simd_ge(Simd::splat(self.fog_distance * self.fog_distance))
            | y_distances_abs.simd_ge(Simd::splat(self.fog_distance));

        if outside_fog_mask.test(0) {
            // early exit
            *results = CombinedTestResults::OUTSIDE;
            return;
        }

        results.set_partial::<{ CombinedTestResults::FOG_BIT }>(outside_fog_mask.test(1));
    }

    fn tile_get_relative_bounds(&self, coords: LocalTileCoords) -> RelativeBoundingBox {
        let pos_int = coords.to_local_block_coords() - self.camera_pos_int.cast::<i16>();
        let pos_float = pos_int.cast::<f32>() - self.camera_pos_frac;

        let tile_size = f32x3::splat(LocalTileCoords::LENGTH_IN_BLOCKS as f32);

        RelativeBoundingBox::new(pos_float, pos_float + tile_size)
    }

    // TODO OPT: add douira's magic visible directions culler
    // TODO OPT: add ray culling
}

/// When using this, it is expected that coordinates are relative to the camera
/// rather than the world origin.
pub struct LocalFrustum {
    plane_xs: f32x6,
    plane_ys: f32x6,
    plane_zs: f32x6,
    plane_ws: f32x6,
}

impl LocalFrustum {
    pub fn new(planes: [f32x6; 4]) -> Self {
        LocalFrustum {
            plane_xs: planes[0],
            plane_ys: planes[1],
            plane_zs: planes[2],
            plane_ws: planes[3],
        }
    }

    // TODO OPT: get rid of W by normalizing plane_xs, ys, zs.
    //  potentially can exclude near and far plane
    pub fn test_box(&self, bb: RelativeBoundingBox, results: &mut CombinedTestResults) {
        const SIGN_BIT: u32x6 = Simd::splat(1 << 31);

        // These mask shenanigans just check if the sign bit is set for each lane.
        // This is faster than doing a float comparison because we can ignore special
        // float values like infinity, and because we can hint to the compiler to use
        // vblendvps on x86.
        let is_neg_x = self.plane_xs.to_bits().simd_ge(SIGN_BIT);
        let is_neg_y = self.plane_ys.to_bits().simd_ge(SIGN_BIT);
        let is_neg_z = self.plane_zs.to_bits().simd_ge(SIGN_BIT);

        let bb_min_x = Simd::splat(bb.min.x());
        let bb_max_x = Simd::splat(bb.max.x());
        let outside_bounds_x = is_neg_x.select(bb_min_x, bb_max_x);

        let bb_min_y = Simd::splat(bb.min.y());
        let bb_max_y = Simd::splat(bb.max.y());
        let outside_bounds_y = is_neg_y.select(bb_min_y, bb_max_y);

        let bb_min_z = Simd::splat(bb.min.z());
        let bb_max_z = Simd::splat(bb.max.z());
        let outside_bounds_z = is_neg_z.select(bb_min_z, bb_max_z);

        let outside_length_sq = self.plane_xs.mul_add_fast(
            outside_bounds_x,
            self.plane_ys
                .mul_add_fast(outside_bounds_y, self.plane_zs * outside_bounds_z),
        );

        // TODO: double check the stuff here
        // if any outside lengths are less than -w, return OUTSIDE
        // if all inside lengths are greater than -w, return INSIDE
        // otherwise, return PARTIAL
        // NOTE: it is impossible for a lane to be both inside and outside at the same
        // time

        // the resize is necessary here because it allows LLVM to generate a vptest on
        // x86
        let any_outside =
            ((outside_length_sq + self.plane_ws).to_bits() & SIGN_BIT).resize(0) != u32x8::splat(0);

        if any_outside {
            // early exit
            *results = CombinedTestResults::OUTSIDE;
            return;
        }

        let inside_bounds_x = is_neg_x.select(bb_max_x, bb_min_x);
        let inside_bounds_y = is_neg_y.select(bb_max_y, bb_min_y);
        let inside_bounds_z = is_neg_z.select(bb_max_z, bb_min_z);

        let inside_length_sq = self.plane_xs.mul_add_fast(
            inside_bounds_x,
            self.plane_ys
                .mul_add_fast(inside_bounds_y, self.plane_zs * inside_bounds_z),
        );

        let any_partial =
            ((inside_length_sq + self.plane_ws).to_bits() & SIGN_BIT).resize(0) != u32x8::splat(0);

        results.set_partial::<{ CombinedTestResults::FRUSTUM_BIT }>(any_partial);
    }
}

// If the value of this is not OUTSIDE, the following applies:
// Each test is represented by a single bit in this bit set. For each test:
// 1-bit = Partially inside, partially outside
// 0-bit = Inside
#[derive(PartialEq, Copy, Clone)]
pub struct CombinedTestResults(u8);

impl CombinedTestResults {
    pub const NONE_INSIDE: Self = Self(0b111);
    pub const HEIGHT_INSIDE: Self = Self(0b011);
    pub const ALL_INSIDE: Self = Self(0b000);
    pub const OUTSIDE: Self = Self(0xFF);

    pub const FRUSTUM_BIT: u8 = 0b001;
    pub const FOG_BIT: u8 = 0b010;
    pub const HEIGHT_BIT: u8 = 0b100;

    pub fn is_partial<const BIT: u8>(self) -> bool {
        bitset::contains(self.0, BIT)
    }

    pub fn set_partial<const BIT: u8>(&mut self, value: bool) {
        self.0 |= (value as u8) << BIT.trailing_zeros();
    }
}

/// Relative to the camera position
#[derive(Clone, Copy)]
pub struct RelativeBoundingBox {
    min: f32x3,
    max: f32x3,
}

impl RelativeBoundingBox {
    const BOUNDING_BOX_EPSILON: f32 = 1.0 + 0.125;

    pub fn new(min: f32x3, max: f32x3) -> Self {
        Self {
            max: max + f32x3::splat(Self::BOUNDING_BOX_EPSILON),
            min: min - f32x3::splat(Self::BOUNDING_BOX_EPSILON),
        }
    }
}
