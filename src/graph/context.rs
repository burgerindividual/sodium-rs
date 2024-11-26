use core::mem::transmute;

use core_simd::simd::prelude::*;
use std_float::StdFloat;

use crate::graph::*;

pub struct GraphSearchContext {
    frustum: LocalFrustum,

    // the camera coords relative to the local origin, which is the (0, 0, 0) point of the
    // graph.
    pub camera_coords_int: u16x3,
    pub camera_coords_frac: f32x3,
    pub camera_section_coords: LocalTileCoords<0>,
    pub camera_section_index: LocalTileIndex<0>,

    pub global_region_offset: i32x3,

    fog_distance_squared: f32,

    // this is the index that encompasses the corner of the view distance bounding box where the
    // coordinate for each axis is closest to negative infinity, and truncated to the origin of the
    // level 3 node it's contained in.
    pub iter_start_index: LocalTileIndex<3>,
    pub level_3_node_iter_counts: LocalTileCoords<3>,
    pub iter_start_section_coords: LocalTileCoords<0>,

    pub use_occlusion_culling: bool,
}

impl GraphSearchContext {
    pub fn new(
        frustum_planes: [f32x6; 4],
        camera_global_coords: f64x3,
        search_distance: f32,
        world_bottom_section_y: i8,
        world_top_section_y: i8,
    ) -> Self {
        // this should never be negative, and we want to truncate (supposedly)
        let section_view_distance = (search_distance / 16.0) as u8;

        // TODO: check against graph size

        let frustum = LocalFrustum::new(frustum_planes);

        // convert Ys from -2048..2047 to 0..4095
        // TO LATER ME: DON'T DO THIS
        let world_bottom_section_y =
            (world_bottom_section_y as u8).wrapping_add(Self::Y_ADD_SECTIONS);
        let world_top_section_y = (world_top_section_y as u8).wrapping_add(Self::Y_ADD_SECTIONS);

        // TODO: catch when the Y axis wraps, then use that for OOB occlusion culling
        let camera_coords = (camera_global_coords + f64x3::from_xyz(0.0, Self::Y_ADD_BLOCKS, 0.0))
            .rem_euclid(f64x3::splat(4096.0))
            .cast::<f32>();

        // shift right by 4 (divide by 16) to go from block to section coords
        let camera_global_section_coords =
            camera_global_coords.floor().cast::<i32>() >> i32x3::splat(4);

        // the cast to u8 puts it in the local coordinate space by effectively doing a
        // mod 256
        let camera_section_coords = LocalTileCoords::<0>::from_raw(
            camera_global_section_coords.cast::<u8>() + u8x3::from_xyz(0, Self::Y_ADD_SECTIONS, 0),
        );
        let camera_section_index = LocalTileIndex::pack(camera_section_coords);

        // this includes the height shift back down by 32 regions
        let origin_global_region_offset = (camera_global_section_coords
            - camera_section_coords.into_raw().cast::<i32>())
            >> REGION_COORD_SHIFT.cast::<i32>();

        let mut iter_start_section_coords_tmp = camera_section_coords.into_raw().cast::<i32>()
            - i32x3::splat(section_view_distance as i32);
        iter_start_section_coords_tmp[Y] = world_bottom_section_y as i32;

        let axis_can_underflow_mask = iter_start_section_coords_tmp.simd_lt(i32x3::splat(0));
        let block_underflow_offset =
            axis_can_underflow_mask.select(f32x3::splat(-4096.0), f32x3::splat(0.0));
        let region_underflow_offset = axis_can_underflow_mask
            .select(-(GRAPH_REGION_DIMENSIONS.cast::<i32>()), i32x3::splat(0));

        let iter_start_node_coords =
            LocalTileCoords::<0>::from_raw(iter_start_section_coords_tmp.cast::<u8>())
                .into_level::<3>();
        let iter_start_index = LocalTileIndex::pack(iter_start_node_coords);
        let iter_start_section_coords = iter_start_node_coords.into_level::<0>();

        let view_cube_length = (section_view_distance * 2) + 1;

        let world_height = world_top_section_y - world_bottom_section_y;

        let iter_end_section_coords_tmp = iter_start_section_coords_tmp
            + u8x3::from_xyz(view_cube_length, world_height, view_cube_length).cast::<i32>();

        // cannot overflow if the axis is already underflowing
        let axis_can_overflow_mask =
            iter_end_section_coords_tmp.simd_gt(Simd::splat(255)) & !axis_can_underflow_mask;
        let block_overflow_offset =
            axis_can_overflow_mask.select(f32x3::splat(4096.0), f32x3::splat(0.0));
        let region_overflow_offset =
            axis_can_overflow_mask.select(GRAPH_REGION_DIMENSIONS.cast::<i32>(), i32x3::splat(0));

        // the add is done to make sure we round up during truncation
        let level_3_node_iter_counts = (LocalTileCoords::<0>::from_raw(
            (iter_end_section_coords_tmp
                + i32x3::splat((LocalTileCoords::<3>::length() - 1) as i32))
            .cast::<u8>(),
        ) - iter_start_section_coords)
            .into_level::<3>();

        let fog_distance_squared = search_distance * search_distance;

        Self {
            frustum,
            camera_coords,
            camera_section_index,
            camera_section_coords,
            global_region_offset: origin_global_region_offset,
            fog_distance_squared,
            world_bottom_section_y,
            world_top_section_y,
            iter_start_index,
            level_3_node_iter_counts,
            iter_start_section_coords,
        }
    }

    pub fn test_node(
        &self,
        coords: LocalTileCoords,
        level: u8,
        parent_test_results: CombinedTestResults,
    ) -> CombinedTestResults {
        let relative_bounds = self.tile_get_relative_bounds(coords, level);

        let mut results = CombinedTestResults(0);

        self.bounds_inside_world_height(coords, level);

        if results == CombinedTestResults::OUTSIDE {
            return results;
        }

        self.frustum.test_box(bounds);

        if results == CombinedTestResults::OUTSIDE {
            return results;
        }

        self.bounds_inside_fog::<LEVEL>(bounds);

        results
    }

    fn bounds_inside_world_height(&self, coords: LocalTileCoords, level: u8) -> BoundsCheckResult {
        let node_min_y = local_section_coords.y() as u32;
        let node_max_y = node_min_y + (1 << LEVEL) - 1;
        let world_max_y = self.world_top_section_y as u32;

        let min_in_bounds = node_min_y <= world_max_y;
        let max_in_bounds = node_max_y <= world_max_y;

        // in normal circumstances, this really shouldn't ever return OUTSIDE
        unsafe { BoundsCheckResult::from_int_unchecked(min_in_bounds as u8 + max_in_bounds as u8) }
    }

    // this only cares about the x and z axis
    // FIXME: use this algo https://github.com/CaffeineMC/sodium-fabric/blob/dd25399c139004e863beb8a2195b9d80b847d95c/common/src/main/java/net/caffeinemc/mods/sodium/client/render/chunk/occlusion/OcclusionCuller.java#L153
    fn bounds_inside_fog(
        &self,
        relative_bounds: RelativeBoundingBox,
        level: u8,
    ) -> BoundsCheckResult {
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
        let axis_distances =
            simd_swizzle!(closest_in_chunk, furthest_in_chunk, [X, 3 + X, Z, 3 + Z]);
        let axis_distances_squared = axis_distances * axis_distances;

        // add Xs and Zs
        let distances_squared = simd_swizzle!(axis_distances_squared, [0, 1])
            + simd_swizzle!(axis_distances_squared, [2, 3]);

        // janky way of calculating the result from the two points
        unsafe {
            BoundsCheckResult::from_int_unchecked(
                distances_squared
                    .simd_lt(f32x2::splat(self.fog_distance_squared))
                    .select(u32x2::splat(1), u32x2::splat(0))
                    .reduce_sum() as u8,
            )
        }
    }

    fn tile_get_relative_bounds(&self, coords: LocalTileCoords, level: u8) -> RelativeBoundingBox {
        let tile_factor = f32x3::splat(LocalTileCoords::block_length(level) as f32);

        let converted_pos = coords.0.cast::<f32>() * tile_factor;

        let min_bound = converted_pos - self.camera_coords;
        let max_bound = min_bound + tile_factor;

        RelativeBoundingBox::new(min_bound, max_bound)
    }

    // TODO OPT: add douira's magic visible directions culler
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
    pub fn test_box(&self, bb: RelativeBoundingBox) -> BoundsCheckResult {
        unsafe {
            // These unsafe mask shenanigans just check if the sign bit is set for each
            // lane. This is faster than doing a manual comparison with
            // something like simd_gt.
            let is_neg_x =
                Mask::from_int_unchecked(self.plane_xs.to_bits().cast::<i32>() >> Simd::splat(31));
            let is_neg_y =
                Mask::from_int_unchecked(self.plane_ys.to_bits().cast::<i32>() >> Simd::splat(31));
            let is_neg_z =
                Mask::from_int_unchecked(self.plane_zs.to_bits().cast::<i32>() >> Simd::splat(31));

            let bb_min_x = Simd::splat(bb.min.x());
            let bb_max_x = Simd::splat(bb.max.x());
            let outside_bounds_x = is_neg_x.select(bb_min_x, bb_max_x);
            let inside_bounds_x = is_neg_x.select(bb_max_x, bb_min_x);

            let bb_min_y = Simd::splat(bb.min.y());
            let bb_max_y = Simd::splat(bb.max.y());
            let outside_bounds_y = is_neg_y.select(bb_min_y, bb_max_y);
            let inside_bounds_y = is_neg_y.select(bb_max_y, bb_min_y);

            let bb_min_z = Simd::splat(bb.min.z());
            let bb_max_z = Simd::splat(bb.max.z());
            let outside_bounds_z = is_neg_z.select(bb_min_z, bb_max_z);
            let inside_bounds_z = is_neg_z.select(bb_max_z, bb_min_z);

            let outside_length_sq = self.plane_xs.mul_add_fast(
                outside_bounds_x,
                self.plane_ys
                    .mul_add_fast(outside_bounds_y, self.plane_zs * outside_bounds_z),
            );

            let inside_length_sq = self.plane_xs.mul_add_fast(
                inside_bounds_x,
                self.plane_ys
                    .mul_add_fast(inside_bounds_y, self.plane_zs * inside_bounds_z),
            );

            // TODO: double check the stuff here
            // if any outside lengths are greater than -w, return OUTSIDE
            // if all inside lengths are greater than -w, return INSIDE
            // otherwise, return PARTIAL
            // NOTE: it is impossible for a lane to be both inside and outside at the same
            // time
            let none_outside = outside_length_sq.simd_ge(-self.plane_ws).to_bitmask() == 0b111111;
            let all_inside = inside_length_sq.simd_ge(-self.plane_ws).to_bitmask() == 0b111111;

            BoundsCheckResult::from_int_unchecked(none_outside as u8 + all_inside as u8)
        }
    }
}

// If the value of this is not OUTSIDE, the following applies:
// Each test is represented by a single bit in this bit set. For each test:
// 1-bit = Partially inside, partially outside
// 0-bit = Inside
#[derive(PartialEq, Copy, Clone)]
pub struct CombinedTestResults(u8);

impl CombinedTestResults {
    pub const OUTSIDE: Self = Self(0xFF);
    pub const FRUSTUM_BIT: u8 = 0b00000001;
    pub const FOG_BIT: u8 = 0b00000010;
    pub const HEIGHT_BIT: u8 = 0b00000100;

    pub fn is_partial<const BIT: u8>(self) -> bool {
        bitset::contains(self.0, BIT)
    }

    pub fn is_inside<const BIT: u8>(self) -> bool {
        !self.is_partial::<BIT>()
    }

    pub fn set<const BIT: u8>(&mut self, value: bool) {
        self.0 |= (value as u8) << BIT.trailing_zeros();
    }
}

#[derive(PartialEq)]
pub enum BoundsCheckResult {
    Outside = 0,
    Partial = 1,
    Inside = 2,
}

impl BoundsCheckResult {
    /// SAFETY: if out of bounds, this will fail to assert in debug mode
    pub unsafe fn from_int_unchecked(val: u8) -> Self {
        debug_assert!(val <= 2);
        transmute(val)
    }

    pub fn combine(self, rhs: Self) -> Self {
        // SAFETY: given 2 valid inputs, the result will always be valid
        unsafe { Self::from_int_unchecked((self as u8).min(rhs as u8)) }
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
