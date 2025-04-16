use std::array;

use core_simd::simd::prelude::*;
use std_float::StdFloat;

use crate::graph::*;

pub struct GraphSearchContext {
    pub frustum: LocalFrustum,

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

        let frustum = LocalFrustum::new(frustum_planes);

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
            outward_direction_masks: tile::gen_outward_direction_masks(camera_section_in_tile),
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
        let tile_min_y = coords[Y];
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

    pub fn relative_tile_pos(&self, coords: LocalTileCoords) -> f32x3 {
        let pos_int = coords.to_local_block_coords() - self.camera_pos_int.cast::<i16>();
        pos_int.cast::<f32>() - self.camera_pos_frac
    }

    #[inline(never)]
    pub fn voxelize_fog_cylinder(&self, relative_tile_pos: f32x3, visible_sections: &mut u8x64) {
        *visible_sections &= tile::voxelize_fog_cylinder(relative_tile_pos, self.fog_distance);
    }

    // TODO OPT: add ray culling
}

/// When using this, it is expected that coordinates are relative to the camera
/// rather than the world origin.
pub struct LocalFrustum {
    #[cfg(debug_assertions)]
    pub planes: [f32x4; DIRECTION_COUNT],

    // Plane data ordered component-wise rather than plane-wise. The contents are transposed from
    // the normal plane array
    planes_cw: [Simd<f32, DIRECTION_COUNT>; 4],
    pub(crate) plane_bb_offsets: [f32x3; DIRECTION_COUNT],
    pub(crate) planes_scaled: [f32x4; DIRECTION_COUNT],
}

impl LocalFrustum {
    pub fn new(planes: [f32x4; 6]) -> Self {
        let planes_cw = array::from_fn(|component_idx| {
            Simd::from_array(planes.map(|plane| plane[component_idx]))
        });
        let plane_bb_offsets = planes.map(|plane| {
            plane
                .resize(Default::default())
                .is_sign_negative_fast()
                .select(
                    Simd::splat(-RelativeBoundingBox::BOUNDING_BOX_EPSILON),
                    Simd::splat(16.0 + RelativeBoundingBox::BOUNDING_BOX_EPSILON),
                )
        });
        let planes_scaled = planes.map(|plane| {
            let nonzero_plane_divisor = if plane[X] == 0.0 {
                // avoids divide by 0 cases, while not being too small as to mess ratios between
                // the planes
                1e-15_f32.copysign(-plane[X])
            } else {
                plane[X] * -16.0
            };
            let mut plane_scaled = plane / Simd::splat(nonzero_plane_divisor);
            // if plane[X] is positive, set plane_scaled[X] to all 1 bits. if plane[X] is
            // negative, set plane_scaled[X] to all 0 bits
            plane_scaled[X] = f32::from_bits(!((plane[X].to_bits() as i32) >> 31) as u32);
            plane_scaled
        });

        LocalFrustum {
            #[cfg(debug_assertions)]
            planes,
            planes_cw,
            plane_bb_offsets,
            planes_scaled,
        }
    }

    // TODO OPT: get rid of W by normalizing plane_xs, ys, zs.
    //  potentially can exclude near and far plane
    pub fn test_box(&self, bb: RelativeBoundingBox, results: &mut CombinedTestResults) {
        // This is faster than doing a float comparison because we can ignore special
        // float values like infinity, and because we can hint to the compiler to use
        // vblendvps on x86.
        let is_neg_x = self.planes_cw[X].is_sign_negative_fast();
        let is_neg_y = self.planes_cw[Y].is_sign_negative_fast();
        let is_neg_z = self.planes_cw[Z].is_sign_negative_fast();

        let bb_min_x = Simd::splat(bb.min[X]);
        let bb_max_x = Simd::splat(bb.max[X]);
        let outside_bounds_x = is_neg_x.select(bb_min_x, bb_max_x);

        let bb_min_y = Simd::splat(bb.min[Y]);
        let bb_max_y = Simd::splat(bb.max[Y]);
        let outside_bounds_y = is_neg_y.select(bb_min_y, bb_max_y);

        let bb_min_z = Simd::splat(bb.min[Z]);
        let bb_max_z = Simd::splat(bb.max[Z]);
        let outside_bounds_z = is_neg_z.select(bb_min_z, bb_max_z);

        let outside_length_sq = self.planes_cw[X].mul_add_fast(
            outside_bounds_x,
            self.planes_cw[Y].mul_add_fast(outside_bounds_y, self.planes_cw[Z] * outside_bounds_z),
        );

        // if any outside lengths are less than -w, return OUTSIDE
        // if all inside lengths are greater than -w, return INSIDE
        // otherwise, return PARTIAL
        // NOTE: it is impossible for a lane to be both inside and outside at the same
        // time

        // the resize is necessary here because it allows LLVM to generate a vptest on
        // x86
        let any_outside = (outside_length_sq + self.planes_cw[W])
            .is_sign_negative_fast()
            .resize::<8>(false)
            .any();

        if any_outside {
            // early exit
            *results = CombinedTestResults::OUTSIDE;
            return;
        }

        let inside_bounds_x = is_neg_x.select(bb_max_x, bb_min_x);
        let inside_bounds_y = is_neg_y.select(bb_max_y, bb_min_y);
        let inside_bounds_z = is_neg_z.select(bb_max_z, bb_min_z);

        let inside_length_sq = self.planes_cw[X].mul_add_fast(
            inside_bounds_x,
            self.planes_cw[Y].mul_add_fast(inside_bounds_y, self.planes_cw[Z] * inside_bounds_z),
        );

        let intersecting_planes = ((inside_length_sq + self.planes_cw[W])
            .is_sign_negative_fast()
            .to_bitmask()
            & 0b111111) as u8;

        results.set_intersecting_planes(intersecting_planes);
    }

    // The inlining of this was pretty aggressive. It's not really necessary and
    // likely helps the code cache this way.
    #[inline(never)]
    pub fn voxelize_planes(
        &self,
        mut planes: u8,
        relative_tile_pos: f32x3,
        visible_sections: &mut u8x64,
    ) {
        while planes != 0 {
            let plane_direction = take_one(&mut planes);
            let plane_idx = to_index(plane_direction);

            let sections_in_plane = tile::voxelize_frustum_plane(
                relative_tile_pos,
                unsafe { *self.planes_scaled.get_unchecked(plane_idx) },
                unsafe { *self.plane_bb_offsets.get_unchecked(plane_idx) },
            );

            #[cfg(debug_assertions)]
            {
                use crate::graph::tile::print_tile;

                let sane_sections_in_plane =
                    tile::voxelize_frustum_plane_slow(relative_tile_pos, unsafe {
                        *self.planes.get_unchecked(plane_idx)
                    });
                if sections_in_plane != sane_sections_in_plane {
                    println!("Relative Coords: {:?}", relative_tile_pos);
                    println!("Frustum: {:#?}", self.planes);

                    let dir_str = to_str(plane_direction);
                    println!("Plane {dir_str} - Sane");
                    print_tile(&sane_sections_in_plane);
                    println!("Plane {dir_str} - Fast");
                    print_tile(&sections_in_plane);

                    panic!("Mismatch between frustum plane voxel representations");
                }
            }

            *visible_sections &= sections_in_plane;
        }
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

/// Relative to the camera position
#[derive(Clone, Copy)]
pub struct RelativeBoundingBox {
    pub(crate) min: f32x3,
    pub(crate) max: f32x3,
}

impl RelativeBoundingBox {
    pub const BOUNDING_BOX_EPSILON: f32 = 1.125;

    pub fn new(min: f32x3, max: f32x3) -> Self {
        Self {
            max: max + f32x3::splat(Self::BOUNDING_BOX_EPSILON),
            min: min - f32x3::splat(Self::BOUNDING_BOX_EPSILON),
        }
    }
}
