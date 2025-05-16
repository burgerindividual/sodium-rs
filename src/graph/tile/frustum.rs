use std::array;

use super::*;
use crate::graph::coords::RelativeBoundingBox;

/// When using this, it is expected that coordinates are relative to the camera
/// rather than the world origin.
pub struct Frustum {
    planes: [f32x4; DIRECTION_COUNT],
    plane_bb_offsets: [f32x3; DIRECTION_COUNT],

    // Plane data ordered component-wise rather than plane-wise. The contents are transposed from
    // the normal plane array
    planes_cw: [Simd<f32, DIRECTION_COUNT>; 4],
}

impl Frustum {
    pub fn new(planes: [f32x4; 6]) -> Self {
        let plane_bb_offsets = planes.map(Self::gen_plane_bb_offsets);
        let planes_cw = array::from_fn(|component_idx| {
            Simd::from_array(planes.map(|plane| plane[component_idx]))
        });

        Frustum {
            planes,
            plane_bb_offsets,
            planes_cw,
        }
    }

    fn gen_plane_bb_offsets(plane: f32x4) -> f32x3 {
        plane
            .resize(Default::default())
            .is_sign_negative_fast()
            .select(
                Simd::splat(-RelativeBoundingBox::BOUNDING_BOX_EXTENSION),
                Simd::splat(16.0 + RelativeBoundingBox::BOUNDING_BOX_EXTENSION),
            )
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

            let sections_in_plane = voxelize_plane(
                relative_tile_pos,
                unsafe { *self.planes.get_unchecked(plane_idx) },
                unsafe { *self.plane_bb_offsets.get_unchecked(plane_idx) },
            );

            *visible_sections &= sections_in_plane;
        }
    }
}

// This function voxelizes one of the six planes that make up the frustum,
// producing a 1 bit if the associated section is inside the plane (with a small
// offset to ensure no false negatives), and a 0 bit if the associated section
// is outside of the plane.
// This function works by solving the plane equation for the X intercept on each
// X-axis row of 8 sections. TODO: continue the comment
fn voxelize_plane(relative_tile_pos: f32x3, plane: f32x4, plane_bb_offsets: f32x3) -> u8x64 {
    let mut section_bb_offsets = relative_tile_pos + plane_bb_offsets;

    let plane_x_scaled = f32x8::splat(plane[X] * -16.0);

    let tile_x_intercepts = u8x64::from_slice(
        array::from_fn::<_, 8, _>(|_y| {
            let section_bb_zs = f32x8::from_array([0.0, 16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0])
                + Simd::splat(section_bb_offsets[Z]);

            let tile_x_intercepts = section_bb_zs.mul_add_fast(
                Simd::splat(plane[Z]),
                Simd::splat(section_bb_offsets[X].mul_add_fast(
                    plane[X],
                    section_bb_offsets[Y].mul_add_fast(plane[Y], plane[W]),
                )),
            ) / plane_x_scaled;

            // Increment Y by length of section in blocks after usage of offsets
            section_bb_offsets += Simd::from_xyz(0.0, 16.0, 0.0);

            // SAFETY: we check if the float value being converted is within the bounds 0 to
            // 8 before using this value. in those cases, this should always produce a
            // correct result.
            let tile_x_intercepts_int = unsafe { tile_x_intercepts.to_int_unchecked::<i32>() };

            let tile_x_intercepts_clamped = tile_x_intercepts
                .simd_lt(Simd::splat(8.0))
                .select(
                    tile_x_intercepts.is_sign_negative_fast().select(
                        Simd::splat(!0), // this index will result in all 0 bits for the mask
                        tile_x_intercepts_int,
                    ),
                    Simd::splat(7), // this index will result in all 1 bits for the mask
                )
                .cast();

            tile_x_intercepts_clamped.to_array()
        })
        .as_flattened(),
    );

    #[cfg(target_feature = "avx2")]
    let tile_x_masks = unsafe {
        use std::arch::x86_64::*;

        let intercepts_halves: [u8x32; 2] = [
            tile_x_intercepts.extract::<0, 32>(),
            tile_x_intercepts.extract::<32, 32>(),
        ];

        let mask_table = _mm256_set1_epi64x(i64::from_le_bytes([
            0b1, 0b11, 0b111, 0b1111, 0b11111, 0b111111, 0b1111111, 0b11111111,
        ]));
        let shuffled_masks_halves: [u8x32; 2] = intercepts_halves
            .map(|intercepts| _mm256_shuffle_epi8(mask_table, intercepts.into()).into());

        simd_swizzle!(
            shuffled_masks_halves[0],
            shuffled_masks_halves[1],
            concat_swizzle_pattern::<64>()
        )
    };

    #[cfg(not(target_feature = "avx2"))]
    let tile_x_masks = {
        let in_bounds_masks = (Simd::splat(0b10) << tile_x_intercepts) - Simd::splat(1);
        tile_x_intercepts
            .simd_lt(Simd::splat(8))
            .select(in_bounds_masks, Simd::splat(0))
    };

    // If plane[X] is positive, this will be all 1 bits. if plane[X] is negative,
    // this will be all 0 bits. This is used to reverse the direction of the mask
    // when needed.
    let plane_x_positive_mask = Simd::splat(!(plane[X].to_bits() as i32 >> 31) as u8);

    tile_x_masks ^ plane_x_positive_mask
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use rand::prelude::*;

    use super::*;
    use crate::TESTS_RANDOM_SEED;

    fn voxelize_plane_slow(relative_tile_pos: f32x3, plane: f32x4, bounds_extension: f32) -> u8x64 {
        let mut visible_sections = SECTIONS_EMPTY;

        for y in 0..8 {
            for z in 0..8 {
                for x in 0..8 {
                    let section_coords = Simd::from_xyz(x, y, z);
                    let section_index = section_index(section_coords);

                    let relative_section_pos = section_coords
                        .cast::<f32>()
                        .mul_add_fast(Simd::splat(16.0), relative_tile_pos);
                    let bb = RelativeBoundingBox::new(
                        relative_section_pos - Simd::splat(bounds_extension),
                        relative_section_pos + Simd::splat(16.0 + bounds_extension),
                    );

                    // let not_outside = plane[X]
                    //     * (if plane[X] < 0.0 { bb.min[X] } else { bb.max[X] })
                    //     + plane[Y] * (if plane[Y] < 0.0 { bb.min[Y] } else { bb.max[Y] })
                    //     + plane[Z] * (if plane[Z] < 0.0 { bb.min[Z] } else { bb.max[Z] })
                    //     >= -plane[W];

                    // this should be a bit more accurate by using FMAs
                    let not_outside = plane[X].mul_add(
                        if plane[X] < 0.0 { bb.min[X] } else { bb.max[X] },
                        plane[Y].mul_add(
                            if plane[Y] < 0.0 { bb.min[Y] } else { bb.max[Y] },
                            plane[Z].mul_add(
                                if plane[Z] < 0.0 { bb.min[Z] } else { bb.max[Z] },
                                plane[W],
                            ),
                        ),
                    ) >= 0.0;

                    modify_bit(&mut visible_sections, section_index, not_outside);
                }
            }
        }

        visible_sections
    }

    #[test]
    fn plane_voxelization_test() {
        const ITERATIONS: u32 = 10000;
        let mut rand = StdRng::seed_from_u64(TESTS_RANDOM_SEED);

        for _ in 0..ITERATIONS {
            // generate random plane from random unit vector
            let theta = rand.random_range(0.0..TAU);
            let z: f32 = rand.random_range(-1.0..1.0);
            let w: f32 = rand.random_range(-10.0..1000.0);

            let z_modified = (1.0 - (z * z)).sqrt();
            let x = z_modified * theta.cos();
            let y = z_modified * theta.sin();

            let plane = Simd::from_array([x, y, z, w]);
            let plane_bb_offsets = Frustum::gen_plane_bb_offsets(plane);

            let relative_tile_pos = Simd::from_xyz(
                rand.random_range(-3000.0_f32..3000.0_f32),
                rand.random_range(-3000.0_f32..3000.0_f32),
                rand.random_range(-3000.0_f32..3000.0_f32),
            );

            let sane_visible_sections_min = voxelize_plane_slow(
                relative_tile_pos,
                plane,
                RelativeBoundingBox::BOUNDING_BOX_EXTENSION_MIN,
            );
            let sane_visible_sections_max = voxelize_plane_slow(
                relative_tile_pos,
                plane,
                RelativeBoundingBox::BOUNDING_BOX_EXTENSION_MAX,
            );
            let test_visible_sections = voxelize_plane(relative_tile_pos, plane, plane_bb_offsets);

            if !test_minimum_maximum(
                &sane_visible_sections_min,
                &sane_visible_sections_max,
                &test_visible_sections,
            ) {
                panic!(
                    "Test results don't fit in sane bounds. Relative Tile Coords: {relative_tile_pos:?}, Plane: {plane:?}",
                );
            }
        }
    }
}
