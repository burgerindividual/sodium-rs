use std::array;

use super::*;
use crate::graph::coords::RelativeBoundingBox;

/// When using this, it is expected that coordinates are relative to the camera
/// rather than the world origin.
pub struct Frustum {
    #[cfg(debug_assertions)]
    pub planes: [f32x4; DIRECTION_COUNT],

    // Plane data ordered component-wise rather than plane-wise. The contents are transposed from
    // the normal plane array
    planes_cw: [Simd<f32, DIRECTION_COUNT>; 4],
    pub(crate) plane_bb_offsets: [f32x3; DIRECTION_COUNT],
    pub(crate) planes_scaled: [f32x4; DIRECTION_COUNT],
}

impl Frustum {
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

        Frustum {
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

            let sections_in_plane = tile::frustum::voxelize_plane(
                relative_tile_pos,
                unsafe { *self.planes_scaled.get_unchecked(plane_idx) },
                unsafe { *self.plane_bb_offsets.get_unchecked(plane_idx) },
            );

            #[cfg(debug_assertions)]
            {
                use crate::graph::tile::print_tile;

                let sane_sections_in_plane =
                    tile::frustum::voxelize_plane_slow(relative_tile_pos, unsafe {
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

fn voxelize_plane(relative_tile_pos: f32x3, plane_scaled: f32x4, plane_bb_offsets: f32x3) -> u8x64 {
    let mut section_bb_offsets = relative_tile_pos + plane_bb_offsets;

    // if plane[X] was positive, this will be all 1 bits. if plane[X] is negative,
    // this will be all 0 bits.
    let plane_x_positive_mask = plane_scaled[X].to_bits() as i32;

    Simd::from_slice(
        array::from_fn::<_, 8, _>(|_| {
            let section_bb_zs = f32x8::from_array([0.0, 16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0])
                + Simd::splat(section_bb_offsets[Z]);

            let tile_x_positions = section_bb_zs.mul_add_fast(
                Simd::splat(plane_scaled[Z]),
                Simd::splat(section_bb_offsets[X].mul_add_fast(
                    const { -1.0 / 16.0 },
                    section_bb_offsets[Y].mul_add_fast(plane_scaled[Y], plane_scaled[W]),
                )),
            );

            // Increment Y by length of section in blocks after usage of offsets
            section_bb_offsets += Simd::from_xyz(0.0, 16.0, 0.0);

            let tile_x_positions_int = unsafe { tile_x_positions.to_int_unchecked::<i32>() };

            #[cfg(target_feature = "avx2")]
            let tile_x_shift: i32x8 = unsafe {
                use std::arch::x86_64::*;
                // this lets us skip having to mask tile_x_positions_int
                _mm256_sllv_epi32(_mm256_set1_epi32(0b10), tile_x_positions_int.into()).into()
            };
            #[cfg(not(target_feature = "avx2"))]
            let tile_x_shift = Simd::splat(0b10) << tile_x_positions_int;
            // TODO: how does this work? why do we not need to unconditionally include the
            // section we derived? and why does this even work with negatives at all??
            // conditionally NOT part of the mask using an XOR
            let tile_x_masks = (tile_x_shift - Simd::splat(1)) ^ Simd::splat(plane_x_positive_mask);

            let tile_x_masks_clamped = (tile_x_positions - Simd::splat(8.0))
                .is_sign_positive_fast()
                .select(
                    !Simd::splat(plane_x_positive_mask),
                    tile_x_positions
                        .is_sign_negative_fast()
                        .select(Simd::splat(plane_x_positive_mask), tile_x_masks),
                )
                .cast();

            tile_x_masks_clamped.to_array()
        })
        .as_flattened(),
    )
}

fn voxelize_plane_slow(relative_tile_pos: f32x3, plane: f32x4) -> u8x64 {
    let mut visible_sections = SECTIONS_EMPTY;

    for y in 0..8 {
        for z in 0..8 {
            for x in 0..8 {
                let min = u8x3::from_xyz(x, y, z)
                    .cast::<f32>()
                    .mul_add_fast(Simd::splat(16.0), relative_tile_pos);
                let bb = RelativeBoundingBox::new(min, min + Simd::splat(16.0));

                let not_outside = plane[X] * (if plane[X] < 0.0 { bb.min[X] } else { bb.max[X] })
                    + plane[Y] * (if plane[Y] < 0.0 { bb.min[Y] } else { bb.max[Y] })
                    + plane[Z] * (if plane[Z] < 0.0 { bb.min[Z] } else { bb.max[Z] })
                    >= -plane[W];

                modify_bit(
                    &mut visible_sections,
                    section_index(Simd::from_xyz(x, y, z)),
                    not_outside,
                );
            }
        }
    }

    visible_sections
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: automate this
    #[test]
    fn frustum_voxelization_test() {
        let relative_tile_pos = Simd::from_xyz(-168.475, -183.705, -63.434998);

        let frustum = Frustum::new([
            Simd::from_array([-0.591241, -0.49853715, 0.6339517, 0.0]),
            Simd::from_array([-0.23236583, 0.1140805, 0.96591496, 0.0]),
            Simd::from_array([-0.19515383, -0.55120045, 0.81122935, -0.049999997]),
            Simd::from_array([0.23822449, -0.49853715, 0.8334925, -0.0]),
            Simd::from_array([-0.06662716, -0.9585686, 0.27696052, -0.0]),
            Simd::from_array([0.1951034, 0.55120337, -0.81123954, 512.102]),
        ]);

        let mut failed = false;
        let mut directions = ALL_DIRECTIONS;
        while directions != 0 {
            let direction = take_one(&mut directions);
            let dir_idx = to_index(direction);

            let sane_visible_sections =
                frustum::voxelize_plane_slow(relative_tile_pos, frustum.planes[dir_idx]);
            let test_visible_sections = frustum::voxelize_plane(
                relative_tile_pos,
                frustum.planes_scaled[dir_idx],
                frustum.plane_bb_offsets[dir_idx],
            );

            if test_visible_sections == sane_visible_sections {
                continue;
            } else {
                failed = true;
            }

            let dir_str = to_str(direction);

            println!("Plane {dir_str} - Sane");
            print_tile(&sane_visible_sections);

            println!("Plane {dir_str} - Test");
            print_tile(&test_visible_sections);
        }

        if failed {
            panic!();
        }
    }
}
