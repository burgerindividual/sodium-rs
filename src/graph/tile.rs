use std::array;

use core_simd::simd::prelude::*;
use core_simd::simd::ToBytes;
use std_float::StdFloat;

use super::context::RelativeBoundingBox;
use super::visibility::*;
use super::{connection_index, u8x3, *};
use crate::bitset;
use crate::bitset::BitSet;
use crate::math::Coords3;

pub const SECTIONS_EMPTY: u8x64 = Simd::splat(0);
pub const SECTIONS_FILLED: u8x64 = Simd::splat(!0);

pub fn section_index(coords: u8x3) -> u16 {
    debug_assert!(coords.simd_lt(Simd::splat(8)).all());

    ((coords[Y] as u16) << 6) | ((coords[Z] as u16) << 3) | (coords[X] as u16)
}

pub fn get_bit(sections: &u8x64, index: u16) -> bool {
    let array_idx = index as usize >> 3;
    let bit_idx = index as u8 & 0b111;
    let byte = unsafe { *sections.as_array().get_unchecked(array_idx) };
    byte.get_bit(bit_idx)
}

pub fn set_bit(sections: &mut u8x64, index: u16) {
    let array_idx = index as usize >> 3;
    let bit_idx = index as u8 & 0b111;
    let byte = unsafe { sections.as_mut_array().get_unchecked_mut(array_idx) };
    byte.set_bit(bit_idx);
}

pub fn clear_bit(sections: &mut u8x64, index: u16) {
    let array_idx = index as usize >> 3;
    let bit_idx = index as u8 & 0b111;
    let byte = unsafe { sections.as_mut_array().get_unchecked_mut(array_idx) };
    byte.clear_bit(bit_idx);
}

pub fn modify_bit(sections: &mut u8x64, index: u16, value: bool) {
    let array_idx = index as usize >> 3;
    let bit_idx = index as u8 & 0b111;
    let byte = unsafe { sections.as_mut_array().get_unchecked_mut(array_idx) };
    byte.modify_bit(bit_idx, value);
}

pub fn or_bit(sections: &mut u8x64, index: u16, value: bool) {
    let array_idx = index as usize >> 3;
    let bit_idx = index as u8 & 0b111;
    let byte = unsafe { sections.as_mut_array().get_unchecked_mut(array_idx) };
    byte.or_bit(bit_idx, value);
}

pub fn print_tile(sections: &u8x64) {
    for y in 0..8 {
        println!("↓Y{y}");
        for z in 0..8 {
            for x in 0..8 {
                print!(
                    "{}",
                    if get_bit(sections, section_index(Simd::from_xyz(x, y, z))) {
                        1_u8
                    } else {
                        0_u8
                    }
                );
            }
            println!(" Z{z}");
        }
    }
}

// TODO: merge the shift methods and move to edge methods together with const
// generics

pub fn edge_neg_to_pos_x(sections: u8x64) -> u8x64 {
    sections << 7
}

pub fn edge_pos_to_neg_x(sections: u8x64) -> u8x64 {
    sections >> 7
}

#[rustfmt::skip]
pub fn edge_neg_to_pos_z(sections: u8x64) -> u8x64 {
    simd_swizzle!(
        sections,
        Simd::splat(0),
        [
            64, 64, 64, 64, 64, 64, 64, 0,
            64, 64, 64, 64, 64, 64, 64, 8,
            64, 64, 64, 64, 64, 64, 64, 16,
            64, 64, 64, 64, 64, 64, 64, 24,
            64, 64, 64, 64, 64, 64, 64, 32,
            64, 64, 64, 64, 64, 64, 64, 40,
            64, 64, 64, 64, 64, 64, 64, 48,
            64, 64, 64, 64, 64, 64, 64, 56,
        ]
    )
}

#[rustfmt::skip]
pub fn edge_pos_to_neg_z(sections: u8x64) -> u8x64 {
    simd_swizzle!(
        sections,
        Simd::splat(0),
        [
            7,  64, 64, 64, 64, 64, 64, 64,
            15, 64, 64, 64, 64, 64, 64, 64,
            23, 64, 64, 64, 64, 64, 64, 64,
            31, 64, 64, 64, 64, 64, 64, 64,
            39, 64, 64, 64, 64, 64, 64, 64,
            47, 64, 64, 64, 64, 64, 64, 64,
            55, 64, 64, 64, 64, 64, 64, 64,
            63, 64, 64, 64, 64, 64, 64, 64,
        ]
    )
}

#[rustfmt::skip]
pub fn edge_neg_to_pos_y(sections: u8x64) -> u8x64 {
    simd_swizzle!(
        sections,
        Simd::splat(0),
        [
            64,  65,  66,  67,  68,  69,  70,  71,
            72,  73,  74,  75,  76,  77,  78,  79,
            80,  81,  82,  83,  84,  85,  86,  87,
            88,  89,  90,  91,  92,  93,  94,  95,
            96,  97,  98,  99,  100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 116, 117, 118, 119,
            0,   1,   2,   3,   4,   5,   6,   7,
        ]
    )
}

#[rustfmt::skip]
pub fn edge_pos_to_neg_y(sections: u8x64) -> u8x64 {
    simd_swizzle!(
        sections,
        Simd::splat(0),
        [
            56,  57,  58,  59,  60,  61,  62,  63,
            64,  65,  66,  67,  68,  69,  70,  71,
            72,  73,  74,  75,  76,  77,  78,  79,
            80,  81,  82,  83,  84,  85,  86,  87,
            88,  89,  90,  91,  92,  93,  94,  95,
            96,  97,  98,  99,  100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 116, 117, 118, 119,
        ]
    )
}

pub fn shift_neg_x(sections: u8x64) -> u8x64 {
    sections >> 1
}

pub fn shift_pos_x(sections: u8x64) -> u8x64 {
    sections << 1
}

#[rustfmt::skip]
pub fn shift_neg_z(sections: u8x64) -> u8x64 {
    // The u8x64 "sections" vector represents an 8x8x8 array of bits, with each
    // bit representing a render section. It is indexed with the pattern
    // YYYZZZXXX. Because of our indexing scheme, we know that each u8 lane
    // in the vector represents a row of sections on the X axis.
    // 
    // The array of indices provided to this swizzle can be read with
    // the following diagram:
    // 
    //     z=0       Z Axis      z=7
    //  y=0|------------------------
    //     |
    //     |
    //  Y  |
    // Axis|
    //     |
    //     |
    // y=7 |
    // 
    // Keep in mind, a swizzle with an array of indices full of only incrementing
    // indices starting at 0 would result in a completely unmodified vector. That
    // array would look like the following:
    //
    // 0,  1,  2,  3,  4,  5,  6,  7, 
    // 8,  9,  10, 11, 12, 13, 14, 15,
    // 16, 17, 18, 19, 20, 21, 22, 23,
    // 24, 25, 26, 27, 28, 29, 30, 31,
    // 32, 33, 34, 35, 36, 37, 38, 39,
    // 40, 41, 42, 43, 44, 45, 46, 47,
    // 48, 49, 50, 51, 52, 53, 54, 55,
    // 56, 57, 58, 59, 60, 61, 62, 63,
    // 
    // By shifting each index in that array to the left by 1, this swizzle
    // operation effectively shifts each X-axis row of sections by -1 on the Z
    // axis. The "64" indices seen in this swizzle are used to fill the empty
    // space that the shift left over with zeroes.
    simd_swizzle!(
        sections,
        Simd::splat(0),
        [
            1,  2,  3,  4,  5,  6,  7,  64,
            9,  10, 11, 12, 13, 14, 15, 64,
            17, 18, 19, 20, 21, 22, 23, 64,
            25, 26, 27, 28, 29, 30, 31, 64,
            33, 34, 35, 36, 37, 38, 39, 64,
            41, 42, 43, 44, 45, 46, 47, 64,
            49, 50, 51, 52, 53, 54, 55, 64,
            57, 58, 59, 60, 61, 62, 63, 64,
        ]
    )
}

#[rustfmt::skip]
pub fn shift_pos_z(sections: u8x64) -> u8x64 {
    simd_swizzle!(
        sections,
        Simd::splat(0),
        [
            64, 0,  1,  2,  3,  4,  5,  6,
            64, 8,  9,  10, 11, 12, 13, 14,
            64, 16, 17, 18, 19, 20, 21, 22,
            64, 24, 25, 26, 27, 28, 29, 30,
            64, 32, 33, 34, 35, 36, 37, 38,
            64, 40, 41, 42, 43, 44, 45, 46,
            64, 48, 49, 50, 51, 52, 53, 54,
            64, 56, 57, 58, 59, 60, 61, 62,
        ]
    )
}

#[rustfmt::skip]
pub fn shift_neg_y(sections: u8x64) -> u8x64 {
    simd_swizzle!(
        sections,
        Simd::splat(0),
        [
            8,  9,  10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 66, 67, 68, 69, 70, 71,
        ]
    )
}

#[rustfmt::skip]
pub fn shift_pos_y(sections: u8x64) -> u8x64 {
    simd_swizzle!(
        sections,
        Simd::splat(0),
        [
            64, 65, 66, 67, 68, 69, 70, 71,
            0,  1,  2,  3,  4,  5,  6,  7,
            8,  9,  10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55,
        ]
    )
}

pub fn voxelize_frustum_plane(
    relative_tile_pos: f32x3,
    plane_scaled: f32x4,
    plane_bb_offsets: f32x3,
) -> u8x64 {
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

pub fn voxelize_frustum_plane_slow(relative_tile_pos: f32x3, plane: f32x4) -> u8x64 {
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

pub fn gen_outward_direction_masks(camera_section_in_tile: u8x3) -> [u8x64; DIRECTION_COUNT] {
    let neg_x_lane = (0b10_u8 << camera_section_in_tile[X]).wrapping_sub(1);
    let neg_x_mask = Simd::splat(neg_x_lane);

    let pos_x_lane = 0xFF << camera_section_in_tile[X];
    let pos_x_mask = Simd::splat(pos_x_lane);

    let neg_y_bitmask = (0b10 << camera_section_in_tile[Y]) - 1;
    let neg_y_mask = mask64x8::from_bitmask(neg_y_bitmask).to_int().to_ne_bytes();

    // Mask is truncated to u8 by from_bitmask
    let pos_y_bitmask = 0xFF << camera_section_in_tile[Y];
    let pos_y_mask = mask64x8::from_bitmask(pos_y_bitmask).to_int().to_ne_bytes();

    // native endianness should be correct here, but it's worth double checking
    let neg_z_bitmask = (0b10 << camera_section_in_tile[Z]) - 1;
    let neg_z_lane = u64::from_ne_bytes(
        mask8x8::from_bitmask(neg_z_bitmask)
            .to_int()
            .to_ne_bytes()
            .to_array(),
    );
    let neg_z_mask = u64x8::splat(neg_z_lane).to_ne_bytes();

    let pos_z_bitmask = 0xFF << camera_section_in_tile[Z];
    let pos_z_lane = u64::from_ne_bytes(
        mask8x8::from_bitmask(pos_z_bitmask)
            .to_int()
            .to_ne_bytes()
            .to_array(),
    );
    let pos_z_mask = u64x8::splat(pos_z_lane).to_ne_bytes();

    [
        neg_x_mask, neg_y_mask, neg_z_mask, pos_x_mask, pos_y_mask, pos_z_mask,
    ]
}

pub fn gen_angle_visibility_masks(relative_tile_pos: f32x3) -> [u8x64; 3] {
    let offsets = relative_tile_pos.mul_add_fast(Simd::splat(1.0 / 16.0), Simd::splat(0.5));

    let (xy_mask_compressed, yx_mask_compressed) =
        gen_compressed_angle_mask_pair(offsets[X], offsets[Y]);
    let xy_mask = expand_xy_angle_mask(xy_mask_compressed);
    let yx_mask = expand_xy_angle_mask(yx_mask_compressed);

    let (xz_mask_compressed, zx_mask_compressed) =
        gen_compressed_angle_mask_pair(offsets[X], offsets[Z]);
    let xz_mask = expand_xz_angle_mask(xz_mask_compressed);
    let zx_mask = expand_xz_angle_mask(zx_mask_compressed);

    let (zy_mask_compressed, yz_mask_compressed) =
        gen_compressed_angle_mask_pair(offsets[Z], offsets[Y]);
    let zy_mask = expand_zy_angle_mask(zy_mask_compressed);
    let yz_mask = expand_zy_angle_mask(yz_mask_compressed);

    let x_mask = yx_mask & zx_mask;
    let y_mask = xy_mask & zy_mask;
    let z_mask = xz_mask & yz_mask;

    [x_mask, y_mask, z_mask]
}

// This *really* doesn't like being inlined for some reason
#[inline(never)]
pub fn gen_compressed_angle_mask_pair(offset_1: f32, offset_2: f32) -> (u8x8, u8x8) {
    let neg_x_offset = Simd::splat(-offset_1);
    let y_offset = Simd::splat(offset_2);
    let ys = Simd::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    let line_1 = neg_x_offset + y_offset + ys;
    let line_2 = neg_x_offset - y_offset - ys;

    let lower_bound = line_1.simd_min(line_2);
    let upper_bound = line_1.simd_max(line_2);

    let (lower_bound_ceil_clamped, upper_bound_floor, lower_bound_mask, upper_bound_mask) =
        rasterize_rows(lower_bound, upper_bound);
    let combined_mask = lower_bound_mask & upper_bound_mask;

    // Get lowest set bit of the mask if the bound falls on an integer.
    let lowest_bit_mask = lower_bound.simd_eq(lower_bound_ceil_clamped).select(
        lower_bound_mask & lower_bound_mask.wrapping_neg(),
        Simd::splat(0),
    );
    // Get highest set bit of the mask if the bound falls on an integer
    let highest_bit_mask = upper_bound
        .simd_eq(upper_bound_floor)
        .select((upper_bound_mask + Simd::splat(1)) >> 1, Simd::splat(0));
    let reverse_mask = lowest_bit_mask | highest_bit_mask | !combined_mask;

    // cut off upper bits
    (combined_mask.cast::<u8>(), reverse_mask.cast::<u8>())
}

pub fn expand_xy_angle_mask(compressed_mask: u8x8) -> u8x64 {
    simd_swizzle!(
        compressed_mask,
        [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
            7, 7, 7, 7, 7, 7,
        ]
    )
}

pub fn expand_xz_angle_mask(compressed_mask: u8x8) -> u8x64 {
    simd_swizzle!(
        compressed_mask,
        [
            0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4,
            5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1,
            2, 3, 4, 5, 6, 7,
        ]
    )
}

pub fn expand_zy_angle_mask(compressed_mask: u8x8) -> u8x64 {
    mask8x64::from_bitmask(u64::from_ne_bytes(compressed_mask.to_array()))
        .to_int()
        .to_ne_bytes()
}

pub fn voxelize_fog_cylinder(relative_tile_pos: f32x3, fog_distance: f32) -> u8x64 {
    const BB_EPSILON: f32 = RelativeBoundingBox::BOUNDING_BOX_EPSILON;
    const BB_EPSILON_SCALED: f32 = BB_EPSILON / 16.0;

    let section_zs = (f32x8::from_array([0.0, 16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0])
        - Simd::splat(BB_EPSILON))
        + Simd::splat(relative_tile_pos[Z]);

    let distance_zs = Simd::splat(0.0)
        .simd_max(section_zs)
        .simd_min(section_zs + Simd::splat(16.0 + (BB_EPSILON * 2.0)));

    let c_squared =
        distance_zs.mul_add_fast(-distance_zs, Simd::splat(fog_distance * fog_distance));
    let c = c_squared.sqrt();

    let upper_bound = (c - Simd::splat(relative_tile_pos[X]))
        .mul_add_fast(Simd::splat(1.0 / 16.0), Simd::splat(BB_EPSILON_SCALED));
    let lower_bound = (c + Simd::splat(relative_tile_pos[X])).mul_add_fast(
        Simd::splat(-1.0 / 16.0),
        Simd::splat(-1.0 - BB_EPSILON_SCALED),
    );

    let (.., lower_bound_mask, upper_bound_mask) = rasterize_rows(lower_bound, upper_bound);
    let out_of_bounds_mask = c_squared.is_sign_positive_fast().to_int().cast::<u32>();
    let combined_mask = (lower_bound_mask & upper_bound_mask & out_of_bounds_mask).cast::<u8>();

    let zx_mask = u64x8::splat(u64::from_ne_bytes(combined_mask.to_array())).to_ne_bytes();

    let y_lower_bound_mask = (0xFF_u32
        << unsafe {
            (-fog_distance - relative_tile_pos[Y])
                .mul_add_fast(1.0 / 16.0, -BB_EPSILON_SCALED)
                .floor()
                .to_int_unchecked::<i32>()
                .clamp(0, 8)
        }) as u8;
    let y_upper_bound_mask = (0xFF_u32
        >> unsafe {
            8 - (fog_distance - relative_tile_pos[Y])
                .mul_add_fast(1.0 / 16.0, BB_EPSILON_SCALED)
                .ceil()
                .to_int_unchecked::<i32>()
                .clamp(0, 8)
        }) as u8;
    let y_mask = y_lower_bound_mask & y_upper_bound_mask;
    let y_mask_expanded = mask64x8::from_bitmask(y_mask as u64).to_int().to_ne_bytes();

    zx_mask & y_mask_expanded
}

pub fn rasterize_rows(lower_bound: f32x8, upper_bound: f32x8) -> (f32x8, f32x8, u32x8, u32x8) {
    let lower_bound_ceil_clamped = lower_bound
        .ceil()
        .simd_clamp(Simd::splat(0.0), Simd::splat(8.0));
    let upper_bound_floor = upper_bound.floor();

    let lower_bound_shifts = unsafe {
        lower_bound_ceil_clamped
            .to_int_unchecked::<i32>()
            .cast::<u32>()
    };
    let upper_bound_shifts = unsafe {
        (upper_bound_floor.to_int_unchecked::<i32>() + Simd::splat(1))
            .simd_clamp(Simd::splat(0), Simd::splat(9))
            .cast::<u32>()
    };

    let lower_bound_mask = Simd::splat(!0) << lower_bound_shifts;
    let upper_bound_mask = !(Simd::splat(!0) << upper_bound_shifts);

    (
        lower_bound_ceil_clamped,
        upper_bound_floor,
        lower_bound_mask,
        upper_bound_mask,
    )
}

#[derive(Debug)]
pub struct Tile {
    // Only changes on section update
    pub connection_section_sets: [u8x64; UNIQUE_CONNECTION_COUNT],

    // Changes every time tile is processed
    pub outgoing_dir_section_sets: [u8x64; DIRECTION_COUNT],
    pub visible_sections: u8x64,

    #[cfg(debug_assertions)]
    pub processed: bool,
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            // fully untraversable by default
            connection_section_sets: [SECTIONS_EMPTY; UNIQUE_CONNECTION_COUNT],
            outgoing_dir_section_sets: [SECTIONS_EMPTY; DIRECTION_COUNT],
            // All sections are visible, and the culling methods mask parts of this
            visible_sections: SECTIONS_FILLED,
            #[cfg(debug_assertions)]
            processed: false,
        }
    }
}

impl Tile {
    pub fn set_empty(&mut self) {
        self.outgoing_dir_section_sets = [SECTIONS_EMPTY; DIRECTION_COUNT];
        self.visible_sections = SECTIONS_EMPTY;
    }

    // TODO: review all fast paths
    // TODO: is it necessary to use tile_incoming_directions for the first
    // iteration?
    pub fn traverse<const TRAVERSAL_DIRS: u8>(
        &mut self,
        start_sections: u8x64,
        mut incoming_dir_section_sets: [u8x64; DIRECTION_COUNT],
        outward_direction_masks: &[u8x64; DIRECTION_COUNT],
        angle_visibility_masks: &[u8x64; 3],
    ) {
        // TODO OPT: consider changing this back to "for _ in 0..24" and measure
        loop {
            let mut incoming_changed = false;

            self.try_traverse_dir::<TRAVERSAL_DIRS, NEG_X>(
                &mut incoming_dir_section_sets,
                outward_direction_masks,
                angle_visibility_masks,
                &mut incoming_changed,
            );
            self.try_traverse_dir::<TRAVERSAL_DIRS, NEG_Y>(
                &mut incoming_dir_section_sets,
                outward_direction_masks,
                angle_visibility_masks,
                &mut incoming_changed,
            );
            self.try_traverse_dir::<TRAVERSAL_DIRS, NEG_Z>(
                &mut incoming_dir_section_sets,
                outward_direction_masks,
                angle_visibility_masks,
                &mut incoming_changed,
            );
            self.try_traverse_dir::<TRAVERSAL_DIRS, POS_X>(
                &mut incoming_dir_section_sets,
                outward_direction_masks,
                angle_visibility_masks,
                &mut incoming_changed,
            );
            self.try_traverse_dir::<TRAVERSAL_DIRS, POS_Y>(
                &mut incoming_dir_section_sets,
                outward_direction_masks,
                angle_visibility_masks,
                &mut incoming_changed,
            );
            self.try_traverse_dir::<TRAVERSAL_DIRS, POS_Z>(
                &mut incoming_dir_section_sets,
                outward_direction_masks,
                angle_visibility_masks,
                &mut incoming_changed,
            );

            if !incoming_changed {
                break;
            }
        }

        // TODO: AND with existing visible nodes when there are more culling stages
        //  okay so this happened, what do we do?
        self.visible_sections = incoming_dir_section_sets
            .iter()
            .fold(start_sections, |a, b| a | b);
    }

    pub fn setup_center_tile(&mut self, section_index: u16) {
        let mut outgoing_dirs = ALL_DIRECTIONS;
        while outgoing_dirs != 0 {
            let outgoing_dir = take_one(&mut outgoing_dirs);
            let sections_outgoing = unsafe {
                self.outgoing_dir_section_sets
                    .get_unchecked_mut(to_index(outgoing_dir))
            };

            let mut incoming_dirs = all_except(outgoing_dir);
            while incoming_dirs != 0 {
                let incoming_dir = take_one(&mut incoming_dirs);

                let connected = get_bit(
                    unsafe {
                        self.connection_section_sets
                            .get_unchecked(connection_index(outgoing_dir, incoming_dir))
                    },
                    section_index,
                );

                or_bit(sections_outgoing, section_index, connected);
            }
        }
    }

    fn try_traverse_dir<const TRAVERSAL_DIRS: u8, const OUTGOING_DIR: u8>(
        &mut self,
        incoming_dir_section_sets: &mut [u8x64; DIRECTION_COUNT],
        outward_direction_masks: &[u8x64; DIRECTION_COUNT],
        angle_visibility_masks: &[u8x64; 3],
        incoming_changed: &mut bool,
    ) {
        if bitset::contains_u8(TRAVERSAL_DIRS, OUTGOING_DIR) {
            let dir_index = to_index(OUTGOING_DIR);
            let axis_index = index_dir_to_axis(dir_index);
            let opposite_dir_index = to_index(opposite(OUTGOING_DIR));

            self.find_outgoing_connections::<TRAVERSAL_DIRS, OUTGOING_DIR>(
                &incoming_dir_section_sets,
                outward_direction_masks[dir_index],
                angle_visibility_masks[axis_index],
            );

            let outgoing_sections = self.outgoing_dir_section_sets[dir_index];
            let shifted_masked = match OUTGOING_DIR {
                NEG_X => shift_neg_x(outgoing_sections),
                NEG_Y => shift_neg_y(outgoing_sections),
                NEG_Z => shift_neg_z(outgoing_sections),
                POS_X => shift_pos_x(outgoing_sections),
                POS_Y => shift_pos_y(outgoing_sections),
                POS_Z => shift_pos_z(outgoing_sections),
                _ => unreachable!(),
            } & self.visible_sections;

            // TODO: does this have to be an OR? I think the answer is yes
            let previous = incoming_dir_section_sets[opposite_dir_index];
            incoming_dir_section_sets[opposite_dir_index] |= shifted_masked;

            *incoming_changed |= incoming_dir_section_sets[opposite_dir_index] != previous;
        }
    }

    fn find_outgoing_connections<const TRAVERSAL_DIRS: u8, const OUTGOING_DIR: u8>(
        &mut self,
        incoming_dir_section_sets: &[u8x64; DIRECTION_COUNT],
        outward_direction_mask: u8x64,
        angle_visibility_mask: u8x64,
    ) {
        let sections_outgoing = &mut self.outgoing_dir_section_sets[to_index(OUTGOING_DIR)];

        let mut incoming_dirs = opposite(TRAVERSAL_DIRS) & !OUTGOING_DIR;
        while incoming_dirs != 0 {
            let incoming_dir = take_one(&mut incoming_dirs);

            let mut connection_sections =
                self.connection_section_sets[connection_index(OUTGOING_DIR, incoming_dir)];

            if incoming_dir == opposite(OUTGOING_DIR) {
                connection_sections &= angle_visibility_mask;
            }

            *sections_outgoing |=
                incoming_dir_section_sets[to_index(incoming_dir)] & connection_sections;
        }

        let opposing_directions =
            bitset::contains_u8(TRAVERSAL_DIRS, OUTGOING_DIR | opposite(OUTGOING_DIR));

        if opposing_directions {
            *sections_outgoing &= outward_direction_mask;
        }
    }
}
