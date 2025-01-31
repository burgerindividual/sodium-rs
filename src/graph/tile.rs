use std::array;

use core_simd::simd::prelude::*;
use core_simd::simd::ToBytes;

use super::context::RelativeBoundingBox;
use super::visibility::*;
use super::{connection_index, u8x3, *};
use crate::bitset;
use crate::bitset::BitSet;
use crate::math::Coords3;

pub const SECTIONS_EMPTY: u8x64 = Simd::splat(0);
pub const SECTIONS_FILLED: u8x64 = Simd::splat(0xFF);

pub fn section_index(coords: u8x3) -> u16 {
    debug_assert!(coords.simd_lt(Simd::splat(8)).all());

    ((coords[Z] as u16) << 6) | ((coords[Y] as u16) << 3) | (coords[X] as u16)
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

pub fn modify_bit(sections: &mut u8x64, index: u16, value: bool) {
    let array_idx = index as usize >> 3;
    let bit_idx = index as u8 & 0b111;
    let byte = unsafe { sections.as_mut_array().get_unchecked_mut(array_idx) };
    byte.modify_bit(bit_idx, value);
}

pub fn print_tile(sections: &u8x64) {
    for z in 0..8 {
        println!("↓Z{z}");
        for y in 0..8 {
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
            println!(" Y{y}");
        }
    }
}

// TODO: merge the shift methods and move to edge methods together with const
// generics

pub fn edge_neg_to_pos_x(sections: u8x64) -> u8x64 {
    sections << Simd::splat(7)
}

pub fn edge_pos_to_neg_x(sections: u8x64) -> u8x64 {
    sections >> Simd::splat(7)
}

#[rustfmt::skip]
pub fn edge_neg_to_pos_y(sections: u8x64) -> u8x64 {
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
pub fn edge_pos_to_neg_y(sections: u8x64) -> u8x64 {
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
pub fn edge_neg_to_pos_z(sections: u8x64) -> u8x64 {
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
pub fn edge_pos_to_neg_z(sections: u8x64) -> u8x64 {
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
    sections >> Simd::splat(1)
}

pub fn shift_pos_x(sections: u8x64) -> u8x64 {
    sections << Simd::splat(1)
}

#[rustfmt::skip]
pub fn shift_neg_y(sections: u8x64) -> u8x64 {
    // The u8x64 "sections" vector represents an 8x8x8 array of bits, with each
    // bit representing a render section. It is indexed with the pattern
    // ZZZYYYXXX. Because of our indexing scheme, we know that each u8 lane
    // in the vector represents a row of sections on the X axis.
    // 
    // The array of indices provided to this swizzle can be read with
    // the following diagram:
    // 
    //     y=0       Y Axis      y=7
    //  z=0|------------------------
    //     |
    //     |
    //  Z  |
    // Axis|
    //     |
    //     |
    // z=7 |
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
    // operation effectively shifts each X-axis row of sections by -1 on the Y
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
pub fn shift_pos_y(sections: u8x64) -> u8x64 {
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
pub fn shift_neg_z(sections: u8x64) -> u8x64 {
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
pub fn shift_pos_z(sections: u8x64) -> u8x64 {
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

#[no_mangle]
// TODO: put this back into Tile, do bitwise AND on visible nodes
// TODO OPT: pre-calculate partial BB offsets for plane
// TODO OPT: pre-divide planes with -plane.x, check if this is accurate enough
pub fn voxelize_frustum_plane(relative_tile_coords: f32x3, plane: f32x4) -> u8x64 {
    const SIGN_BIT: u32 = 1 << 31;

    let mut section_bb_offsets = relative_tile_coords
        + plane
            .resize(0.0)
            .to_bits()
            .simd_ge(Simd::splat(SIGN_BIT))
            .select(
                Simd::splat(16.0 + RelativeBoundingBox::BOUNDING_BOX_EPSILON),
                Simd::splat(-RelativeBoundingBox::BOUNDING_BOX_EPSILON),
            );

    Simd::from_slice(
        array::from_fn::<_, 8, _>(|_| {
            let section_bb_ys = f32x8::from_array([0.0, 16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0])
                + Simd::splat(section_bb_offsets[Y]);

            let dot_products = section_bb_ys.mul_add_fast(
                Simd::splat(plane[Y]),
                Simd::splat(
                    plane[X].mul_add_fast(section_bb_offsets[X], plane[Z] * section_bb_offsets[Z])
                        + plane[W],
                ),
            );

            // Increment Z by length of section in blocks after usage of offsets
            section_bb_offsets += Simd::from_xyz(0.0, 0.0, 16.0);

            let tile_x_positions = -dot_products / Simd::splat(plane[X] * 16.0);

            let tile_x_masks = (Simd::splat(1_i32)
                << (unsafe { tile_x_positions.to_int_unchecked() } + Simd::splat(1)))
                - Simd::splat(1);
            let tile_x_masks_clamped = tile_x_positions
                .simd_ge(Simd::splat(8.0))
                .select(
                    Simd::splat(-1_i32),
                    tile_x_positions
                        .to_bits()
                        .simd_ge(Simd::splat(SIGN_BIT))
                        .select(Simd::splat(0_i32), tile_x_masks),
                )
                .cast();

            tile_x_masks_clamped.to_array()
        })
        .as_flattened(),
    )
}

pub fn voxelize_frustum_plane_slow(relative_tile_coords: f32x3, plane: f32x4) -> u8x64 {
    let mut visible_sections = SECTIONS_EMPTY;

    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                let min = u8x3::from_xyz(x, y, z)
                    .cast::<f32>()
                    .mul_add_fast(Simd::splat(16.0), relative_tile_coords);
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

// TODO: verify these are correct
pub fn create_camera_direction_masks(camera_section_in_tile: u8x3) -> [u8x64; DIRECTION_COUNT] {
    let neg_x_lane = (0b10_u8 << camera_section_in_tile[X]).wrapping_sub(1);
    let neg_x_mask = Simd::splat(neg_x_lane);

    let pos_x_lane = 0xFF << camera_section_in_tile[X];
    let pos_x_mask = Simd::splat(pos_x_lane);

    // native endianness should be correct here, but it's worth double checking
    let neg_y_bitmask = (0b10 << camera_section_in_tile[Y]) - 1;
    let neg_y_lane = u64::from_ne_bytes(
        mask8x8::from_bitmask(neg_y_bitmask)
            .to_int()
            .to_ne_bytes()
            .to_array(),
    );
    let neg_y_mask = u64x8::splat(neg_y_lane).to_ne_bytes();

    let pos_y_bitmask = 0xFF << camera_section_in_tile[Y];
    let pos_y_lane = u64::from_ne_bytes(
        mask8x8::from_bitmask(pos_y_bitmask)
            .to_int()
            .to_ne_bytes()
            .to_array(),
    );
    let pos_y_mask = u64x8::splat(pos_y_lane).to_ne_bytes();

    let neg_z_bitmask = (0b10 << camera_section_in_tile[Z]) - 1;
    let neg_z_mask = mask64x8::from_bitmask(neg_z_bitmask).to_int().to_ne_bytes();

    let pos_z_bitmask = 0xFF << camera_section_in_tile[Z];
    let pos_z_mask = mask64x8::from_bitmask(pos_z_bitmask).to_int().to_ne_bytes();

    [
        neg_x_mask, neg_y_mask, neg_z_mask, pos_x_mask, pos_y_mask, pos_z_mask,
    ]
}

// TODO: switch to YZX or YXZ indexing from ZYX to allow faster splitting into
// regions of 8x4x8
// TODO: maybe just make tiles the size of regions?
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
            // TODO: should this start out as all 1s?
            visible_sections: SECTIONS_EMPTY,
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
    // TODO: use existing visible nodes as masks when earlier stages exist
    // TODO: is it necessary to use tile_incoming_directions for the first
    // iteration?
    pub fn find_visible_sections<const TRAVERSAL_DIRS: u8>(
        &mut self,
        start_visible_sections: u8x64,
        mut incoming_dir_section_sets: [u8x64; DIRECTION_COUNT],
        traversal_direction_masks: &[u8x64; DIRECTION_COUNT],
    ) {
        // TODO OPT: consider changing this back to "for _ in 0..24" and measure
        loop {
            let mut incoming_changed = false;

            self.try_traverse_dir::<TRAVERSAL_DIRS, NEG_X>(
                &mut incoming_dir_section_sets,
                &traversal_direction_masks,
                &mut incoming_changed,
            );
            self.try_traverse_dir::<TRAVERSAL_DIRS, NEG_Y>(
                &mut incoming_dir_section_sets,
                &traversal_direction_masks,
                &mut incoming_changed,
            );
            self.try_traverse_dir::<TRAVERSAL_DIRS, NEG_Z>(
                &mut incoming_dir_section_sets,
                &traversal_direction_masks,
                &mut incoming_changed,
            );
            self.try_traverse_dir::<TRAVERSAL_DIRS, POS_X>(
                &mut incoming_dir_section_sets,
                &traversal_direction_masks,
                &mut incoming_changed,
            );
            self.try_traverse_dir::<TRAVERSAL_DIRS, POS_Y>(
                &mut incoming_dir_section_sets,
                &traversal_direction_masks,
                &mut incoming_changed,
            );
            self.try_traverse_dir::<TRAVERSAL_DIRS, POS_Z>(
                &mut incoming_dir_section_sets,
                &traversal_direction_masks,
                &mut incoming_changed,
            );

            if !incoming_changed {
                break;
            }
        }

        // TODO: AND with existing visible nodes when there are more culling stages
        self.visible_sections = incoming_dir_section_sets
            .iter()
            .fold(start_visible_sections, |a, b| a | b);
    }

    pub fn setup_center_tile(&mut self, visible_sections: u8x64) {
        let fake_section_sets = [visible_sections; DIRECTION_COUNT];

        // TODO: should the traversal masks actually be something here?
        self.find_outgoing_connections::<ALL_DIRECTIONS, NEG_X>(
            &fake_section_sets,
            Simd::splat(0xFF),
        );
        self.find_outgoing_connections::<ALL_DIRECTIONS, NEG_Y>(
            &fake_section_sets,
            Simd::splat(0xFF),
        );
        self.find_outgoing_connections::<ALL_DIRECTIONS, NEG_Z>(
            &fake_section_sets,
            Simd::splat(0xFF),
        );
        self.find_outgoing_connections::<ALL_DIRECTIONS, POS_X>(
            &fake_section_sets,
            Simd::splat(0xFF),
        );
        self.find_outgoing_connections::<ALL_DIRECTIONS, POS_Y>(
            &fake_section_sets,
            Simd::splat(0xFF),
        );
        self.find_outgoing_connections::<ALL_DIRECTIONS, POS_Z>(
            &fake_section_sets,
            Simd::splat(0xFF),
        );
    }

    fn try_traverse_dir<const TRAVERSAL_DIRS: u8, const OUTGOING_DIR: u8>(
        &mut self,
        incoming_dir_section_sets: &mut [u8x64; DIRECTION_COUNT],
        traversal_direction_masks: &[u8x64; DIRECTION_COUNT],
        incoming_changed: &mut bool,
    ) {
        if bitset::contains(TRAVERSAL_DIRS, OUTGOING_DIR) {
            let dir_index = to_index(OUTGOING_DIR);
            let opposite_dir_index = to_index(opposite(OUTGOING_DIR));

            self.find_outgoing_connections::<TRAVERSAL_DIRS, OUTGOING_DIR>(
                &incoming_dir_section_sets,
                traversal_direction_masks[dir_index],
            );

            let outgoing_sections = self.outgoing_dir_section_sets[dir_index];
            let shifted = match OUTGOING_DIR {
                NEG_X => shift_neg_x(outgoing_sections),
                NEG_Y => shift_neg_y(outgoing_sections),
                NEG_Z => shift_neg_z(outgoing_sections),
                POS_X => shift_pos_x(outgoing_sections),
                POS_Y => shift_pos_y(outgoing_sections),
                POS_Z => shift_pos_z(outgoing_sections),
                _ => unreachable!(),
            };
            // TODO: does this have to be an OR? I think the answer is yes
            let previous = incoming_dir_section_sets[opposite_dir_index];
            incoming_dir_section_sets[opposite_dir_index] |= shifted;

            *incoming_changed |= incoming_dir_section_sets[opposite_dir_index] != previous;
        }
    }

    fn find_outgoing_connections<const TRAVERSAL_DIRS: u8, const OUTGOING_DIR: u8>(
        &mut self,
        incoming_dir_section_sets: &[u8x64; DIRECTION_COUNT],
        traversal_mask: u8x64,
    ) {
        let opposing_directions =
            bitset::contains(TRAVERSAL_DIRS, OUTGOING_DIR | opposite(OUTGOING_DIR));

        let sections_outgoing = &mut self.outgoing_dir_section_sets[to_index(OUTGOING_DIR)];

        let mut masked_traversal_dirs = opposite(TRAVERSAL_DIRS) & !OUTGOING_DIR;
        while masked_traversal_dirs != 0 {
            let incoming_dir = take_one(&mut masked_traversal_dirs);

            let mut connection_sections =
                self.connection_section_sets[connection_index(OUTGOING_DIR, incoming_dir)];

            if opposing_directions {
                connection_sections &= traversal_mask;
            }

            *sections_outgoing |=
                incoming_dir_section_sets[to_index(incoming_dir)] & connection_sections;
        }
    }
}
