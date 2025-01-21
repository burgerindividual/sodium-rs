use core_simd::simd::prelude::*;
use core_simd::simd::ToBytes;

use super::visibility::*;
use super::{connection_index, u8x3, *};
use crate::bitset;
use crate::bitset::BitSet;
use crate::math::Coords3;

pub const SECTIONS_EMPTY: u8x64 = Simd::splat(0);
pub const SECTIONS_FILLED: u8x64 = Simd::splat(0xFF);

pub fn section_index(coords: u8x3) -> u16 {
    debug_assert!(coords.simd_lt(Simd::splat(8)).all());

    ((coords.z() as u16) << 6) | ((coords.y() as u16) << 3) | (coords.x() as u16)
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

// TODO: verify these are correct
pub fn create_camera_direction_masks(camera_section_in_tile: u8x3) -> [u8x64; DIRECTION_COUNT] {
    let neg_x_lane = (0b10_u8 << camera_section_in_tile.x()).wrapping_sub(1);
    let neg_x_mask = Simd::splat(neg_x_lane);

    let pos_x_lane = 0xFF << camera_section_in_tile.x();
    let pos_x_mask = Simd::splat(pos_x_lane);

    // native endianness should be correct here, but it's worth double checking
    let neg_y_bitmask = (0b10 << camera_section_in_tile.y()) - 1;
    let neg_y_lane = u64::from_ne_bytes(
        mask8x8::from_bitmask(neg_y_bitmask)
            .to_int()
            .to_ne_bytes()
            .to_array(),
    );
    let neg_y_mask = u64x8::splat(neg_y_lane).to_ne_bytes();

    let pos_y_bitmask = 0xFF << camera_section_in_tile.y();
    let pos_y_lane = u64::from_ne_bytes(
        mask8x8::from_bitmask(pos_y_bitmask)
            .to_int()
            .to_ne_bytes()
            .to_array(),
    );
    let pos_y_mask = u64x8::splat(pos_y_lane).to_ne_bytes();

    let neg_z_bitmask = (0b10 << camera_section_in_tile.z()) - 1;
    let neg_z_mask = mask64x8::from_bitmask(neg_z_bitmask).to_int().to_ne_bytes();

    let pos_z_bitmask = 0xFF << camera_section_in_tile.z();
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
    pub connection_section_sets: [u8x64; UNIQUE_CONNECTION_COUNT],

    pub outgoing_dir_section_sets: [u8x64; DIRECTION_COUNT],
    pub visible_sections: u8x64,
    // the last timestamp where either traversal_status, traversed_nodes, or visible_nodes changed
    pub last_change_timestamp: u64,
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            // fully untraversable by default
            connection_section_sets: [SECTIONS_EMPTY; UNIQUE_CONNECTION_COUNT],
            outgoing_dir_section_sets: [SECTIONS_EMPTY; DIRECTION_COUNT],
            // TODO: should this start out as all 1s?
            visible_sections: SECTIONS_EMPTY,
            last_change_timestamp: 0,
        }
    }
}

impl Tile {
    pub fn set_empty(&mut self) {
        self.outgoing_dir_section_sets = [SECTIONS_EMPTY; DIRECTION_COUNT];
        self.visible_sections = SECTIONS_EMPTY;
    }

    pub fn clear_if_outdated(&mut self, current_timestamp: u64) {
        if self.last_change_timestamp != current_timestamp {
            self.last_change_timestamp = current_timestamp;
            self.set_empty();
        }
    }

    // TODO: review all fast paths
    // TODO: use existing visible nodes as masks when earlier stages exist
    // TODO: is it necessary to use tile_incoming_directions for the first
    // iteration?
    pub fn find_visible_sections<const TRAVERSAL_DIRS: u8>(
        &mut self,
        mut incoming_dir_section_sets: [u8x64; DIRECTION_COUNT],
        traversal_direction_masks: &[u8x64; DIRECTION_COUNT],
    ) {
        let traverse_neg_x = bitset::contains(TRAVERSAL_DIRS, NEG_X);
        let traverse_pos_x = bitset::contains(TRAVERSAL_DIRS, NEG_Y);
        let traverse_neg_y = bitset::contains(TRAVERSAL_DIRS, NEG_Z);
        let traverse_pos_y = bitset::contains(TRAVERSAL_DIRS, POS_X);
        let traverse_neg_z = bitset::contains(TRAVERSAL_DIRS, POS_Y);
        let traverse_pos_z = bitset::contains(TRAVERSAL_DIRS, POS_Z);

        let connection_section_sets =
            self.mask_connection_section_sets::<TRAVERSAL_DIRS>(traversal_direction_masks);

        let mut visible_sections = SECTIONS_EMPTY;

        // maximum of 24 steps to complete the bfs (TODO: is this really faster than a
        // normal loop?)
        for _ in 0..24 {
            let mut new_visible_sections = SECTIONS_EMPTY;

            if traverse_neg_x {
                self.update_outgoing_dirs::<TRAVERSAL_DIRS, NEG_X>(
                    &incoming_dir_section_sets,
                    &connection_section_sets,
                );
                let incoming_sections =
                    shift_neg_x(self.outgoing_dir_section_sets[to_index(NEG_X)]);
                incoming_dir_section_sets[to_index(POS_X)] = incoming_sections;
                new_visible_sections |= incoming_sections;
            }
            if traverse_neg_y {
                self.update_outgoing_dirs::<TRAVERSAL_DIRS, NEG_Y>(
                    &incoming_dir_section_sets,
                    &connection_section_sets,
                );
                let incoming_sections =
                    shift_neg_y(self.outgoing_dir_section_sets[to_index(NEG_Y)]);
                incoming_dir_section_sets[to_index(POS_Y)] = incoming_sections;
                new_visible_sections |= incoming_sections;
            }
            if traverse_neg_z {
                self.update_outgoing_dirs::<TRAVERSAL_DIRS, NEG_Z>(
                    &incoming_dir_section_sets,
                    &connection_section_sets,
                );
                let incoming_sections =
                    shift_neg_z(self.outgoing_dir_section_sets[to_index(NEG_Z)]);
                incoming_dir_section_sets[to_index(POS_Z)] = incoming_sections;
                new_visible_sections |= incoming_sections;
            }
            if traverse_pos_x {
                self.update_outgoing_dirs::<TRAVERSAL_DIRS, POS_X>(
                    &incoming_dir_section_sets,
                    &connection_section_sets,
                );
                let incoming_sections =
                    shift_pos_x(self.outgoing_dir_section_sets[to_index(POS_X)]);
                incoming_dir_section_sets[to_index(NEG_X)] = incoming_sections;
                new_visible_sections |= incoming_sections;
            }
            if traverse_pos_y {
                self.update_outgoing_dirs::<TRAVERSAL_DIRS, POS_Y>(
                    &incoming_dir_section_sets,
                    &connection_section_sets,
                );
                let incoming_sections =
                    shift_pos_y(self.outgoing_dir_section_sets[to_index(POS_Y)]);
                incoming_dir_section_sets[to_index(NEG_Y)] = incoming_sections;
                new_visible_sections |= incoming_sections;
            }
            if traverse_pos_z {
                self.update_outgoing_dirs::<TRAVERSAL_DIRS, POS_Z>(
                    &incoming_dir_section_sets,
                    &connection_section_sets,
                );
                let incoming_sections =
                    shift_pos_z(self.outgoing_dir_section_sets[to_index(POS_Z)]);
                incoming_dir_section_sets[to_index(NEG_Z)] = incoming_sections;
                new_visible_sections |= incoming_sections;
            }

            // TODO: are we sure this can go at the end, or do we have to do it directly
            // after the ORs? If we did do it directly after the ORs, then if no incoming
            // data is provided, it'll stop early.
            if visible_sections == new_visible_sections {
                break;
            }

            visible_sections = new_visible_sections;
        }

        // TODO: AND with existing visible nodes when there are more culling stages
        self.visible_sections = visible_sections;
    }

    fn update_outgoing_dirs<const TRAVERSAL_DIRS: u8, const OUTGOING_DIR: u8>(
        &mut self,
        incoming_dir_section_sets: &[u8x64; DIRECTION_COUNT],
        connection_section_sets: &[u8x64; UNIQUE_CONNECTION_COUNT],
    ) {
        let sections_outgoing = &mut self.outgoing_dir_section_sets[to_index(OUTGOING_DIR)];

        let mut masked_traversal_dirs = TRAVERSAL_DIRS & !OUTGOING_DIR;
        while masked_traversal_dirs != 0 {
            let incoming_dir = take_one(&mut masked_traversal_dirs);

            *sections_outgoing |= incoming_dir_section_sets[to_index(incoming_dir)]
                & connection_section_sets[connection_index(OUTGOING_DIR, incoming_dir)];
        }
    }

    // we need to add direction-specific masks when there are pairs of opposing
    // directions
    fn mask_connection_section_sets<const TRAVERSAL_DIRS: u8>(
        &self,
        traversal_direction_masks: &[u8x64; DIRECTION_COUNT],
    ) -> [u8x64; UNIQUE_CONNECTION_COUNT] {
        let use_x_mask = bitset::contains(TRAVERSAL_DIRS, NEG_X | POS_X);
        let use_y_mask = bitset::contains(TRAVERSAL_DIRS, NEG_Y | POS_Y);
        let use_z_mask = bitset::contains(TRAVERSAL_DIRS, NEG_Z | POS_Z);

        let mut masked_connection_section_sets = self.connection_section_sets;

        if use_x_mask {
            for connection_idx in NEG_X_MASK_CONNECTION_INDICES {
                masked_connection_section_sets[connection_idx] &=
                    traversal_direction_masks[to_index(NEG_X)];
            }
            for connection_idx in POS_X_MASK_CONNECTION_INDICES {
                masked_connection_section_sets[connection_idx] &=
                    traversal_direction_masks[to_index(POS_X)];
            }
        }

        if use_y_mask {
            for connection_idx in NEG_Y_MASK_CONNECTION_INDICES {
                masked_connection_section_sets[connection_idx] &=
                    traversal_direction_masks[to_index(NEG_Y)];
            }
            for connection_idx in POS_Y_MASK_CONNECTION_INDICES {
                masked_connection_section_sets[connection_idx] &=
                    traversal_direction_masks[to_index(POS_Y)];
            }
        }

        if use_z_mask {
            for connection_idx in NEG_Z_MASK_CONNECTION_INDICES {
                masked_connection_section_sets[connection_idx] &=
                    traversal_direction_masks[to_index(NEG_Z)];
            }
            for connection_idx in POS_Z_MASK_CONNECTION_INDICES {
                masked_connection_section_sets[connection_idx] &=
                    traversal_direction_masks[to_index(POS_Z)];
            }
        }

        masked_connection_section_sets
    }
}
