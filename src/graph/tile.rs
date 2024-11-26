use std::mem::transmute;
use std::ptr::addr_of_mut;

use core_simd::simd::prelude::*;

use crate::bitset;

use super::coords::{LocalTileCoords, LocalTileIndex};
use super::{direction, u16x3};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeStorage(pub u8x64);

impl NodeStorage {
    const CHILD_TRAVERSE_THRESHOLD: f32 = 0.8;

    #[cfg(test)]
    pub fn index(x: u8, y: u8, z: u8) -> u16 {
        debug_assert!(x < 8);
        debug_assert!(y < 8);
        debug_assert!(z < 8);

        ((z as u16) << 6) | ((y as u16) << 3) | (x as u16)
    }

    #[cfg(test)]
    pub fn get_bit(&self, index: u16) -> bool {
        let array_idx = index >> 3;
        let bit_idx = index & 0b111;

        let byte = self.0[array_idx as usize];
        ((byte >> bit_idx) & 0b1) != 0
    }

    #[cfg(test)]
    pub fn put_bit(&mut self, index: u16, value: bool) {
        let array_idx = index >> 3;
        let bit_idx = index & 0b111;

        let byte = &mut self.0[array_idx as usize];
        *byte &= !(0b1 << bit_idx);
        *byte |= (value as u8) << bit_idx;
    }

    pub fn downscale_to_octant<const OPAQUE: bool>(&mut self, src: Self, dst_octant: u8) -> bool {
        let src_pairs_mask = simd_swizzle!(
            u8x4::from_array([0b11111100, 0b11110011, 0b11001111, 0b00111111]),
            [
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3, 0, 1, 2, 3,
            ],
        );

        let first_pair_nodes = simd_swizzle!(
            src.0,
            [
                0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 16, 16, 16, 16, 18, 18, 18, 18, 20,
                20, 20, 20, 22, 22, 22, 22, 32, 32, 32, 32, 34, 34, 34, 34, 36, 36, 36, 36, 38, 38,
                38, 38, 48, 48, 48, 48, 50, 50, 50, 50, 52, 52, 52, 52, 54, 54, 54, 54,
            ]
        );

        let second_pair_nodes = simd_swizzle!(
            src.0,
            [
                1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 17, 17, 17, 17, 19, 19, 19, 19, 21,
                21, 21, 21, 23, 23, 23, 23, 33, 33, 33, 33, 35, 35, 35, 35, 37, 37, 37, 37, 39, 39,
                39, 39, 49, 49, 49, 49, 51, 51, 51, 51, 53, 53, 53, 53, 55, 55, 55, 55,
            ]
        );

        let third_pair_nodes = simd_swizzle!(
            src.0,
            [
                8, 8, 8, 8, 10, 10, 10, 10, 12, 12, 12, 12, 14, 14, 14, 14, 24, 24, 24, 24, 26, 26,
                26, 26, 28, 28, 28, 28, 30, 30, 30, 30, 40, 40, 40, 40, 42, 42, 42, 42, 44, 44, 44,
                44, 46, 46, 46, 46, 56, 56, 56, 56, 58, 58, 58, 58, 60, 60, 60, 60, 62, 62, 62, 62,
            ]
        );

        let fourth_pair_nodes = simd_swizzle!(
            src.0,
            [
                9, 9, 9, 9, 11, 11, 11, 11, 13, 13, 13, 13, 15, 15, 15, 15, 25, 25, 25, 25, 27, 27,
                27, 27, 29, 29, 29, 29, 31, 31, 31, 31, 41, 41, 41, 41, 43, 43, 43, 43, 45, 45, 45,
                45, 47, 47, 47, 47, 57, 57, 57, 57, 59, 59, 59, 59, 61, 61, 61, 61, 63, 63, 63, 63,
            ]
        );

        let masked_pairs =
            (first_pair_nodes & second_pair_nodes & third_pair_nodes & fourth_pair_nodes)
                | src_pairs_mask;
        let set_cubes = if OPAQUE {
            // downscaling opaque blocks
            masked_pairs.simd_eq(u8x64::splat(0b11111111))
        } else {
            // downscaling traversal data
            masked_pairs.simd_ne(u8x64::splat(0b00000000))
        }
        .to_bitmask();

        // quadrant X determines whether left or right nibble is the destination
        let dst_x_shift = dst_octant & 0b100;

        let nibbles_1 = set_cubes << dst_x_shift;
        let nibbles_2 = set_cubes >> (dst_x_shift ^ 0b100);

        let shuffled_bytes = simd_swizzle!(
            u8x8::from_array(nibbles_1.to_le_bytes()),
            u8x8::from_array(nibbles_2.to_le_bytes()),
            [
                0, 8, 1, 9, 0, 8, 1, 9, 2, 10, 3, 11, 2, 10, 3, 11, 4, 12, 5, 13, 4, 12, 5, 13, 6,
                14, 7, 15, 6, 14, 7, 15,
            ]
        );

        // if the quadrant Y coordinate is 1, toggle each lane's on or off state, shifting the
        // output by 4
        let y_shift = (dst_octant & 0b010) != 0;
        let dst_bytes_mask = (mask8x32::from_array([
            true, true, true, true, false, false, false, false, true, true, true, true, false,
            false, false, false, true, true, true, true, false, false, false, false, true, true,
            true, true, false, false, false, false,
        ]) ^ Mask::splat(y_shift))
        .to_int()
        .cast::<u8>();

        let dst_nibble_mask: u8 = 0b1111 << dst_x_shift;

        let dst_full_mask = Simd::splat(dst_nibble_mask) & dst_bytes_mask;

        let masked_bytes = shuffled_bytes & dst_full_mask;

        // if the quadrant Z coordinate is 1, shift each index by 32.
        let z_shift = (dst_octant & 0b001) as usize;
        let dst_halves = unsafe {
            (addr_of_mut!(self.0) as *mut [u8x32; 2])
                .as_mut()
                .unwrap_unchecked()
        };
        let dst_half: &mut u8x32 = dst_halves.get_mut(z_shift).unwrap();

        // erase bits first before ORing them
        *dst_half &= !dst_full_mask;
        *dst_half |= masked_bytes;

        if OPAQUE {
            let original_population = src.population();
            let downscaled_population = set_cubes.count_ones() as u8;

            (original_population as f32 / downscaled_population as f32)
                < (Self::CHILD_TRAVERSE_THRESHOLD / 8.0)
        } else {
            true
        }
    }

    pub fn upscale(self, src_octant: u8) -> Self {
        // if the quadrant Z coordinate is 1, shift each index by 32.
        // we can replicate this by choosing either the upper or lower half of the node data.
        let z_shift = (src_octant & 0b001) as usize;
        let src_half: u8x32 = unsafe {
            let halves: [u8x32; 2] = transmute(self.0);
            halves[z_shift]
        };

        // if the quadrant Y coordinate is 1, shift each index by 4
        let y_shift = if (src_octant & 0b010) != 0 { 4 } else { 0 };
        let indices = u8x16::from_array([0, 0, 1, 1, 2, 2, 3, 3, 8, 8, 9, 9, 10, 10, 11, 11])
            + Simd::splat(y_shift);

        // By dividing the swizzles into 128-bit chunks, we are guaranteeing that the indices do
        // not cross 128-bit lane boundaries.
        // On x86, this helps us, as VPSHUFB doesn't actually shuffle between 128-bit lanes.
        // These swizzles, when unoptimized, would usually produce separate PSHUFB
        // instructions, but we're hoping that the optimizer can pick up what we've
        // done and combine them.
        let shuffled: u8x32 = unsafe {
            let batches: [u8x16; 2] = transmute(src_half);
            let processed = batches.map(|vec| vec.swizzle_dyn(indices));
            transmute(processed)
        };

        // quadrant X determines whether left or right nibble is chosen
        let x_shift = src_octant & 0b100;
        // the shift is done on a vector of u32s because it produces the same output
        // after the bit AND, while producing much better codegen.
        let shifted: u8x32 = unsafe {
            let reinterpreted: u32x8 = transmute(shuffled);
            let processed = reinterpreted >> Simd::splat(x_shift as u32);
            transmute(processed)
        };
        let nibbles = shifted & Simd::splat(0b1111);

        // repeat each bit in nibble twice using a lookup table
        const EXPAND_LUT: u8x16 = Simd::from_array([
            0b00000000, 0b00000011, 0b00001100, 0b00001111, 0b00110000, 0b00110011, 0b00111100,
            0b00111111, 0b11000000, 0b11000011, 0b11001100, 0b11001111, 0b11110000, 0b11110011,
            0b11111100, 0b11111111,
        ]);

        // See previous comment to see why we split it like this
        let expanded_bits: u8x32 = unsafe {
            let batches: [u8x16; 2] = transmute(nibbles);
            let processed = batches.map(|vec| EXPAND_LUT.swizzle_dyn(vec));
            transmute(processed)
        };

        let expanded_lanes = simd_swizzle!(
            expanded_bits,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 24, 25, 26, 27, 28, 29, 30, 31,
            ]
        );

        Self(expanded_lanes)
    }

    // LLVM seems to auto-vectorize this function with good results on x86
    fn population(self) -> u16 {
        // we transmute the struct directly because of its strict alignment requirements
        let u64_vec: u64x8 = unsafe { transmute(self) };

        let mut count = 0_u16;
        for element in u64_vec.to_array() {
            count += element.count_ones() as u16;
        }
        count
    }

    pub fn edge_neg_to_pos_x(data: u8x64) -> u8x64 {
        data << Simd::splat(7)
    }

    pub fn edge_pos_to_neg_x(data: u8x64) -> u8x64 {
        data >> Simd::splat(7)
    }

    #[rustfmt::skip]
    pub fn edge_neg_to_pos_y(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
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
    pub fn edge_pos_to_neg_y(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
            Simd::splat(0),
            [
                7, 64, 64, 64, 64, 64, 64, 64,
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
    pub fn edge_neg_to_pos_z(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
            Simd::splat(0),
            [
                64, 65, 66, 67, 68, 69, 70, 71,
                72, 73, 74, 75, 76, 77, 78, 79,
                80, 81, 82, 83, 84, 85, 86, 87,
                88, 89, 90, 91, 92, 93, 94, 95,
                96, 97, 98, 99, 100, 101, 102, 103,
                104, 105, 106, 107, 108, 109, 110, 111,
                112, 113, 114, 115, 116, 117, 118, 119,
                0, 1, 2, 3, 4, 5, 6, 7,
            ]
        )
    }

    #[rustfmt::skip]
    pub fn edge_pos_to_neg_z(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
            Simd::splat(0),
            [
                56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 66, 67, 68, 69, 70, 71,
                72, 73, 74, 75, 76, 77, 78, 79,
                80, 81, 82, 83, 84, 85, 86, 87,
                88, 89, 90, 91, 92, 93, 94, 95,
                96, 97, 98, 99, 100, 101, 102, 103,
                104, 105, 106, 107, 108, 109, 110, 111,
                112, 113, 114, 115, 116, 117, 118, 119,
            ]
        )
    }

    pub fn shift_neg_x(data: u8x64) -> u8x64 {
        data >> Simd::splat(1)
    }

    pub fn shift_pos_x(data: u8x64) -> u8x64 {
        data << Simd::splat(1)
    }

    #[rustfmt::skip]
    pub fn shift_neg_y(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
            Simd::splat(0),
            [
                1, 2, 3, 4, 5, 6, 7, 64,
                9, 10, 11, 12, 13, 14, 15, 64,
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
    pub fn shift_pos_y(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
            Simd::splat(0),
            [
                64, 0, 1, 2, 3, 4, 5, 6,
                64, 8, 9, 10, 11, 12, 13, 14,
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
    pub fn shift_neg_z(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
            Simd::splat(0),
            [
                8, 9, 10, 11, 12, 13, 14, 15,
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
    pub fn shift_pos_z(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
            Simd::splat(0),
            [
                64, 65, 66, 67, 68, 69, 70, 71,
                0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55,
            ]
        )
    }
}

// level 0 is the highest resolution level, and each level up has half
// the resolution on each axis
#[derive(Debug)]
pub struct Tile {
    pub traversed_nodes: NodeStorage,
    pub opaque_nodes: NodeStorage,
    // There are 8 possible child nodes, each bit represents a child.
    // Indices are formatted as XYZ.
    // Not necessary on level 0, possibly remove this?
    pub children_to_traverse: u8,
    pub traversal_status: TraversalStatus,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TraversalStatus {
    Uninitialized,
    Traversed { children_upmipped: u8 },
    // is this one really necessary? it could probably be merged with Traversed and renamed
    Upmipped { children_upmipped: u8 },

    // for this to be worth using, the skipped status needs to be propagated down immediately (i
    // think?). when a tile is marked as skipped, the contents of the first byte will either
    // be all 0s or all 1s.
    //
    // Skipped,
    Downmipped,
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            traversed_nodes: NodeStorage(Simd::splat(0)),
            // visibility fully blocked by default
            opaque_nodes: NodeStorage(Simd::splat(0xFF)),
            children_to_traverse: 0,
            traversal_status: TraversalStatus::Uninitialized,
        }
    }
}

impl Tile {
    pub fn traverse<const DIRECTION_SET: u8>(
        &mut self,
        combined_edge_data: u8x64,
        direction_masks: &[u8x64; 6],
    ) -> bool {
        // FAST PATH: if we start with all 0s, we'll end with all 0s
        // if combined_edge_data == Simd::splat(0) {
        //     // TODO: is it necessary to set this?
        //     self.traversed_nodes = NodeStorage(Simd::splat(0));

        //     self.traversal_status = TraversalStatus::Skipped {
        //         children_downmipped: 0,
        //     };

        //     return false;
        // }

        self.traversal_status = TraversalStatus::Traversed {
            children_upmipped: 0,
        };

        let opaque_nodes = self.opaque_nodes.0;

        // if DIRECTION_SET.count_ones() == 3 {
        // TODO OPT: fast path for air in octants: if opaque is all 0s, and the corner
        // bit in the edge data is 1, the whole thing will be 1s
        // }

        let mut neg_x_mask = opaque_nodes;
        let mut neg_y_mask = opaque_nodes;
        let mut neg_z_mask = opaque_nodes;
        let mut pos_x_mask = opaque_nodes;
        let mut pos_y_mask = opaque_nodes;
        let mut pos_z_mask = opaque_nodes;

        // we need to add direction-specific masks when there are pairs of opposing directions
        if bitset::contains(DIRECTION_SET, direction::NEG_X | direction::POS_X) {
            neg_x_mask |= direction_masks[direction::NEG_X as usize];
            pos_x_mask |= direction_masks[direction::POS_X as usize];
        }
        if bitset::contains(DIRECTION_SET, direction::NEG_Y | direction::POS_Y) {
            neg_y_mask |= direction_masks[direction::NEG_Y as usize];
            pos_y_mask |= direction_masks[direction::POS_Y as usize];
        }
        if bitset::contains(DIRECTION_SET, direction::NEG_Z | direction::POS_Y) {
            neg_z_mask |= direction_masks[direction::NEG_Z as usize];
            pos_z_mask |= direction_masks[direction::POS_Z as usize];
        }

        let mut traversed_nodes = combined_edge_data & opaque_nodes;

        // maximum of 24 steps to complete the bfs (is this really faster than a normal loop?)
        for _ in 0..24 {
            let previous_traversed_nodes = traversed_nodes;

            if bitset::contains(DIRECTION_SET, direction::NEG_X) {
                traversed_nodes |= NodeStorage::shift_neg_x(traversed_nodes) & neg_x_mask;
            }
            if bitset::contains(DIRECTION_SET, direction::NEG_Y) {
                traversed_nodes |= NodeStorage::shift_neg_y(traversed_nodes) & neg_y_mask;
            }
            if bitset::contains(DIRECTION_SET, direction::NEG_Z) {
                traversed_nodes |= NodeStorage::shift_neg_z(traversed_nodes) & neg_z_mask;
            }
            if bitset::contains(DIRECTION_SET, direction::POS_X) {
                traversed_nodes |= NodeStorage::shift_pos_x(traversed_nodes) & pos_x_mask;
            }
            if bitset::contains(DIRECTION_SET, direction::POS_Y) {
                traversed_nodes |= NodeStorage::shift_pos_y(traversed_nodes) & pos_y_mask;
            }
            if bitset::contains(DIRECTION_SET, direction::POS_Z) {
                traversed_nodes |= NodeStorage::shift_pos_z(traversed_nodes) & pos_z_mask;
            }

            if traversed_nodes == previous_traversed_nodes {
                break;
            }
        }

        self.traversed_nodes = NodeStorage(traversed_nodes);

        // if there is any edge data, the tile is visible
        // (this whole method could probably be skipped for tiles that won't be relied upon)
        // TODO: make sure this is sound with up and downscaling
        return true;
    }

    pub fn sorted_child_iter(
        &self,
        index: LocalTileIndex,
        coords: LocalTileCoords,
        camera_block_coords: u16x3,
        level: u8,
    ) -> SortedChildIterator {
        SortedChildIterator::new(
            index,
            coords,
            camera_block_coords,
            self.children_to_traverse,
            level,
        )
    }
}

pub struct SortedChildIterator {
    parent_index_high_bits: u32,
    children_index_low_bits: u32,
    parent_coords_high_bits: u16x3,
    // this would fit in a u8x3, but keeping the elements as u16 makes some things faster
    children_coords_low_bits: u16x3,
    children_present: u8,
}

impl SortedChildIterator {
    pub fn new(
        index: LocalTileIndex,
        coords: LocalTileCoords,
        camera_block_coords: u16x3,
        children_present: u8,
        level: u8,
    ) -> Self {
        let parent_index_high_bits = index.to_child_level().0;
        let parent_coords_high_bits = coords.to_child_level().0;

        let middle_coords = coords.to_block_coords(level)
            + Simd::splat(LocalTileCoords::block_length(level) as u16 / 2);
        let first_child_coords = camera_block_coords.simd_ge(middle_coords);
        // this will create an index of a child with the bit order XYZ
        let first_child_index = first_child_coords.to_bitmask() as u32;

        // broadcast first child to 8 "lanes".
        // we only use the bottom 3 bits of each lane, but the lane width is 4 bits to allow for
        // faster indexing.
        let mut children_index_low_bits =
            first_child_index * 0b0001_0001_0001_0001_0001_0001_0001_0001;
        // toggle different bits on specific axis to replicate addition or subtraction
        // in the first lane, the value is the original.
        // in the next 3 lanes, the value is moved on 1 axis.
        // in the following 3 lanes, the value is moved on 2 axes.
        // in the final lane, the value is moved on 3 axes.
        // this stays sorted by manhattan distance, because each move on an axis counts as 1 extra
        // distance
        children_index_low_bits ^= 0b0111_0101_0110_0011_0100_0010_0001_0000;

        // same concept as previous, but vertical instead of horizontal
        let children_coords_low_bits = first_child_coords.to_int().cast::<u16>()
            ^ Simd::from_array([
                // top bits need to be set to 0
                0b11111111_11101000,
                0b11111111_10110100,
                0b11111111_11010010,
            ]);

        Self {
            parent_index_high_bits,
            children_index_low_bits,
            parent_coords_high_bits,
            children_coords_low_bits,
            children_present,
        }
    }
}

impl Iterator for SortedChildIterator {
    type Item = (LocalTileIndex, LocalTileCoords);

    fn next(&mut self) -> Option<Self::Item> {
        // Description of the iteration approach on daniel lemire's blog
        // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
        if self.children_present != 0 {
            let child_number = self.children_present.trailing_zeros();

            let child_index_low_bits = (self.children_index_low_bits >> (child_number * 4)) & 0b111;
            let child_index = LocalTileIndex(self.parent_index_high_bits | child_index_low_bits);

            let child_coords_low_bits = (self.children_coords_low_bits
                >> Simd::splat(child_number as u16))
                & Simd::splat(0b1);
            let child_coords =
                LocalTileCoords(self.parent_coords_high_bits | child_coords_low_bits);

            self.children_present &= self.children_present - 1;

            Some((child_index, child_coords))
        } else {
            None
        }
    }
}
