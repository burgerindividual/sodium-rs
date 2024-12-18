use std::array;
use std::mem::transmute;
use std::ptr::addr_of_mut;

use core_simd::simd::prelude::*;
use core_simd::simd::ToBytes;

use crate::bitset;
use crate::graph::direction::*;
use crate::math::{u16x3, Coords3};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
// TODO: switch to YZX or YXZ indexing from ZYX to allow faster splitting into
// regions of 8x4x8
pub struct NodeStorage(pub u8x64);

impl NodeStorage {
    pub const EMPTY: Self = Self(Simd::splat(0));
    pub const FILLED: Self = Self(Simd::splat(0xFF));

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

        // if the quadrant Y coordinate is 1, toggle each lane's on or off state,
        // shifting the output by 4
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
        // we can replicate this by choosing either the upper or lower half of the node
        // data.
        let z_shift = (src_octant & 0b001) as usize;
        let src_half: u8x32 = unsafe {
            let halves: [u8x32; 2] = transmute(self.0);
            halves[z_shift]
        };

        // if the quadrant Y coordinate is 1, shift each index by 4
        let y_shift = if (src_octant & 0b010) != 0 { 4 } else { 0 };
        let indices = u8x16::from_array([0, 0, 1, 1, 2, 2, 3, 3, 8, 8, 9, 9, 10, 10, 11, 11])
            + Simd::splat(y_shift);

        // By dividing the swizzles into 128-bit chunks, we are guaranteeing that the
        // indices do not cross 128-bit lane boundaries.
        // On x86, this helps us, as VPSHUFB doesn't actually shuffle between 128-bit
        // lanes. These swizzles, when unoptimized, would usually produce
        // separate PSHUFB instructions, but we're hoping that the optimizer can
        // pick up what we've done and combine them.
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
    pub fn edge_neg_to_pos_z(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
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
    pub fn edge_pos_to_neg_z(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
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

    pub fn shift_neg_x(data: u8x64) -> u8x64 {
        data >> Simd::splat(1)
    }

    pub fn shift_pos_x(data: u8x64) -> u8x64 {
        data << Simd::splat(1)
    }

    #[rustfmt::skip]
    pub fn shift_neg_y(data: u8x64) -> u8x64 {
        // The u8x64 "data" vector represents an 8x8x8 array of bits, with each bit
        // representing a node. It is indexed with the pattern ZZZYYYXXX. Because
        // of our indexing scheme, we know that each u8 lane in the vector represents
        // a row of data on the X axis.
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
        // By shifting each index in that array to the left by 1, this swizzle operation
        // effectively shifts each X-axis row of data by -1 on the Y axis. The "64" indices
        // seen in this swizzle are used to fill the empty space that the shift left over
        // with zeroes.
        simd_swizzle!(
            data,
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
    pub fn shift_pos_y(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
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
    pub fn shift_neg_z(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
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
    pub fn shift_pos_z(data: u8x64) -> u8x64 {
        simd_swizzle!(
            data,
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

    // a mask is necessary for each direction (6 directions) on each level (5
    // levels)
    pub fn create_direction_masks(local_camera_pos_int: u16x3) -> [[u8x64; 6]; 5] {
        array::from_fn(|level| {
            // a tile has bounds of 8x8x8, so we restrict each axis to 0-7
            let pos_in_tile = (local_camera_pos_int >> Simd::splat(level as u16)).cast::<u8>()
                & Simd::splat(0b111);

            let neg_x_lane = (1 << pos_in_tile.x()) - 1;
            let neg_x_mask = Simd::splat(neg_x_lane);

            let pos_x_lane = 0b11111110 << pos_in_tile.x();
            let pos_x_mask = Simd::splat(pos_x_lane);

            // native endianness should be correct here, but it's worth double checking
            let neg_y_bitmask = (1 << pos_in_tile.y()) - 1;
            let neg_y_lane = u64::from_ne_bytes(
                mask8x8::from_bitmask(neg_y_bitmask)
                    .to_int()
                    .to_ne_bytes()
                    .to_array(),
            );
            let neg_y_mask = u64x8::splat(neg_y_lane).to_ne_bytes();

            let pos_y_bitmask = 0b11111110 << pos_in_tile.y();
            let pos_y_lane = u64::from_ne_bytes(
                mask8x8::from_bitmask(pos_y_bitmask)
                    .to_int()
                    .to_ne_bytes()
                    .to_array(),
            );
            let pos_y_mask = u64x8::splat(pos_y_lane).to_ne_bytes();

            let neg_z_bitmask = (1 << pos_in_tile.z()) - 1;
            let neg_z_mask = mask64x8::from_bitmask(neg_z_bitmask).to_int().to_ne_bytes();

            let pos_z_bitmask = 0b11111110 << pos_in_tile.z();
            let pos_z_mask = mask64x8::from_bitmask(pos_z_bitmask).to_int().to_ne_bytes();

            [
                neg_x_mask, neg_y_mask, neg_z_mask, pos_x_mask, pos_y_mask, pos_z_mask,
            ]
        })
    }
}

// level 0 is the highest resolution level, and each level up has half
// the resolution on each axis
#[derive(Debug)]
pub struct Tile {
    pub traversed_nodes: NodeStorage,
    pub visible_nodes: NodeStorage,
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
    Processed { children_upmipped: u8 },

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
            traversed_nodes: NodeStorage::EMPTY,
            visible_nodes: NodeStorage::EMPTY,
            // visibility fully blocked by default
            opaque_nodes: NodeStorage::FILLED,
            children_to_traverse: 0,
            traversal_status: TraversalStatus::Uninitialized,
        }
    }
}

impl Tile {
    pub fn set_empty_traversal(&mut self) {
        self.traversed_nodes = NodeStorage::EMPTY;
        self.visible_nodes = NodeStorage::EMPTY;
    }

    // TODO: make sure this is sound with up and downscaling
    // TODO: review all fast paths
    pub fn find_visible_nodes<const TRAVERSAL_DIRECTIONS: u8>(
        &mut self,
        incoming_traversed_nodes: u8x64,
        direction_masks: &[u8x64; 6],
    ) {
        // the uses of these variables should optimize out, as they're evaluated at
        // comptime
        let do_shift_neg_x = const { bitset::contains(TRAVERSAL_DIRECTIONS, NEG_X) };
        let do_shift_pos_x = const { bitset::contains(TRAVERSAL_DIRECTIONS, NEG_Y) };
        let do_shift_neg_y = const { bitset::contains(TRAVERSAL_DIRECTIONS, NEG_Z) };
        let do_shift_pos_y = const { bitset::contains(TRAVERSAL_DIRECTIONS, POS_X) };
        let do_shift_neg_z = const { bitset::contains(TRAVERSAL_DIRECTIONS, POS_Y) };
        let do_shift_pos_z = const { bitset::contains(TRAVERSAL_DIRECTIONS, POS_Z) };

        let use_x_mask = const { bitset::contains(TRAVERSAL_DIRECTIONS, NEG_X | POS_X) };
        let use_y_mask = const { bitset::contains(TRAVERSAL_DIRECTIONS, NEG_Y | POS_Y) };
        let use_z_mask = const { bitset::contains(TRAVERSAL_DIRECTIONS, NEG_Z | POS_Z) };

        // these should be optimized out if they aren't used
        let neg_x_mask = direction_masks[to_index(NEG_X) as usize];
        let pos_x_mask = direction_masks[to_index(POS_X) as usize];
        let neg_y_mask = direction_masks[to_index(NEG_Y) as usize];
        let pos_y_mask = direction_masks[to_index(POS_Y) as usize];
        let neg_z_mask = direction_masks[to_index(NEG_Z) as usize];
        let pos_z_mask = direction_masks[to_index(POS_Z) as usize];

        let opaque_nodes = self.opaque_nodes.0;

        // if TRAVERSAL_DIRECTIONS.count_ones() == 3 {
        // TODO OPT: fast path for air in octants: if opaque is all 0s, and the corner
        // bit in the edge data is 1, the whole thing will be 1s
        // }

        // the traversal masks always are used, and are combined prior to traversal with
        // the opaque nodes mask.
        let mut neg_x_combined_mask = opaque_nodes;
        let mut neg_y_combined_mask = opaque_nodes;
        let mut neg_z_combined_mask = opaque_nodes;
        let mut pos_x_combined_mask = opaque_nodes;
        let mut pos_y_combined_mask = opaque_nodes;
        let mut pos_z_combined_mask = opaque_nodes;

        // we need to add direction-specific masks when there are pairs of opposing
        // directions
        if use_x_mask {
            neg_x_combined_mask |= direction_masks[to_index(NEG_X) as usize];
            pos_x_combined_mask |= direction_masks[to_index(POS_X) as usize];
        }
        if use_y_mask {
            neg_y_combined_mask |= direction_masks[to_index(NEG_Y) as usize];
            pos_y_combined_mask |= direction_masks[to_index(POS_Y) as usize];
        }
        if use_z_mask {
            neg_z_combined_mask |= direction_masks[to_index(NEG_Z) as usize];
            pos_z_combined_mask |= direction_masks[to_index(POS_Z) as usize];
        }

        let mut traversed_nodes = incoming_traversed_nodes;

        // maximum of 24 steps to complete the bfs (TODO: is this really faster than a
        // normal loop?)
        for _ in 0..24 {
            let previous_traversed_nodes = traversed_nodes;

            if do_shift_neg_x {
                traversed_nodes |= NodeStorage::shift_neg_x(traversed_nodes) & neg_x_combined_mask;
            }
            if do_shift_neg_y {
                traversed_nodes |= NodeStorage::shift_neg_y(traversed_nodes) & neg_y_combined_mask;
            }
            if do_shift_neg_z {
                traversed_nodes |= NodeStorage::shift_neg_z(traversed_nodes) & neg_z_combined_mask;
            }
            if do_shift_pos_x {
                traversed_nodes |= NodeStorage::shift_pos_x(traversed_nodes) & pos_x_combined_mask;
            }
            if do_shift_pos_y {
                traversed_nodes |= NodeStorage::shift_pos_y(traversed_nodes) & pos_y_combined_mask;
            }
            if do_shift_pos_z {
                traversed_nodes |= NodeStorage::shift_pos_z(traversed_nodes) & pos_z_combined_mask;
            }

            if traversed_nodes == previous_traversed_nodes {
                break;
            }
        }

        self.traversed_nodes = NodeStorage(traversed_nodes);

        // we have to do one more traversal step, individually shifting the traversed
        // nodes each direction without masking the opaque nodes. This gives us the
        // visible, possibly opaque neighbors of the currently traversed nodes.
        let mut visible_nodes = traversed_nodes;

        if do_shift_neg_x {
            let mut shifted = NodeStorage::shift_neg_x(traversed_nodes);

            if use_x_mask {
                shifted &= neg_x_mask;
            }

            visible_nodes |= shifted;
        }
        if do_shift_neg_y {
            let mut shifted = NodeStorage::shift_neg_y(traversed_nodes);

            if use_y_mask {
                shifted &= neg_y_mask;
            }

            visible_nodes |= shifted;
        }
        if do_shift_neg_z {
            let mut shifted = NodeStorage::shift_neg_z(traversed_nodes);

            if use_z_mask {
                shifted &= neg_z_mask;
            }

            visible_nodes |= shifted;
        }
        if do_shift_pos_x {
            let mut shifted = NodeStorage::shift_pos_x(traversed_nodes);

            if use_x_mask {
                shifted &= pos_x_mask;
            }

            visible_nodes |= shifted;
        }
        if do_shift_pos_y {
            let mut shifted = NodeStorage::shift_pos_y(traversed_nodes);

            if use_y_mask {
                shifted &= pos_y_mask;
            }

            visible_nodes |= shifted;
        }
        if do_shift_pos_z {
            let mut shifted = NodeStorage::shift_pos_z(traversed_nodes);

            if use_z_mask {
                shifted &= pos_z_mask;
            }

            visible_nodes |= shifted;
        }

        self.visible_nodes = NodeStorage(visible_nodes);
    }
}
