use std::mem::MaybeUninit;

use core_simd::simd::prelude::*;

use super::{direction, i16x3, i32x3, i8x3, u16x3, u8x3, Coords3};

pub struct GraphCoordSpace {
    morton_swizzle_pattern: u8x16,
    morton_bitmasks: u8x16,

    tile_bitmask: i8x3,
    block_bitmask: u16x3,

    pub world_bottom_section_y: i8,
    pub world_top_section_y: i8,
}

impl GraphCoordSpace {
    pub fn new(
        mut x_bits: u8,
        mut y_bits: u8,
        mut z_bits: u8,
        world_bottom_section_y: i8,
        world_top_section_y: i8,
    ) -> Self {
        // NOTE: extra bits need to be present in the LocalTileCoords, but not in the
        // LocalTileIndex, as long as we're not doing any unpacking.
        let bit_counts = u8x3::from_xyz(x_bits, y_bits, z_bits);

        let mut coord_space = Self {
            // setting the top bit to 1 results in 0 being placed in a dynamic shuffle
            morton_swizzle_pattern: Simd::splat(0b10000000),
            morton_bitmasks: Simd::splat(0),
            block_bitmask: (Simd::splat(0b1) << (bit_counts.cast::<u16>() + Simd::splat(7)))
                - Simd::splat(1),
            tile_bitmask: (Simd::splat(0b1) << bit_counts.cast::<i8>()) - Simd::splat(1),
            world_bottom_section_y,
            world_top_section_y,
        };

        let mut idx: usize = 0;
        let mut cur_x_bit: u8 = 0;
        let mut cur_y_bit: u8 = 0;
        let mut cur_z_bit: u8 = 0;

        // bits roughly in 0b...XYZXYZ order

        while z_bits != 0 || y_bits != 0 || x_bits != 0 {
            assert!(idx < 16, "Total index bits exceeds 16");

            if z_bits != 0 {
                // choose the first or second u8 of the u16 to sample
                coord_space.morton_swizzle_pattern[idx] = if cur_z_bit < 8 { 4 } else { 5 };
                coord_space.morton_bitmasks[idx] = 1 << (cur_z_bit & 0b111);
                idx += 1;
                cur_z_bit += 1;
                z_bits -= 1;
            }

            if y_bits != 0 {
                coord_space.morton_swizzle_pattern[idx] = if cur_y_bit < 8 { 2 } else { 3 };
                coord_space.morton_bitmasks[idx] = 1 << (cur_y_bit & 0b111);
                idx += 1;
                cur_y_bit += 1;
                y_bits -= 1;
            }

            if x_bits != 0 {
                coord_space.morton_swizzle_pattern[idx] = if cur_x_bit < 8 { 0 } else { 1 };
                coord_space.morton_bitmasks[idx] = 1 << (cur_x_bit & 0b111);
                idx += 1;
                cur_x_bit += 1;
                x_bits -= 1;
            }
        }

        coord_space
    }

    pub fn pack_index(&self, coords: LocalTileCoords) -> LocalTileIndex {
        // this produces the best codegen, and should always be safe due to how we
        // populate morton_swizzle_pattern
        #[allow(invalid_value)] // yeah, we know
        let broadcasted_coords: i8x16 = coords
            .0
            .resize(unsafe { MaybeUninit::uninit().assume_init() });

        #[cfg(target_feature = "ssse3")]
        let packed_morton_bits = unsafe {
            use std::arch::x86_64::*;
            // allocate one byte per bit for each element. each element is still has its
            // individual bits in linear ordering, but the bytes in the vector are in morton
            // ordering.
            let expanded_bytes = _mm_shuffle_epi8(
                broadcasted_coords.into(),
                self.morton_swizzle_pattern.into(),
            );

            // isolate each bit necessary for morton ordering
            let expanded_morton_bits = _mm_and_si128(expanded_bytes, self.morton_bitmasks.into());

            // check if masked bit is set (!= 0) or unset (== 0) for each lane, then pack
            // each lane into one bit.
            !_mm_movemask_epi8(_mm_cmpeq_epi8(expanded_morton_bits, _mm_setzero_si128())) as u16
        };

        #[cfg(not(target_feature = "ssse3"))]
        let packed_morton_bits = {
            // allocate one byte per bit for each element. each element is still has its
            // individual bits in linear ordering, but the bytes in the vector
            // are in morton ordering.
            let expanded_bytes = broadcasted_coords
                .cast::<u8>()
                .swizzle_dyn(self.morton_swizzle_pattern);

            // isolate each bit necessary for morton ordering
            let expanded_morton_bits = expanded_bytes & self.morton_bitmasks;

            // check if masked bit is set (!= 0) or unset (== 0) for each lane, then pack
            // each lane into one bit.
            expanded_morton_bits.simd_ne(Simd::splat(0)).to_bitmask() as u16
        };

        LocalTileIndex(packed_morton_bits)
    }

    pub fn section_to_tile_coords(&self, section_coords: i32x3) -> LocalTileCoords {
        let shifted_coords =
            section_coords - i32x3::from_xyz(0, self.world_bottom_section_y as i32, 0);
        LocalTileCoords((shifted_coords >> Simd::splat(3)).cast::<i8>() & self.tile_bitmask)
    }

    pub fn block_to_local_coords(&self, block_coords: i32x3) -> u16x3 {
        let world_bottom_block_y = (self.world_bottom_section_y as i32) << 4;
        let shifted_coords = block_coords - i32x3::from_xyz(0, world_bottom_block_y, 0);
        shifted_coords.cast::<u16>() & self.block_bitmask
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(align(8))]
pub struct LocalTileCoords(pub i8x3);

impl LocalTileCoords {
    pub const LENGTH_IN_BLOCKS: u8 = 128;
    pub const LENGTH_IN_SECTIONS: u8 = 8;

    pub fn step(self, direction: u8) -> Self {
        // position a 1-byte mask within a 6-byte SWAR vector, with each of the 6 bytes
        // representing a direction
        let dir_index = direction::to_index(direction);
        let shifted_byte = 0xFF_u64 << (dir_index * 8);

        // positive directions (indices 3, 4, and 5) need to be shifted into the lower
        // half. this lets us convert it to a 3-byte vector.
        // the mask is used to turn each present value in the mask into a positive 1.
        let pos_selected = (shifted_byte >> 24) as u32 & 0x01_01_01;

        // negative directions (indices 0, 1, and 2) are already in the bottom half, so
        // we mask out the top half. the mask here is also used to turn each present
        // value in the mask into a negative 1, or 0xFF in hex.
        let neg_selected = shifted_byte as u32 & 0xFF_FF_FF;

        // because we only allow 1 direction to be passed to this function, we know that
        // one of the two vectors will be empty. we can combine the positive and
        // negative vectors to get a vector that we know contains our increment value.
        let collapsed_selected = pos_selected | neg_selected;

        // each byte in the SWAR register is actually meant to represent an i8, so we
        // turn the bytes into a vector and cast it as such.
        let offset_vec = Simd::from_array(collapsed_selected.to_le_bytes())
            .resize(0)
            .cast::<i8>();

        Self(self.0 + offset_vec)
    }

    // TODO: should we just inline this? we've pretty much inlined all the other
    // conversions
    pub fn to_local_block_coords(self) -> i16x3 {
        self.0.cast::<i16>() << Simd::splat(7)
    }
}

impl Coords3<i8> for LocalTileCoords {
    fn from_xyz(x: i8, y: i8, z: i8) -> Self {
        Self(Simd::from_xyz(x, y, z))
    }

    fn into_tuple(self) -> (i8, i8, i8) {
        self.0.into_tuple()
    }

    fn x(&self) -> i8 {
        self.0.x()
    }

    fn y(&self) -> i8 {
        self.0.y()
    }

    fn z(&self) -> i8 {
        self.0.z()
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LocalTileIndex(pub u16);

impl LocalTileIndex {
    pub fn to_usize(self) -> usize {
        self.0 as usize
    }
}
