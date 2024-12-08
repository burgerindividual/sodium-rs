use std::hint::assert_unchecked;
use std::mem::MaybeUninit;

use core_simd::simd::prelude::*;

use super::{direction, i32x3, u16x3, u8x3, Coords3, Graph};

pub struct GraphCoordSpace {
    morton_swizzle_pattern: u8x32,
    morton_bitmasks: u8x32,

    section_bitmask: u16x3,
    block_bitmask: u16x3,
    level_0_tile_bitmask: u16x3,

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
        let bit_counts = u8x3::from_xyz(x_bits, y_bits, z_bits).cast::<u16>();

        let mut indexer = Self {
            // setting the top bit to 1 results in 0 being placed in a dynamic shuffle
            morton_swizzle_pattern: u8x32::splat(0b10000000),
            morton_bitmasks: u8x32::splat(0),
            // the amount of bits are specified in terms of graph level 0.
            // we need to shift them to the right by 1 extra, because we're dealing with sections,
            // which are represented as graph level 1.
            section_bitmask: (u16x3::splat(0b1) << (bit_counts - u16x3::splat(1)))
                - u16x3::splat(1),
            block_bitmask: (u16x3::splat(0b1) << (bit_counts + u16x3::splat(3))) - u16x3::splat(1),
            level_0_tile_bitmask: (u16x3::splat(0b1) << bit_counts) - u16x3::splat(1),
            world_bottom_section_y,
            world_top_section_y,
        };

        let mut idx: u32 = 0;
        let mut cur_x_bit: u8 = 0;
        let mut cur_y_bit: u8 = 0;
        let mut cur_z_bit: u8 = 0;

        // bits roughly in 0b...XYZXYZ order

        while z_bits != 0 || y_bits != 0 || x_bits != 0 {
            assert!(idx < 32, "Total index bits exceeds 32");

            if z_bits != 0 {
                indexer.morton_swizzle_pattern[idx as usize] = if cur_z_bit < 8 { 4 } else { 5 };
                indexer.morton_bitmasks[idx as usize] = 1 << (cur_z_bit & 0b111);
                idx += 1;
                cur_z_bit += 1;
                z_bits -= 1;
            }

            if y_bits != 0 {
                indexer.morton_swizzle_pattern[idx as usize] = if cur_y_bit < 8 { 2 } else { 3 };
                indexer.morton_bitmasks[idx as usize] = 1 << (cur_y_bit & 0b111);
                idx += 1;
                cur_y_bit += 1;
                y_bits -= 1;
            }

            if x_bits != 0 {
                indexer.morton_swizzle_pattern[idx as usize] = if cur_x_bit < 8 { 0 } else { 1 };
                indexer.morton_bitmasks[idx as usize] = 1 << (cur_x_bit & 0b111);
                idx += 1;
                cur_x_bit += 1;
                x_bits -= 1;
            }
        }

        indexer
    }

    pub fn pack_index(&self, coords: LocalTileCoords) -> LocalTileIndex {
        #[cfg(target_feature = "avx2")]
        unsafe {
            use std::arch::x86_64::*;

            // the compiler is known to shit itself when coming across the line below. it's
            // very fragile. currently, this produces optimal codegen.
            #[allow(invalid_value)] // yeah, we know
            let broadcasted_coords = simd_swizzle!(
                coords.0,
                MaybeUninit::uninit().assume_init(),
                [0, 1, 2, 3, 3, 3, 3, 3, 0, 1, 2, 3, 3, 3, 3, 3,]
            )
            .into();

            // allocate one byte per bit for each element. each element is still has its
            // individual bits in linear ordering, but the bytes in the vector
            // are in morton ordering.
            let expanded_bytes =
                _mm256_shuffle_epi8(broadcasted_coords, self.morton_swizzle_pattern.into());

            // isolate each bit necessary for morton ordering
            let expanded_morton_bits =
                _mm256_and_si256(expanded_bytes, self.morton_bitmasks.into());

            // check if masked bit is set (!= 0) or unset (== 0) for each lane, then pack
            // each lane into one bit.
            let packed_morton_bits = !_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                expanded_morton_bits,
                _mm256_setzero_si256(),
            )) as u32;

            LocalTileIndex(packed_morton_bits)
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            let broadcasted_coords =
                simd_swizzle!(coords.0, Simd::splat(0), [0, 1, 2, 3, 3, 3, 3, 3,]);

            // allocate one byte per bit for each element. each element is still has its
            // individual bits in linear ordering, but the bytes in the vector
            // are in morton ordering.
            let expanded_bytes: u8x32 = unsafe {
                let broadcasted_coords_bytes: u8x16 = transmute(broadcasted_coords);
                let index_batches: [u8x16; 2] =
                    transmute(self.morton_swizzle_pattern & Simd::splat(0b111));
                transmute([
                    broadcasted_coords_bytes.swizzle_dyn(index_batches[0]),
                    broadcasted_coords_bytes.swizzle_dyn(index_batches[1]),
                ])
            };

            // isolate each bit necessary for morton ordering
            let expanded_morton_bits = expanded_bytes & self.morton_bitmasks;

            // check if masked bit is set (!= 0) or unset (== 0) for each lane, then pack
            // each lane into one bit.
            // simd_eq and a NOT on the bitmask is used here, because on x86, it's faster
            // than simd_neq
            let packed_morton_bits =
                !(expanded_morton_bits.simd_eq(Simd::splat(0)).to_bitmask() as u32);

            LocalTileIndex(packed_morton_bits)
        }
    }

    pub fn section_to_local_coords(&self, section_coords: i32x3) -> LocalTileCoords {
        let shifted_coords =
            section_coords - i32x3::from_xyz(0, self.world_bottom_section_y as i32, 0);
        LocalTileCoords(shifted_coords.cast::<u16>() & self.section_bitmask)
    }

    pub fn block_to_local_coords(&self, block_coords: i32x3) -> u16x3 {
        let world_bottom_block_y = (self.world_bottom_section_y as i32) << 4;
        let shifted_coords = block_coords - i32x3::from_xyz(0, world_bottom_block_y, 0);
        shifted_coords.cast::<u16>() & self.block_bitmask
    }

    pub fn coords_bitmask(&self, level: u8) -> u16x3 {
        unsafe { assert_unchecked(level <= Graph::HIGHEST_LEVEL) }
        self.level_0_tile_bitmask >> Simd::splat(level as u16)
    }

    pub fn step_wrapping(
        &self,
        coords: LocalTileCoords,
        direction: u8,
        level: u8,
    ) -> LocalTileCoords {
        let dir_index = direction::to_index(direction);
        let shifted_byte = 0xFF_u64 << (dir_index * 8);
        let pos_selected = (shifted_byte >> 24) as u32 & 0x01_01_01;
        let neg_selected = shifted_byte as u32 & 0xFF_FF_FF;
        let collapsed_selected = pos_selected | neg_selected;
        let offset_vec = Simd::from_array(collapsed_selected.to_le_bytes()).resize(0);
        let sum = (coords.0.cast::<i16>() + offset_vec.cast::<i16>()).cast::<u16>();
        LocalTileCoords(sum & self.coords_bitmask(level))
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(align(8))]
pub struct LocalTileCoords(pub u16x3);

impl LocalTileCoords {
    pub fn to_parent_level(self) -> Self {
        Self(self.0 >> Simd::splat(1))
    }

    pub fn to_child_level(self) -> Self {
        Self(self.0 << Simd::splat(1))
    }

    pub fn to_block_coords(self, level: u8) -> u16x3 {
        self.0 << Simd::splat(level as u16 + 3)
    }

    pub fn block_length(level: u8) -> u8 {
        8 << level
    }

    // FIXME
    // pub fn size() -> u32 {
    //     1 << (LEVEL * 3)
    // }
}

impl Coords3<u16> for LocalTileCoords {
    fn from_xyz(x: u16, y: u16, z: u16) -> Self {
        Self(Simd::from_xyz(x, y, z))
    }

    fn into_tuple(self) -> (u16, u16, u16) {
        self.0.into_tuple()
    }

    fn x(&self) -> u16 {
        self.0.x()
    }

    fn y(&self) -> u16 {
        self.0.y()
    }

    fn z(&self) -> u16 {
        self.0.z()
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LocalTileIndex(pub u32);

impl LocalTileIndex {
    pub fn to_parent_level(self) -> LocalTileIndex {
        Self(self.0 >> 3)
    }

    pub fn to_child_level(self) -> LocalTileIndex {
        Self(self.0 << 3)
    }

    pub fn to_usize(self) -> usize {
        self.0 as usize
    }

    pub fn child_number(self) -> u8 {
        self.0 as u8 & 0b111
    }

    pub fn unordered_child_iter(self, children_present: u8) -> UnorderedChildIter {
        UnorderedChildIter::new(self, children_present)
    }
}

pub struct UnorderedChildIter {
    parent_index_high_bits: u32,
    children_present: u8,
}

impl UnorderedChildIter {
    pub fn new(index: LocalTileIndex, children_present: u8) -> Self {
        Self {
            parent_index_high_bits: index.to_child_level().0,
            children_present,
        }
    }
}

impl Iterator for UnorderedChildIter {
    type Item = (LocalTileIndex, u8);

    fn next(&mut self) -> Option<Self::Item> {
        // Description of the iteration approach on daniel lemire's blog
        // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
        if self.children_present != 0 {
            let child_number = self.children_present.trailing_zeros();
            self.children_present &= self.children_present - 1;
            let child_index = LocalTileIndex(self.parent_index_high_bits | child_number);
            Some((child_index, child_number as u8))
        } else {
            None
        }
    }
}
