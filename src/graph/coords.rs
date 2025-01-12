use std::hint::assert_unchecked;

use core_simd::simd::prelude::*;

use super::{direction, i16x3, i32x3, u16x3, u8x3, Coords3, Graph};

pub struct GraphCoordSpace {
    morton_swizzle_pattern: u8x32,
    morton_bitmasks: u8x32,

    block_bitmask: u16x3,
    section_bitmask: i16x3,

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
        let bit_counts = u8x3::from_xyz(x_bits, y_bits, z_bits).cast::<i16>();

        let mut coord_space = Self {
            // setting the top bit to 1 results in 0 being placed in a dynamic shuffle
            morton_swizzle_pattern: Simd::splat(0b10000000),
            morton_bitmasks: u8x32::splat(0),
            // the amount of bits are specified in terms of graph level 0.
            // we need to shift them to the right by 1 extra, because we're dealing with sections,
            // which are represented as graph level 1.
            block_bitmask: (u16x3::splat(0b1) << (bit_counts.cast::<u16>() + u16x3::splat(3)))
                - u16x3::splat(1),
            section_bitmask: (i16x3::splat(0b1) << (bit_counts - i16x3::splat(1)))
                - i16x3::splat(1),
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
                // choose the first or second u8 of the u16 to sample
                coord_space.morton_swizzle_pattern[idx as usize] =
                    if cur_z_bit < 8 { 4 } else { 5 };
                coord_space.morton_bitmasks[idx as usize] = 1 << (cur_z_bit & 0b111);
                idx += 1;
                cur_z_bit += 1;
                z_bits -= 1;
            }

            if y_bits != 0 {
                coord_space.morton_swizzle_pattern[idx as usize] =
                    if cur_y_bit < 8 { 2 } else { 3 };
                coord_space.morton_bitmasks[idx as usize] = 1 << (cur_y_bit & 0b111);
                idx += 1;
                cur_y_bit += 1;
                y_bits -= 1;
            }

            if x_bits != 0 {
                coord_space.morton_swizzle_pattern[idx as usize] =
                    if cur_x_bit < 8 { 0 } else { 1 };
                coord_space.morton_bitmasks[idx as usize] = 1 << (cur_x_bit & 0b111);
                idx += 1;
                cur_x_bit += 1;
                x_bits -= 1;
            }
        }

        coord_space
    }

    pub fn pack_index(&self, coords: LocalTileCoords, level: u8) -> LocalTileIndex {
        unsafe {
            assert_unchecked(level <= Graph::HIGHEST_LEVEL);
        }
        let shifted_coords = coords.0 << Simd::splat(level as i16);

        #[cfg(target_feature = "avx2")]
        let packed_morton_bits = unsafe {
            use std::arch::x86_64::*;
            use std::mem::MaybeUninit;

            // the compiler is known to shit itself when coming across the line below. it's
            // very fragile. currently, this produces optimal codegen.
            #[allow(invalid_value)] // yeah, we know
            let broadcasted_coords = simd_swizzle!(
                shifted_coords,
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
            !_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                expanded_morton_bits,
                _mm256_setzero_si256(),
            )) as u32
        };

        #[cfg(not(target_feature = "avx2"))]
        let packed_morton_bits = {
            use std::mem::transmute;

            let broadcasted_coords =
                simd_swizzle!(shifted_coords, Simd::splat(0), [0, 1, 2, 3, 3, 3, 3, 3,]);

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
            !(expanded_morton_bits.simd_eq(Simd::splat(0)).to_bitmask() as u32)
        };

        LocalTileIndex(packed_morton_bits >> level * 3)
    }

    pub fn section_to_local_coords(&self, section_coords: i32x3) -> LocalTileCoords {
        let shifted_coords =
            section_coords - i32x3::from_xyz(0, self.world_bottom_section_y as i32, 0);
        LocalTileCoords(shifted_coords.cast::<i16>() & self.section_bitmask)
    }

    pub fn block_to_local_coords(&self, block_coords: i32x3) -> u16x3 {
        let world_bottom_block_y = (self.world_bottom_section_y as i32) << 4;
        let shifted_coords = block_coords - i32x3::from_xyz(0, world_bottom_block_y, 0);
        shifted_coords.cast::<u16>() & self.block_bitmask
    }

    pub fn step(&self, coords: LocalTileCoords, direction: u8) -> LocalTileCoords {
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

        // sign-extend the offset vec and add it to the coords.
        let sum = coords.0 + offset_vec.cast::<i16>();
        LocalTileCoords(sum)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(align(8))]
pub struct LocalTileCoords(pub i16x3);

impl LocalTileCoords {
    pub fn to_parent_level(self) -> Self {
        Self(self.0 >> Simd::splat(1))
    }

    pub fn to_child_level(self) -> Self {
        Self(self.0 << Simd::splat(1))
    }

    pub fn to_block_coords(self, level: u8) -> i16x3 {
        unsafe {
            assert_unchecked(level <= Graph::HIGHEST_LEVEL);
        }
        self.0 << Simd::splat(level as i16 + 3)
    }

    pub fn block_length(level: u8) -> u8 {
        8 << level
    }
}

impl Coords3<i16> for LocalTileCoords {
    fn from_xyz(x: i16, y: i16, z: i16) -> Self {
        Self(Simd::from_xyz(x, y, z))
    }

    fn into_tuple(self) -> (i16, i16, i16) {
        self.0.into_tuple()
    }

    fn x(&self) -> i16 {
        self.0.x()
    }

    fn y(&self) -> i16 {
        self.0.y()
    }

    fn z(&self) -> i16 {
        self.0.z()
    }
}

#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
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

    pub fn child_octant(self) -> u8 {
        self.0 as u8 & 0b111
    }

    pub fn unordered_child_iter(self) -> CountingChildIter {
        CountingChildIter::new(self)
    }
}

pub struct CountingChildIter {
    parent_index_high_bits: u32,
    current_child_octant: u8,
}

impl CountingChildIter {
    pub fn new(index: LocalTileIndex) -> Self {
        Self {
            parent_index_high_bits: index.to_child_level().0,
            current_child_octant: 0,
        }
    }
}

impl Iterator for CountingChildIter {
    type Item = LocalTileIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_child_octant < 8 {
            let child_index =
                LocalTileIndex(self.parent_index_high_bits | self.current_child_octant as u32);
            self.current_child_octant += 1;
            Some(child_index)
        } else {
            None
        }
    }
}

pub struct PosSortedChildIterator {
    iter_count: u8,

    sorted_child_octants: u32,
    // this would fit in a u8x3, but keeping the elements as u16 makes some things faster
    sorted_child_octant_coords: u16x3,

    parent_index_high_bits: u32,
    parent_coords_high_bits: i16x3,
}

impl PosSortedChildIterator {
    pub fn new(
        index: LocalTileIndex,
        coords: LocalTileCoords,
        level: u8,
        camera_block_coords: i16x3,
    ) -> Self {
        let parent_index_high_bits = index.to_child_level().0;
        let parent_coords_high_bits = coords.to_child_level().0;

        let middle_coords = coords.to_block_coords(level)
            + Simd::splat(LocalTileCoords::block_length(level) as i16 / 2);
        let first_child_coords = camera_block_coords.simd_ge(middle_coords);
        // this will create the octant index of a child with the bit order XYZ. the call
        // to reverse is necessary to get the correct bit order.
        let first_child_octant = first_child_coords.reverse().to_bitmask() as u32;

        // broadcast first child to 8 "lanes" of 3 buts.
        let mut sorted_child_octants = first_child_octant * 0b001_001_001_001_001_001_001_001;
        // toggle different bits on specific axis to replicate addition or subtraction
        // in the first lane, the value is the original.
        // in the next 3 lanes, the value is moved on 1 axis.
        // in the following 3 lanes, the value is moved on 2 axes.
        // in the final lane, the value is moved on 3 axes.
        // this stays sorted by manhattan distance, because each move on an axis counts
        // as 1 extra distance
        sorted_child_octants ^= 0b111_101_110_011_100_010_001_000;

        // same concept as previous, but vertical instead of horizontal
        let sorted_child_octant_coords = first_child_coords.to_int().cast::<u16>()
            ^ Simd::from_array([
                // top bits need to be set to 0
                0b11111111_11101000,
                0b11111111_10110100,
                0b11111111_11010010,
            ]);

        Self {
            iter_count: 0,
            sorted_child_octants,
            sorted_child_octant_coords,
            parent_index_high_bits,
            parent_coords_high_bits,
        }
    }
}

impl Iterator for PosSortedChildIterator {
    // We return both the indices and coordinates because it's easy to calculate
    // both at once
    type Item = (LocalTileIndex, LocalTileCoords);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iter_count < 8 {
            let child_octant = self.sorted_child_octants & 0b111;
            let child_octant_coords = self.sorted_child_octant_coords & Simd::splat(0b1);

            let child_index = LocalTileIndex(self.parent_index_high_bits | child_octant);
            let child_coords =
                LocalTileCoords(self.parent_coords_high_bits | child_octant_coords.cast::<i16>());

            // pop child octant from list of octants by shifting it out of bounds
            self.sorted_child_octants >>= 3;
            self.sorted_child_octant_coords >>= Simd::splat(1);
            self.iter_count += 1;
            Some((child_index, child_coords))
        } else {
            None
        }
    }
}
