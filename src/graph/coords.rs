use std::ops::Index;

use core_simd::simd::prelude::*;

use super::{direction, i16x3, i32x3, i8x3, u16x3, u8x3, Coords3};
use crate::math::*;

pub struct GraphCoordSpace {
    pub axis_lengths_in_tiles: u8x3,
    axis_lengths_in_tiles_extended: u16x3,
    modulo_magics: u16x3,
    index_axis_scales: u16x3,

    pub world_bottom_section_y: i8,
    pub world_top_section_y: i8,
}

impl GraphCoordSpace {
    /// The lengths provided must be greater than or equal to 2, and less
    /// than or equal to 128. The lengths multiplied together must be
    /// less than or equal to 65536
    pub fn new(
        x_length_tiles: u8,
        y_length_tiles: u8,
        z_length_tiles: u8,
        world_bottom_section_y: i8,
        world_top_section_y: i8,
    ) -> Self {
        let axis_lengths_in_tiles = u8x3::from_xyz(x_length_tiles, y_length_tiles, z_length_tiles);
        Self {
            axis_lengths_in_tiles,
            axis_lengths_in_tiles_extended: axis_lengths_in_tiles.cast(),
            modulo_magics: u16x3::from_xyz(
                Self::compute_magic(x_length_tiles),
                Self::compute_magic(y_length_tiles),
                Self::compute_magic(z_length_tiles),
            ),
            index_axis_scales: u16x3::from_xyz(
                1,
                x_length_tiles as u16 * z_length_tiles as u16,
                x_length_tiles as u16,
            ),
            world_bottom_section_y,
            world_top_section_y,
        }
    }

    fn compute_magic(denom: u8) -> u16 {
        let base = (u16::MAX / (denom as u16)).wrapping_add(1);
        let po2_modifier = if denom & (denom - 1) == 0 { 1 } else { 0 };
        base + po2_modifier
    }

    pub fn pack_index(&self, coords: LocalTileCoords) -> LocalTileIndex {
        let coords_extended = coords.0.cast::<i16>();
        // add -1 if negative
        let coords_shifted = coords_extended - (coords_extended >> 15);
        let low_bits = coords_shifted.cast::<u16>() * self.modulo_magics;
        let high_bits =
            ((low_bits.cast::<u32>() * self.axis_lengths_in_tiles_extended.cast::<u32>()) >> 16)
                .cast::<u16>();
        let wrapped = coords_extended.simd_eq(Simd::splat(-1)).select(
            self.axis_lengths_in_tiles_extended - Simd::splat(1),
            high_bits,
        );
        LocalTileIndex((wrapped * self.index_axis_scales).reduce_sum())
    }

    pub fn section_to_tile_coords(&self, section_coords: i32x3) -> (LocalTileCoords, u8x3) {
        let shifted_coords =
            section_coords - i32x3::from_xyz(0, self.world_bottom_section_y as i32, 0);
        let tile_coords = LocalTileCoords(
            (shifted_coords >> 3)
                .rem_euclid(self.axis_lengths_in_tiles.cast())
                .cast::<i8>(),
        );
        let section_coords_in_tile = shifted_coords.cast::<u8>() & Simd::splat(0b111);
        (tile_coords, section_coords_in_tile)
    }

    pub fn block_to_local_coords(&self, block_coords: i32x3) -> u16x3 {
        let world_bottom_block_y = (self.world_bottom_section_y as i32) << 4;
        let shifted_coords = block_coords - i32x3::from_xyz(0, world_bottom_block_y, 0);
        shifted_coords
            .rem_euclid(self.axis_lengths_in_tiles.cast() << 7)
            .cast::<u16>()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(align(8))] // speeds up packing and stepping slightly
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

    pub fn to_local_block_coords(self) -> i16x3 {
        self.0.cast::<i16>() << 7
    }
}

impl Coords3<i8> for LocalTileCoords {
    fn from_xyz(x: i8, y: i8, z: i8) -> Self {
        Self(Simd::from_xyz(x, y, z))
    }
}

impl Index<usize> for LocalTileCoords {
    type Output = i8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[derive(Clone, Copy, PartialEq, Debug, Hash, Eq)]
pub struct LocalTileIndex(pub u16);

impl LocalTileIndex {
    pub fn to_usize(self) -> usize {
        self.0 as usize
    }
}
