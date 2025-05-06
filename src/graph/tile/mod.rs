pub mod angle;
pub mod fog;
pub mod frustum;
pub mod height;
pub mod traversal;

use core_simd::simd::prelude::*;
use core_simd::simd::ToBytes;
use std_float::StdFloat;

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

#[derive(Debug)]
pub struct Tile {
    // Only changes on section update
    pub connection_section_sets: [u8x64; UNIQUE_CONNECTION_COUNT],
    // Changes every time tile is processed
    pub outgoing_dir_section_sets: [u8x64; DIRECTION_COUNT],
    // visible_sections can be added back here to do visibility tests. for now, this is not
    // necessary
    #[cfg(debug_assertions)]
    pub processed: bool,
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            // fully untraversable by default
            connection_section_sets: [SECTIONS_EMPTY; UNIQUE_CONNECTION_COUNT],
            outgoing_dir_section_sets: [SECTIONS_EMPTY; DIRECTION_COUNT],
            #[cfg(debug_assertions)]
            processed: false,
        }
    }
}

impl Tile {
    pub fn set_empty(&mut self) {
        self.outgoing_dir_section_sets = [SECTIONS_EMPTY; DIRECTION_COUNT];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outward_direction_mask_test() {
        for camera_x in 0..8 {
            for camera_y in 0..8 {
                for camera_z in 0..8 {
                    let camera_tile_coords = u8x3::from_xyz(camera_x, camera_y, camera_z);

                    let mut sane_camera_direction_masks = [SECTIONS_EMPTY; DIRECTION_COUNT];

                    for tile_x in 0..8 {
                        for tile_y in 0..8 {
                            for tile_z in 0..8 {
                                let other_tile_coords = Simd::from_xyz(tile_x, tile_y, tile_z);

                                let negative = other_tile_coords.simd_le(camera_tile_coords);
                                let positive = other_tile_coords.simd_ge(camera_tile_coords);
                                let traversal_directions = negative.to_bitmask() as u8
                                    | ((positive.to_bitmask() as u8) << 3);

                                let section_index = section_index(other_tile_coords);
                                for dir_idx in 0..6 {
                                    modify_bit(
                                        &mut sane_camera_direction_masks[dir_idx as usize],
                                        section_index,
                                        traversal_directions.get_bit(dir_idx),
                                    );
                                }
                            }
                        }
                    }

                    let test_camera_direction_masks =
                        gen_outward_direction_masks(camera_tile_coords);

                    let mut directions = ALL_DIRECTIONS;
                    while directions != 0 {
                        let direction = take_one(&mut directions);
                        let dir_idx = to_index(direction);
                        assert_eq!(
                            sane_camera_direction_masks[dir_idx],
                            test_camera_direction_masks[dir_idx],
                            "sane != test, Camera Coords: {:?}, Direction: {}",
                            camera_tile_coords,
                            to_str(direction)
                        );
                    }
                }
            }
        }
    }
}
