// Directions and Direction Sets are represented as raw u8s to work seamlessly
// with the current state of const generics.

use std::num::NonZero;

pub const NEG_X: u8 = 0b000001;
pub const NEG_Y: u8 = 0b000010;
pub const NEG_Z: u8 = 0b000100;
pub const POS_X: u8 = 0b001000;
pub const POS_Y: u8 = 0b010000;
pub const POS_Z: u8 = 0b100000;

pub const ALL_DIRECTIONS: u8 = 0b111111;

pub const fn to_index(direction: u8) -> u8 {
    unsafe { NonZero::new_unchecked(direction) }.trailing_zeros() as u8
}

pub const fn inverse_set(direction_set: u8) -> u8 {
    !direction_set & 0b111111
}

/// Removes a direction from the direction set, and returns it
pub const fn take_dir(direction_set: &mut u8) -> u8 {
    let prev_set = *direction_set;
    *direction_set &= *direction_set - 1;
    let direction = *direction_set ^ prev_set;
    direction
}
