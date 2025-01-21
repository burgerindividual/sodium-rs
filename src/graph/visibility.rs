use crate::graph::direction::*;

// Minecraft's Direction enum uses the following order:
// -Y, +Y, -Z, +Z, -X, +X
//
// This clashes with our direction format. because of this, it makes sense
// to just provide all 15 possible direction combinations as constants,
// calculated with the following:
// smaller_dir_ordinal * 6 + larger_dir_ordinal
pub const BIT_IDX_NEG_Y_NEG_X: u8 = 4;
pub const BIT_IDX_NEG_Z_NEG_X: u8 = 16;
pub const BIT_IDX_NEG_Z_NEG_Y: u8 = 2;
pub const BIT_IDX_POS_X_NEG_X: u8 = 29;
pub const BIT_IDX_POS_X_NEG_Y: u8 = 5;
pub const BIT_IDX_POS_X_NEG_Z: u8 = 17;
pub const BIT_IDX_POS_Y_NEG_X: u8 = 10;
pub const BIT_IDX_POS_Y_NEG_Y: u8 = 1;
pub const BIT_IDX_POS_Y_NEG_Z: u8 = 8;
pub const BIT_IDX_POS_Y_POS_X: u8 = 11;
pub const BIT_IDX_POS_Z_NEG_X: u8 = 22;
pub const BIT_IDX_POS_Z_NEG_Y: u8 = 3;
pub const BIT_IDX_POS_Z_NEG_Z: u8 = 15;
pub const BIT_IDX_POS_Z_POS_X: u8 = 23;
pub const BIT_IDX_POS_Z_POS_Y: u8 = 9;

pub const UNIQUE_CONNECTION_COUNT: usize = 15;

// returns the index in the upper right triangle in which a > b.
// undefined for a == b.
pub const fn connection_index(dir_1: u8, dir_2: u8) -> usize {
    debug_assert!(dir_1 != dir_2);

    let dir_1_idx = to_index(dir_1);
    let dir_2_idx = to_index(dir_2);

    let (small_idx, large_idx) = if dir_1 > dir_2 {
        (dir_1_idx, dir_2_idx)
    } else {
        (dir_2_idx, dir_1_idx)
    };

    (((5 - small_idx) * 5) + large_idx) - (0b1100 >> small_idx)
}

pub const NEG_X_MASK_CONNECTION_INDICES: [usize; 5] = [0, 1, 3, 6, 10];
pub const NEG_Y_MASK_CONNECTION_INDICES: [usize; 5] = [0, 2, 4, 7, 11];
pub const NEG_Z_MASK_CONNECTION_INDICES: [usize; 5] = [1, 2, 5, 8, 12];
pub const POS_X_MASK_CONNECTION_INDICES: [usize; 5] = [3, 4, 5, 9, 13];
pub const POS_Y_MASK_CONNECTION_INDICES: [usize; 5] = [6, 7, 8, 9, 14];
pub const POS_Z_MASK_CONNECTION_INDICES: [usize; 5] = [10, 11, 12, 13, 14];
