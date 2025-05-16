use super::*;

pub fn test_coords(
    coord_space: &GraphCoordSpace,
    coords: LocalTileCoords,
    results: &mut CombinedTestResults,
) {
    let tile_y = coords[Y];
    let world_max_y = (coord_space.axis_lengths_in_tiles[Y] - 1) as i8;

    // out of bounds
    if tile_y > world_max_y {
        // early exit
        // TODO: should this ever happen?
        *results = CombinedTestResults::OUTSIDE;
        return;
    }

    // if height checks are on, we know that tiles at the maximum Y coord will be
    // partially outside of the world.
    results.set_partial::<{ CombinedTestResults::HEIGHT_BIT }>(tile_y == world_max_y);
}

pub fn gen_top_tile_visibility_mask(section_height_in_top_tile: u16) -> u8x64 {
    let height_mask_small = (1_u8 << section_height_in_top_tile) - 1;
    mask64x8::from_bitmask(height_mask_small as u64)
        .to_int()
        .to_le_bytes()
}

// TODO: implement
pub fn gen_out_of_bounds_above_incoming_sections(section_height_in_top_tile: u16) -> u8x64 {
    todo!();
    let height_mask_small = (1_u8 << section_height_in_top_tile) - 1;
    mask64x8::from_bitmask(height_mask_small as u64)
        .to_int()
        .to_le_bytes()
}
