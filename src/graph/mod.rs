use context::{CombinedTestResults, GraphSearchContext};
use coords::{GraphCoordSpace, LocalTileIndex};
use core_simd::simd::prelude::*;
use direction::*;
use tile::{Tile, SECTIONS_EMPTY, SECTIONS_FILLED};
use visibility::*;

use self::coords::LocalTileCoords;
use crate::bitset::{self, BitSet};
use crate::ffi::FFIVisibleSectionsTile;
use crate::math::*;

pub mod context;
pub mod coords;
pub mod direction;
pub mod tile;
pub mod visibility;

macro_rules! iterate_dirs {
    ($graph:ident, $context:ident, $($dir:expr),+) => {{
        const DIRS_SLICE: &[u8] = &[$($dir),+];
        const INCOMING_DIRS: u8 = opposite(bitset::from_u8_slice(DIRS_SLICE));
        const TRAVERSAL_DIRS: u8 = all_except(INCOMING_DIRS);

        $graph.iterate_dirs(
            $context,
            $context.camera_tile_coords,
            DIRS_SLICE,
            Self::process_tile::<INCOMING_DIRS, TRAVERSAL_DIRS>,
        );
    }};
}

pub struct Graph {
    tiles: Box<[Tile]>,

    pub coord_space: GraphCoordSpace,
    do_height_checks: bool,
    // TODO: add world height masks here
    pub visible_tiles: Vec<FFIVisibleSectionsTile>,
}

impl Graph {
    pub fn new(render_distance: u8, world_bottom_section_y: i8, world_top_section_y: i8) -> Self {
        // Same as Minecraft's ClientChunkCache.calculateStorageRange
        let storage_distance = render_distance.max(2) + 3;
        let y_length_sections =
            (world_top_section_y as i16 - world_bottom_section_y as i16 + 1) as u16;
        let xz_length_sections = (storage_distance as u16 * 2) + 1;

        assert!(y_length_sections > 0 && xz_length_sections > 0,
            "Invalid graph size. RD: {render_distance}, Bottom Section: {world_bottom_section_y}, Top Section: {world_top_section_y}"
        );

        // the minimum size of the graph is 2x2x2 tiles, so we can guarantee that each
        // tile will only be processed once. if any axis were allowed to have a
        // size of 1, when the graph search wraps past the edge of the graph, we
        // would land on the same tile that was just processed.
        let y_length_tiles = (y_length_sections.next_multiple_of(8) >> 3).max(2);
        let xz_length_tiles = (xz_length_sections.next_multiple_of(8) >> 3).max(2);

        let graph_total_tiles = y_length_tiles as usize * (xz_length_tiles as usize).pow(2);

        // Make sure graph bounds can be represented with i8 coordinates, and u16
        // indices.
        // TODO: should max axis length be smaller to prevent wrapping on step?
        const MAX_AXIS_LENGTH: u16 = i8::MAX as u16 + 1;
        const MAX_TOTAL_TILES: usize = u16::MAX as usize + 1;
        assert!(
            y_length_tiles <= MAX_AXIS_LENGTH && xz_length_tiles <= MAX_AXIS_LENGTH && graph_total_tiles <= MAX_TOTAL_TILES,
            "Graph size is too large. Y Length (tiles): {y_length_tiles}, XZ Length (tiles): {xz_length_tiles}"
        );

        let tiles = unsafe {
            let mut tiles_uninit = Box::<[Tile]>::new_uninit_slice(graph_total_tiles);

            for tile_uninit in tiles_uninit.iter_mut() {
                tile_uninit.write(Default::default());
            }

            tiles_uninit.assume_init()
        };

        let do_height_checks = y_length_sections % LocalTileCoords::LENGTH_IN_SECTIONS as u16 != 0;

        Self {
            tiles,
            coord_space: GraphCoordSpace::new(
                xz_length_tiles as u8,
                y_length_tiles as u8,
                xz_length_tiles as u8,
                world_bottom_section_y,
                world_top_section_y,
            ),
            do_height_checks,
            visible_tiles: Vec::with_capacity(128), // probably not a bad start
        }
    }

    pub fn cull(&mut self, context: &GraphSearchContext) {
        self.clear();

        self.iterate_tiles(context);
    }

    pub fn clear(&mut self) {
        self.visible_tiles.clear();

        #[cfg(debug_assertions)]
        for tile in &mut self.tiles {
            tile.processed = false;
        }
    }

    fn iterate_tiles(&mut self, context: &GraphSearchContext) {
        // Center
        self.process_tile::<0, ALL_DIRECTIONS>(
            context,
            self.coord_space.pack_index(context.camera_tile_coords),
            context.camera_tile_coords,
        );

        // Axes
        iterate_dirs!(self, context, POS_X);
        iterate_dirs!(self, context, NEG_X);
        iterate_dirs!(self, context, POS_Z);
        iterate_dirs!(self, context, NEG_Z);
        iterate_dirs!(self, context, POS_Y);
        iterate_dirs!(self, context, NEG_Y);

        // Planes
        iterate_dirs!(self, context, POS_X, POS_Y);
        iterate_dirs!(self, context, NEG_X, POS_Y);
        iterate_dirs!(self, context, POS_X, NEG_Y);
        iterate_dirs!(self, context, NEG_X, NEG_Y);

        iterate_dirs!(self, context, POS_X, POS_Z);
        iterate_dirs!(self, context, NEG_X, POS_Z);
        iterate_dirs!(self, context, POS_X, NEG_Z);
        iterate_dirs!(self, context, NEG_X, NEG_Z);

        iterate_dirs!(self, context, POS_Z, POS_Y);
        iterate_dirs!(self, context, POS_Z, NEG_Y);
        iterate_dirs!(self, context, NEG_Z, POS_Y);
        iterate_dirs!(self, context, NEG_Z, NEG_Y);

        // Octants
        iterate_dirs!(self, context, POS_X, POS_Z, POS_Y);
        iterate_dirs!(self, context, NEG_X, POS_Z, POS_Y);
        iterate_dirs!(self, context, NEG_X, POS_Z, NEG_Y);
        iterate_dirs!(self, context, POS_X, POS_Z, NEG_Y);
        iterate_dirs!(self, context, POS_X, NEG_Z, POS_Y);
        iterate_dirs!(self, context, NEG_X, NEG_Z, POS_Y);
        iterate_dirs!(self, context, NEG_X, NEG_Z, NEG_Y);
        iterate_dirs!(self, context, POS_X, NEG_Z, NEG_Y);
    }

    /// dirs must not be empty when calling this
    fn iterate_dirs(
        &mut self,
        context: &GraphSearchContext,
        start_coords: LocalTileCoords,
        dirs: &[u8],
        process_tile_fn: fn(&mut Self, &GraphSearchContext, LocalTileIndex, LocalTileCoords),
    ) {
        let last_direction = dirs.len() == 1;
        let direction = dirs[0];
        let steps = context.direction_step_counts[to_index(direction)];
        let mut coords = start_coords;

        for _ in 0..steps {
            coords = coords.step(direction);

            // if the direction set is empty, we should stop recursing, and start processing
            // tiles
            if last_direction {
                let index = self.coord_space.pack_index(coords);

                process_tile_fn(self, context, index, coords);
            } else {
                self.iterate_dirs(context, coords, &dirs[1..], process_tile_fn);
            }
        }
    }

    fn process_tile<const INCOMING_DIRS: u8, const TRAVERSAL_DIRS: u8>(
        &mut self,
        context: &GraphSearchContext,
        index: LocalTileIndex,
        coords: LocalTileCoords,
    ) {
        #[cfg(debug_assertions)]
        println!("Current Tile - Coords: {:?} Index: {:?}", coords.0, index.0);

        // try to quickly determine whether we need to actually traverse the tile using
        // the frustum, fog, etc
        let relative_tile_pos = context.relative_tile_pos(coords);
        let test_result = context.test_tile(
            &self.coord_space,
            coords,
            relative_tile_pos,
            self.do_height_checks,
        );

        // tile needs to be re-borrowed multiple times in this method, due to borrow
        // checker rules. these should get optimized out.
        let tile = self.get_tile_mut(index);

        #[cfg(debug_assertions)]
        {
            assert!(!tile.processed);
            tile.processed = true;
        }

        if test_result == CombinedTestResults::OUTSIDE {
            // early exit
            tile.set_empty();
            return;
        }

        tile.visible_sections = SECTIONS_FILLED;

        let intersecting_planes = test_result.get_intersecting_planes();
        if intersecting_planes != 0 {
            context.frustum.voxelize_planes(
                intersecting_planes,
                relative_tile_pos,
                &mut tile.visible_sections,
            );
        }

        if test_result.is_partial::<{ CombinedTestResults::FOG_BIT }>() {
            context.voxelize_fog_cylinder(relative_tile_pos, &mut tile.visible_sections);
        }

        // if test_result.is_partial::<{ CombinedTestResults::HEIGHT_BIT }>() {
        //     todo!();
        // }

        if context.use_occlusion_culling {
            let visibility_mask = tile.visible_sections;
            let mut traverse_start_sections = SECTIONS_EMPTY;
            let mut incoming_dir_section_sets = [SECTIONS_EMPTY; DIRECTION_COUNT];
            tile.outgoing_dir_section_sets = [SECTIONS_EMPTY; DIRECTION_COUNT];

            // the center tile has no incoming directions, so there will be no data from
            // neighboring tiles. instead, we have to place the first set section manually.
            if INCOMING_DIRS == 0 {
                let tile = self.get_tile_mut(index);
                let section_index = tile::section_index(context.camera_section_in_tile);

                tile::set_bit(&mut traverse_start_sections, section_index);
                tile.setup_center_tile(section_index);
            } else {
                self.get_incoming_edges::<INCOMING_DIRS>(
                    coords,
                    visibility_mask,
                    &mut traverse_start_sections,
                    &mut incoming_dir_section_sets,
                );

                // FAST PATH: if we start the traversal with all 0s, we'll end with all 0s.
                if traverse_start_sections == SECTIONS_EMPTY {
                    // early exit
                    let tile = self.get_tile_mut(index);
                    tile.set_empty();
                    return;
                }
            }

            // if we've hit this point, we know that there's atleast 1 section that has been
            // traversed in this tile. because of this, we know atleast part of
            // it is visible.

            let tile = self.get_tile_mut(index);

            let angle_visibility_masks = tile::gen_angle_visibility_masks(relative_tile_pos);

            tile.traverse::<TRAVERSAL_DIRS>(
                traverse_start_sections,
                incoming_dir_section_sets,
                &context.outward_direction_masks,
                &angle_visibility_masks,
            );

            for sections in tile.outgoing_dir_section_sets {
                debug_assert_eq!(sections & visibility_mask, sections, "after traversal");
            }
        }

        let tile = self.get_tile(index);

        if tile.visible_sections != SECTIONS_EMPTY {
            let local_section_coords = coords.0.cast::<i32>() << 3;
            let global_section_coords = context.global_section_offset + local_section_coords;

            let visible_sections_ptr = &raw const tile.visible_sections;

            self.visible_tiles.push(FFIVisibleSectionsTile::new(
                global_section_coords,
                visible_sections_ptr,
            ));
        }
    }

    fn get_incoming_edges<const INCOMING_DIRS: u8>(
        &self,
        coords: LocalTileCoords,
        visibility_mask: u8x64,
        traverse_start_sections: &mut u8x64,
        incoming_dir_section_sets: &mut [u8x64; DIRECTION_COUNT],
    ) {
        if bitset::contains_u8(INCOMING_DIRS, NEG_X) {
            let incoming_edge = self.get_incoming_edge::<NEG_X>(coords) & visibility_mask;
            *traverse_start_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(NEG_X)] = incoming_edge;
        }

        if bitset::contains_u8(INCOMING_DIRS, NEG_Y) {
            let incoming_edge = self.get_incoming_edge::<NEG_Y>(coords) & visibility_mask;
            *traverse_start_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(NEG_Y)] = incoming_edge;
        }

        if bitset::contains_u8(INCOMING_DIRS, NEG_Z) {
            let incoming_edge = self.get_incoming_edge::<NEG_Z>(coords) & visibility_mask;
            *traverse_start_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(NEG_Z)] = incoming_edge;
        }

        if bitset::contains_u8(INCOMING_DIRS, POS_X) {
            let incoming_edge = self.get_incoming_edge::<POS_X>(coords) & visibility_mask;
            *traverse_start_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(POS_X)] = incoming_edge;
        }

        if bitset::contains_u8(INCOMING_DIRS, POS_Y) {
            let incoming_edge = self.get_incoming_edge::<POS_Y>(coords) & visibility_mask;
            *traverse_start_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(POS_Y)] = incoming_edge;
        }

        if bitset::contains_u8(INCOMING_DIRS, POS_Z) {
            let incoming_edge = self.get_incoming_edge::<POS_Z>(coords) & visibility_mask;
            *traverse_start_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(POS_Z)] = incoming_edge;
        }
    }

    fn get_incoming_edge<const DIRECTION: u8>(&self, coords: LocalTileCoords) -> u8x64 {
        let neighbor_coords = coords.step(DIRECTION);
        let neighbor_index = self.coord_space.pack_index(neighbor_coords);
        let neighbor_tile = self.get_tile(neighbor_index);

        let neighbor_outgoing_sections =
            neighbor_tile.outgoing_dir_section_sets[to_index(opposite(DIRECTION))];

        match DIRECTION {
            NEG_X => tile::edge_pos_to_neg_x(neighbor_outgoing_sections),
            NEG_Y => tile::edge_pos_to_neg_y(neighbor_outgoing_sections),
            NEG_Z => tile::edge_pos_to_neg_z(neighbor_outgoing_sections),
            POS_X => tile::edge_neg_to_pos_x(neighbor_outgoing_sections),
            POS_Y => tile::edge_neg_to_pos_y(neighbor_outgoing_sections),
            POS_Z => tile::edge_neg_to_pos_z(neighbor_outgoing_sections),
            _ => unreachable!(),
        }
    }

    fn get_tile_mut(&mut self, index: LocalTileIndex) -> &mut Tile {
        unsafe { self.tiles.get_unchecked_mut(index.to_usize()) }
    }

    fn get_tile(&self, index: LocalTileIndex) -> &Tile {
        unsafe { self.tiles.get_unchecked(index.to_usize()) }
    }

    pub fn set_section(&mut self, section_coords: i32x3, visibility_data: u64) {
        let (tile_coords, section_coords_in_tile) =
            self.coord_space.section_to_tile_coords(section_coords);
        let tile_index = self.coord_space.pack_index(tile_coords);
        let section_index = tile::section_index(section_coords_in_tile);

        #[cfg(debug_assertions)]
        println!(
            "Set Section - Section Coords: {:?}, Tile Coords: {:?}, Tile Index: {:?}, Section Index: {:?}, Vis: {}",
            section_coords, tile_coords.0, tile_index.0, section_index, visibility_data
        );

        let tile = self.get_tile_mut(tile_index);

        for (array_idx, &bit_idx) in ARRAY_TO_BIT_IDX.iter().enumerate() {
            tile::modify_bit(
                &mut tile.connection_section_sets[array_idx],
                section_index,
                visibility_data.get_bit(bit_idx),
            );
        }
    }
}
