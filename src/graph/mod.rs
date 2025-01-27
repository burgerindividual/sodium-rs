use context::{CombinedTestResults, GraphSearchContext};
use coords::{GraphCoordSpace, LocalTileIndex};
use core_simd::simd::prelude::*;
use direction::*;
use tile::{Tile, SECTIONS_EMPTY};
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
    ($graph:ident, $context:ident, $iter_dirs:expr) => {{
        const INCOMING_DIRS: u8 = opposite($iter_dirs);
        const TRAVERSAL_DIRS: u8 = all_except(INCOMING_DIRS);

        $graph.iterate_dirs(
            $context,
            $context.camera_tile_coords,
            $iter_dirs,
            Self::process_tile::<INCOMING_DIRS, TRAVERSAL_DIRS>,
        );
    }};
}

pub struct Graph {
    tiles: Box<[Tile]>,

    pub coord_space: GraphCoordSpace,
    do_height_checks: bool,

    pub visible_tiles: Vec<FFIVisibleSectionsTile>,
}

impl Graph {
    pub fn new(render_distance: u8, world_bottom_section_y: i8, world_top_section_y: i8) -> Self {
        let world_y_len_sections =
            (world_top_section_y as i16 - world_bottom_section_y as i16 + 1) as u16;
        let world_xz_len_sections = (render_distance as u16 * 2) + 1;

        assert!(world_y_len_sections > 0 && world_xz_len_sections > 0,
            "Invalid graph size. RD: {render_distance}, Bottom Section: {world_bottom_section_y}, Top Section: {world_top_section_y}"
        );

        // the minimum size of the graph is 2x2x2 so we can guarantee that each tile
        // will only be processed once. if any axis had a length of 1, when the graph
        // search wraps past the edge of the graph, we would land on the same tile that
        // was just processed.
        let graph_y_bits =
            (u16::BITS as u8 - (world_y_len_sections - 1).leading_zeros() as u8).max(5) - 3;
        let graph_xz_bits =
            (u16::BITS as u8 - (world_xz_len_sections - 1).leading_zeros() as u8).max(5) - 3;

        let graph_y_len_tiles = 1_usize << graph_y_bits;
        let graph_xz_len_tiles = 1_usize << graph_xz_bits;

        let tiles = unsafe {
            let mut tiles_uninit =
                Box::<[Tile]>::new_uninit_slice(graph_y_len_tiles * graph_xz_len_tiles.pow(2));

            for tile_uninit in tiles_uninit.iter_mut() {
                tile_uninit.write(Default::default());
            }

            tiles_uninit.assume_init()
        };

        let do_height_checks = world_y_len_sections & 0b111 != 0;

        Self {
            tiles,
            coord_space: GraphCoordSpace::new(
                graph_xz_bits,
                graph_y_bits,
                graph_xz_bits,
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
            tile.set_empty();
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
        iterate_dirs!(self, context, POS_Y);
        iterate_dirs!(self, context, POS_Z);
        iterate_dirs!(self, context, NEG_X);
        iterate_dirs!(self, context, NEG_Y);
        iterate_dirs!(self, context, NEG_Z);

        // Planes
        iterate_dirs!(self, context, POS_X | POS_Y);
        iterate_dirs!(self, context, NEG_X | POS_Y);
        iterate_dirs!(self, context, POS_X | NEG_Y);
        iterate_dirs!(self, context, NEG_X | NEG_Y);

        iterate_dirs!(self, context, POS_X | POS_Z);
        iterate_dirs!(self, context, NEG_X | POS_Z);
        iterate_dirs!(self, context, POS_X | NEG_Z);
        iterate_dirs!(self, context, NEG_X | NEG_Z);

        iterate_dirs!(self, context, POS_Y | POS_Z);
        iterate_dirs!(self, context, NEG_Y | POS_Z);
        iterate_dirs!(self, context, POS_Y | NEG_Z);
        iterate_dirs!(self, context, NEG_Y | NEG_Z);

        // Octants
        iterate_dirs!(self, context, POS_X | POS_Y | POS_Z);
        iterate_dirs!(self, context, NEG_X | POS_Y | POS_Z);
        iterate_dirs!(self, context, NEG_X | NEG_Y | POS_Z);
        iterate_dirs!(self, context, POS_X | NEG_Y | POS_Z);
        iterate_dirs!(self, context, POS_X | POS_Y | NEG_Z);
        iterate_dirs!(self, context, NEG_X | POS_Y | NEG_Z);
        iterate_dirs!(self, context, NEG_X | NEG_Y | NEG_Z);
        iterate_dirs!(self, context, POS_X | NEG_Y | NEG_Z);
    }

    fn iterate_dirs(
        &mut self,
        context: &GraphSearchContext,
        start_coords: LocalTileCoords,
        mut iter_directions: u8,
        process_tile_fn: fn(&mut Self, &GraphSearchContext, LocalTileIndex, LocalTileCoords),
    ) {
        let direction = take_one(&mut iter_directions);
        let steps = context.direction_step_counts[to_index(direction)];
        let mut coords = start_coords;

        for _ in 0..steps {
            coords = coords.step(direction);

            // if the direction set is empty, we should stop recursing, and start processing
            // tiles
            if iter_directions != 0 {
                self.iterate_dirs(context, coords, iter_directions, process_tile_fn);
            } else {
                let index = self.coord_space.pack_index(coords);

                process_tile_fn(self, context, index, coords);
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
        let test_result = context.test_tile(&self.coord_space, coords, self.do_height_checks);

        // tile needs to be re-borrowed multiple times in this method, due to borrow
        // checker rules. these should get optimized out.
        let tile = self.get_tile_mut(index);

        debug_assert_eq!(
            tile.outgoing_dir_section_sets,
            [SECTIONS_EMPTY; DIRECTION_COUNT]
        );
        debug_assert_eq!(tile.visible_sections, SECTIONS_EMPTY);

        if test_result == CombinedTestResults::OUTSIDE {
            // early exit
            tile.set_empty();
            return;
        }

        let mut start_visible_sections = SECTIONS_EMPTY;
        let mut incoming_dir_section_sets = [SECTIONS_EMPTY; DIRECTION_COUNT];

        // the center tile has no incoming directions, so there will be no data from
        // neighboring tiles. instead, we have to place the first set section manually.
        if INCOMING_DIRS == 0 {
            let tile = self.get_tile_mut(index);
            let section_idx = tile::section_index(context.camera_section_in_tile);

            tile::set_bit(&mut start_visible_sections, section_idx);
            tile.setup_center_tile(start_visible_sections);
        } else {
            self.get_incoming_edges::<INCOMING_DIRS>(
                coords,
                &mut start_visible_sections,
                &mut incoming_dir_section_sets,
            );

            // FAST PATH: if we start the traversal with all 0s, we'll end with all 0s.
            if start_visible_sections == SECTIONS_EMPTY {
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

        tile.find_visible_sections::<TRAVERSAL_DIRS>(
            start_visible_sections,
            incoming_dir_section_sets,
            &context.camera_direction_masks,
        );

        if tile.visible_sections != SECTIONS_EMPTY {
            let local_region_coords = coords.0.cast::<i32>() << Simd::from_xyz(0, 1, 0);
            let global_region_coords = context.global_region_offset + local_region_coords;

            let visible_sections_ptr = &raw const tile.visible_sections;

            self.visible_tiles.push(FFIVisibleSectionsTile::new(
                global_region_coords,
                visible_sections_ptr,
            ));
        }
    }

    fn get_incoming_edges<const INCOMING_DIRS: u8>(
        &mut self,
        coords: LocalTileCoords,
        visible_sections: &mut u8x64,
        incoming_dir_section_sets: &mut [u8x64; DIRECTION_COUNT],
    ) {
        if bitset::contains(INCOMING_DIRS, NEG_X) {
            let incoming_edge = self.get_incoming_edge::<NEG_X>(coords);
            *visible_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(NEG_X)] = incoming_edge;
        }

        if bitset::contains(INCOMING_DIRS, NEG_Y) {
            let incoming_edge = self.get_incoming_edge::<NEG_Y>(coords);
            *visible_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(NEG_Y)] = incoming_edge;
        }

        if bitset::contains(INCOMING_DIRS, NEG_Z) {
            let incoming_edge = self.get_incoming_edge::<NEG_Z>(coords);
            *visible_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(NEG_Z)] = incoming_edge;
        }

        if bitset::contains(INCOMING_DIRS, POS_X) {
            let incoming_edge = self.get_incoming_edge::<POS_X>(coords);
            *visible_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(POS_X)] = incoming_edge;
        }

        if bitset::contains(INCOMING_DIRS, POS_Y) {
            let incoming_edge = self.get_incoming_edge::<POS_Y>(coords);
            *visible_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(POS_Y)] = incoming_edge;
        }

        if bitset::contains(INCOMING_DIRS, POS_Z) {
            let incoming_edge = self.get_incoming_edge::<POS_Z>(coords);
            *visible_sections |= incoming_edge;
            incoming_dir_section_sets[to_index(POS_Z)] = incoming_edge;
        }
    }

    fn get_incoming_edge<const DIRECTION: u8>(&mut self, coords: LocalTileCoords) -> u8x64 {
        let neighbor_coords = coords.step(DIRECTION);
        let neighbor_index = self.coord_space.pack_index(neighbor_coords);
        let neighbor_tile = self.get_tile_mut(neighbor_index);

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
        let tile_coords = self.coord_space.section_to_tile_coords(section_coords);
        let index = self.coord_space.pack_index(tile_coords);

        #[cfg(debug_assertions)]
        println!(
            "Set Section - Section Coords: {:?}, Tile Coords: {:?}, Tile Index: {:?}, Vis: {}",
            section_coords, tile_coords.0, index.0, visibility_data
        );

        let tile = self.get_tile_mut(index);

        let section_coords_in_tile = section_coords.cast::<u8>() & Simd::splat(0b111);
        let section_idx = tile::section_index(section_coords_in_tile);

        for (array_idx, &bit_idx) in ARRAY_TO_BIT_IDX.iter().enumerate() {
            tile::modify_bit(
                &mut tile.connection_section_sets[array_idx],
                section_idx,
                visibility_data.get_bit(bit_idx),
            );
        }
    }
}
