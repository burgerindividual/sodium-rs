use std::hint::unreachable_unchecked;

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

    pub current_timestamp: u64,

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
            (u16::BITS as u8 - (world_y_len_sections - 1).leading_zeros() as u8).max(2);
        let graph_xz_bits =
            (u16::BITS as u8 - (world_xz_len_sections - 1).leading_zeros() as u8).max(2);

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
            current_timestamp: 0,
            visible_tiles: Vec::with_capacity(128), // probably not a bad start
        }
    }

    pub fn cull(&mut self, context: &GraphSearchContext) {
        self.clear();

        self.iterate_tiles(context);
    }

    pub fn clear(&mut self) {
        self.visible_tiles.clear();

        // if the cull is being run 1000 times a second, this'll take 584942415 years to
        // overflow
        self.current_timestamp += 1;
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

        let current_timestamp = self.current_timestamp;

        // tile needs to be re-borrowed multiple times in this method, due to borrow
        // checker rules. these should get optimized out.
        let tile = self.get_tile_mut(index);

        tile.last_change_timestamp = current_timestamp;

        // try to quickly determine whether we need to actually traverse the tile using
        // the frustum, fog, etc
        let test_result = context.test_tile(&self.coord_space, coords, self.do_height_checks);

        let tile = self.get_tile_mut(index);

        if test_result == CombinedTestResults::OUTSIDE {
            // early exit
            tile.set_empty();
            return;
        }

        let mut incoming_dir_section_sets = [SECTIONS_EMPTY; DIRECTION_COUNT];

        // the center tile has no incoming directions, so there will be no data from
        // neighboring tiles. instead, we have to place the first set section manually.
        if INCOMING_DIRS == 0 {
            let tile = self.get_tile_mut(index);
            let section_idx = tile::section_index(context.camera_section_in_tile);

            for outgoing_sections in &mut tile.outgoing_dir_section_sets {
                *outgoing_sections = SECTIONS_EMPTY;
                tile::set_bit(outgoing_sections, section_idx);
            }
        } else {
            let all_edges_empty =
                self.get_incoming_edges(coords, INCOMING_DIRS, &mut incoming_dir_section_sets);

            // FAST PATH: if we start the traversal with all 0s, we'll end with all 0s.
            if all_edges_empty {
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
            incoming_dir_section_sets,
            &context.camera_direction_masks,
        );

        if tile.visible_sections != SECTIONS_EMPTY {
            let origin_region_coords =
                (context.global_tile_offset + coords.0.cast::<i32>()) << Simd::from_xyz(0, 1, 0);

            let visible_sections_ptr = &raw const tile.visible_sections;

            self.visible_tiles.push(FFIVisibleSectionsTile::new(
                origin_region_coords,
                visible_sections_ptr,
            ));
        }
    }

    fn get_incoming_edges(
        &mut self,
        coords: LocalTileCoords,
        mut incoming_directions: u8,
        dest: &mut [u8x64; DIRECTION_COUNT],
    ) -> bool {
        let mut all_edges_empty = true;

        while incoming_directions != 0 {
            let direction = take_one(&mut incoming_directions);
            let incoming_edge = self.get_incoming_edge(coords, direction);
            all_edges_empty &= incoming_edge == SECTIONS_EMPTY;
            dest[to_index(direction)] = incoming_edge;
        }

        all_edges_empty
    }

    fn get_incoming_edge(&mut self, coords: LocalTileCoords, direction: u8) -> u8x64 {
        let current_timestamp = self.current_timestamp;
        let neighbor_coords = coords.step(direction);
        let neighbor_index = self.coord_space.pack_index(neighbor_coords);
        let neighbor_tile = self.get_tile_mut(neighbor_index);
        neighbor_tile.clear_if_outdated(current_timestamp);

        match direction {
            NEG_X => {
                tile::edge_pos_to_neg_x(neighbor_tile.outgoing_dir_section_sets[to_index(POS_X)])
            }
            NEG_Y => {
                tile::edge_pos_to_neg_y(neighbor_tile.outgoing_dir_section_sets[to_index(POS_Y)])
            }
            NEG_Z => {
                tile::edge_pos_to_neg_z(neighbor_tile.outgoing_dir_section_sets[to_index(POS_Z)])
            }
            POS_X => {
                tile::edge_neg_to_pos_x(neighbor_tile.outgoing_dir_section_sets[to_index(NEG_X)])
            }
            POS_Y => {
                tile::edge_neg_to_pos_y(neighbor_tile.outgoing_dir_section_sets[to_index(NEG_Y)])
            }
            POS_Z => {
                tile::edge_neg_to_pos_z(neighbor_tile.outgoing_dir_section_sets[to_index(NEG_Z)])
            }
            _ => unsafe { unreachable_unchecked() },
        }
    }

    fn get_tile_mut(&mut self, index: LocalTileIndex) -> &mut Tile {
        unsafe { self.tiles.get_unchecked_mut(index.to_usize()) }
    }

    fn get_tile(&self, index: LocalTileIndex) -> &Tile {
        unsafe { self.tiles.get_unchecked(index.to_usize()) }
    }

    pub fn set_section(&mut self, section_coords: i32x3, visibility_data: u64) {
        #[cfg(debug_assertions)]
        println!("Set Section - Coords: {:?}", section_coords);

        let tile_coords = self.coord_space.section_to_tile_coords(section_coords);
        let index = self.coord_space.pack_index(tile_coords);
        let tile = self.get_tile_mut(index);

        let section_coords_in_tile = section_coords.cast::<u8>() & Simd::splat(0b111);
        let section_idx = tile::section_index(section_coords_in_tile);

        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(NEG_Y, NEG_X)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_NEG_Y_NEG_X),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(NEG_Z, NEG_X)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_NEG_Z_NEG_X),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(NEG_Z, NEG_Y)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_NEG_Z_NEG_Y),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_X, NEG_X)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_X_NEG_X),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_X, NEG_Y)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_X_NEG_Y),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_X, NEG_Z)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_X_NEG_Z),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_Y, NEG_X)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_Y_NEG_X),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_Y, NEG_Y)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_Y_NEG_Y),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_Y, NEG_Z)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_Y_NEG_Z),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_Y, POS_X)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_Y_POS_X),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_Z, NEG_X)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_Z_NEG_X),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_Z, NEG_Y)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_Z_NEG_Y),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_Z, NEG_Z)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_Z_NEG_Z),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_Z, POS_X)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_Z_POS_X),
        );
        tile::modify_bit(
            &mut tile.connection_section_sets[connection_index(POS_Z, POS_Y)],
            section_idx,
            visibility_data.get_bit(BIT_IDX_POS_Z_POS_Y),
        );
    }
}
