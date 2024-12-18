use std::array;
use std::hint::unreachable_unchecked;
use std::num::NonZero;

use context::{CombinedTestResults, GraphSearchContext};
use coords::{GraphCoordSpace, LocalTileIndex, SortedChildIterator};
use core_simd::simd::prelude::*;
use direction::*;
use tile::{NodeStorage, Tile, TraversalStatus};

use self::coords::LocalTileCoords;
use crate::bitset;
use crate::ffi::FFIVisibleSectionsTile;
use crate::math::*;

pub mod context;
pub mod coords;
pub mod direction;
pub mod tile;

macro_rules! iterate_dirs {
    ($graph:ident, $context:ident, $iter_dirs:expr) => {{
        const INCOMING_DIRECTIONS: u8 = opposite($iter_dirs);
        const TRAVERSAL_DIRECTIONS: u8 = all_except(INCOMING_DIRECTIONS);

        $graph.iterate_dirs_recursive(
            $context,
            $context.iter_start_tile,
            $iter_dirs,
            Self::process_tile::<INCOMING_DIRECTIONS, TRAVERSAL_DIRECTIONS>,
        );
    }};
}

pub struct Graph {
    tile_levels: [Box<[Tile]>; 5],

    pub coord_space: GraphCoordSpace,
    pub render_distance: u8,

    // TODO: include references to visible section arrays
    pub visible_tiles: Vec<FFIVisibleSectionsTile>,
}

impl Graph {
    const HIGHEST_LEVEL: u8 = 4;
    const EARLY_CHECKS_LOWEST_LEVEL: u8 = 1;

    pub fn new(render_distance: u8, world_bottom_section_y: i8, world_top_section_y: i8) -> Self {
        let world_height_level_0 =
            (world_top_section_y as i16 - world_bottom_section_y as i16) as u16 * 2;
        let world_width_level_0 = (render_distance as u16 * 4) + 2;

        // the size of each dimension of the graph has to be able to account for 1 full
        // tile at the lowest resolution.
        let graph_height_bits_level_0 = (u16::BITS as u8
            - unsafe { NonZero::<u16>::new_unchecked(world_height_level_0) }.leading_zeros() as u8)
            .max(Self::HIGHEST_LEVEL);
        let graph_width_bits_level_0 = (u16::BITS as u8
            - unsafe { NonZero::<u16>::new_unchecked(world_width_level_0) }.leading_zeros() as u8)
            .max(Self::HIGHEST_LEVEL);

        let graph_height_level_0 = 1_usize << graph_height_bits_level_0;
        let graph_width_level_0 = 1_usize << graph_width_bits_level_0;

        let tile_levels: [Box<[Tile]>; 5] = array::from_fn(|level| {
            let mut tiles = Box::<[Tile]>::new_uninit_slice(
                (graph_height_level_0 >> level) * (graph_width_level_0 >> level).pow(2),
            );

            for tile_uninit in tiles.iter_mut() {
                tile_uninit.write(Default::default());
            }

            unsafe { tiles.assume_init() }
        });

        Self {
            tile_levels,
            coord_space: GraphCoordSpace::new(
                graph_width_bits_level_0,
                graph_height_bits_level_0,
                graph_width_bits_level_0,
                world_bottom_section_y,
                world_top_section_y,
            ),
            render_distance,
            visible_tiles: Vec::with_capacity(128), // probably not a bad start
        }
    }

    pub fn cull(&mut self, context: &GraphSearchContext) {
        self.clear();

        self.iterate_tiles(context);
    }

    pub fn clear(&mut self) {
        self.visible_tiles.clear();

        for tile_level in self.tile_levels.iter_mut() {
            for tile in tile_level.iter_mut() {
                tile.traversal_status = TraversalStatus::Uninitialized;
                tile.traversed_nodes = NodeStorage::EMPTY;
                tile.visible_nodes = NodeStorage::EMPTY;
            }
        }
    }

    fn iterate_tiles(&mut self, context: &GraphSearchContext) {
        // Center
        self.process_upper_tile(
            context,
            self.coord_space.pack_index(context.iter_start_tile),
            context.iter_start_tile,
            Self::process_tile::<0, ALL_DIRECTIONS>,
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

    fn iterate_dirs_recursive(
        &mut self,
        context: &GraphSearchContext,
        start_coords: LocalTileCoords,
        mut iter_directions: u8,
        inner_process_fn: fn(
            &mut Graph,
            &GraphSearchContext,
            LocalTileIndex,
            LocalTileCoords,
            u8,
            CombinedTestResults,
        ),
    ) {
        let direction = direction::take_one(&mut iter_directions);
        let steps = context.direction_step_counts[direction::to_index(direction) as usize];
        let mut coords = start_coords;

        for _ in 0..steps {
            coords = self
                .coord_space
                .step_wrapping(coords, direction, Self::HIGHEST_LEVEL);

            // if the direction set is empty, we should stop recursing, and start processing
            // tiles
            if iter_directions != 0 {
                self.iterate_dirs_recursive(context, coords, iter_directions, inner_process_fn);
            } else {
                let index = self.coord_space.pack_index(start_coords);

                self.process_upper_tile(context, index, coords, inner_process_fn);
            }
        }
    }

    fn process_upper_tile(
        &mut self,
        context: &GraphSearchContext,
        index: LocalTileIndex,
        coords: LocalTileCoords,
        inner_process_fn: fn(
            &mut Graph,
            &GraphSearchContext,
            LocalTileIndex,
            LocalTileCoords,
            u8,
            CombinedTestResults,
        ),
    ) {
        inner_process_fn(
            self,
            context,
            index,
            coords,
            Self::HIGHEST_LEVEL,
            CombinedTestResults::ALL_PARTIAL,
        );

        let tile = self.get_tile(index, Self::HIGHEST_LEVEL);

        if tile.visible_nodes != NodeStorage::EMPTY {
            self.visible_tiles.push(FFIVisibleSectionsTile::new(
                &raw const tile.visible_nodes,
                coords,
            ));
        }
    }

    fn process_tile<const INCOMING_DIRECTIONS: u8, const TRAVERSAL_DIRECTIONS: u8>(
        &mut self,
        context: &GraphSearchContext,
        index: LocalTileIndex,
        coords: LocalTileCoords,
        level: u8,
        parent_test_results: CombinedTestResults,
    ) {
        // tile needs to be re-borrowed multiple times in this method, due to borrow
        // checker rules. these should get optimized out.
        let tile = self.get_tile_mut(index, level);

        tile.traversal_status = TraversalStatus::Processed {
            children_upmipped: 0,
        };

        // try to quickly determine whether we need to actually traverse the tile using
        // the frustum, fog, etc
        // TODO: do the height test unconditionally?
        let test_result = if level >= Self::EARLY_CHECKS_LOWEST_LEVEL {
            context.test_tile(&self.coord_space, coords, level, parent_test_results)
        } else {
            CombinedTestResults::ALL_INSIDE
        };

        let tile = self.get_tile_mut(index, level);

        if test_result == CombinedTestResults::OUTSIDE {
            // early exit
            tile.set_empty_traversal();
            return;
        }

        let children_to_traverse = tile.children_to_traverse;

        // if all children are present, we don't have to process this tile, and we can
        // solely process the children.
        if children_to_traverse != 0b11111111 {
            let combined_edge_data =
                self.combine_incoming_edges::<INCOMING_DIRECTIONS>(coords, level);

            let tile = self.get_tile_mut(index, level);
            let incoming_traversed_nodes = combined_edge_data & tile.opaque_nodes.0;

            // FAST PATH: if we start the traversal with all 0s, we'll end with all 0s.
            if incoming_traversed_nodes == Simd::splat(0) {
                // early exit
                tile.set_empty_traversal();
                return;
            }

            // if we've hit this point, we know that there's atleast 1 node that has been
            // traversed in this tile. because of this, we know atleast part of
            // it is visible, so we must mark that accordingly in the results.

            let direction_masks = &context.direction_masks[level as usize];
            tile.find_visible_nodes::<TRAVERSAL_DIRECTIONS>(
                incoming_traversed_nodes,
                direction_masks,
            );

            // self.results.set_tile(coords, level, traversed_nodes,
            // !children_to_traverse);
        }

        // this should always be false for level 0 tiles
        if children_to_traverse != 0b00000000 {
            let child_iter = SortedChildIterator::new(
                index,
                coords,
                context.camera_pos_int,
                children_to_traverse,
                level,
            );
            let child_level = level - 1;

            for (child_index, child_coords) in child_iter {
                self.process_tile::<TRAVERSAL_DIRECTIONS, INCOMING_DIRECTIONS>(
                    context,
                    child_index,
                    child_coords,
                    child_level,
                    parent_test_results,
                );

                // unconditionally downscale visibility data from children to propogate visible
                // sections up
                let child_tile = self.get_tile(child_index, child_level);
                let child_visible_nodes = child_tile.visible_nodes;

                let tile = self.get_tile_mut(index, level);
                tile.visible_nodes
                    .downscale_to_octant::<false>(child_visible_nodes, child_index.child_number());
            }
        }
    }

    fn combine_incoming_edges<const INCOMING_DIRECTIONS: u8>(
        &mut self,
        coords: LocalTileCoords,
        level: u8,
    ) -> u8x64 {
        let mut combined_edge_data = Simd::splat(0);

        if bitset::contains(INCOMING_DIRECTIONS, NEG_X) {
            combined_edge_data |= self.get_incoming_edge::<{ NEG_X }>(coords, level);
        }
        if bitset::contains(INCOMING_DIRECTIONS, direction::NEG_Y) {
            combined_edge_data |= self.get_incoming_edge::<{ direction::NEG_Y }>(coords, level);
        }
        if bitset::contains(INCOMING_DIRECTIONS, direction::NEG_Z) {
            combined_edge_data |= self.get_incoming_edge::<{ direction::NEG_Z }>(coords, level);
        }
        if bitset::contains(INCOMING_DIRECTIONS, direction::POS_X) {
            combined_edge_data |= self.get_incoming_edge::<{ direction::POS_X }>(coords, level);
        }
        if bitset::contains(INCOMING_DIRECTIONS, direction::POS_Y) {
            combined_edge_data |= self.get_incoming_edge::<{ direction::POS_Y }>(coords, level);
        }
        if bitset::contains(INCOMING_DIRECTIONS, direction::POS_Z) {
            combined_edge_data |= self.get_incoming_edge::<{ direction::POS_Z }>(coords, level);
        }

        combined_edge_data
    }

    fn get_incoming_edge<const DIRECTION: u8>(
        &mut self,
        coords: LocalTileCoords,
        level: u8,
    ) -> u8x64 {
        let neighbor_coords = self.coord_space.step_wrapping(coords, DIRECTION, level);
        let neighbor_index = self.coord_space.pack_index(neighbor_coords);
        let neighbor_tile = self.get_tile(neighbor_index, level);
        let neighbor_traversal_status = neighbor_tile.traversal_status;

        let neighbor_traversed_nodes = match neighbor_traversal_status {
            TraversalStatus::Uninitialized => self.get_traversed_nodes_up(neighbor_index, level),
            TraversalStatus::Downmipped => neighbor_tile.traversed_nodes,
            TraversalStatus::Processed {
                children_upmipped: _,
            } => self.get_traversed_nodes_down::<DIRECTION>(neighbor_index, level),
        }
        .0;

        match DIRECTION {
            NEG_X => NodeStorage::edge_neg_to_pos_x(neighbor_traversed_nodes),
            NEG_Y => NodeStorage::edge_neg_to_pos_y(neighbor_traversed_nodes),
            NEG_Z => NodeStorage::edge_neg_to_pos_z(neighbor_traversed_nodes),
            POS_X => NodeStorage::edge_pos_to_neg_x(neighbor_traversed_nodes),
            POS_Y => NodeStorage::edge_pos_to_neg_y(neighbor_traversed_nodes),
            POS_Z => NodeStorage::edge_pos_to_neg_z(neighbor_traversed_nodes),
            _ => unsafe { unreachable_unchecked() },
        }
    }

    fn get_traversed_nodes_down<const DIRECTION: u8>(
        &mut self,
        index: LocalTileIndex,
        level: u8,
    ) -> NodeStorage {
        let tile = self.get_tile(index, level);

        let edge_mask = match DIRECTION {
            NEG_X => 0b00001111,
            direction::NEG_Y => 0b00110011,
            direction::NEG_Z => 0b01010101,
            direction::POS_X => 0b11110000,
            direction::POS_Y => 0b11001100,
            direction::POS_Z => 0b10101010,
            _ => unsafe { unreachable_unchecked() },
        };

        let children_on_edge = tile.children_to_traverse & edge_mask;
        let children_upmipped =
            if let TraversalStatus::Processed { children_upmipped } = tile.traversal_status {
                children_upmipped
            } else {
                unsafe { unreachable_unchecked() }
            };
        let children_to_recurse = children_on_edge & !children_upmipped;

        if children_to_recurse == 0 {
            return tile.traversed_nodes;
        }

        let mut traversed_nodes_updated = tile.traversed_nodes;
        let child_level = level - 1;
        for (child_index, child) in index.unordered_child_iter(children_to_recurse) {
            let child_traversed_nodes =
                self.get_traversed_nodes_down::<DIRECTION>(child_index, child_level);

            traversed_nodes_updated.downscale_to_octant::<false>(child_traversed_nodes, child);
        }

        let tile = self.get_tile_mut(index, level);

        let children_upmipped =
            if let TraversalStatus::Processed { children_upmipped } = &mut tile.traversal_status {
                children_upmipped
            } else {
                unsafe { unreachable_unchecked() }
            };
        *children_upmipped |= children_to_recurse;
        tile.traversed_nodes = traversed_nodes_updated;

        traversed_nodes_updated
    }

    // the upward search here assumes that there is an upper level which has been
    // traversed.
    fn get_traversed_nodes_up(&mut self, index: LocalTileIndex, level: u8) -> NodeStorage {
        let parent_index = index.to_parent_level();
        let parent_level = level + 1;
        let parent_tile = self.get_tile(parent_index, parent_level);
        let parent_traversal_status = parent_tile.traversal_status;

        let parent_traversed_nodes = if parent_traversal_status == TraversalStatus::Uninitialized {
            self.get_traversed_nodes_up(parent_index, parent_level)
        } else {
            parent_tile.traversed_nodes
        };

        let tile = self.get_tile_mut(index, level);

        let mut traversed_nodes = parent_traversed_nodes.upscale(index.child_number());
        traversed_nodes.0 |= tile.opaque_nodes.0;

        tile.traversed_nodes = traversed_nodes;
        tile.traversal_status = TraversalStatus::Downmipped;

        traversed_nodes
    }

    fn get_tile_mut(&mut self, index: LocalTileIndex, level: u8) -> &mut Tile {
        unsafe {
            self.tile_levels
                .get_unchecked_mut(level as usize)
                .get_unchecked_mut(index.to_usize())
        }
    }

    fn get_tile(&self, index: LocalTileIndex, level: u8) -> &Tile {
        unsafe {
            self.tile_levels
                .get_unchecked(level as usize)
                .get_unchecked(index.to_usize())
        }
    }

    // TODO: should this reset the traversal status of the nodes it affects? that'll
    // get reset anyway, right?
    // NOTE: not thread safe, make sure
    pub fn set_section(&mut self, section_coords: i32x3, opaque_block_bytes: &[u8; 512]) {
        let level_1_coords = self.coord_space.section_to_local_coords(section_coords);
        let level_1_index = self.coord_space.pack_index(level_1_coords);
        let parent_index_high_bits = level_1_index.to_child_level().0;

        let mut level_1_children = 0;

        for (level_1_child, tile_bytes) in opaque_block_bytes.chunks_exact(64).enumerate() {
            let level_0_index = LocalTileIndex(parent_index_high_bits | level_1_child as u32);
            let level_0_opaque_nodes = NodeStorage(u8x64::from_slice(tile_bytes));

            let level_0_tile = &mut self.tile_levels[0][level_0_index.to_usize()];
            level_0_tile.opaque_nodes = level_0_opaque_nodes;

            let level_1_tile = &mut self.tile_levels[1][level_1_index.to_usize()];
            let level_1_traverse_child = level_1_tile
                .opaque_nodes
                .downscale_to_octant::<true>(level_0_opaque_nodes, level_1_child as u8);

            level_1_children |= (level_1_traverse_child as u8) << level_1_child;
        }

        let level_1_tile = &mut self.tile_levels[1][level_1_index.to_usize()];
        level_1_tile.children_to_traverse = level_1_children;

        let level_1_opaque_nodes = level_1_tile.opaque_nodes;

        self.propagate_opaque_nodes_up(level_1_index, level_1_opaque_nodes, 1);
    }

    pub fn propagate_opaque_nodes_up(
        &mut self,
        index: LocalTileIndex,
        opaque_nodes: NodeStorage,
        level: u8,
    ) {
        let parent_level = level + 1;
        let parent_index = index.to_parent_level();
        let child = index.child_number();
        let parent_tile = self.get_tile_mut(parent_index, parent_level);
        let should_traverse_child = parent_tile
            .opaque_nodes
            .downscale_to_octant::<true>(opaque_nodes, child);
        parent_tile.children_to_traverse &= !(0b1 << child);
        parent_tile.children_to_traverse |= (should_traverse_child as u8) << child;

        let parent_opaque_nodes = parent_tile.opaque_nodes;

        if parent_level < Self::HIGHEST_LEVEL {
            self.propagate_opaque_nodes_up(parent_index, parent_opaque_nodes, parent_level);
        }
    }

    pub fn remove_section(&mut self, section_coord: i32x3) {
        // a removed section should be fully opaque to prevent traversal (maybe don't do
        // this? hm.)
        // TODO: write this directly to just set the tiles to Default::default()
        // TODO: can the level 0 nodes just be kept the same, and the level 1 children
        // just be toggled off?
        self.set_section(section_coord, &[0xFF; 512]);
    }
}
