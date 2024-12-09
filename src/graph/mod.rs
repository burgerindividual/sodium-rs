use std::hint::unreachable_unchecked;

use context::{CombinedTestResults, GraphSearchContext};
use coords::{GraphCoordSpace, LocalTileIndex};
use core_simd::simd::prelude::*;
use direction::*;
use tile::{NodeStorage, Tile, TraversalStatus};

use self::coords::LocalTileCoords;
use crate::bitset;
use crate::math::*;
use crate::results::SectionBitArray;

pub mod context;
pub mod coords;
pub mod direction;
pub mod tile;

pub struct Graph {
    level_0: Box<[Tile]>,
    level_1: Box<[Tile]>,
    level_2: Box<[Tile]>,
    level_3: Box<[Tile]>,
    level_4: Box<[Tile]>,

    pub coord_space: GraphCoordSpace,
    pub render_distance: u8,

    pub results: SectionBitArray,
}

impl Graph {
    const HIGHEST_LEVEL: u8 = 4;
    const EARLY_CHECKS_LOWEST_LEVEL: u8 = 1;

    pub fn new() -> Self {
        Self {
            level_0: todo!(),
            level_1: todo!(),
            level_2: todo!(),
            level_3: todo!(),
            level_4: todo!(),
            coord_space: todo!(),
            render_distance: todo!(),
            results: todo!(),
        }
    }

    pub fn cull(&mut self, context: &GraphSearchContext) {
        self.clear();

        self.iterate_tiles(context);
    }

    pub fn clear(&mut self) {
        self.results.clear();

        for tile in &mut self.level_0 {
            tile.traversal_status = TraversalStatus::Uninitialized;
            tile.traversed_nodes = NodeStorage(Simd::splat(0));
        }

        for tile in &mut self.level_1 {
            tile.traversal_status = TraversalStatus::Uninitialized;
            tile.traversed_nodes = NodeStorage(Simd::splat(0));
        }

        for tile in &mut self.level_2 {
            tile.traversal_status = TraversalStatus::Uninitialized;
            tile.traversed_nodes = NodeStorage(Simd::splat(0));
        }

        for tile in &mut self.level_3 {
            tile.traversal_status = TraversalStatus::Uninitialized;
            tile.traversed_nodes = NodeStorage(Simd::splat(0));
        }

        for tile in &mut self.level_4 {
            tile.traversal_status = TraversalStatus::Uninitialized;
            tile.traversed_nodes = NodeStorage(Simd::splat(0));
        }
    }

    fn iterate_tiles(&mut self, context: &GraphSearchContext) {
        // Center
        self.process_tile::<ALL_DIRECTIONS>(
            context,
            self.coord_space.pack_index(context.iter_start_tile),
            context.iter_start_tile,
            Self::HIGHEST_LEVEL,
            CombinedTestResults::ALL_PARTIAL,
        );

        // Axes
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_X,
            Self::process_tile::<{ direction::inverse_set(POS_X) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_Y,
            Self::process_tile::<{ direction::inverse_set(POS_Y) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_Z,
            Self::process_tile::<{ direction::inverse_set(POS_Z) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_X,
            Self::process_tile::<{ direction::inverse_set(NEG_X) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_Y,
            Self::process_tile::<{ direction::inverse_set(NEG_Y) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_Z,
            Self::process_tile::<{ direction::inverse_set(NEG_Z) }>,
        );

        // Planes
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_X | POS_Y,
            Self::process_tile::<{ direction::inverse_set(NEG_X | NEG_Y) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_X | POS_Y,
            Self::process_tile::<{ direction::inverse_set(POS_X | NEG_Y) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_X | NEG_Y,
            Self::process_tile::<{ direction::inverse_set(NEG_X | POS_Y) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_X | NEG_Y,
            Self::process_tile::<{ direction::inverse_set(POS_X | POS_Y) }>,
        );

        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_X | POS_Z,
            Self::process_tile::<{ direction::inverse_set(NEG_X | NEG_Z) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_X | POS_Z,
            Self::process_tile::<{ direction::inverse_set(POS_X | NEG_Z) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_X | NEG_Z,
            Self::process_tile::<{ direction::inverse_set(NEG_X | POS_Z) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_X | NEG_Z,
            Self::process_tile::<{ direction::inverse_set(POS_X | POS_Z) }>,
        );

        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_Y | POS_Z,
            Self::process_tile::<{ direction::inverse_set(NEG_Y | NEG_Z) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_Y | POS_Z,
            Self::process_tile::<{ direction::inverse_set(POS_Y | NEG_Z) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_Y | NEG_Z,
            Self::process_tile::<{ direction::inverse_set(NEG_Y | POS_Z) }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_Y | NEG_Z,
            Self::process_tile::<{ direction::inverse_set(POS_Y | POS_Z) }>,
        );

        // Octants
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_X | POS_Y | POS_Z,
            Self::process_tile::<{ POS_X | POS_Y | POS_Z }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_X | POS_Y | POS_Z,
            Self::process_tile::<{ NEG_X | POS_Y | POS_Z }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_X | NEG_Y | POS_Z,
            Self::process_tile::<{ NEG_X | NEG_Y | POS_Z }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_X | NEG_Y | POS_Z,
            Self::process_tile::<{ POS_X | NEG_Y | POS_Z }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_X | POS_Y | NEG_Z,
            Self::process_tile::<{ POS_X | POS_Y | NEG_Z }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_X | POS_Y | NEG_Z,
            Self::process_tile::<{ NEG_X | POS_Y | NEG_Z }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            NEG_X | NEG_Y | NEG_Z,
            Self::process_tile::<{ NEG_X | NEG_Y | NEG_Z }>,
        );
        self.iterate_dirs(
            context,
            context.iter_start_tile,
            POS_X | NEG_Y | NEG_Z,
            Self::process_tile::<{ POS_X | NEG_Y | NEG_Z }>,
        );
    }

    fn iterate_dirs(
        &mut self,
        context: &GraphSearchContext,
        start_coords: LocalTileCoords,
        mut iter_directions: u8,
        process_fn: fn(
            &mut Graph,
            &GraphSearchContext,
            LocalTileIndex,
            LocalTileCoords,
            u8,
            CombinedTestResults,
        ),
    ) {
        let direction = direction::take_dir(&mut iter_directions);
        let steps = context.direction_step_counts[direction::to_index(direction) as usize];
        let mut coords = start_coords;

        for _ in 0..steps {
            coords = self
                .coord_space
                .step_wrapping(coords, direction, Self::HIGHEST_LEVEL);

            // if the direction set is empty, we should stop recursing, and start processing
            // tiles
            if iter_directions != 0 {
                self.iterate_dirs(context, coords, iter_directions, process_fn);
            } else {
                let index = self.coord_space.pack_index(start_coords);

                process_fn(
                    self,
                    context,
                    index,
                    coords,
                    Self::HIGHEST_LEVEL,
                    CombinedTestResults::ALL_PARTIAL,
                );
            }
        }
    }

    fn process_tile<const DIRECTION_SET: u8>(
        &mut self,
        context: &GraphSearchContext,
        index: LocalTileIndex,
        coords: LocalTileCoords,
        level: u8,
        parent_test_results: CombinedTestResults,
    ) {
        // tile needs to be re-borrowed 3 times in this method, due to borrow checker
        // rules. these should be optimized out.
        let tile = self.get_tile_mut(index, level);

        tile.traversal_status = TraversalStatus::Processed {
            children_upmipped: 0,
        };

        // try to quickly determine whether we need to actually traverse the tile using
        // the frustum, fog, etc
        let test_result = if level >= Self::EARLY_CHECKS_LOWEST_LEVEL {
            context.test_tile(&self.coord_space, coords, level, parent_test_results)
        } else {
            CombinedTestResults::ALL_INSIDE
        };

        let tile = self.get_tile_mut(index, level);

        if test_result == CombinedTestResults::OUTSIDE {
            // early exit
            tile.traversed_nodes = NodeStorage(Simd::splat(0));
            return;
        }

        // if all children are present, we don't have to process this tile, and we can
        // solely process the children.
        if tile.children_to_traverse == 0b11111111 {
            // early exit
            // we don't have to set the traversed nodes here, because they'll be overwritten
            // by upmipping when needed
            return;
        }

        let combined_edge_data = self.combine_incoming_edges::<DIRECTION_SET>(coords, level);

        let tile = self.get_tile_mut(index, level);

        // FAST PATH: if we start with all 0s on the edges, we'll end with all 0s
        if combined_edge_data == Simd::splat(0) {
            // early exit
            tile.traversed_nodes = NodeStorage(Simd::splat(0));
            return;
        }

        let direction_masks = &context.direction_masks[level as usize];
        tile.traverse::<DIRECTION_SET>(combined_edge_data, direction_masks);

        if level > 0 && tile.children_to_traverse != 0b00000000 {
            let child_iter = tile.sorted_child_iter(index, coords, context.camera_pos_int, level);
            let child_level = level - 1;

            for (child_index, child_coords) in child_iter {
                self.process_tile::<DIRECTION_SET>(
                    context,
                    child_index,
                    child_coords,
                    child_level,
                    parent_test_results,
                );
            }
        } else {
            // end of the downward traversal
            self.results.set_tile(coords, level);
        }
    }

    fn combine_incoming_edges<const DIRECTION_SET: u8>(
        &mut self,
        coords: LocalTileCoords,
        level: u8,
    ) -> u8x64 {
        let mut combined_edge_data = Simd::splat(0);

        if bitset::contains(DIRECTION_SET, NEG_X) {
            combined_edge_data |= self.get_incoming_edge::<{ NEG_X }>(coords, level);
        }
        if bitset::contains(DIRECTION_SET, direction::NEG_Y) {
            combined_edge_data |= self.get_incoming_edge::<{ direction::NEG_Y }>(coords, level);
        }
        if bitset::contains(DIRECTION_SET, direction::NEG_Z) {
            combined_edge_data |= self.get_incoming_edge::<{ direction::NEG_Z }>(coords, level);
        }
        if bitset::contains(DIRECTION_SET, direction::POS_X) {
            combined_edge_data |= self.get_incoming_edge::<{ direction::POS_X }>(coords, level);
        }
        if bitset::contains(DIRECTION_SET, direction::POS_Y) {
            combined_edge_data |= self.get_incoming_edge::<{ direction::POS_Y }>(coords, level);
        }
        if bitset::contains(DIRECTION_SET, direction::POS_Z) {
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
            direction::NEG_Y => NodeStorage::edge_neg_to_pos_y(neighbor_traversed_nodes),
            direction::NEG_Z => NodeStorage::edge_neg_to_pos_z(neighbor_traversed_nodes),
            direction::POS_X => NodeStorage::edge_pos_to_neg_x(neighbor_traversed_nodes),
            direction::POS_Y => NodeStorage::edge_pos_to_neg_y(neighbor_traversed_nodes),
            direction::POS_Z => NodeStorage::edge_pos_to_neg_z(neighbor_traversed_nodes),
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
            match level {
                0 => self.level_0.get_mut(index.to_usize()).unwrap_unchecked(),
                1 => self.level_1.get_mut(index.to_usize()).unwrap_unchecked(),
                2 => self.level_2.get_mut(index.to_usize()).unwrap_unchecked(),
                3 => self.level_3.get_mut(index.to_usize()).unwrap_unchecked(),
                4 => self.level_4.get_mut(index.to_usize()).unwrap_unchecked(),
                _ => unreachable_unchecked(),
            }
        }
    }

    fn get_tile(&self, index: LocalTileIndex, level: u8) -> &Tile {
        unsafe {
            match level {
                0 => self.level_0.get(index.to_usize()).unwrap_unchecked(),
                1 => self.level_1.get(index.to_usize()).unwrap_unchecked(),
                2 => self.level_2.get(index.to_usize()).unwrap_unchecked(),
                3 => self.level_3.get(index.to_usize()).unwrap_unchecked(),
                4 => self.level_4.get(index.to_usize()).unwrap_unchecked(),
                _ => unreachable_unchecked(),
            }
        }
    }

    // TODO: should this reset the traversal status of the nodes it affects? that'll
    // get reset anyway, right?
    // NOTE: not thread safe, make sure
    pub fn set_section(&mut self, section_coords: i32x3, opaque_block_bytes: &[u8; 512]) {
        let level_1_coords = self.coord_space.section_to_local_coords(section_coords);
        let level_1_index = self.coord_space.pack_index(level_1_coords);
        let parent_index_high_bits = level_1_index.to_child_level().0;

        let level_1_tile = &mut self.level_1[level_1_index.to_usize()];
        let mut level_1_children = 0;

        for (level_1_child, tile_bytes) in opaque_block_bytes.chunks_exact(64).enumerate() {
            let level_0_index = LocalTileIndex(parent_index_high_bits | level_1_child as u32);
            let level_0_opaque_nodes = NodeStorage(u8x64::from_slice(tile_bytes));

            self.level_0[level_0_index.to_usize()].opaque_nodes = level_0_opaque_nodes;

            let level_1_traverse_child = level_1_tile
                .opaque_nodes
                .downscale_to_octant::<true>(level_0_opaque_nodes, level_1_child as u8);

            level_1_children |= (level_1_traverse_child as u8) << level_1_child;
        }

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
