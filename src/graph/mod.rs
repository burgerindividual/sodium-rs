use std::array;
use std::hint::{assert_unchecked, unreachable_unchecked};
use std::num::NonZero;

use context::{CombinedTestResults, GraphSearchContext};
use coords::{GraphCoordSpace, LocalTileIndex, PosSortedChildIterator};
use core_simd::simd::prelude::*;
use direction::*;
use tile::{NodeStorage, Tile, TraversalStatus};

use self::coords::LocalTileCoords;
use crate::bitset::{self};
use crate::ffi::FFIVisibleSectionsTile;
use crate::math::*;

pub mod context;
pub mod coords;
pub mod direction;
pub mod tile;

// NOTE: this structure is NOT THREAD SAFE on its own. you must manually do
// synchronization if you will be accessing this structure from multiple
// threads.
pub struct Graph {
    tile_levels: [Box<[Tile]>; 5],

    pub coord_space: GraphCoordSpace,
    pub render_distance: u8,
    enabled_tile_checks: CombinedTestResults,

    pub current_timestamp: u64,

    pub visible_tiles: Vec<FFIVisibleSectionsTile>,
}

impl Graph {
    const HIGHEST_LEVEL: u8 = 4;
    const EARLY_CHECKS_LOWEST_LEVEL: u8 = 1;

    pub fn new(render_distance: u8, world_bottom_section_y: i8, world_top_section_y: i8) -> Self {
        let world_height_level_0 =
            (world_top_section_y as i16 - world_bottom_section_y as i16 + 1) as u16 * 2;
        let world_width_level_0 = (render_distance as u16 * 4) + 2;

        // the size of each dimension of the graph has to be able to account for 2 full
        // tiles at the lowest resolution.
        let graph_height_bits_level_0 = (u16::BITS as u8
            - unsafe { NonZero::<u16>::new_unchecked(world_height_level_0) }.leading_zeros() as u8)
            .max(Self::HIGHEST_LEVEL + 1);
        let graph_width_bits_level_0 = (u16::BITS as u8
            - unsafe { NonZero::<u16>::new_unchecked(world_width_level_0) }.leading_zeros() as u8)
            .max(Self::HIGHEST_LEVEL + 1);

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

        let skip_height_checks =
            world_height_level_0 == (world_height_level_0 & (-1_i16 << Self::HIGHEST_LEVEL) as u16);

        let enabled_tile_checks = if skip_height_checks {
            CombinedTestResults::HEIGHT_INSIDE
        } else {
            CombinedTestResults::NONE_INSIDE
        };

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
            enabled_tile_checks,
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

        // for tile_level in self.tile_levels.iter_mut() {
        //     for tile in tile_level.iter_mut() {
        //         tile.traversal_status = TraversalStatus::Uninitialized;
        //         tile.traversed_nodes = NodeStorage::EMPTY;
        //         tile.visible_nodes = NodeStorage::EMPTY;
        //     }
        // }
    }

    fn iterate_tiles(&mut self, context: &GraphSearchContext) {
        // Center
        self.process_upper_tile(
            context,
            self.coord_space
                .pack_index(context.iter_start_tile, Self::HIGHEST_LEVEL),
            context.iter_start_tile,
        );

        // Axes
        self.iterate_dirs(context, context.iter_start_tile, POS_X);
        self.iterate_dirs(context, context.iter_start_tile, POS_Y);
        self.iterate_dirs(context, context.iter_start_tile, POS_Z);
        self.iterate_dirs(context, context.iter_start_tile, NEG_X);
        self.iterate_dirs(context, context.iter_start_tile, NEG_Y);
        self.iterate_dirs(context, context.iter_start_tile, NEG_Z);

        // Planes
        self.iterate_dirs(context, context.iter_start_tile, POS_X | POS_Y);
        self.iterate_dirs(context, context.iter_start_tile, NEG_X | POS_Y);
        self.iterate_dirs(context, context.iter_start_tile, POS_X | NEG_Y);
        self.iterate_dirs(context, context.iter_start_tile, NEG_X | NEG_Y);

        self.iterate_dirs(context, context.iter_start_tile, POS_X | POS_Z);
        self.iterate_dirs(context, context.iter_start_tile, NEG_X | POS_Z);
        self.iterate_dirs(context, context.iter_start_tile, POS_X | NEG_Z);
        self.iterate_dirs(context, context.iter_start_tile, NEG_X | NEG_Z);

        self.iterate_dirs(context, context.iter_start_tile, POS_Y | POS_Z);
        self.iterate_dirs(context, context.iter_start_tile, NEG_Y | POS_Z);
        self.iterate_dirs(context, context.iter_start_tile, POS_Y | NEG_Z);
        self.iterate_dirs(context, context.iter_start_tile, NEG_Y | NEG_Z);

        // Octants
        self.iterate_dirs(context, context.iter_start_tile, POS_X | POS_Y | POS_Z);
        self.iterate_dirs(context, context.iter_start_tile, NEG_X | POS_Y | POS_Z);
        self.iterate_dirs(context, context.iter_start_tile, NEG_X | NEG_Y | POS_Z);
        self.iterate_dirs(context, context.iter_start_tile, POS_X | NEG_Y | POS_Z);
        self.iterate_dirs(context, context.iter_start_tile, POS_X | POS_Y | NEG_Z);
        self.iterate_dirs(context, context.iter_start_tile, NEG_X | POS_Y | NEG_Z);
        self.iterate_dirs(context, context.iter_start_tile, NEG_X | NEG_Y | NEG_Z);
        self.iterate_dirs(context, context.iter_start_tile, POS_X | NEG_Y | NEG_Z);
    }

    // fn iterate_dirs_wrapped(
    //     &mut self,
    //     context: &GraphSearchContext,
    //     start_coords: LocalTileCoords,
    //     iter_directions: u8,
    // ) {
    //     #[cfg(debug_assertions)]
    //     {
    //         print!("Iterating dirs");
    //         let mut tmp = iter_directions;
    //         while tmp != 0 {
    //             let direction = take_one(&mut tmp);
    //             print!(" {}", to_str(direction));
    //         }
    //         println!();
    //     }

    //     self.iterate_dirs(context, start_coords, iter_directions);
    // }

    fn iterate_dirs(
        &mut self,
        context: &GraphSearchContext,
        start_coords: LocalTileCoords,
        mut iter_directions: u8,
    ) {
        let direction = take_one(&mut iter_directions);
        let steps = context.direction_step_counts[to_index(direction) as usize];
        let mut coords = start_coords;

        for _ in 0..steps {
            coords = self.coord_space.step(coords, direction);

            // if the direction set is empty, we should stop recursing, and start processing
            // tiles
            if iter_directions != 0 {
                self.iterate_dirs(context, coords, iter_directions);
            } else {
                let index = self.coord_space.pack_index(coords, Self::HIGHEST_LEVEL);

                self.process_upper_tile(context, index, coords);
            }
        }
    }

    fn process_upper_tile(
        &mut self,
        context: &GraphSearchContext,
        index: LocalTileIndex,
        coords: LocalTileCoords,
    ) {
        self.process_tile(
            context,
            index,
            coords,
            Self::HIGHEST_LEVEL,
            self.enabled_tile_checks,
        );

        let tile = self.get_tile(index, Self::HIGHEST_LEVEL);

        if tile.visible_nodes != NodeStorage::EMPTY {
            let mut tile_section_coords = coords;
            // go from the highest level to level 1, which is the section level
            for _ in 1..Self::HIGHEST_LEVEL {
                tile_section_coords = tile_section_coords.to_child_level();
            }

            let origin_section_coords =
                context.global_section_offset + tile_section_coords.0.cast::<i32>();

            self.visible_tiles.push(FFIVisibleSectionsTile::new(
                origin_section_coords,
                &raw const tile.visible_nodes,
            ));
        }
    }

    fn process_tile(
        &mut self,
        context: &GraphSearchContext,
        index: LocalTileIndex,
        coords: LocalTileCoords,
        level: u8,
        parent_test_results: CombinedTestResults,
    ) {
        #[cfg(debug_assertions)]
        println!(
            "Current Tile - Level {:?} Coords: {:?} Index: {:?}",
            level, coords.0, index.0
        );

        let current_timestamp = self.current_timestamp;

        // tile needs to be re-borrowed multiple times in this method, due to borrow
        // checker rules. these should get optimized out.
        let tile = self.get_tile_mut(index, level);

        tile.traversal_status = TraversalStatus::Processed {
            children_traversal_upmipped: 0,
        };
        tile.last_change_timestamp = current_timestamp;

        // try to quickly determine whether we need to actually traverse the tile using
        // the frustum, fog, etc
        let test_result = context.test_tile(&self.coord_space, coords, level, parent_test_results);

        let tile = self.get_tile_mut(index, level);

        if test_result == CombinedTestResults::OUTSIDE {
            // early exit
            tile.set_empty();
            return;
        }

        // Process tile if children don't need to be processed. This should always be
        // false for level 0 tiles
        if tile.should_traverse_children {
            let child_iter = PosSortedChildIterator::new(
                index,
                coords,
                level,
                context.camera_pos_int.cast::<i16>(),
            );
            let child_level = level - 1;

            for (child_index, child_coords) in child_iter {
                self.process_tile(
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
                    .downscale_to_octant(child_visible_nodes, child_index.child_octant());
            }
        } else {
            let incoming_directions = context.get_incoming_directions(coords, level);
            let traversal_directions = all_except(incoming_directions);

            let incoming_traversed_nodes = self.get_incoming_traversed_nodes(
                context,
                index,
                coords,
                level,
                incoming_directions,
            );

            let tile = self.get_tile_mut(index, level);

            // FAST PATH: if we start the traversal with all 0s, we'll end with all 0s.
            if incoming_traversed_nodes == Simd::splat(0) {
                // early exit
                tile.set_empty();
                return;
            }

            // if we've hit this point, we know that there's atleast 1 node that has been
            // traversed in this tile. because of this, we know atleast part of
            // it is visible.

            let direction_masks = unsafe { context.direction_masks.get_unchecked(level as usize) };
            tile.find_visible_nodes(
                incoming_traversed_nodes,
                direction_masks,
                traversal_directions,
            );
        }
    }

    fn get_incoming_traversed_nodes(
        &mut self,
        context: &GraphSearchContext,
        index: LocalTileIndex,
        coords: LocalTileCoords,
        level: u8,
        mut incoming_directions: u8,
    ) -> u8x64 {
        // the center tile has no incoming directions. instead, we have to place the
        // first set node manually.
        if incoming_directions == 0 {
            unsafe {
                assert_unchecked(level <= Graph::HIGHEST_LEVEL);
            }
            let camera_coords_in_tile =
                (context.camera_pos_int >> Simd::splat(level as u16)) & Simd::splat(0b111);
            let camera_node_index = NodeStorage::index(
                camera_coords_in_tile.x() as u8,
                camera_coords_in_tile.y() as u8,
                camera_coords_in_tile.z() as u8,
            );

            let mut traversed_nodes = NodeStorage::EMPTY;
            traversed_nodes.put_bit(camera_node_index, true);

            // TODO: get rid of this, and find a better way to handle when the camera is
            // inside an non-traversable node
            #[cfg(debug_assertions)]
            println!(
                "Camera inside traversable node: {}",
                self.get_tile(index, level)
                    .traversable_nodes
                    .get_bit(camera_node_index)
            );

            traversed_nodes.0
        } else {
            let mut combined_incoming_edges = Simd::splat(0);

            while incoming_directions != 0 {
                let direction = take_one(&mut incoming_directions);

                combined_incoming_edges |= self.get_incoming_edge(coords, level, direction);
            }

            let tile = self.get_tile(index, level);

            combined_incoming_edges & tile.traversable_nodes.0
        }
    }

    fn get_incoming_edge(&mut self, coords: LocalTileCoords, level: u8, direction: u8) -> u8x64 {
        let current_timestamp = self.current_timestamp;
        let neighbor_coords = self.coord_space.step(coords, direction);
        let neighbor_index = self.coord_space.pack_index(neighbor_coords, level);
        let neighbor_tile = self.get_tile_mut(neighbor_index, level);
        neighbor_tile.clear_if_outdated(current_timestamp);
        let neighbor_traversal_status = neighbor_tile.traversal_status;

        let neighbor_traversed_nodes = match neighbor_traversal_status {
            TraversalStatus::Uninitialized => self.get_traversed_nodes_up(neighbor_index, level),
            TraversalStatus::Downmipped => neighbor_tile.traversed_nodes,
            TraversalStatus::Processed {
                children_traversal_upmipped: _,
            } => {
                // the edge mask is for the opposite of the incoming direction
                let edge_mask = match direction {
                    POS_X => 0b00001111,
                    POS_Y => 0b00110011,
                    POS_Z => 0b01010101,
                    NEG_X => 0b11110000,
                    NEG_Y => 0b11001100,
                    NEG_Z => 0b10101010,
                    _ => unsafe { unreachable_unchecked() },
                };

                self.get_traversed_nodes_down(neighbor_index, level, edge_mask)
            }
        }
        .0;

        let result = match direction {
            NEG_X => NodeStorage::edge_pos_to_neg_x(neighbor_traversed_nodes),
            NEG_Y => NodeStorage::edge_pos_to_neg_y(neighbor_traversed_nodes),
            NEG_Z => NodeStorage::edge_pos_to_neg_z(neighbor_traversed_nodes),
            POS_X => NodeStorage::edge_neg_to_pos_x(neighbor_traversed_nodes),
            POS_Y => NodeStorage::edge_neg_to_pos_y(neighbor_traversed_nodes),
            POS_Z => NodeStorage::edge_neg_to_pos_z(neighbor_traversed_nodes),
            _ => unsafe { unreachable_unchecked() },
        };

        result
    }

    fn get_traversed_nodes_down(
        &mut self,
        index: LocalTileIndex,
        level: u8,
        edge_mask: u8,
    ) -> NodeStorage {
        #[cfg(debug_assertions)]
        println!("Traversed Down - Level {:?} Index: {:?}", level, index.0);

        let current_timestamp = self.current_timestamp;
        let tile = self.get_tile_mut(index, level);
        tile.clear_if_outdated(current_timestamp);

        // stop downward traversal for empty nodes, as their children should also be
        // empty
        if tile.is_empty() {
            return NodeStorage::EMPTY;
        }

        // stop downward traversal when we know children won't be populated
        if !tile.should_traverse_children {
            return tile.traversed_nodes;
        }

        let children_upmipped = if let TraversalStatus::Processed {
            children_traversal_upmipped: children_upmipped,
        } = tile.traversal_status
        {
            children_upmipped
        } else {
            unsafe { unreachable_unchecked() }
        };
        let children_to_recurse = edge_mask & !children_upmipped;

        if children_to_recurse == 0 {
            return tile.traversed_nodes;
        }

        let mut traversed_nodes_updated = tile.traversed_nodes;
        let child_level = level - 1;
        for child_index in index.unordered_child_iter() {
            let child_traversed_nodes =
                self.get_traversed_nodes_down(child_index, child_level, edge_mask);

            traversed_nodes_updated
                .downscale_to_octant(child_traversed_nodes, child_index.child_octant());
        }

        let tile = self.get_tile_mut(index, level);

        let children_upmipped = if let TraversalStatus::Processed {
            children_traversal_upmipped: children_upmipped,
        } = &mut tile.traversal_status
        {
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
        let current_timestamp = self.current_timestamp;
        let parent_index = index.to_parent_level();
        let parent_level = level + 1;
        let parent_tile = self.get_tile_mut(parent_index, parent_level);
        parent_tile.clear_if_outdated(current_timestamp);
        let parent_traversal_status = parent_tile.traversal_status;

        #[cfg(debug_assertions)]
        println!(
            "Traversed Up - Level {:?} Index: {:?} Parent Index: {}",
            level, index.0, parent_index.0
        );

        let parent_traversed_nodes = if parent_traversal_status == TraversalStatus::Uninitialized {
            self.get_traversed_nodes_up(parent_index, parent_level)
        } else {
            parent_tile.traversed_nodes
        };

        let tile = self.get_tile_mut(index, level);

        let mut upscaled_traversed_nodes =
            parent_traversed_nodes.upscale_octant(index.child_octant());
        upscaled_traversed_nodes.0 &= tile.traversable_nodes.0;

        tile.traversed_nodes = upscaled_traversed_nodes;
        tile.traversal_status = TraversalStatus::Downmipped;

        upscaled_traversed_nodes
    }

    fn get_tile_mut(&mut self, index: LocalTileIndex, level: u8) -> &mut Tile {
        debug_assert!(level <= Graph::HIGHEST_LEVEL, "Level {level} out of bounds");

        unsafe {
            self.tile_levels
                .get_unchecked_mut(level as usize)
                .get_unchecked_mut(index.to_usize())
        }
    }

    fn get_tile(&self, index: LocalTileIndex, level: u8) -> &Tile {
        debug_assert!(level <= Graph::HIGHEST_LEVEL, "Level {level} out of bounds");

        unsafe {
            self.tile_levels
                .get_unchecked(level as usize)
                .get_unchecked(index.to_usize())
        }
    }

    // TODO: should this reset the traversal status of the nodes it affects? that'll
    // get reset anyway, right?
    pub fn set_section(&mut self, section_coords: i32x3, traversable_block_bytes: &[u8; 512]) {
        #[cfg(debug_assertions)]
        println!("Set Section - Coords: {:?}", section_coords);

        let level_1_coords = self.coord_space.section_to_local_coords(section_coords);
        let level_1_index = self.coord_space.pack_index(level_1_coords, 1);
        let level_1_tile = self.get_tile_mut(level_1_index, 1);

        let level_1_prev_traversable_nodes = level_1_tile.traversable_nodes;
        let level_1_prev_traversable_blocks_count = level_1_tile.traversable_blocks_count;
        let level_1_prev_should_traverse_children = level_1_tile.should_traverse_children;

        let mut level_1_should_traverse_children = false;
        let mut level_1_traversable_blocks_count = 0;

        for (level_0_index, tile_bytes) in level_1_index
            .unordered_child_iter()
            .zip(traversable_block_bytes.chunks_exact(64))
        {
            let child_octant = level_0_index.child_octant();
            let level_0_tile = self.get_tile_mut(level_0_index, 0);

            let level_0_traversable_nodes = NodeStorage(u8x64::from_slice(tile_bytes));
            let level_0_traversable_blocks_count = level_0_traversable_nodes.population() as u32;

            level_0_tile.traversable_nodes = level_0_traversable_nodes;
            // TODO: nothing currently uses this for level 0 nodes, consider omitting
            level_0_tile.traversable_blocks_count = level_0_traversable_blocks_count;
            level_1_traversable_blocks_count += level_0_traversable_blocks_count;

            let level_1_tile = self.get_tile_mut(level_1_index, 1);
            let downscaled_population = level_1_tile
                .traversable_nodes
                .downscale_to_octant(level_0_traversable_nodes, child_octant);

            level_1_should_traverse_children |=
                Tile::should_traverse(0, level_0_traversable_blocks_count, downscaled_population);
        }

        let level_1_tile = self.get_tile_mut(level_1_index, 1);
        level_1_tile.should_traverse_children = level_1_should_traverse_children;
        // TODO: nothing currently uses this for level 1 nodes, consider omitting
        level_1_tile.traversable_blocks_count = level_1_traversable_blocks_count;

        let level_1_traversable_nodes = level_1_tile.traversable_nodes;

        if level_1_traversable_nodes != level_1_prev_traversable_nodes
            || level_1_should_traverse_children != level_1_prev_should_traverse_children
            || level_1_traversable_blocks_count != level_1_prev_traversable_blocks_count
        {
            self.propagate_tile_change_up(
                level_1_index,
                1,
                level_1_traversable_nodes,
                level_1_prev_traversable_blocks_count,
                level_1_traversable_blocks_count,
                level_1_should_traverse_children,
            );
        }
    }

    // TODO: skip parts of this when possible, or split this out into multiple
    // methods
    pub fn propagate_tile_change_up(
        &mut self,
        index: LocalTileIndex,
        level: u8,
        traversable_nodes: NodeStorage,
        prev_traversable_blocks_count: u32,
        traversable_blocks_count: u32,
        needs_traversal: bool,
    ) {
        #[cfg(debug_assertions)]
        println!("Propagate Section - Index: {0}, {0:#b}", index.0);

        let child_octant = index.child_octant();
        let parent_index = index.to_parent_level();
        let parent_level = level + 1;
        let parent_tile = self.get_tile_mut(parent_index, parent_level);

        let parent_prev_traversable_blocks_count = parent_tile.traversable_blocks_count;
        let parent_traversable_blocks_count = parent_prev_traversable_blocks_count
            - prev_traversable_blocks_count
            + traversable_blocks_count;
        parent_tile.traversable_blocks_count = parent_traversable_blocks_count;

        let parent_prev_traversable_nodes = parent_tile.traversable_nodes;
        let downscaled_population = parent_tile
            .traversable_nodes
            .downscale_to_octant(traversable_nodes, child_octant);
        let parent_traversable_nodes = parent_tile.traversable_nodes;

        let parent_prev_should_traverse_children = parent_tile.should_traverse_children;
        let parent_should_traverse_children = parent_prev_should_traverse_children
            || needs_traversal
            || Tile::should_traverse(level, traversable_blocks_count, downscaled_population);
        parent_tile.should_traverse_children = parent_should_traverse_children;

        if parent_level < Self::HIGHEST_LEVEL
            && (parent_tile.traversable_nodes != parent_prev_traversable_nodes
                || parent_tile.should_traverse_children != parent_prev_should_traverse_children
                || parent_tile.traversable_blocks_count != parent_prev_traversable_blocks_count)
        {
            self.propagate_tile_change_up(
                parent_index,
                parent_level,
                parent_traversable_nodes,
                parent_prev_traversable_blocks_count,
                parent_traversable_blocks_count,
                parent_should_traverse_children,
            );
        }
    }

    pub fn remove_section(&mut self, section_coord: i32x3) {
        // TODO: write this directly to just set the tiles to Default::default()
        // TODO: can the level 0 nodes just be kept the same, and the level 1 children
        // just be toggled off?
        self.set_section(section_coord, &[0xFF; 512]);
    }
}
