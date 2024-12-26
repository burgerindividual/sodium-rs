use std::array;
use std::hint::{assert_unchecked, unreachable_unchecked};
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

// NOTE: this structure is NOT THREAD SAFE on its own. you must manually do
// synchronization if you will be accessing this structure from multiple
// threads.
pub struct Graph {
    tile_levels: [Box<[Tile]>; 5],

    pub coord_space: GraphCoordSpace,
    pub render_distance: u8,
    upper_tile_tests: CombinedTestResults,

    pub visible_tiles: Vec<FFIVisibleSectionsTile>,
}

impl Graph {
    const HIGHEST_LEVEL: u8 = 4;
    const EARLY_CHECKS_LOWEST_LEVEL: u8 = 1;

    pub fn new(render_distance: u8, world_bottom_section_y: i8, world_top_section_y: i8) -> Self {
        let world_height_level_0 =
            (world_top_section_y as i16 - world_bottom_section_y as i16 + 1) as u16 * 2;
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

        let skip_height_checks =
            world_height_level_0 == (world_height_level_0 & (-1_i16 << Self::HIGHEST_LEVEL) as u16);

        let upper_tile_tests = if skip_height_checks {
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
            upper_tile_tests,
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

    fn iterate_dirs(
        &mut self,
        context: &GraphSearchContext,
        start_coords: LocalTileCoords,
        mut iter_directions: u8,
    ) {
        let direction = take_any(&mut iter_directions);
        println!("Iterating {}", to_str(direction));
        let steps = context.direction_step_counts[to_index(direction) as usize];
        let mut coords = start_coords;

        for _ in 0..steps {
            coords = self
                .coord_space
                .step_wrapping(coords, direction, Self::HIGHEST_LEVEL);

            // if the direction set is empty, we should stop recursing, and start processing
            // tiles
            if iter_directions != 0 {
                self.iterate_dirs(context, coords, iter_directions);
            } else {
                let index = self.coord_space.pack_index(coords);

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
            self.upper_tile_tests,
        );

        let tile = self.get_tile(index, Self::HIGHEST_LEVEL);

        if tile.visible_nodes != NodeStorage::EMPTY {
            self.visible_tiles.push(FFIVisibleSectionsTile::new(
                &raw const tile.visible_nodes,
                coords,
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
        println!(
            "Current Tile - Level {:?} Coords: {:?} Index: {:?}",
            level, coords.0, index.0
        );

        // tile needs to be re-borrowed multiple times in this method, due to borrow
        // checker rules. these should get optimized out.
        let tile = self.get_tile_mut(index, level);

        tile.traversal_status = TraversalStatus::Processed {
            children_upmipped: 0,
        };

        // try to quickly determine whether we need to actually traverse the tile using
        // the frustum, fog, etc
        let test_result = context.test_tile(&self.coord_space, coords, level, parent_test_results);

        let tile = self.get_tile_mut(index, level);

        if test_result == CombinedTestResults::OUTSIDE {
            // early exit
            tile.set_empty();
            return;
        }

        let children_to_traverse = tile.children_to_traverse;

        // if all children are present, we don't have to process this tile, and we can
        // solely process the children.
        if children_to_traverse != 0b11111111 {
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
            // it is visible, so we must mark that accordingly in the results.

            let direction_masks = unsafe { context.direction_masks.get_unchecked(level as usize) };
            tile.find_visible_nodes(
                incoming_traversed_nodes,
                direction_masks,
                traversal_directions,
            );

            // a dedicated set function, if the visible nodes were to be
            // refactored, should be called here
        }

        // this should always be false for level 0 tiles
        if children_to_traverse != 0b00000000 {
            let child_iter = SortedChildIterator::new(
                index,
                coords,
                level,
                context.camera_pos_int,
                children_to_traverse,
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
            let camera_coords_in_tile = (context.camera_pos_int >> Simd::splat(level as u16))
                .cast::<u8>()
                & Simd::splat(0b111);
            let camera_node_index = NodeStorage::index(
                camera_coords_in_tile.x(),
                camera_coords_in_tile.y(),
                camera_coords_in_tile.z(),
            );

            let mut traversed_nodes = NodeStorage::EMPTY;
            traversed_nodes.put_bit(camera_node_index, true);

            // TODO: get rid of this, and find a better way to handle when the camera is
            // inside an non-traversable node
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
                let direction = take_any(&mut incoming_directions);

                combined_incoming_edges |= self.get_incoming_edge(coords, level, direction);
            }

            let tile = self.get_tile(index, level);

            // TODO: is this masking necessary?
            combined_incoming_edges & tile.traversable_nodes.0
        }
    }

    fn get_incoming_edge(&mut self, coords: LocalTileCoords, level: u8, direction: u8) -> u8x64 {
        let neighbor_coords = self.coord_space.step_wrapping(coords, direction, level);
        let neighbor_index = self.coord_space.pack_index(neighbor_coords);
        let neighbor_tile = self.get_tile(neighbor_index, level);
        let neighbor_traversal_status = neighbor_tile.traversal_status;

        let neighbor_traversed_nodes = match neighbor_traversal_status {
            TraversalStatus::Uninitialized => self.get_traversed_nodes_up(neighbor_index, level),
            TraversalStatus::Downmipped => neighbor_tile.traversed_nodes,
            TraversalStatus::Processed {
                children_upmipped: _,
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

        match direction {
            NEG_X => NodeStorage::edge_neg_to_pos_x(neighbor_traversed_nodes),
            NEG_Y => NodeStorage::edge_neg_to_pos_y(neighbor_traversed_nodes),
            NEG_Z => NodeStorage::edge_neg_to_pos_z(neighbor_traversed_nodes),
            POS_X => NodeStorage::edge_pos_to_neg_x(neighbor_traversed_nodes),
            POS_Y => NodeStorage::edge_pos_to_neg_y(neighbor_traversed_nodes),
            POS_Z => NodeStorage::edge_pos_to_neg_z(neighbor_traversed_nodes),
            _ => unsafe { unreachable_unchecked() },
        }
    }

    fn get_traversed_nodes_down(
        &mut self,
        index: LocalTileIndex,
        level: u8,
        edge_mask: u8,
    ) -> NodeStorage {
        println!("Traversed Down - Level {:?} Index: {:?}", level, index.0);

        let tile = self.get_tile(index, level);

        // stop downward traversal for empty nodes, the children are likely left
        // unpopulated
        if tile.is_empty() {
            return NodeStorage::EMPTY;
        }

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
        for child_index in index.unordered_child_iter(children_to_recurse) {
            let child_traversed_nodes =
                self.get_traversed_nodes_down(child_index, child_level, edge_mask);

            traversed_nodes_updated
                .downscale_to_octant(child_traversed_nodes, child_index.child_octant());
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
        upscaled_traversed_nodes.0 &= !tile.traversable_nodes.0;

        tile.traversed_nodes = upscaled_traversed_nodes;
        tile.traversal_status = TraversalStatus::Downmipped;

        upscaled_traversed_nodes
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
    pub fn set_section(&mut self, section_coords: i32x3, traversable_block_bytes: &[u8; 512]) {
        let level_1_coords = self.coord_space.section_to_local_coords(section_coords);
        let level_1_index = self.coord_space.pack_index(level_1_coords);
        let parent_index_high_bits = level_1_index.to_child_level().0;

        let mut level_1_children = 0;

        for (level_1_child, tile_bytes) in traversable_block_bytes.chunks_exact(64).enumerate() {
            let level_0_index = LocalTileIndex(parent_index_high_bits | level_1_child as u32);
            let level_0_traversable_nodes = NodeStorage(u8x64::from_slice(tile_bytes));

            let level_0_tile = &mut self.tile_levels[0][level_0_index.to_usize()];
            level_0_tile.traversable_nodes = level_0_traversable_nodes;

            let level_1_tile = &mut self.tile_levels[1][level_1_index.to_usize()];
            let level_1_traverse_child = level_1_tile
                .traversable_nodes
                .downscale_to_octant(level_0_traversable_nodes, level_1_child as u8);

            level_1_children |= (level_1_traverse_child as u8) << level_1_child;
        }

        let level_1_tile = &mut self.tile_levels[1][level_1_index.to_usize()];
        level_1_tile.children_to_traverse = level_1_children;

        let level_1_traversable_nodes = level_1_tile.traversable_nodes;

        self.propagate_traversable_nodes_up(level_1_index, level_1_traversable_nodes, 1);
    }

    pub fn propagate_traversable_nodes_up(
        &mut self,
        index: LocalTileIndex,
        traversable_nodes: NodeStorage,
        level: u8,
    ) {
        let parent_level = level + 1;
        let parent_index = index.to_parent_level();
        let child_octant = index.child_octant();
        let parent_tile = self.get_tile_mut(parent_index, parent_level);
        let should_traverse_child = parent_tile
            .traversable_nodes
            .downscale_to_octant(traversable_nodes, child_octant);
        // clear slot in children to traverse, then set it to the value returned by the
        // downscale
        parent_tile.children_to_traverse &= !(0b1 << child_octant);
        parent_tile.children_to_traverse |= (should_traverse_child as u8) << child_octant;

        let parent_traversable_nodes = parent_tile.traversable_nodes;

        if parent_level < Self::HIGHEST_LEVEL {
            self.propagate_traversable_nodes_up(
                parent_index,
                parent_traversable_nodes,
                parent_level,
            );
        }
    }

    pub fn remove_section(&mut self, section_coord: i32x3) {
        // TODO: a removed section should be fully opaque to prevent traversal (maybe
        // don't do this? hm.)
        // TODO: write this directly to just set the tiles to Default::default()
        // TODO: can the level 0 nodes just be kept the same, and the level 1 children
        // just be toggled off?
        self.set_section(section_coord, &[0xFF; 512]);
    }
}
