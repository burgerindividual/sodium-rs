use context::GraphSearchContext;
use coords::{GraphCoordSpace, LocalTileIndex};
use core_simd::simd::prelude::*;
use tile::{NodeStorage, Tile, TraversalStatus};

use self::coords::LocalTileCoords;
use crate::bit_array::SectionBitArray;
use crate::math::*;
use crate::unreachable_debug;
use crate::unwrap_debug;

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

    coord_space: GraphCoordSpace,

    render_distance: u8,

    pub results: SectionBitArray,
}

impl Graph {
    const HIGHEST_LEVEL: u8 = 4;

    pub fn new() -> Self {
        Self {
            level_0: (),
            level_1: (),
            level_2: (),
            level_3: (),
            level_4: (),
            coord_space: GraphCoordSpace::new(x_bits, y_bits, z_bits),
            results: Vec::with_capacity(64),
        }
    }

    pub fn cull(&mut self, context: &GraphSearchContext) {
        self.clear();

        self.iterate_tiles(context);
    }

    pub fn clear(&mut self) {
        self.results.clear();
        // TODO: reset all traversal status
    }

    fn iterate_tiles(&mut self, context: &GraphSearchContext) {}

    fn process_tile(&mut self, context: &GraphSearchContext, index: LocalTileCoords, level: u8) {
        // Test frustum and fog first, before touching any edge data
        // TODO OPT: if all input edges are blank, immediately mark the tile skipped and move on
    }

    pub fn get_incoming_edge<const DIRECTION: u8>(
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
            TraversalStatus::Traversed { children_upmipped } => todo!(),
            TraversalStatus::Upmipped { children_upmipped } => todo!(),
        };
    }

    // the upward search here assumes that there is an upper level which has been traversed.
    fn get_traversed_nodes_down<const DIRECTION: u8>(
        &mut self,
        index: LocalTileIndex,
        level: u8,
    ) -> NodeStorage {
        let parent_index = index.to_parent_level();
        let parent_level = level + 1;
        let parent_tile = self.get_tile(parent_index, parent_level);
        let parent_traversal_status = parent_tile.traversal_status;

        debug_assert!(
            !matches!(
                parent_traversal_status,
                TraversalStatus::Upmipped {
                    children_upmipped: _
                }
            ),
            "Upward traversal should not encounter tile of type \"Upmipped\". Tile Found: {:?}",
            parent_tile,
        );

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

    // the upward search here assumes that there is an upper level which has been traversed.
    fn get_traversed_nodes_up(&mut self, index: LocalTileIndex, level: u8) -> NodeStorage {
        let parent_index = index.to_parent_level();
        let parent_level = level + 1;
        let parent_tile = self.get_tile(parent_index, parent_level);
        let parent_traversal_status = parent_tile.traversal_status;

        debug_assert!(
            !matches!(
                parent_traversal_status,
                TraversalStatus::Upmipped {
                    children_upmipped: _
                }
            ),
            "Upward traversal should not encounter tile of type \"Upmipped\". Tile Found: {:?}",
            parent_tile,
        );

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
                0 => unwrap_debug!(self.level_0.get_mut(index.to_usize())),
                1 => unwrap_debug!(self.level_1.get_mut(index.to_usize())),
                2 => unwrap_debug!(self.level_2.get_mut(index.to_usize())),
                3 => unwrap_debug!(self.level_3.get_mut(index.to_usize())),
                4 => unwrap_debug!(self.level_4.get_mut(index.to_usize())),
                _ => unreachable_debug!(),
            }
        }
    }

    fn get_tile(&self, index: LocalTileIndex, level: u8) -> &Tile {
        unsafe {
            match level {
                0 => unwrap_debug!(self.level_0.get(index.to_usize())),
                1 => unwrap_debug!(self.level_1.get(index.to_usize())),
                2 => unwrap_debug!(self.level_2.get(index.to_usize())),
                3 => unwrap_debug!(self.level_3.get(index.to_usize())),
                4 => unwrap_debug!(self.level_4.get(index.to_usize())),
                _ => unreachable_debug!(),
            }
        }
    }

    // TODO: should this reset the traversal status of the nodes it affects? that'll get reset
    // anyway, right?
    // NOTE: not thread safe, make sure
    pub fn set_section(&mut self, section_coords: i32x3, opaque_block_bytes: &[u8; 512]) {
        let local_coords = self.coord_space.section_to_local_coords(section_coords);
        let level_1_index = self.coord_space.pack_index(local_coords);
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
        // a removed section should be fully opaque to prevent traversal (maybe don't do this? hm.)
        // TODO: write this directly to just set the tiles to Default::default()
        // TODO: can the level 0 nodes just be kept the same, and the level 1 children just be
        // toggled off?
        self.set_section(section_coord, &[0xFF; 512]);
    }
}
