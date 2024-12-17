use std::array;
use std::collections::HashSet;
use std::hint::unreachable_unchecked;

use core_simd::simd::Simd;

use crate::graph::coords::LocalTileCoords;
use crate::graph::tile::NodeStorage;

// TODO OPT: organize into 8x4x8 section regions, represented as 256-bit masks
// pub struct SectionBitArray {
//     pub section_count: u32,
//     pub data: Box<[u64]>,

//     graph_origin_offset: u16x3,
//     dimensions: u16x3,
// }

// impl SectionBitArray {
//     pub fn new() -> Self {
//         Self {
//             section_count: todo!(),
//             data: todo!(),
//             graph_origin_offset: todo!(),
//             dimensions: todo!(),
//         }
//     }

//     pub fn clear(&mut self) {
//         self.data.fill(0);
//     }

//     pub fn set_tile(&mut self, coords: LocalTileCoords, level: u8) {
//         std::hint::black_box(self);
//         std::hint::black_box(coords);
//         std::hint::black_box(level);
//     }
// }

pub struct SectionBitArray {
    // TODO: this won't work, need to switch to an actual array based solution (i think)
    pub level_sets: [HashSet<LocalTileCoords>; 4],
}

impl SectionBitArray {
    pub fn new() -> Self {
        Self {
            level_sets: array::from_fn(|_| HashSet::new()),
        }
    }

    pub fn clear(&mut self) {
        for set in self.level_sets.iter_mut() {
            set.clear();
        }
    }

    pub fn set_tile(
        &mut self,
        coords: LocalTileCoords,
        level: u8,
        traversal_data: NodeStorage,
        child_mask: u8,
    ) {
        match level {
            0 => {
                let parent_coords = coords.to_parent_level();

                if traversal_data.0 != Simd::splat(0) {
                    self.level_sets[0].insert(parent_coords);
                }
            }
            1 => {
                if traversal_data.0 != Simd::splat(0) {
                    self.level_sets[0].insert(coords);
                }
            }
            2 => {}
            3 => {}
            4 => {
                // each node in the tile is the size of 1 section
            }
            _ => unsafe { unreachable_unchecked() },
        }
    }

    pub fn get_count(&self) -> u32 {
        let mut count: u32 = 0;

        for (level, set) in self.level_sets.iter().enumerate() {
            count += set.len() as u32 * (level as u32 + 1).pow(3)
        }

        count
    }

    pub fn verify(&self) {
        // TODO
    }
}
