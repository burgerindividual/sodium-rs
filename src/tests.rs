#![cfg(test)]

use std::collections::HashSet;

use core_simd::simd::*;
use num::SimdUint;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use crate::graph::coords::{GraphCoordSpace, LocalTileCoords, SortedChildIterator};
use crate::graph::direction::*;
use crate::graph::tile::NodeStorage;
use crate::math::Coords3;

const RANDOM_SEED: u64 = 8427234087098706983;

// #[test]
// fn pack_local_tile_index() {
//     for x in 0..=255 {
//         for y in 0..=255 {
//             for z in 0..=255 {
//                 let local_coord = LocalTileCoords::from_xyz(x, y, z);

//                 let index = LocalTileIndex::<0>::pack(local_coord);
//                 let unpacked = index.unpack();

//                 assert_eq!(unpacked, local_coord);
//             }
//         }
//     }
// }

#[test]
fn upscale_tile() {
    const ITERATIONS: u32 = 10000;
    let mut rand = StdRng::seed_from_u64(RANDOM_SEED);

    for src_octant in 0..8 {
        for _ in 0..ITERATIONS {
            let mut src_node_storage = NodeStorage::EMPTY;

            rand.fill_bytes(src_node_storage.0.as_mut_array());

            let quad_x_mod = if src_octant & 0b100 == 0 { 0 } else { 4 };
            let quad_y_mod = if src_octant & 0b010 == 0 { 0 } else { 4 };
            let quad_z_mod = if src_octant & 0b001 == 0 { 0 } else { 4 };

            let dst_node_storage = src_node_storage.upscale_octant(src_octant);

            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        let src_idx =
                            NodeStorage::index(x + quad_x_mod, y + quad_y_mod, z + quad_z_mod);
                        let src_val = src_node_storage.get_bit(src_idx);

                        for z2 in 0..2 {
                            for y2 in 0..2 {
                                for x2 in 0..2 {
                                    let dst_idx = NodeStorage::index(
                                        (x * 2) + x2,
                                        (y * 2) + y2,
                                        (z * 2) + z2,
                                    );
                                    let dst_val = dst_node_storage.get_bit(dst_idx);
                                    assert_eq!(src_val, dst_val);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn downscale_tile() {
    const ITERATIONS: u32 = 10000;
    let mut rand = StdRng::seed_from_u64(RANDOM_SEED);

    for dst_octant in 0..8 {
        for _ in 0..ITERATIONS {
            let mut src_node_storage = NodeStorage::EMPTY;
            let mut dst_node_storage = NodeStorage::EMPTY;

            rand.fill_bytes(src_node_storage.0.as_mut_array());
            rand.fill_bytes(dst_node_storage.0.as_mut_array());

            let mut dst_node_storage_modified = dst_node_storage;

            dst_node_storage_modified.downscale_to_octant(src_node_storage, dst_octant);

            for z in 0..8 {
                for y in 0..8 {
                    for x in 0..8 {
                        let cur_octant = (x & 0b100) | ((y & 0b100) >> 1) | ((z & 0b100) >> 2);
                        let dst_idx = NodeStorage::index(x, y, z);
                        let dst_val_modified = dst_node_storage_modified.get_bit(dst_idx);

                        if cur_octant == dst_octant {
                            let mut src_val = false;
                            for z2 in 0..2 {
                                for y2 in 0..2 {
                                    for x2 in 0..2 {
                                        let src_idx = NodeStorage::index(
                                            ((x & 0b11) * 2) + x2,
                                            ((y & 0b11) * 2) + y2,
                                            ((z & 0b11) * 2) + z2,
                                        );
                                        src_val |= src_node_storage.get_bit(src_idx);
                                    }
                                }
                            }
                            assert_eq!(dst_val_modified, src_val,
                                "Source and Destination don't match at {dst_idx:b}, x: {x} y: {y} z: {z}"
                            );
                        } else {
                            let orig_val_modified = dst_node_storage.get_bit(dst_idx);
                            assert_eq!(dst_val_modified, orig_val_modified,
                                "Destination and Original don't match at {dst_idx}, x: {x} y: {y} z: {z}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn shifts_test() {
    const ITERATIONS: u32 = 10000;
    let mut rand = StdRng::seed_from_u64(RANDOM_SEED);

    for _ in 0..ITERATIONS {
        let mut src = NodeStorage(u8x64::splat(0));

        rand.fill_bytes(src.0.as_mut_array());

        {
            let mut dst_sane_neg_x = NodeStorage(u8x64::splat(0));
            for z in 0..8 {
                for y in 0..8 {
                    for x in 1..8 {
                        dst_sane_neg_x.put_bit(
                            NodeStorage::index(x - 1, y, z),
                            src.get_bit(NodeStorage::index(x, y, z)),
                        );
                    }
                }
            }

            let dst_test_neg_x = NodeStorage::shift_neg_x(src.0);

            assert_eq!(dst_sane_neg_x.0, dst_test_neg_x);
        }

        {
            let mut dst_sane_pos_x = NodeStorage(u8x64::splat(0));
            for z in 0..8 {
                for y in 0..8 {
                    for x in 0..7 {
                        dst_sane_pos_x.put_bit(
                            NodeStorage::index(x + 1, y, z),
                            src.get_bit(NodeStorage::index(x, y, z)),
                        );
                    }
                }
            }

            let dst_test_pos_x = NodeStorage::shift_pos_x(src.0);

            assert_eq!(dst_sane_pos_x.0, dst_test_pos_x);
        }

        {
            let mut dst_sane_neg_y = NodeStorage(u8x64::splat(0));
            for z in 0..8 {
                for y in 1..8 {
                    for x in 0..8 {
                        dst_sane_neg_y.put_bit(
                            NodeStorage::index(x, y - 1, z),
                            src.get_bit(NodeStorage::index(x, y, z)),
                        );
                    }
                }
            }

            let dst_test_neg_y = NodeStorage::shift_neg_y(src.0);

            assert_eq!(dst_sane_neg_y.0, dst_test_neg_y);
        }

        {
            let mut dst_sane_pos_y = NodeStorage(u8x64::splat(0));
            for z in 0..8 {
                for y in 0..7 {
                    for x in 0..8 {
                        dst_sane_pos_y.put_bit(
                            NodeStorage::index(x, y + 1, z),
                            src.get_bit(NodeStorage::index(x, y, z)),
                        );
                    }
                }
            }

            let dst_test_pos_y = NodeStorage::shift_pos_y(src.0);

            assert_eq!(dst_sane_pos_y.0, dst_test_pos_y);
        }

        {
            let mut dst_sane_neg_z = NodeStorage(u8x64::splat(0));
            for z in 1..8 {
                for y in 0..8 {
                    for x in 0..8 {
                        dst_sane_neg_z.put_bit(
                            NodeStorage::index(x, y, z - 1),
                            src.get_bit(NodeStorage::index(x, y, z)),
                        );
                    }
                }
            }

            let dst_test_neg_z = NodeStorage::shift_neg_z(src.0);

            assert_eq!(dst_sane_neg_z.0, dst_test_neg_z);
        }

        {
            let mut dst_sane_pos_z = NodeStorage(u8x64::splat(0));
            for z in 0..7 {
                for y in 0..8 {
                    for x in 0..8 {
                        dst_sane_pos_z.put_bit(
                            NodeStorage::index(x, y, z + 1),
                            src.get_bit(NodeStorage::index(x, y, z)),
                        );
                    }
                }
            }

            let dst_test_pos_z = NodeStorage::shift_pos_z(src.0);

            assert_eq!(dst_sane_pos_z.0, dst_test_pos_z);
        }
    }
}

#[test]
fn edge_move_test() {
    const ITERATIONS: u32 = 10000;
    let mut rand = StdRng::seed_from_u64(RANDOM_SEED);

    for _ in 0..ITERATIONS {
        let mut src = NodeStorage(u8x64::splat(0));

        rand.fill_bytes(src.0.as_mut_array());

        {
            let mut dst_sane_neg_to_pos_x = NodeStorage(u8x64::splat(0));

            for z in 0..8 {
                for y in 0..8 {
                    dst_sane_neg_to_pos_x.put_bit(
                        NodeStorage::index(7, y, z),
                        src.get_bit(NodeStorage::index(0, y, z)),
                    );
                }
            }

            let dst_test_neg_to_pos_x = NodeStorage::edge_neg_to_pos_x(src.0);

            assert_eq!(dst_sane_neg_to_pos_x.0, dst_test_neg_to_pos_x);
        }

        {
            let mut dst_sane_pos_to_neg_x = NodeStorage(u8x64::splat(0));

            for z in 0..8 {
                for y in 0..8 {
                    dst_sane_pos_to_neg_x.put_bit(
                        NodeStorage::index(0, y, z),
                        src.get_bit(NodeStorage::index(7, y, z)),
                    );
                }
            }

            let dst_test_pos_to_neg_x = NodeStorage::edge_pos_to_neg_x(src.0);

            assert_eq!(dst_sane_pos_to_neg_x.0, dst_test_pos_to_neg_x);
        }

        {
            let mut dst_sane_neg_to_pos_y = NodeStorage(u8x64::splat(0));

            for z in 0..8 {
                for x in 0..8 {
                    dst_sane_neg_to_pos_y.put_bit(
                        NodeStorage::index(x, 7, z),
                        src.get_bit(NodeStorage::index(x, 0, z)),
                    );
                }
            }

            let dst_test_neg_to_pos_y = NodeStorage::edge_neg_to_pos_y(src.0);

            assert_eq!(dst_sane_neg_to_pos_y.0, dst_test_neg_to_pos_y);
        }

        {
            let mut dst_sane_pos_to_neg_y = NodeStorage(u8x64::splat(0));

            for z in 0..8 {
                for x in 0..8 {
                    dst_sane_pos_to_neg_y.put_bit(
                        NodeStorage::index(x, 0, z),
                        src.get_bit(NodeStorage::index(x, 7, z)),
                    );
                }
            }

            let dst_test_pos_to_neg_y = NodeStorage::edge_pos_to_neg_y(src.0);

            assert_eq!(dst_sane_pos_to_neg_y.0, dst_test_pos_to_neg_y);
        }

        {
            let mut dst_sane_neg_to_pos_z = NodeStorage(u8x64::splat(0));

            for y in 0..8 {
                for x in 0..8 {
                    dst_sane_neg_to_pos_z.put_bit(
                        NodeStorage::index(x, y, 7),
                        src.get_bit(NodeStorage::index(x, y, 0)),
                    );
                }
            }

            let dst_test_neg_to_pos_z = NodeStorage::edge_neg_to_pos_z(src.0);

            assert_eq!(dst_sane_neg_to_pos_z.0, dst_test_neg_to_pos_z);
        }

        {
            let mut dst_sane_pos_to_neg_z = NodeStorage(u8x64::splat(0));

            for y in 0..8 {
                for x in 0..8 {
                    dst_sane_pos_to_neg_z.put_bit(
                        NodeStorage::index(x, y, 0),
                        src.get_bit(NodeStorage::index(x, y, 7)),
                    );
                }
            }

            let dst_test_pos_to_neg_z = NodeStorage::edge_pos_to_neg_z(src.0);

            assert_eq!(dst_sane_pos_to_neg_z.0, dst_test_pos_to_neg_z);
        }
    }
}

// TODO: actually test the order here
#[test]
fn sorted_iter_test() {
    let coord_space = GraphCoordSpace::new(6, 6, 6, -4, 19);
    let coords = LocalTileCoords(Simd::from_xyz(10, 15, 31));
    let level = 1;
    let index = coord_space.pack_index(coords, level);
    let camera_block_coords = coord_space.block_to_local_coords(Simd::from_xyz(10, 30, 50));
    let children_present = 0b10101111;
    let child_level = level - 1;

    let mut sorted_indices = HashSet::new();
    let mut unsorted_indices = HashSet::new();

    let iter_sorted = SortedChildIterator::new(
        index,
        coords,
        level,
        camera_block_coords.cast::<i16>(),
        children_present,
    );
    for (child_index, child_coords) in iter_sorted {
        let calculated_child_index = coord_space.pack_index(child_coords, child_level);
        assert_eq!(calculated_child_index, child_index);
        println!("{:?} {:?}", child_coords, child_index);
        sorted_indices.insert(child_index);
    }

    let iter_unsorted = index.unordered_child_iter(children_present);
    for child_index in iter_unsorted {
        unsorted_indices.insert(child_index);
    }

    assert_eq!(sorted_indices, unsorted_indices);
}

// TODO: make this automatic
#[test]
fn step_test() {
    let coord_space = GraphCoordSpace::new(6, 6, 6, -4, 19);
    let coords = LocalTileCoords(Simd::from_xyz(10, 15, 31));

    let mut direction_set = ALL_DIRECTIONS;
    while direction_set != 0 {
        let direction = take_one(&mut direction_set);
        let stepped = coord_space.step(coords, direction);
        println!("{} {:?}", to_str(direction), stepped);
    }
}

#[test]
fn hmm_test() {
    let coord_space = GraphCoordSpace::new(4, 6, 4, -4, 19);
    let coords = LocalTileCoords(Simd::from_xyz(9, 23, 8));
    let stepped = coord_space.step(coords, POS_Z);
    let stepped_index = coord_space.pack_index(stepped, 0);
    println!("{:#026b}", stepped_index.0);
}

// TODO: test clearing the graph, test searching traversed nodes, test axis and
// plane masks
