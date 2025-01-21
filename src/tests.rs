#![cfg(test)]

use core_simd::simd::prelude::*;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use crate::bitset::BitSet;
use crate::graph::coords::LocalTileCoords;
use crate::graph::direction::*;
use crate::graph::tile::*;
use crate::math::{u8x3, Coords3};

const RANDOM_SEED: u64 = 8427234087098706983;

// #[test]
// fn pack_tile_index() {
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
fn shifts_test() {
    const ITERATIONS: u32 = 10000;
    let mut rand = StdRng::seed_from_u64(RANDOM_SEED);

    for _ in 0..ITERATIONS {
        let mut src = u8x64::splat(0);

        rand.fill_bytes(src.as_mut_array());

        {
            let mut dst_sane_neg_x = u8x64::splat(0);
            for z in 0..8 {
                for y in 0..8 {
                    for x in 1..8 {
                        modify_bit(
                            &mut dst_sane_neg_x,
                            section_index(Simd::from_xyz(x - 1, y, z)),
                            get_bit(&src, section_index(Simd::from_xyz(x, y, z))),
                        );
                    }
                }
            }

            let dst_test_neg_x = shift_neg_x(src);

            assert_eq!(dst_sane_neg_x, dst_test_neg_x);
        }

        {
            let mut dst_sane_pos_x = u8x64::splat(0);
            for z in 0..8 {
                for y in 0..8 {
                    for x in 0..7 {
                        modify_bit(
                            &mut dst_sane_pos_x,
                            section_index(Simd::from_xyz(x + 1, y, z)),
                            get_bit(&src, section_index(Simd::from_xyz(x, y, z))),
                        );
                    }
                }
            }

            let dst_test_pos_x = shift_pos_x(src);

            assert_eq!(dst_sane_pos_x, dst_test_pos_x);
        }

        {
            let mut dst_sane_neg_y = u8x64::splat(0);
            for z in 0..8 {
                for y in 1..8 {
                    for x in 0..8 {
                        modify_bit(
                            &mut dst_sane_neg_y,
                            section_index(Simd::from_xyz(x, y - 1, z)),
                            get_bit(&src, section_index(Simd::from_xyz(x, y, z))),
                        );
                    }
                }
            }

            let dst_test_neg_y = shift_neg_y(src);

            assert_eq!(dst_sane_neg_y, dst_test_neg_y);
        }

        {
            let mut dst_sane_pos_y = u8x64::splat(0);
            for z in 0..8 {
                for y in 0..7 {
                    for x in 0..8 {
                        modify_bit(
                            &mut dst_sane_pos_y,
                            section_index(Simd::from_xyz(x, y + 1, z)),
                            get_bit(&src, section_index(Simd::from_xyz(x, y, z))),
                        );
                    }
                }
            }

            let dst_test_pos_y = shift_pos_y(src);

            assert_eq!(dst_sane_pos_y, dst_test_pos_y);
        }

        {
            let mut dst_sane_neg_z = u8x64::splat(0);
            for z in 1..8 {
                for y in 0..8 {
                    for x in 0..8 {
                        modify_bit(
                            &mut dst_sane_neg_z,
                            section_index(Simd::from_xyz(x, y, z - 1)),
                            get_bit(&src, section_index(Simd::from_xyz(x, y, z))),
                        );
                    }
                }
            }

            let dst_test_neg_z = shift_neg_z(src);

            assert_eq!(dst_sane_neg_z, dst_test_neg_z);
        }

        {
            let mut dst_sane_pos_z = u8x64::splat(0);
            for z in 0..7 {
                for y in 0..8 {
                    for x in 0..8 {
                        modify_bit(
                            &mut dst_sane_pos_z,
                            section_index(Simd::from_xyz(x, y, z + 1)),
                            get_bit(&src, section_index(Simd::from_xyz(x, y, z))),
                        );
                    }
                }
            }

            let dst_test_pos_z = shift_pos_z(src);

            assert_eq!(dst_sane_pos_z, dst_test_pos_z);
        }
    }
}

#[test]
fn edge_move_test() {
    const ITERATIONS: u32 = 10000;
    let mut rand = StdRng::seed_from_u64(RANDOM_SEED);

    for _ in 0..ITERATIONS {
        let mut src = u8x64::splat(0);

        rand.fill_bytes(src.as_mut_array());

        {
            let mut dst_sane_neg_to_pos_x = u8x64::splat(0);

            for z in 0..8 {
                for y in 0..8 {
                    modify_bit(
                        &mut dst_sane_neg_to_pos_x,
                        section_index(Simd::from_xyz(7, y, z)),
                        get_bit(&src, section_index(Simd::from_xyz(0, y, z))),
                    );
                }
            }

            let dst_test_neg_to_pos_x = edge_neg_to_pos_x(src);

            assert_eq!(dst_sane_neg_to_pos_x, dst_test_neg_to_pos_x);
        }

        {
            let mut dst_sane_pos_to_neg_x = u8x64::splat(0);

            for z in 0..8 {
                for y in 0..8 {
                    modify_bit(
                        &mut dst_sane_pos_to_neg_x,
                        section_index(Simd::from_xyz(0, y, z)),
                        get_bit(&src, section_index(Simd::from_xyz(7, y, z))),
                    );
                }
            }

            let dst_test_pos_to_neg_x = edge_pos_to_neg_x(src);

            assert_eq!(dst_sane_pos_to_neg_x, dst_test_pos_to_neg_x);
        }

        {
            let mut dst_sane_neg_to_pos_y = u8x64::splat(0);

            for z in 0..8 {
                for x in 0..8 {
                    modify_bit(
                        &mut dst_sane_neg_to_pos_y,
                        section_index(Simd::from_xyz(x, 7, z)),
                        get_bit(&src, section_index(Simd::from_xyz(x, 0, z))),
                    );
                }
            }

            let dst_test_neg_to_pos_y = edge_neg_to_pos_y(src);

            assert_eq!(dst_sane_neg_to_pos_y, dst_test_neg_to_pos_y);
        }

        {
            let mut dst_sane_pos_to_neg_y = u8x64::splat(0);

            for z in 0..8 {
                for x in 0..8 {
                    modify_bit(
                        &mut dst_sane_pos_to_neg_y,
                        section_index(Simd::from_xyz(x, 0, z)),
                        get_bit(&src, section_index(Simd::from_xyz(x, 7, z))),
                    );
                }
            }

            let dst_test_pos_to_neg_y = edge_pos_to_neg_y(src);

            assert_eq!(dst_sane_pos_to_neg_y, dst_test_pos_to_neg_y);
        }

        {
            let mut dst_sane_neg_to_pos_z = u8x64::splat(0);

            for y in 0..8 {
                for x in 0..8 {
                    modify_bit(
                        &mut dst_sane_neg_to_pos_z,
                        section_index(Simd::from_xyz(x, y, 7)),
                        get_bit(&src, section_index(Simd::from_xyz(x, y, 0))),
                    );
                }
            }

            let dst_test_neg_to_pos_z = edge_neg_to_pos_z(src);

            assert_eq!(dst_sane_neg_to_pos_z, dst_test_neg_to_pos_z);
        }

        {
            let mut dst_sane_pos_to_neg_z = u8x64::splat(0);

            for y in 0..8 {
                for x in 0..8 {
                    modify_bit(
                        &mut dst_sane_pos_to_neg_z,
                        section_index(Simd::from_xyz(x, y, 0)),
                        get_bit(&src, section_index(Simd::from_xyz(x, y, 7))),
                    );
                }
            }

            let dst_test_pos_to_neg_z = edge_pos_to_neg_z(src);

            assert_eq!(dst_sane_pos_to_neg_z, dst_test_pos_to_neg_z);
        }
    }
}

#[test]
fn direction_mask_test() {
    for camera_x in 0..8 {
        for camera_y in 0..8 {
            for camera_z in 0..8 {
                let camera_tile_coords = u8x3::from_xyz(camera_x, camera_y, camera_z);

                let mut sane_camera_direction_masks = [SECTIONS_EMPTY; DIRECTION_COUNT];

                for tile_x in 0..8 {
                    for tile_y in 0..8 {
                        for tile_z in 0..8 {
                            let other_tile_coords = Simd::from_xyz(tile_x, tile_y, tile_z);

                            let negative = other_tile_coords.simd_le(camera_tile_coords);
                            let positive = other_tile_coords.simd_ge(camera_tile_coords);
                            let traversal_directions =
                                negative.to_bitmask() as u8 | ((positive.to_bitmask() as u8) << 3);

                            let section_idx = section_index(other_tile_coords);
                            for dir_idx in 0..6 {
                                modify_bit(
                                    &mut sane_camera_direction_masks[dir_idx as usize],
                                    section_idx,
                                    traversal_directions.get_bit(dir_idx),
                                );
                            }
                        }
                    }
                }

                let test_camera_direction_masks = create_camera_direction_masks(camera_tile_coords);

                let mut directions = ALL_DIRECTIONS;
                while directions != 0 {
                    let direction = take_one(&mut directions);
                    let dir_idx = to_index(direction);
                    assert_eq!(
                        sane_camera_direction_masks[dir_idx], test_camera_direction_masks[dir_idx],
                        "sane != test, Camera Coords: {:?}, Direction: {}",
                        camera_tile_coords,
                        to_str(direction)
                    );
                }
            }
        }
    }
}

// TODO: make this automatic
#[test]
fn step_test() {
    let coords = LocalTileCoords(Simd::from_xyz(10, 15, 31));

    let mut direction_set = ALL_DIRECTIONS;
    while direction_set != 0 {
        let direction = take_one(&mut direction_set);
        let stepped = coords.step(direction);
        println!("{} {:?}", to_str(direction), stepped);
    }
}

// TODO: test clearing the graph, test searching traversed nodes, test axis and
// plane masks, test sorted child iterator, test packing indices
