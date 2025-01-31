#![cfg(test)]

use std::collections::HashMap;

use core_simd::simd::prelude::*;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use crate::bitset::BitSet;
use crate::graph::context::LocalFrustum;
use crate::graph::coords::{GraphCoordSpace, LocalTileCoords, LocalTileIndex};
use crate::graph::direction::*;
use crate::graph::tile::*;
use crate::math::{u8x3, Coords3};

const RANDOM_SEED: u64 = 8427234087098706983;

#[test]
fn pack_index_test() {
    let graph_y_bits = 2;
    let graph_xz_bits = 3;

    let graph_y_len_tiles = 1 << graph_y_bits;
    let graph_xz_len_tiles = 1 << graph_xz_bits;

    let coord_space = GraphCoordSpace::new(graph_xz_bits, graph_y_bits, graph_xz_bits, -4, 19);
    let mut index_coords_map = HashMap::<LocalTileIndex, LocalTileCoords>::new();

    for x in 0..graph_xz_len_tiles {
        for y in 0..graph_y_len_tiles {
            for z in 0..graph_xz_len_tiles {
                let coords = LocalTileCoords::from_xyz(x, y, z);
                let index = coord_space.pack_index(coords);

                let entry = index_coords_map.get(&index);
                if let Some(&existing_coords) = entry {
                    panic!(
                        "Duplicate Tile Index Found: {:?}\nCoords: {:?} and {:?}",
                        index.0, existing_coords.0, coords.0
                    );
                } else {
                    index_coords_map.insert(index, coords);
                }
            }
        }
    }

    // test wrapping on edges
}

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
                        sane_camera_direction_masks[dir_idx],
                        test_camera_direction_masks[dir_idx],
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

// TODO: automate this
#[test]
fn frustum_voxelization_test() {
    let relative_tile_coords = Simd::from_xyz(-552.477356, -55.7096558, 59.6260223);

    // TODO: the order of these is wrong
    let frustum = LocalFrustum::new([
        Simd::from_array([-0.24678199, -0.241355747, -0.938533962, 0.0]),
        Simd::from_array([-0.594573379, 0.415058464, -0.688628316, 0.0]),
        Simd::from_array([-0.629826427, -0.266851544, -0.729458034, -0.0500000082]),
        Simd::from_array([-0.892519951, -0.241355777, -0.380992979, -0.0]),
        Simd::from_array([-0.370376676, -0.823898673, -0.428966165, -0.0]),
        Simd::from_array([0.629539371, 0.267188221, 0.729582489, 2046.93347]),
    ]);

    let mut failed = false;
    let mut directions = ALL_DIRECTIONS;
    while directions != 0 {
        let direction = take_one(&mut directions);
        let dir_idx = to_index(direction);

        let sane_visible_sections =
            voxelize_frustum_plane_slow(relative_tile_coords, frustum.planes[dir_idx]);
        let test_visible_sections = voxelize_frustum_plane(
            relative_tile_coords,
            frustum.planes_scaled[dir_idx],
            frustum.planes_bb_offsets[dir_idx],
        );

        if test_visible_sections == sane_visible_sections {
            continue;
        } else {
            failed = true;
        }

        let dir_str = to_str(direction);

        println!("Plane {dir_str} - Sane");
        print_tile(&sane_visible_sections);

        println!("Plane {dir_str} - Test");
        print_tile(&test_visible_sections);
    }

    if failed {
        panic!();
    }
}

// TODO: test clearing the graph, test searching traversed nodes, test axis and
// plane masks, test sorted child iterator, test packing indices
