#![cfg(test)]

use core::panic;
use std::collections::HashMap;

use core_simd::simd::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

use crate::bitset::BitSet;
use crate::graph::context::{LocalFrustum, RelativeBoundingBox};
use crate::graph::coords::{GraphCoordSpace, LocalTileCoords, LocalTileIndex};
use crate::graph::direction::*;
use crate::graph::tile::*;
use crate::math::*;

const RANDOM_SEED: u64 = 0x0c41ce821df0e3a9;

#[test]
fn pack_index_test() {
    let storage_distance = 20;
    let y_length_sections = 24_u16;
    let xz_length_sections = (storage_distance as u16 * 2) + 1;

    let y_length_tiles = (y_length_sections.next_multiple_of(8) >> 3).max(2);
    let xz_length_tiles = (xz_length_sections.next_multiple_of(8) >> 3).max(2);

    let graph_total_tiles = y_length_tiles as u32 * (xz_length_tiles as u32).pow(2);

    let coord_space = GraphCoordSpace::new(
        xz_length_tiles as u8,
        y_length_tiles as u8,
        xz_length_tiles as u8,
        -4,
        19,
    );
    let mut index_coords_map = HashMap::<LocalTileIndex, LocalTileCoords>::new();

    for y in 0..y_length_tiles {
        for z in 0..xz_length_tiles {
            for x in 0..xz_length_tiles {
                let coords = LocalTileCoords::from_xyz(x as i8, y as i8, z as i8);
                let index = coord_space.pack_index(coords);

                assert!(
                    (index.0 as u32) < graph_total_tiles,
                    "Index too large. Index: {:#018b}, Max: {:#018b}",
                    index.0,
                    graph_total_tiles
                );

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

    // test a stray out of bounds index to see if it's handled
    {
        let coords = LocalTileCoords::from_xyz(-1, -1, -1);
        let index = coord_space.pack_index(coords);
        assert!(
            (index.0 as u32) < graph_total_tiles,
            "Index too large. Index: {:#018b}, Max: {:#018b}",
            index.0,
            graph_total_tiles
        );
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

                            let section_index = section_index(other_tile_coords);
                            for dir_idx in 0..6 {
                                modify_bit(
                                    &mut sane_camera_direction_masks[dir_idx as usize],
                                    section_index,
                                    traversal_directions.get_bit(dir_idx),
                                );
                            }
                        }
                    }
                }

                let test_camera_direction_masks = gen_outward_direction_masks(camera_tile_coords);

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
    let relative_tile_pos = Simd::from_xyz(-168.475, -183.705, -63.434998);

    let frustum = LocalFrustum::new([
        Simd::from_array([-0.591241, -0.49853715, 0.6339517, 0.0]),
        Simd::from_array([-0.23236583, 0.1140805, 0.96591496, 0.0]),
        Simd::from_array([-0.19515383, -0.55120045, 0.81122935, -0.049999997]),
        Simd::from_array([0.23822449, -0.49853715, 0.8334925, -0.0]),
        Simd::from_array([-0.06662716, -0.9585686, 0.27696052, -0.0]),
        Simd::from_array([0.1951034, 0.55120337, -0.81123954, 512.102]),
    ]);

    let mut failed = false;
    let mut directions = ALL_DIRECTIONS;
    while directions != 0 {
        let direction = take_one(&mut directions);
        let dir_idx = to_index(direction);

        let sane_visible_sections =
            voxelize_frustum_plane_slow(relative_tile_pos, frustum.planes[dir_idx]);
        let test_visible_sections = voxelize_frustum_plane(
            relative_tile_pos,
            frustum.planes_scaled[dir_idx],
            frustum.plane_bb_offsets[dir_idx],
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

#[test]
fn angle_visibility_masks_test() {
    const ITERATIONS: u32 = 10000;
    let mut rand = StdRng::seed_from_u64(RANDOM_SEED);

    for _ in 0..ITERATIONS {
        let relative_tile_pos = Simd::from_xyz(
            // (rand.random_range(-20_i8..20_i8) as f32) * 16.0,
            // (rand.random_range(-20_i8..20_i8) as f32) * 16.0,
            // (rand.random_range(-20_i8..20_i8) as f32) * 16.0,
            rand.random_range(-300.0_f32..300.0_f32),
            rand.random_range(-300.0_f32..300.0_f32),
            rand.random_range(-300.0_f32..300.0_f32),
        );

        let test_masks = gen_angle_visibility_masks(relative_tile_pos);
        let sane_masks = gen_angle_visibility_masks_slow(relative_tile_pos);

        if sane_masks != test_masks {
            println!("Sane X Mask");
            print_tile(&sane_masks[X]);
            println!();
            println!("Sane Y Mask");
            print_tile(&sane_masks[Y]);
            println!();
            println!("Sane Z Mask");
            print_tile(&sane_masks[Z]);
            println!();
            println!("Test X Mask");
            print_tile(&test_masks[X]);
            println!();
            println!("Test Y Mask");
            print_tile(&test_masks[X]);
            println!();
            println!("Test Z Mask");
            print_tile(&test_masks[X]);
            println!();
            panic!(
                "sane != test, Relative Tile Coords: {:?}",
                relative_tile_pos,
            );
        }
    }
}

fn gen_angle_visibility_masks_slow(relative_tile_pos: f32x3) -> [u8x64; 3] {
    let mut x_mask = SECTIONS_FILLED;
    let mut y_mask = SECTIONS_FILLED;
    let mut z_mask = SECTIONS_FILLED;

    for y in 0..8_u8 {
        for z in 0..8_u8 {
            for x in 0..8_u8 {
                let section_coords = Simd::from_xyz(x, y, z);
                let section_index = section_index(section_coords);
                let relative_section_center = relative_tile_pos
                    + Simd::splat(8.0)
                    + (section_coords.cast::<f32>() * Simd::splat(16.0));

                let distances = relative_section_center.abs();

                if distances[X] > distances[Y] || distances[Z] > distances[Y] {
                    clear_bit(&mut y_mask, section_index)
                }
                if distances[X] > distances[Z] || distances[Y] > distances[Z] {
                    clear_bit(&mut z_mask, section_index)
                }
                if distances[Y] > distances[X] || distances[Z] > distances[X] {
                    clear_bit(&mut x_mask, section_index)
                }
            }
        }
    }

    [x_mask, y_mask, z_mask]
}

#[test]
fn fog_voxelization_test() {
    const ITERATIONS: u32 = 10000;
    let mut rand = StdRng::seed_from_u64(RANDOM_SEED);

    for _ in 0..ITERATIONS {
        let relative_tile_pos = Simd::from_xyz(
            // (rand.random_range(-20_i8..20_i8) as f32) * 16.0,
            // (rand.random_range(-20_i8..20_i8) as f32) * 16.0,
            // (rand.random_range(-20_i8..20_i8) as f32) * 16.0,
            rand.random_range(-3000.0_f32..3000.0_f32),
            rand.random_range(-3000.0_f32..3000.0_f32),
            rand.random_range(-3000.0_f32..3000.0_f32),
        );
        let fog_distance = rand.random_range(0.0_f32..900.0_f32);

        let test_result = voxelize_fog_cylinder(relative_tile_pos, fog_distance);
        let sane_result = voxelize_fog_cylinder_slow(relative_tile_pos, fog_distance);

        if sane_result != test_result {
            println!("Sane Result");
            print_tile(&sane_result);
            println!();
            println!("Test Result");
            print_tile(&test_result);
            println!();
            panic!(
                "sane != test, Relative Tile Coords: {:?}, Fog Distance: {fog_distance}",
                relative_tile_pos,
            );
        }
    }
}

fn voxelize_fog_cylinder_slow(relative_tile_pos: f32x3, fog_distance: f32) -> u8x64 {
    let mut visible_sections = SECTIONS_EMPTY;

    for y in 0..8 {
        for z in 0..8 {
            for x in 0..8 {
                let section_coords = Simd::from_xyz(x, y, z);
                let section_index = section_index(section_coords);

                let relative_section_pos = section_coords
                    .cast::<f32>()
                    .mul_add_fast(Simd::splat(16.0), relative_tile_pos);
                let relative_bounds = RelativeBoundingBox::new(
                    relative_section_pos,
                    relative_section_pos + Simd::splat(16.0),
                );

                let closest_in_chunk = f32x3::splat(0.0)
                    .simd_max(relative_bounds.min)
                    .simd_min(relative_bounds.max);

                let distances_squared = closest_in_chunk * closest_in_chunk;

                let inside_fog = (distances_squared[X] + distances_squared[Z])
                    < (fog_distance * fog_distance)
                    && closest_in_chunk[Y].abs() < fog_distance;

                modify_bit(&mut visible_sections, section_index, inside_fog);
            }
        }
    }

    visible_sections
}

// #[test]
// fn test_modulo() {
//     for denom in (19995..=20000).rev() {
//         for i in -30_000_000_i32..=30_000_000_i32 {
//             let sane = i.rem_euclid(denom);
//             let test = i32x1::splat(i).modulo(i32x1::splat(denom))[0];
//             assert_eq!(sane, test);
//         }
//     }
// }

// TODO: test bfs
