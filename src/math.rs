#![allow(non_camel_case_types)]

use std::ops::{Add, Mul};

use core_simd::simd::prelude::*;
use core_simd::simd::*;
use std_float::StdFloat;

pub const X: usize = 0;
pub const Y: usize = 1;
pub const Z: usize = 2;

// the most common non-po2 length we use is 3, so we create shorthands for it
pub type i8x3 = Simd<i8, 3>;
pub type i16x3 = Simd<i16, 3>;
pub type i32x3 = Simd<i32, 3>;

pub type u8x3 = Simd<u8, 3>;
pub type u16x3 = Simd<u16, 3>;

pub type f32x3 = Simd<f32, 3>;
pub type f64x3 = Simd<f64, 3>;

// additional useful shorthands
pub type u8x6 = Simd<u8, 6>;
pub type f32x6 = Simd<f32, 6>;
pub type u32x6 = Simd<u32, 6>;

pub trait Coords3<T> {
    fn from_xyz(x: T, y: T, z: T) -> Self;
    fn into_tuple(self) -> (T, T, T);
    fn x(&self) -> T;
    fn y(&self) -> T;
    fn z(&self) -> T;
}

impl<T> Coords3<T> for Simd<T, 3>
where
    T: SimdElement,
{
    fn from_xyz(x: T, y: T, z: T) -> Self {
        Simd::from_array([x, y, z])
    }

    fn into_tuple(self) -> (T, T, T) {
        self.to_array().into()
    }

    fn x(&self) -> T {
        self[X]
    }

    fn y(&self) -> T {
        self[Y]
    }

    fn z(&self) -> T {
        self[Z]
    }
}

impl<T> Coords3<bool> for Mask<T, 3>
where
    T: MaskElement,
{
    fn from_xyz(x: bool, y: bool, z: bool) -> Self {
        Mask::from_array([x, y, z])
    }

    fn into_tuple(self) -> (bool, bool, bool) {
        self.to_array().into()
    }

    fn x(&self) -> bool {
        self.test(X)
    }

    fn y(&self) -> bool {
        self.test(Y)
    }

    fn z(&self) -> bool {
        self.test(Z)
    }
}

pub trait MulAddFast {
    fn mul_add_fast(self, mul: Self, add: Self) -> Self;
}

impl<T: StdFloat + Mul<Output = T> + Add<Output = T>> MulAddFast for T {
    fn mul_add_fast(self, mul: Self, add: Self) -> Self {
        // this could probably have better detection
        if cfg!(target_feature = "fma") || cfg!(target_feature = "neon") {
            self.mul_add(mul, add)
        } else {
            self * mul + add
        }
    }
}
