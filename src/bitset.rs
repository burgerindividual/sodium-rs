use std::hint::assert_unchecked;
use std::ops::{BitAnd, BitOr, Not, Shl, Shr};

pub const fn contains(bitset: u8, other_bitset: u8) -> bool {
    bitset & other_bitset == other_bitset
}

pub trait BitSet {
    fn get_bit(self, idx: u8) -> bool;
    fn set_bit(&mut self, idx: u8);
    fn clear_bit(&mut self, idx: u8);
    fn modify_bit(&mut self, idx: u8, value: bool);
}

impl<
        T: Copy
            + From<bool>
            + Shr<u8, Output = T>
            + Shl<u8, Output = T>
            + BitAnd<T, Output = T>
            + BitOr<T, Output = T>
            + PartialEq<T>
            + Not<Output = T>,
    > BitSet for T
{
    fn get_bit(self, idx: u8) -> bool {
        unsafe { assert_unchecked(idx < (size_of::<T>() as u8 * 8)) };
        ((self >> idx) & T::from(true)) != T::from(false)
    }

    fn set_bit(&mut self, idx: u8) {
        unsafe { assert_unchecked(idx < (size_of::<T>() as u8 * 8)) };
        *self = *self | (T::from(true) << idx);
    }

    fn clear_bit(&mut self, idx: u8) {
        unsafe { assert_unchecked(idx < (size_of::<T>() as u8 * 8)) };
        *self = *self & !(T::from(true) << idx);
    }

    fn modify_bit(&mut self, idx: u8, value: bool) {
        unsafe { assert_unchecked(idx < (size_of::<T>() as u8 * 8)) };
        *self = (*self & !(T::from(true) << idx)) | (T::from(value) << idx);
    }
}
