pub const fn contains(bitset: u8, other_bitset: u8) -> bool {
    bitset & other_bitset == other_bitset
}

pub fn modify_bit(bitset: &mut u8, idx: u8, value: bool) {
    unsafe {
        std::hint::assert_unchecked(idx < 8);
    }

    *bitset = (*bitset & !(1 << idx)) | ((value as u8) << idx);
}
