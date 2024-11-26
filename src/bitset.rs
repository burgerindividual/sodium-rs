pub const fn contains(bitset: u8, other_bitset: u8) -> bool {
    bitset & other_bitset == other_bitset
}
