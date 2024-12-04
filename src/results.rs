// TODO OPT: organize into 8x4x8 section regions, represented as 256-bit masks
pub struct SectionBitArray {
    pub section_count: u32,
    pub data: Box<[u64]>,
}

impl SectionBitArray {
    pub fn clear(&mut self) {
        self.data.fill(0);
    }
}
