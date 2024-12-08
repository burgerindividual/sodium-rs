use crate::{graph::coords::LocalTileCoords, math::u16x3};

// TODO OPT: organize into 8x4x8 section regions, represented as 256-bit masks
pub struct SectionBitArray {
    pub section_count: u32,
    pub data: Box<[u64]>,

    graph_origin_offset: u16x3,
    dimensions: u16x3, 
}

impl SectionBitArray {
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    pub fn set_tile(&mut self, coords: LocalTileCoords, level: u8) {
    }
}
