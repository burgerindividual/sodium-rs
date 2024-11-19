// Directions and Direction Sets are represented as raw u8s to work seamlessly
// with the current state of const generics.

pub const NEG_X: u8 = 0b000001;
pub const NEG_Y: u8 = 0b000010;
pub const NEG_Z: u8 = 0b000100;
pub const POS_X: u8 = 0b001000;
pub const POS_Y: u8 = 0b010000;
pub const POS_Z: u8 = 0b100000;

pub const fn contains(direction_set: u8, other_direction_set: u8) -> bool {
    direction_set & other_direction_set == other_direction_set
}

pub struct DirectionSetIter(pub u8);

impl Iterator for DirectionSetIter {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        // Description of the iteration approach on daniel lemire's blog
        // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
        if self.0 != 0 {
            let direction = self.0.trailing_zeros() as u8;
            self.0 &= self.0 - 1;
            Some(direction)
        } else {
            None
        }
    }
}
