// Directions and Direction Sets are represented as raw u8s to work seamlessly
// with the current state of const generics.

pub const NEG_X: u8 = 0b000001;
pub const NEG_Y: u8 = 0b000010;
pub const NEG_Z: u8 = 0b000100;
pub const POS_X: u8 = 0b001000;
pub const POS_Y: u8 = 0b010000;
pub const POS_Z: u8 = 0b100000;

pub struct DirectionSetIter(pub u8);

impl Iterator for DirectionSetIter {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 != 0 {
            let prev_set = self.0;
            self.0 &= self.0 - 1;
            let direction = self.0 ^ prev_set;
            Some(direction)
        } else {
            None
        }
    }
}
