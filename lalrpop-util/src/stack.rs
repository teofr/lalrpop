use std::mem::size_of;
use std::ptr;

/// A type-erased heterogeneous stack.
///
/// This stack holds values of different types. Push stores a value in a buffer,
/// and pop retrieves it by computing the offset directly from the type's size
/// and alignment.
///
/// The buffer grows as values are pushed. Each value is placed at the next
/// aligned offset. On pop, the offset of the top value is computed from its
/// type's layout.
///
/// # Safety
///
/// The caller is responsible for popping values in the exact type they were
/// pushed. Popping as the wrong type is undefined behavior. The implementation
/// performs no type validation.
pub struct HeterogeneousStack<Location> {
    buffer: Vec<u8>,
    phantom: std::marker::PhantomData<Location>,
    // locations: Vec<(Location, Location)>,
}

impl<Location> HeterogeneousStack<Location> {
    /// Creates a new empty stack.
    pub fn new() -> Self {
        HeterogeneousStack {
            buffer: Vec::new(),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn push<T>(&mut self, l: Location, value: T, r: Location) {
        let size = size_of::<T>();

        self._push(value);
        self._push(size);
        self._push(l);
        self._push(r);

    }

    fn _push<T>(&mut self, value: T) {
        let size = size_of::<T>();
        let align = align_of::<T>();

        // Compute aligned offset for this value.
        let offset = self.buffer.len();

        // Resize buffer to accommodate padding and value.
        // println!("Pushing value of size {} and align {} at offset {}", size, align, offset);
        self.buffer.resize(offset + size, 0);

        // Write the value.
        unsafe {
            ptr::write(self.buffer.as_mut_ptr().add(offset) as *mut T, value);
        }
    }

    pub fn pop<T>(&mut self) -> (Location, T, Location) {
        let r: Location = self._pop();
        let l: Location = self._pop();
        let _: usize = self._pop();
        let value: T = self._pop();

        (l, value, r)
    }

    fn _pop<T>(&mut self) -> T {
        let size = size_of::<T>();
        let align = align_of::<T>();

        // The top value starts at the last aligned offset that fits this type.
        let offset = self.buffer.len().saturating_sub(size);

        assert!(offset + size <= self.buffer.len(), "buffer too small for pop");

        // println!("Popping value of size {} and align {} at offset {}", size, align, offset);
        let value = unsafe { ptr::read(self.buffer.as_ptr().add(offset) as *const T) };
        self.buffer.truncate(offset);
        value
    }

    fn _peek<T>(&self) -> &T {
        let size = size_of::<T>();

        // The top value starts at the last aligned offset that fits this type.
        let offset = self.buffer.len().saturating_sub(size);

        assert!(offset + size <= self.buffer.len(), "buffer too small for peek");
        unsafe { &*(self.buffer.as_ptr().add(offset) as *const T) }
    }

    fn _peek_two<T, S>(&self) -> (&S, &T) {
        let size_t = size_of::<T>();
        let size_s = size_of::<S>();

        // The top value starts at the last aligned offset that fits this type.
        let offset_t = self.buffer.len().saturating_sub(size_t);
        let offset_s = offset_t.saturating_sub(size_s);

        assert!(offset_t + size_t <= self.buffer.len(), "buffer too small for peek_two T");
        assert!(offset_s + size_s <= self.buffer.len(), "buffer too small for peek_two S");

        let t_ref = unsafe { &*(self.buffer.as_ptr().add(offset_t) as *const T) };
        let s_ref = unsafe { &*(self.buffer.as_ptr().add(offset_s) as *const S) };

        (s_ref, t_ref)
    }

    /// Returns true if the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the current buffer size in bytes.
    pub fn len(&self) -> usize {
        // This is wrong, but for now
        self.buffer.len()
    }

    /// Clears the stack.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    // This shouldnt be mut
    pub fn last_location(&mut self) -> Option<(&Location, &Location)> {
        if self.buffer.is_empty() {
            return None;
        }
        // Use a peek

        Some( self._peek_two::<Location, Location>() )
    }

}

impl<Location> Default for HeterogeneousStack<Location> {
    fn default() -> Self {
        Self::new()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_push_pop_primitives() {
//         let mut stack = HeterogeneousStack::new();

//         stack.push(42i32);
//         stack.push(3.14f64);
//         stack.push(true);

//         assert!(stack.pop::<bool>());
//         assert_eq!(stack.pop::<f64>(), 3.14);
//         assert_eq!(stack.pop::<i32>(), 42);
//     }

//     #[test]
//     fn test_push_pop_strings() {
//         let mut stack = HeterogeneousStack::new();

//         stack.push("hello".to_string());
//         stack.push(vec![1, 2, 3]);

//         let v = stack.pop::<Vec<i32>>();
//         assert_eq!(v, vec![1, 2, 3]);

//         let s = stack.pop::<String>();
//         assert_eq!(s, "hello");
//     }

//     #[test]
//     fn test_empty() {
//         let stack = HeterogeneousStack::new();
//         assert!(stack.is_empty());
//     }

//     #[test]
//     fn test_alignment() {
//         let mut stack = HeterogeneousStack::new();

//         stack.push(1u8);
//         stack.push(42u64);
//         stack.push(2u8);

//         assert_eq!(stack.pop::<u8>(), 2);
//         assert_eq!(stack.pop::<u64>(), 42);
//         assert_eq!(stack.pop::<u8>(), 1);
//     }
// }
