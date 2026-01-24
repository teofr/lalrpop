use std::mem::size_of;
use std::ptr;
use std::fmt::Debug;

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
struct Metadata<Location>
where Location: Debug {
    offset: usize, // offset where the value begins
    l: Location,
    r: Location,
}

type Base = u8;
pub struct HeterogeneousStack<Location>
where Location:Debug {
    buffer: Vec<Base>,
    phantom: std::marker::PhantomData<Location>,
    /// Number of elements in the stack
    size: usize,
}


impl<Location: Debug> HeterogeneousStack<Location> {

    /// Creates a new empty stack.
    pub fn new() -> Self {
        HeterogeneousStack {
            buffer: Vec::new(),
            phantom: std::marker::PhantomData,
            size: 0,
        }
    }

    pub fn push<T>(&mut self, l: Location, value: T, r: Location) {
        let value_offset = self.buffer.len();
        let size_value = size_of::<T>();

        // Calculate aligned offset for metadata (add padding after value)
        let after_value = value_offset + size_value;
        let metadata_align = std::mem::align_of::<Metadata<Location>>();
        let metadata_offset = (after_value + metadata_align - 1) & !(metadata_align - 1);

        // println!("Pushing value of size {} and align {} at offset {}", size_of::<T>(), std::mem::align_of::<T>(), value_offset);
        // println!("  with l: {:?}, r: {:?}", &l, &r);

        let metadata = Metadata {
            offset: value_offset,
            l,
            r,
        };

        let total_size = metadata_offset + size_of::<Metadata<Location>>();


        self.buffer.reserve(total_size - self.buffer.len());
        unsafe {
            self.buffer.set_len(total_size);
        }

        unsafe {
            ptr::write_unaligned(self.buffer.as_mut_ptr().add(value_offset) as *mut T, value);
            // Metadata is aligned, but buffer base may not be, so still use write_unaligned
            ptr::write_unaligned(self.buffer.as_mut_ptr().add(metadata_offset) as *mut Metadata<Location>, metadata);
        }
        self.size += 1;
    }

    pub fn pop<T>(&mut self) -> (Location, T, Location) {
        let len = self.buffer.len();
        let metadata_size = size_of::<Metadata<Location>>();
        let metadata_offset = len - metadata_size;

        let Metadata { offset: value_offset, l, r } = unsafe {
            ptr::read_unaligned(self.buffer.as_ptr().add(metadata_offset) as *const Metadata<Location>)
        };

        // println!("Popping value of size {} and align {} at offset {}", size_of::<T>(), std::mem::align_of::<T>(), value_offset);
        let value = unsafe { ptr::read_unaligned(self.buffer.as_ptr().add(value_offset) as *const T) };

        unsafe {
            self.buffer.set_len(value_offset);
        }
        self.size -= 1;

        (l, value, r)
    }

    fn _peek<T>(&self) -> &T {
        let size = size_of::<T>();

        // The top value starts at the last aligned offset that fits this type.
        let offset = self.buffer.len() - size;

        // assert!(offset + size <= self.buffer.len(), "buffer too small for peek");
        unsafe { &*(self.buffer.as_ptr().add(offset) as *const T) }
    }

    /// Returns true if the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the number of elements in the stack.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Clears the stack.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.size = 0;
    }

    // This shouldnt be mut
    pub fn last_location(&mut self) -> Option<(&Location, &Location)> {
        if self.buffer.is_empty() {
            return None;
        }
        // Use a peek

        let Metadata { offset: _, l, r } = self._peek();

        Some( (l, r) )
    }

    /// Returns the location of the element at the given index (0-indexed from the start).
    pub fn get_nth_location(&self, index: usize) -> Option<(&Location, &Location)> {
        if index >= self.size {
            return None;
        }
        // Convert to steps from the end: to get element at index `i` from start,
        // we need to traverse `size - 1 - i` elements from the end
        let mut steps_from_end = self.size - 1 - index;

        let metadata_size = size_of::<Metadata<Location>>();
        let mut metadata_offset = self.buffer.len();

        loop {
            if metadata_offset == 0 {
                // There's not enough data left
                return None;
            }
            metadata_offset -= metadata_size;
            let Metadata { offset: value_offset, l, r } = unsafe {
                &*(self.buffer.as_ptr().add(metadata_offset) as *const Metadata<Location>)
            };
            if steps_from_end == 0 {
                return Some((l, r))
            }
            // Previous metadata ends at value_offset, so it starts at value_offset - metadata_size
            metadata_offset = *value_offset;
            steps_from_end -= 1;
        }
    }

    pub fn truncate_last(&mut self, count: usize) {
        // We're assuming that the buffer has sufficient length, no checks here
        let metadata_size = size_of::<Metadata<Location>>();

        let mut truncate_offset = self.buffer.len();
        let mut remaining = count;

        while remaining > 0 {
            let metadata_offset = truncate_offset - metadata_size;
            let Metadata { offset: value_offset, l: _, r: _ } = unsafe {
                &*(self.buffer.as_ptr().add(metadata_offset) as *const Metadata<Location>)
            };
            // Previous metadata ends at value_offset
            truncate_offset = *value_offset;
            remaining -= 1;
        }

        println!("Truncating at {truncate_offset}");

        self.buffer.truncate(truncate_offset);
        self.size -= count;
    }

}

impl<Location: Debug> Default for HeterogeneousStack<Location> {
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
