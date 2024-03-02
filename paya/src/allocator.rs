use std::sync::Arc;

use ash::vk;
use bitflags::bitflags;

use crate::device::DeviceInner;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocationId(u64);

impl AllocationId {
    pub fn new(is_dedicated: bool, index: usize) -> Self {
        let mut i = 0u64;
        if is_dedicated {
            i |= 1 << 63;
        }

        if index >= (1 << 63) {
            panic!("allocation index is too big ");
        }

        i |= index as u64;

        Self(i)
    }

    pub fn is_dedicated(&self) -> bool {
        (self.0 & (1 << 63)) != 0
    }
}
pub struct GpuAllocator {
    device_dep: Arc<DeviceInner>,
    dedicated_allocations: Vec<Allocation>,
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    pub struct MemoryFlags: u32 {
        const DEVICE_LOCAL = vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw();
        const HOST_VISIBLE = vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw();
        const HOST_COHERENT = vk::MemoryPropertyFlags::HOST_COHERENT.as_raw();
        const HOST_CACHED = vk::MemoryPropertyFlags::HOST_CACHED.as_raw();
    }
}

impl Into<vk::MemoryPropertyFlags> for MemoryFlags {
    fn into(self) -> vk::MemoryPropertyFlags {
        let mut flags = vk::MemoryPropertyFlags::empty();
        if self.contains(MemoryFlags::DEVICE_LOCAL) {
            flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
        }
        if self.contains(MemoryFlags::HOST_VISIBLE) {
            flags |= vk::MemoryPropertyFlags::HOST_VISIBLE;
        }
        if self.contains(MemoryFlags::HOST_COHERENT) {
            flags |= vk::MemoryPropertyFlags::HOST_COHERENT;
        }
        if self.contains(MemoryFlags::HOST_CACHED) {
            flags |= vk::MemoryPropertyFlags::HOST_CACHED;
        }
        flags
    }
}

#[derive(Clone)]
pub struct Allocation {
    pub id: AllocationId,
    pub memory: vk::DeviceMemory,
    pub offset: vk::DeviceSize,
}

impl GpuAllocator {
    pub(crate) fn new(device_dep: Arc<DeviceInner>) -> Self {
        GpuAllocator {
            device_dep,
            dedicated_allocations: Vec::new(),
        }
    }

    pub(crate) fn allocate_memory(
        &mut self,
        memory_requirements: vk::MemoryRequirements,
        memory_flags: MemoryFlags,
        vk_memory_allocate_flags: vk::MemoryAllocateFlags,
    ) -> Allocation {
        let device = &self.device_dep;
        let memory_type_bits = memory_requirements.memory_type_bits;
        let memory_type_index =
            Self::find_memory_type(device, memory_type_bits, memory_flags.into());

        let mut memory_allocate_flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk_memory_allocate_flags);
        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .push_next(&mut memory_allocate_flags_info)
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe {
            device
                .device
                .allocate_memory(&memory_allocate_info, None)
                .unwrap()
        };

        let allocation = Allocation {
            id: AllocationId::new(true, self.dedicated_allocations.len()),
            memory,
            offset: 0,
        };

        self.dedicated_allocations.push(allocation.clone());

        allocation
    }

    pub(crate) fn deallocate_memory(&mut self, allocation: Allocation) {
        let id = allocation.id;
        if id.is_dedicated() {
            unsafe { self.device_dep.device.free_memory(allocation.memory, None) };
        } else {
            todo!("Make non dedicated allocs")
        }
    }

    fn find_memory_type(
        device: &DeviceInner,
        memory_type_bits: u32,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> u32 {
        let memory_properties = device.physical_device_memory_properties;
        for i in 0..memory_properties.memory_type_count {
            if (memory_type_bits & (1 << i)) != 0
                && memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(memory_flags)
            {
                return i;
            }
        }
        panic!("Failed to find suitable memory type")
    }
}
