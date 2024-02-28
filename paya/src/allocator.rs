use std::sync::Arc;

use ash::vk;
use bitflags::bitflags;

use crate::device::DeviceInner;

pub struct GpuAllocator {
    device_dep: Arc<DeviceInner>,
}

bitflags! {
    pub struct MemoryFlags: u32 {
        const DEVICE_LOCAL = 1;
        const HOST_VISIBLE = 2;
        const HOST_COHERENT = 4;
        const HOST_CACHED = 8;
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

pub struct Allocation {
    pub memory: vk::DeviceMemory,
    pub offset: vk::DeviceSize,
}

impl GpuAllocator {
    pub(crate) fn new(device_dep: Arc<DeviceInner>) -> Self {
        GpuAllocator { device_dep }
    }

    pub(crate) fn allocate_memory(
        &mut self,
        memory_requirements: vk::MemoryRequirements,
        memory_flags: MemoryFlags,
    ) -> Allocation {
        let device = &self.device_dep;
        let memory_type_bits = memory_requirements.memory_type_bits;
        let memory_type_index = Self::find_memory_type(
            device,
            memory_requirements,
            memory_type_bits,
            memory_flags.into(),
        );

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe {
            device
                .device
                .allocate_memory(&memory_allocate_info, None)
                .unwrap()
        };

        Allocation { memory, offset: 0 }
    }

    fn find_memory_type(
        device: &DeviceInner,
        memory_requirements: vk::MemoryRequirements,
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
