use std::sync::Arc;

use ash::vk;
use bitflags::bitflags;

use crate::{
    device::{Device, DeviceInner},
    gpu_resources::{BufferId, ImageId},
};

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
    gpu_allocator: gpu_allocator::vulkan::Allocator, //dedicated_allocations: Vec<Allocation>,
}

#[derive(Clone, Debug)]
pub(crate) enum MemoryType {
    Managed,
    DedicatedBuffer(vk::Buffer),
    DedicatedImage(vk::Image),
}

impl MemoryType {
    pub(crate) fn into_gpu_allocator_type(self) -> gpu_allocator::vulkan::AllocationScheme {
        match self {
            MemoryType::Managed => gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            MemoryType::DedicatedImage(image) => {
                gpu_allocator::vulkan::AllocationScheme::DedicatedImage(image)
            }
            MemoryType::DedicatedBuffer(buffer) => {
                gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(buffer)
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum MemoryLocation {
    GpuOnly,
    CpuToGpu,
    GpuToCpu,
}

impl From<MemoryLocation> for gpu_allocator::MemoryLocation {
    fn from(memory_type: MemoryLocation) -> Self {
        match memory_type {
            MemoryLocation::GpuOnly => gpu_allocator::MemoryLocation::GpuOnly,
            MemoryLocation::CpuToGpu => gpu_allocator::MemoryLocation::CpuToGpu,
            MemoryLocation::GpuToCpu => gpu_allocator::MemoryLocation::GpuToCpu,
        }
    }
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

pub struct Allocation {
    pub(crate) allocation: gpu_allocator::vulkan::Allocation,
}

impl Allocation {
    pub fn memory(&self) -> vk::DeviceMemory {
        unsafe { self.allocation.memory() }
    }

    pub fn offset(&self) -> vk::DeviceSize {
        self.allocation.offset()
    }
}

impl GpuAllocator {
    pub(crate) fn new(device_dep: Arc<DeviceInner>) -> Self {
        GpuAllocator {
            device_dep: device_dep.clone(),
            gpu_allocator: gpu_allocator::vulkan::Allocator::new(
                &gpu_allocator::vulkan::AllocatorCreateDesc {
                    instance: device_dep.instance_dep.instance.clone(),
                    device: device_dep.device.clone(),
                    physical_device: device_dep.physical_device,
                    buffer_device_address: true,
                    debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
                    allocation_sizes: gpu_allocator::AllocationSizes::default(),
                },
            )
            .expect("Failed to create allocator."),
            // dedicated_allocations: Vec::new(),
        }
    }

    pub(crate) fn allocate_memory(
        &mut self,
        name: impl Into<String>,
        linear: bool,
        location: MemoryLocation,
        mem_type: MemoryType,
        requirements: vk::MemoryRequirements,
    ) -> Allocation {
        let allocation = self
            .gpu_allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: &name.into(),
                requirements,
                location: location.into(),
                linear,
                allocation_scheme: mem_type.into_gpu_allocator_type(),
            })
            .expect("coudlnt make alloc");

        Allocation { allocation }
    }

    pub(crate) fn deallocate_memory(&mut self, allocation: Allocation) {
        self.gpu_allocator.free(allocation.allocation);
    }
}
