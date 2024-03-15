use std::{ffi::CString, sync::Arc};

use ash::vk::{self};

use crate::{
    allocator::{Allocation, GpuAllocator, MemoryFlags},
    common::{BufferUsageFlags, ImageUsageFlags},
    device::{DeviceInner, Image, ImageInfo},
};

pub const MAX_BUFFERS: u64 = 1000;
pub const MAX_IMAGES: u64 = 1000;

pub const BUFFER_ADDRESSES_BINDING: u32 = 0;
pub const STORAGE_IMAGE_BINDING: u32 = 1;

#[derive(Clone, Copy, Debug)]
pub struct ImageId(pub(crate) GpuResourceId);

impl ImageId {
    pub fn pack(&self) -> PackedGpuResourceId {
        PackedGpuResourceId::new(self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BufferId(pub(crate) GpuResourceId);

impl BufferId {
    pub fn pack(&self) -> PackedGpuResourceId {
        PackedGpuResourceId::new(self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GpuResourceId {
    index: u32,
    version: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct PackedGpuResourceId(u32);

impl PackedGpuResourceId {
    fn new(id: GpuResourceId) -> Self {
        PackedGpuResourceId(id.index)
    }
}

pub enum GpuResourceType {
    Undefined = 0,
    StorageImage = 1,
    Buffer = 2,
}

pub enum ResourceEntry<T> {
    Occupied(T),
    Free(usize),
}

pub struct ResourceVersionEntry<T> {
    entry: ResourceEntry<T>,
    version: u16,
}

pub struct ResourceSlot<T> {
    entries: Vec<ResourceVersionEntry<T>>,
    free_head: usize,
}

impl<T> ResourceSlot<T> {
    fn new() -> Self {
        Self {
            entries: vec![ResourceVersionEntry {
                entry: ResourceEntry::Free(usize::MAX),
                version: 0,
            }],
            free_head: 0,
        }
    }

    fn insert_resource(&mut self, resource: T) -> GpuResourceId {
        if self.free_head == usize::MAX {
            self.entries.push(ResourceVersionEntry {
                entry: ResourceEntry::Occupied(resource),
                version: 0,
            });

            return GpuResourceId {
                index: self.entries.len() as u32 - 1,
                version: 0,
            };
        }

        let free_entry = &mut self.entries[self.free_head];
        free_entry.version = (free_entry.version + 1) % u16::MAX;
        let ResourceEntry::Free(next_free) = free_entry.entry else {
            panic!("This entry should be free");
        };
        free_entry.entry = ResourceEntry::Occupied(resource);
        let index = self.free_head as u32;
        self.free_head = next_free as usize;

        GpuResourceId {
            index,
            version: free_entry.version,
        }
    }

    fn get_resource(&self, id: GpuResourceId) -> &T {
        let Some(versioned_entry) = self.entries.get(id.index as usize) else {
            panic!("Could not get resource by id")
        };

        if versioned_entry.version != id.version {
            panic!("Version does not match")
        }

        let ResourceEntry::Occupied(resource) = &versioned_entry.entry else {
            panic!("Resource does not exist")
        };

        return resource;
    }

    fn remove_resource(&mut self, id: GpuResourceId) -> T {
        let Some(versioned_entry) = self.entries.get_mut(id.index as usize) else {
            panic!("Could not get resource by id")
        };

        if versioned_entry.version != id.version {
            panic!("Version does not match")
        }

        match std::mem::replace(&mut versioned_entry.entry, ResourceEntry::Free(usize::MAX)) {
            ResourceEntry::Free(_) => panic!(""),
            ResourceEntry::Occupied(resource) => {
                if self.free_head > id.index as usize {
                    self.entries[id.index as usize].entry = ResourceEntry::Free(self.free_head);
                    self.free_head = id.index as usize;
                } else {
                    let mut p = self.free_head;
                    while p != usize::MAX {
                        let ResourceEntry::Free(next) = self.entries[p].entry else {
                            panic!("should never point to a free entry.");
                        };

                        if next > id.index as usize {
                            self.entries[p].entry = ResourceEntry::Free(id.index as usize);
                            self.entries[id.index as usize].entry = ResourceEntry::Free(next);
                            break;
                        }

                        p = next as usize;
                    }
                }

                resource
            }
        }
    }

    fn collect_existing(&mut self) -> Vec<T> {
        self.entries
            .drain(0..)
            .filter_map(|entry| match entry.entry {
                ResourceEntry::Occupied(data) => Some(data),
                _ => None,
            })
            .collect()
    }
}

/// This will hold all the resources that we will use in the renderer.
pub struct GpuResourcePool {
    device_dep: Arc<DeviceInner>,
    allocator: GpuAllocator,
    descriptor_pool: vk::DescriptorPool,

    pub(crate) bindless_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) descriptor_set: vk::DescriptorSet,
    buffer_addresses_buffer: Buffer,
    buffer_addresses_buffer_ptr: BufferAddressPtr,

    images: ResourceSlot<Image>,
    buffers: ResourceSlot<Buffer>,
}

impl GpuResourcePool {
    pub fn new(device_dep: Arc<DeviceInner>) -> Self {
        let device_inner = &device_dep;

        let descriptor_pool = Self::create_descriptor_pool(device_inner);

        let descriptor_set_layout =
            Self::create_bindless_descriptor_set_layout(device_inner, vk::ShaderStageFlags::ALL);

        let descriptor_set = unsafe {
            device_inner.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[descriptor_set_layout]),
            )
        }
        .unwrap()[0];

        let mut allocator = GpuAllocator::new(device_dep.clone());

        let buffer_addresses_buffer = {
            let info = BufferInfo {
                name: "paya_buffer_addresses_buffer".to_owned(),
                size: MAX_BUFFERS * std::mem::size_of::<u64>() as u64,
                memory_flags: MemoryFlags::DEVICE_LOCAL | MemoryFlags::HOST_VISIBLE,
                usage: BufferUsageFlags::STORAGE,
            };
            let create_info = vk::BufferCreateInfo::default()
                .size(info.size)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .usage(info.usage.into());

            let buffer = unsafe { device_dep.device.create_buffer(&create_info, None) }
                .expect("Failed to make the buffer lol");

            let memory_requirements =
                unsafe { device_dep.device.get_buffer_memory_requirements(buffer) };

            let allocation = allocator.allocate_memory(
                memory_requirements,
                info.memory_flags,
                vk::MemoryAllocateFlags::empty(),
            );

            unsafe {
                device_dep
                    .device
                    .bind_buffer_memory(buffer, allocation.memory, allocation.offset)
            }
            .expect("failed to bind memory to buffer");

            Buffer {
                allocation,
                size: info.size,
                info,
                offset: 0,
                handle: buffer,
            }
        };

        let buffer_addresses_buffer_ptr = unsafe {
            device_dep.device.map_memory(
                buffer_addresses_buffer.allocation.memory,
                0,
                buffer_addresses_buffer.info.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .expect("Failed to map buffer address ptr")
            as *mut u64;

        let buffer_write_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer_addresses_buffer.handle)
            .range(vk::WHOLE_SIZE)
            .offset(0)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(BUFFER_ADDRESSES_BINDING)
            .buffer_info(&buffer_write_info)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)];

        unsafe {
            device_dep.device.update_descriptor_sets(&writes, &[]);
        }

        GpuResourcePool {
            device_dep,
            allocator,
            descriptor_pool,
            bindless_descriptor_set_layout: descriptor_set_layout,
            descriptor_set,
            buffer_addresses_buffer,
            buffer_addresses_buffer_ptr: BufferAddressPtr(buffer_addresses_buffer_ptr),
            images: ResourceSlot::new(),
            buffers: ResourceSlot::new(),
        }
    }

    fn create_descriptor_pool(device_inner: &DeviceInner) -> vk::DescriptorPool {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1000,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1000,
            },
        ];

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);

        unsafe {
            device_inner
                .device
                .create_descriptor_pool(&create_info, None)
                .expect("Failed to create descriptor pool")
        }
    }

    fn create_bindless_descriptor_set_layout(
        device_inner: &DeviceInner,
        stage_flags: vk::ShaderStageFlags,
    ) -> vk::DescriptorSetLayout {
        let bindings = vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(BUFFER_ADDRESSES_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(stage_flags),
            vk::DescriptorSetLayoutBinding::default()
                .binding(STORAGE_IMAGE_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(MAX_IMAGES as u32)
                .stage_flags(stage_flags),
        ];
        let binding_flags = bindings
            .iter()
            .map(|_| {
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            })
            .collect::<Vec<_>>();

        let mut flags_create_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut flags_create_info);

        unsafe {
            device_inner
                .device
                .create_descriptor_set_layout(&create_info, None)
                .expect("Failed to create descriptor set layout")
        }
    }

    pub fn create_image(&mut self, existing_image: Option<vk::Image>, info: &ImageInfo) -> ImageId {
        let handle = existing_image.unwrap_or_else(|| {
            let vk_create_info = vk::ImageCreateInfo::default()
                .image_type(match info.dimensions {
                    1 => vk::ImageType::TYPE_1D,
                    2 => vk::ImageType::TYPE_2D,
                    3 => vk::ImageType::TYPE_3D,
                    _ => panic!("Invalid image dimensions, must be 1, 2, or 3"),
                })
                .format(info.format.into())
                .extent(info.extent.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(info.usage.into())
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            unsafe { self.device_dep.device.create_image(&vk_create_info, None) }
                .expect("Failed to create image")
        });

        let allocation = if existing_image.is_none() {
            let memory_requirements =
                unsafe { self.device_dep.device.get_image_memory_requirements(handle) };
            let allocation = self.allocator.allocate_memory(
                memory_requirements,
                MemoryFlags::DEVICE_LOCAL,
                vk::MemoryAllocateFlags::empty(),
            );

            unsafe {
                self.device_dep.device.bind_image_memory(
                    handle,
                    allocation.memory,
                    allocation.offset,
                )
            }
            .expect("Failed to bind image memory");

            Some(allocation)
        } else {
            None
        };

        let view = info.usage.needs_view().then(|| {
            let vk_image_view_create_info = vk::ImageViewCreateInfo::default()
                .image(handle)
                .view_type(match info.dimensions {
                    1 => vk::ImageViewType::TYPE_1D,
                    2 => vk::ImageViewType::TYPE_2D,
                    3 => vk::ImageViewType::TYPE_3D,
                    _ => panic!("Invalid image dimensions, must be 1, 2, or 3"),
                })
                .format(info.format.into())
                .components(vk::ComponentMapping::default())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            unsafe {
                self.device_dep
                    .device
                    .create_image_view(&vk_image_view_create_info, None)
            }
            .expect("Failed to create image view")
        });

        let index = self.images.insert_resource(Image {
            handle,
            view,
            info: info.clone(),
            allocation,
            is_swapchain_image: existing_image.is_some(),
        });

        if let Some(view) = view {
            let write_image_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(view)
                .sampler(vk::Sampler::null())];
            let mut writes = vec![];
            if info.usage.contains(ImageUsageFlags::STORAGE) {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_set)
                        .dst_binding(STORAGE_IMAGE_BINDING)
                        .dst_array_element(index.index)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&write_image_info),
                );
            }
            unsafe { self.device_dep.device.update_descriptor_sets(&writes, &[]) };
        }

        ImageId(index)
    }

    pub fn get_image(&self, id: ImageId) -> &Image {
        self.images.get_resource(id.0)
    }

    pub fn destroy_image(&mut self, id: ImageId) {
        let image = self.images.remove_resource(id.0);
        self.destroy_image_raw(image);
    }

    fn destroy_image_raw(&mut self, image: Image) {
        if let Some(view) = image.view {
            unsafe { self.device_dep.device.destroy_image_view(view, None) };
        }
        if !image.is_swapchain_image {
            unsafe { self.device_dep.device.destroy_image(image.handle, None) };
        }
        if let Some(allocation) = image.allocation.clone() {
            self.allocator.deallocate_memory(allocation);
        }
    }

    pub fn create_buffer(&mut self, info: &BufferInfo) -> BufferId {
        let buffer = {
            let vk_usage: vk::BufferUsageFlags = info.usage.into();
            let create_info = vk::BufferCreateInfo::default()
                .size(info.size)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .usage(vk_usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

            unsafe { self.device_dep.device.create_buffer(&create_info, None) }
        }
        .expect("Failed to make the buffer lol");

        let c_string_name = CString::new(info.name.clone()).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(buffer)
            .object_name(&c_string_name);
        unsafe {
            let _ = self
                .device_dep
                .instance_dep
                .debug_utils
                .set_debug_utils_object_name(self.device_dep.device.handle(), &name_info);
        }

        let memory_requirements = unsafe {
            self.device_dep
                .device
                .get_buffer_memory_requirements(buffer)
        };

        let allocation = self.allocator.allocate_memory(
            memory_requirements,
            info.memory_flags,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
        );

        unsafe {
            self.device_dep
                .device
                .bind_buffer_memory(buffer, allocation.memory, allocation.offset)
        }
        .expect("failed to bind memory to buffer");

        let index = self.buffers.insert_resource(Buffer {
            info: info.clone(),
            handle: buffer,
            allocation,
            offset: 0,
            size: info.size,
        });

        let buffer_address = unsafe {
            self.device_dep
                .device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };

        self.buffer_addresses_buffer_ptr
            .write_buffer_address(index.index as usize, buffer_address);

        BufferId(index)
    }

    pub fn get_buffer(&self, id: BufferId) -> &Buffer {
        self.buffers.get_resource(id.0)
    }

    pub fn destroy_buffer(&mut self, id: BufferId) {
        let buffer = self.buffers.remove_resource(id.0);
        self.destroy_buffer_raw(buffer);
    }

    fn destroy_buffer_raw(&mut self, buffer: Buffer) {
        unsafe { self.device_dep.device.destroy_buffer(buffer.handle, None) };
        self.allocator.deallocate_memory(buffer.allocation)
    }
}

impl Drop for GpuResourcePool {
    fn drop(&mut self) {
        unsafe { self.device_dep.device.device_wait_idle() }.expect("failed to idle");
        for image in self.images.collect_existing() {
            self.destroy_image_raw(image);
        }
        for buffer in self.buffers.collect_existing() {
            self.destroy_buffer_raw(buffer);
        }
        self.destroy_buffer_raw(self.buffer_addresses_buffer.clone());

        unsafe {
            self.device_dep
                .device
                .destroy_descriptor_set_layout(self.bindless_descriptor_set_layout, None);
        }
        unsafe {
            self.device_dep
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

#[derive(Clone, Debug)]
pub struct BufferInfo {
    pub name: String,
    pub size: u64,
    pub memory_flags: MemoryFlags,
    pub usage: BufferUsageFlags,
}

#[derive(Clone)]
pub struct Buffer {
    pub handle: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
    pub allocation: Allocation,
    pub info: BufferInfo,
}

struct BufferAddressPtr(*mut u64);

// Safety: we only mutate this memory and GpuResources follows mutalibity rules i think
unsafe impl Send for BufferAddressPtr {}
unsafe impl Sync for BufferAddressPtr {}

impl BufferAddressPtr {
    fn write_buffer_address(&mut self, index: usize, address: u64) {
        unsafe { self.0.offset(index as isize).write(address) };
    }
}
