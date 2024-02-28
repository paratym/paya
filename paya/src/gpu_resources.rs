use std::sync::Arc;

use ash::vk::{self, DescriptorSetAllocateInfo};

use crate::{
    allocator::{GpuAllocator, MemoryFlags},
    device::{Device, DeviceInner, Image, ImageInfo},
};

#[derive(Clone, Copy, Debug)]
pub struct ImageId(pub(crate) u32);

/// A packed u32 integer that represents as the following.
/// - 0.=19 bits: The index of the resource in the pool.
/// - 20.=23 bits: The type of the resource.
/// - 24: If the resource is mutable or not.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct GpuResourceId(u32);

pub enum GpuResourceType {
    StorageImage = 0,
    Buffer = 1,
}

/// This will hold all the resources that we will use in the renderer.
pub struct GpuResourcePool {
    device_dep: Arc<DeviceInner>,
    allocator: GpuAllocator,
    descriptor_pool: vk::DescriptorPool,

    pub(crate) bindless_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) descriptor_set: vk::DescriptorSet,

    images: Vec<Image>,
}

impl GpuResourcePool {
    pub fn new(device_dep: Arc<DeviceInner>) -> Self {
        let device_inner = &device_dep;

        let descriptor_pool = Self::create_descriptor_pool(device_inner);

        let descriptor_set_layout =
            Self::create_bindless_descriptor_set_layout(device_inner, vk::ShaderStageFlags::ALL);

        let descriptor_set = unsafe {
            device_inner.device.allocate_descriptor_sets(
                &DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[descriptor_set_layout]),
            )
        }
        .unwrap()[0];

        let allocator = GpuAllocator::new(device_dep.clone());

        GpuResourcePool {
            device_dep,
            allocator,
            descriptor_pool,
            bindless_descriptor_set_layout: descriptor_set_layout,
            descriptor_set,
            images: Vec::new(),
        }
    }

    fn create_descriptor_pool(device_inner: &DeviceInner) -> vk::DescriptorPool {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 100,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 100,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 100,
            },
        ];

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(100)
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
        let result = [vk::DescriptorType::STORAGE_IMAGE]
            .into_iter()
            .enumerate()
            .map(|(i, ty)| {
                let binding = vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(ty)
                    .descriptor_count(100)
                    .stage_flags(stage_flags);

                let flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;

                (binding, flags)
            })
            .collect::<Vec<_>>();

        let bindings = result
            .iter()
            .map(|(binding, _)| binding.clone())
            .collect::<Vec<_>>();
        let binding_flags = result
            .iter()
            .map(|(_, flags)| flags.clone())
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
            let allocation = self
                .allocator
                .allocate_memory(memory_requirements, MemoryFlags::DEVICE_LOCAL);

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

        self.images.push(Image {
            handle,
            view,
            info: info.clone(),
            allocation,
        });

        let index = self.images.len() as u32 - 1;

        if let Some(view) = view {
            let info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(view)
                .sampler(vk::Sampler::null())];
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .dst_array_element(index)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&info)];
            unsafe { self.device_dep.device.update_descriptor_sets(&writes, &[]) };
        }

        ImageId(index)
    }

    pub fn get_image(&self, id: ImageId) -> &Image {
        self.images.get(id.0 as usize).expect("Failed to get image")
    }

    pub(crate) fn create_gpu_id(&self, id: u32, ty: GpuResourceType) -> GpuResourceId {
        GpuResourceId(id | (ty as u32) << 20)
    }

    pub fn destroy_image(&mut self, id: ImageId) {
        todo!();
    }
}

impl Drop for GpuResourcePool {
    fn drop(&mut self) {
        unsafe {
            self.device_dep
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
