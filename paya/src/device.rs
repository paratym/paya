use std::{
    collections::HashMap,
    ffi::{c_void, CString},
    sync::Arc,
};

use ash::{extensions::khr, vk};
use slotmap::{new_key_type, SlotMap};

use crate::{
    allocator::{Allocation, GpuAllocator},
    command_recorder::{CommandList, CommandRecorder, CommandRecorderId, CommandRecorderPool},
    common::{Extent3D, Format, ImageUsageFlags},
    gpu_resources::{
        Buffer, BufferId, BufferInfo, GpuResourceId, GpuResourcePool, GpuResourceType, ImageId,
    },
    instance::{Instance, InstanceInner},
    pipeline::{ComputePipeline, ComputePipelineInfo, PipelineInner},
    swapchain::{Swapchain, SwapchainCreateInfo},
    sync::{BinarySemaphore, TimelineSemaphore},
};

pub struct DeviceProperties {
    pub device_type: DeviceType,
    pub device_name: String,
}

impl From<vk::PhysicalDeviceProperties> for DeviceProperties {
    fn from(properties: vk::PhysicalDeviceProperties) -> Self {
        DeviceProperties {
            device_type: match properties.device_type {
                vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::Integrated,
                vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::Discrete,
                _ => DeviceType::Other,
            },
            device_name: unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_str()
                .expect("Failed to convert device name to string")
                .to_owned(),
        }
    }
}

pub enum DeviceType {
    Integrated,
    Discrete,
    Other,
}

#[derive(Clone)]
pub struct DeviceInner {
    pub(crate) instance_dep: Arc<InstanceInner>,
    pub(crate) device: ash::Device,
    pub(crate) main_queue_family_index: u32,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) physical_device_properties: vk::PhysicalDeviceProperties,
    pub(crate) physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
}

pub struct Device {
    // We need to keep a reference to the instance to ensure it is not dropped before the device
    inner: Arc<DeviceInner>,

    main_queue: vk::Queue,

    pub(crate) gpu_resources: GpuResourcePool,
    command_recorder_pool: CommandRecorderPool,

    deferred_destruct_recorders: HashMap<u64, Vec<CommandRecorderId>>,
    deferred_destruct_buffers: HashMap<u64, Vec<BufferId>>,

    frame_index: u64,
}

impl Device {
    pub fn new<Selector>(instance: &Instance, selector: Selector) -> Self
    where
        Selector: Fn(&DeviceProperties) -> i32,
    {
        let physical_devices = unsafe {
            instance
                .handle()
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };

        let physical_device = physical_devices
            .into_iter()
            .max_by_key(|physical_device| {
                let properties = unsafe {
                    instance
                        .handle()
                        .get_physical_device_properties(*physical_device)
                };

                selector(&DeviceProperties::from(properties))
            })
            .expect("Failed to find suitable physical device");

        let physical_device_properties = unsafe {
            instance
                .handle()
                .get_physical_device_properties(physical_device)
        };
        let physical_device_memory_properties = unsafe {
            instance
                .handle()
                .get_physical_device_memory_properties(physical_device)
        };

        let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&[1.0])];

        let device_extensions = vec![ash::extensions::khr::Swapchain::NAME.as_ptr()];

        let mut descriptor_indexing_features =
            vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        let mut timeline_semaphore_features =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        timeline_semaphore_features.p_next = &mut descriptor_indexing_features
            as *mut vk::PhysicalDeviceDescriptorIndexingFeatures
            as *mut c_void;
        let mut buffer_device_address_features =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        buffer_device_address_features.p_next = &mut timeline_semaphore_features
            as *mut vk::PhysicalDeviceTimelineSemaphoreFeatures
            as *mut c_void;

        let mut device_features =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut buffer_device_address_features);

        unsafe {
            instance
                .handle()
                .get_physical_device_features2(physical_device, &mut device_features);
        }

        let device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut device_features)
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions);

        let device = unsafe {
            instance
                .handle()
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create logical device")
        };

        let main_queue = unsafe { device.get_device_queue(0, 0) };
        let main_queue_family_index = 0;

        let inner_device = DeviceInner {
            instance_dep: instance.create_dep(),
            device,
            main_queue_family_index,
            physical_device,
            physical_device_properties,
            physical_device_memory_properties,
        };

        let deferred_destruct_recorders = HashMap::new();

        let device_dep = Arc::new(inner_device);
        let gpu_resources = GpuResourcePool::new(device_dep.clone());

        Device {
            inner: device_dep.clone(),
            main_queue,
            gpu_resources,
            command_recorder_pool: CommandRecorderPool::new(device_dep.clone()),
            deferred_destruct_recorders,
            deferred_destruct_buffers: HashMap::new(),
            frame_index: 0,
        }
    }

    pub fn create_swapchain(&mut self, create_info: SwapchainCreateInfo<'_>) -> Swapchain {
        Swapchain::new(self, create_info)
    }

    pub(crate) fn create_swapchain_image(
        &mut self,
        image_handle: vk::Image,
        info: &ImageInfo,
    ) -> ImageId {
        self.gpu_resources.create_image(Some(image_handle), info)
    }

    pub fn create_image(&mut self, info: ImageInfo) -> ImageId {
        self.gpu_resources.create_image(None, &info)
    }

    pub fn get_image(&self, id: ImageId) -> &Image {
        self.gpu_resources.get_image(id)
    }

    pub fn destroy_image(&mut self, id: ImageId) {
        self.gpu_resources.destroy_image(id);
    }

    pub fn create_buffer(&mut self, info: BufferInfo) -> BufferId {
        self.gpu_resources.create_buffer(&info)
    }

    pub fn get_buffer(&self, id: BufferId) -> &Buffer {
        self.gpu_resources.get_buffer(id)
    }

    pub fn destroy_buffer(&mut self, id: BufferId) {
        self.gpu_resources.destroy_buffer(id);
    }

    pub fn map_buffer_typed<T>(&self, id: BufferId) -> TypedMappedPtr<'_, T> {
        let buffer = self.gpu_resources.get_buffer(id);
        let ptr = unsafe {
            self.handle().map_memory(
                buffer.allocation.memory,
                buffer.allocation.offset,
                buffer.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .expect("Failed to map typed buf memory");

        TypedMappedPtr {
            ptr: ptr as *mut T,
            device: self,
            memory: buffer.allocation.memory,
        }
    }

    pub fn create_command_recorder(&mut self) -> CommandRecorder {
        self.command_recorder_pool.create_command_recorder()
    }

    pub fn submit(&mut self, info: SubmitInfo) {
        let wait_semaphores = info
            .wait_semaphores
            .iter()
            .map(|semaphore| semaphore.handle())
            .collect::<Vec<_>>();

        let (signal_semaphores, signal_values): (Vec<vk::Semaphore>, Vec<u64>) = info
            .signal_semaphores
            .iter()
            .map(|semaphore| (semaphore.handle(), 0))
            .chain(
                info.signal_timeline_semaphores
                    .iter()
                    .map(|(semaphore, value)| (semaphore.handle(), *value)),
            )
            .unzip();

        let command_buffers = info
            .commands
            .iter()
            .map(|command_list| command_list.handle())
            .collect::<Vec<_>>();

        let mut timeline_submit_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);

        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_submit_info)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::BOTTOM_OF_PIPE])
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        self.deferred_destruct_recorders
            .entry(self.frame_index + 1)
            .or_default()
            .extend(info.commands.iter().map(|list| list.id));

        self.deferred_destruct_buffers
            .entry(self.frame_index + 1)
            .or_default()
            .extend(
                info.commands
                    .iter()
                    .flat_map(|list| list.deferred_delete_buffers.clone()),
            );

        unsafe {
            self.handle()
                .queue_submit(self.main_queue, &[submit_info], vk::Fence::null())
                .expect("Failed to submit queue");
        }

        self.frame_index += 1;
    }

    pub fn present(&self, info: PresentInfo) {
        let wait_semaphores = info
            .wait_semaphores
            .iter()
            .map(|semaphore| semaphore.handle())
            .collect::<Vec<_>>();

        let swapchains = [info.swapchain.handle()];
        let image_indices = [info.swapchain.last_aquired_image_index().unwrap()];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            info.swapchain
                .loader()
                .queue_present(self.main_queue, &present_info)
                .expect("Failed to present queue");
        }
    }

    pub fn collect_garbage(&mut self, timeline_semaphore: &TimelineSemaphore) {
        let gpu_count = unsafe {
            self.handle()
                .get_semaphore_counter_value(timeline_semaphore.handle())
        }
        .expect("Couldn't get semaphore value");

        for recorder_id in self
            .deferred_destruct_recorders
            .get_mut(&gpu_count)
            .unwrap_or(&mut Vec::new())
            .drain(0..)
        {
            self.command_recorder_pool
                .free_command_recorder(recorder_id);
        }

        for buffer_id in self
            .deferred_destruct_buffers
            .get_mut(&gpu_count)
            .unwrap_or(&mut Vec::new())
            .drain(0..)
            .collect::<Vec<_>>()
        {
            self.destroy_buffer(buffer_id);
        }
    }

    pub fn create_compute_pipeline(&self, info: ComputePipelineInfo) -> ComputePipeline {
        let shader_module_create_info =
            vk::ShaderModuleCreateInfo::default().code(info.shader.byte_code.as_slice());

        let shader_module = unsafe {
            self.handle()
                .create_shader_module(&shader_module_create_info, None)
        }
        .unwrap();

        let push_constant_ranges = if info.push_constant_size > 0 {
            vec![vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(info.push_constant_size)]
        } else {
            vec![]
        };

        let set_layouts = [self.gpu_resources.bindless_descriptor_set_layout];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&set_layouts);

        let pipeline_layout = unsafe {
            self.handle()
                .create_pipeline_layout(&pipeline_layout_create_info, None)
        }
        .unwrap();

        let shader_entry_cstring = std::ffi::CString::new(info.shader.entry_point.as_str())
            .expect("Failed to convert entry point to CString");

        let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(shader_module)
                    .name(shader_entry_cstring.as_c_str()),
            )
            .layout(pipeline_layout);

        let pipeline = unsafe {
            self.handle().create_compute_pipelines(
                vk::PipelineCache::null(),
                &[compute_pipeline_create_info],
                None,
            )
        }
        .unwrap()[0];

        unsafe {
            self.handle().destroy_shader_module(shader_module, None);
        }

        ComputePipeline {
            device_dep: self.create_dep(),
            inner: PipelineInner {
                pipeline,
                pipeline_layout,
            },
        }
    }

    pub fn cpu_frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn create_dep(&self) -> Arc<DeviceInner> {
        self.inner.clone()
    }

    pub fn inner(&self) -> &DeviceInner {
        self.inner.as_ref()
    }

    pub fn instance(&self) -> &InstanceInner {
        self.inner.instance_dep.as_ref()
    }

    pub fn handle(&self) -> &ash::Device {
        &self.inner.device
    }

    pub fn main_queue(&self) -> vk::Queue {
        self.main_queue
    }

    pub fn main_queue_family_index(&self) -> u32 {
        self.inner.main_queue_family_index
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
    ) -> vk::DescriptorSetLayout {
        let result = [
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, ty)| {
            let binding = vk::DescriptorSetLayoutBinding::default()
                .binding(i as u32)
                .descriptor_type(ty)
                .descriptor_count(1000)
                .stage_flags(vk::ShaderStageFlags::ALL);

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
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub dimensions: u32,
    pub extent: Extent3D,
    pub format: Format,
    pub usage: ImageUsageFlags,
}

impl ImageInfo {
    pub fn dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = dimensions;
        self
    }

    pub fn extent(mut self, extent: Extent3D) -> Self {
        self.extent = extent;
        self
    }

    pub fn format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    pub fn usage(mut self, usage: ImageUsageFlags) -> Self {
        self.usage = usage;
        self
    }
}

impl Default for ImageInfo {
    fn default() -> Self {
        ImageInfo {
            dimensions: 2,
            extent: Extent3D::new(0, 0, 0),
            format: Format::R8G8B8A8Unorm,
            usage: ImageUsageFlags::empty(),
        }
    }
}

pub struct Image {
    pub handle: vk::Image,
    pub view: Option<vk::ImageView>,
    pub info: ImageInfo,
    pub allocation: Option<Allocation>,
    pub is_swapchain_image: bool,
}

pub struct SubmitInfo<'a> {
    pub commands: Vec<CommandList>,
    pub wait_semaphores: Vec<&'a BinarySemaphore>,
    pub signal_semaphores: Vec<&'a BinarySemaphore>,
    pub signal_timeline_semaphores: Vec<(&'a TimelineSemaphore, u64)>,
}

pub struct PresentInfo<'a> {
    pub swapchain: &'a Swapchain,
    pub wait_semaphores: Vec<&'a BinarySemaphore>,
}

pub struct TypedMappedPtr<'a, T> {
    ptr: *mut T,
    memory: vk::DeviceMemory,
    device: &'a Device,
}

impl<T> std::ops::Deref for TypedMappedPtr<'_, T> {
    type Target = *mut T;
    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl<T> Drop for TypedMappedPtr<'_, T> {
    fn drop(&mut self) {
        unsafe { self.device.handle().unmap_memory(self.memory) };
    }
}

pub struct MappedPtr<'a> {
    ptr: *mut u8,
    memory: vk::DeviceMemory,
    device: &'a Device,
}

impl std::ops::Deref for MappedPtr<'_> {
    type Target = *mut u8;
    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl Drop for MappedPtr<'_> {
    fn drop(&mut self) {
        unsafe { self.device.handle().unmap_memory(self.memory) };
    }
}
