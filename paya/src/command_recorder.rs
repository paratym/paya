use std::sync::Arc;

use ash::vk;

use crate::{
    common::ImageTransition,
    device::{Device, DeviceInner},
    gpu_resources::ImageId,
    pipeline::ComputePipeline,
};

pub struct CommandList {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
}

impl CommandList {
    pub fn handle(&self) -> vk::CommandBuffer {
        self.command_buffer
    }
}

pub struct CommandRecorder {
    device_dep: Arc<DeviceInner>,
    pool: vk::CommandPool,
    current_command_list: CommandList,
}

impl CommandRecorder {
    pub(crate) fn new(device: &Device) -> Self {
        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.main_queue_family_index())
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool = unsafe {
            device
                .handle()
                .create_command_pool(&command_pool_create_info, None)
                .unwrap()
        };

        let mut s = CommandRecorder {
            device_dep: device.create_dep(),
            pool: command_pool,
            current_command_list: CommandList {
                command_pool,
                command_buffer: vk::CommandBuffer::null(),
            },
        };

        s.new_command_list(device);
        s
    }

    fn new_command_list(&mut self, device: &Device) {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            device
                .handle()
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()[0]
        };

        unsafe {
            device
                .handle()
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }

        self.current_command_list = CommandList {
            command_pool: self.pool,
            command_buffer,
        };
    }

    pub fn clear_color_image(
        &mut self,
        device: &Device,
        image: ImageId,
        red: f32,
        green: f32,
        blue: f32,
        alpha: f32,
    ) {
        let clear_color = vk::ClearColorValue {
            float32: [red, green, blue, alpha],
        };

        let image_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        unsafe {
            device.handle().cmd_clear_color_image(
                self.current_command_list.command_buffer,
                device.get_image(image).handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &[image_subresource_range],
            );
        }
    }

    pub fn blit_image_to_image(&mut self, device: &Device, src: ImageId, dst: ImageId) {
        let src_image = device.get_image(src);
        let dst_image = device.get_image(dst);

        let src_subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);
        let dst_subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);

        let region = vk::ImageBlit::default()
            .src_subresource(src_subresource)
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: src_image.info.extent.width as i32,
                    y: src_image.info.extent.height as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource)
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: dst_image.info.extent.width as i32,
                    y: dst_image.info.extent.height as i32,
                    z: 1,
                },
            ]);

        unsafe {
            device.handle().cmd_blit_image(
                self.current_command_list.command_buffer,
                src_image.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
                vk::Filter::LINEAR,
            );
        }
    }

    pub fn pipeline_barrier_image_transition(
        &mut self,
        device: &Device,
        transition: ImageTransition,
    ) {
        let image = device.get_image(transition.image);

        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(transition.src_access.into())
            .dst_access_mask(transition.dst_access.into())
            .old_layout(transition.src_layout.into())
            .new_layout(transition.dst_layout.into())
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.handle)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        unsafe {
            device.handle().cmd_pipeline_barrier(
                self.current_command_list.command_buffer,
                transition.src_access.vk_stages(),
                transition.dst_access.vk_stages(),
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    pub fn bind_compute_pipeline(&mut self, device: &Device, pipeline: &ComputePipeline) {
        unsafe {
            device.handle().cmd_bind_pipeline(
                self.current_command_list.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.inner.pipeline,
            );

            device.handle().cmd_bind_descriptor_sets(
                self.current_command_list.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.inner.pipeline_layout,
                0,
                &[device.gpu_resources.descriptor_set],
                &[],
            );
        }
    }

    pub fn upload_push_constants<T>(
        &mut self,
        device: &Device,
        pipeline: &ComputePipeline,
        data: &T,
    ) {
        let data = std::slice::from_ref(data);
        unsafe {
            device.handle().cmd_push_constants(
                self.current_command_list.command_buffer,
                pipeline.inner.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)),
            );
        }
    }

    pub fn dispatch(&mut self, device: &Device, x: u32, y: u32, z: u32) {
        unsafe {
            device
                .handle()
                .cmd_dispatch(self.current_command_list.command_buffer, x, y, z);
        }
    }

    pub fn finish(mut self, device: &Device) -> CommandList {
        unsafe {
            device
                .handle()
                .end_command_buffer(self.current_command_list.command_buffer)
                .unwrap();
        }

        self.current_command_list
    }
}
