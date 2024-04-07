use std::{collections::HashSet, sync::Arc};

use ash::vk;

use crate::{
    common::{
        AttachmentLoadOp, AttachmentStoreOp, BufferTransition, ClearValue, Extent2D, ImageLayout,
        ImageTransition,
    },
    device::{Device, DeviceInner},
    gpu_resources::{BufferId, ImageId},
    pipeline::{ComputePipeline, Pipeline, RasterPipeline},
};

#[derive(Clone)]
pub struct CommandList {
    pub(crate) id: CommandRecorderId,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    pub(crate) deferred_delete_buffers: Vec<BufferId>,
}

impl CommandList {
    pub fn handle(&self) -> vk::CommandBuffer {
        self.command_buffer
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommandRecorderId(u32);

pub struct CommandRecorderPool {
    device_dep: Arc<DeviceInner>,
    recorders: Vec<CommandRecorder>,
    free_recorders: Vec<CommandRecorderId>,
}

impl CommandRecorderPool {
    pub(crate) fn new(device_dep: Arc<DeviceInner>) -> Self {
        Self {
            device_dep,
            recorders: Vec::new(),
            free_recorders: Vec::new(),
        }
    }

    pub(crate) fn create_command_recorder(&mut self) -> CommandRecorder {
        if self.free_recorders.is_empty() {
            let id = CommandRecorderId(self.recorders.len() as u32);
            self.recorders
                .push(CommandRecorder::new(self.device_dep.clone(), id));
            self.free_recorders.push(id);
        }

        let recorder_id = self
            .free_recorders
            .pop()
            .expect("Failed to create a command recorder");

        self.get_recorder(recorder_id).clone()
    }

    pub(crate) fn free_command_recorder(&mut self, id: CommandRecorderId) {
        self.get_recorder(id).reset();
        self.free_recorders.push(id);
    }

    fn get_recorder(&self, id: CommandRecorderId) -> &CommandRecorder {
        &self.recorders[id.0 as usize]
    }
}

impl Drop for CommandRecorderPool {
    fn drop(&mut self) {
        unsafe { self.device_dep.device.device_wait_idle() }.expect("Faile dto wait for device");
        for recorder in &self.recorders {
            unsafe {
                self.device_dep
                    .device
                    .destroy_command_pool(recorder.pool, None)
            };
        }
    }
}

#[derive(Clone)]
pub struct CommandRecorder {
    device_dep: Arc<DeviceInner>,
    id: CommandRecorderId,
    pool: vk::CommandPool,
    current_command_list: CommandList,
}

impl CommandRecorder {
    pub(crate) fn new(device_dep: Arc<DeviceInner>, id: CommandRecorderId) -> Self {
        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device_dep.main_queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool = unsafe {
            device_dep
                .device
                .create_command_pool(&command_pool_create_info, None)
                .unwrap()
        };

        let mut s = CommandRecorder {
            device_dep,
            pool: command_pool,
            id,
            current_command_list: CommandList {
                deferred_delete_buffers: Vec::new(),
                id,
                command_pool,
                command_buffer: vk::CommandBuffer::null(),
            },
        };

        s.new_command_list();
        s
    }

    fn reset(&self) {
        unsafe {
            self.device_dep
                .device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
        }
        .expect("Couldnt reset command pool");

        unsafe {
            self.device_dep.device.begin_command_buffer(
                self.current_command_list.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .expect("Couldnt reset and re begin command buffer");
    }

    fn new_command_list(&mut self) {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            self.device_dep
                .device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()[0]
        };

        unsafe {
            self.device_dep
                .device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }

        self.current_command_list = CommandList {
            deferred_delete_buffers: Vec::new(),
            id: self.id,
            command_pool: self.pool,
            command_buffer,
        };
    }

    pub fn destroy_buffer_deferred(&mut self, id: BufferId) {
        self.current_command_list.deferred_delete_buffers.push(id);
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

    pub fn copy_buffer_to_buffer(
        &mut self,
        device: &Device,
        src: BufferId,
        src_offset: u64,
        dst: BufferId,
        dst_offset: u64,
        size: u64,
    ) {
        let src_buffer = device.get_buffer(src);
        let dst_buffer = device.get_buffer(dst);

        unsafe {
            device.handle().cmd_copy_buffer(
                self.current_command_list.handle(),
                src_buffer.handle,
                dst_buffer.handle,
                &[vk::BufferCopy::default()
                    .size(size)
                    .src_offset(src_offset)
                    .dst_offset(dst_offset)],
            )
        }
    }

    pub fn copy_buffer_to_buffer_multiple(
        &mut self,
        device: &Device,
        src: BufferId,
        dst: BufferId,
        regions: Vec<CopyRegion>,
    ) {
        let src_buffer = device.get_buffer(src);
        let dst_buffer = device.get_buffer(dst);

        let vk_regions = regions
            .into_iter()
            .map(|region| region.into())
            .collect::<Vec<vk::BufferCopy>>();

        unsafe {
            device.handle().cmd_copy_buffer(
                self.current_command_list.handle(),
                src_buffer.handle,
                dst_buffer.handle,
                &vk_regions,
            )
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

    pub fn pipeline_barrier_buffer_transition(
        &mut self,
        device: &Device,
        transition: BufferTransition,
    ) {
        let buffer = device.get_buffer(transition.buffer);

        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(buffer.handle)
            .size(buffer.size)
            .offset(0);

        unsafe {
            device.handle().cmd_pipeline_barrier(
                self.current_command_list.command_buffer,
                transition.src_access.vk_stages(),
                transition.dst_access.vk_stages(),
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
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

    pub fn bind_graphics_pipeline(&mut self, device: &Device, pipeline: &RasterPipeline) {
        unsafe {
            device.handle().cmd_bind_pipeline(
                self.current_command_list.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.inner.pipeline,
            );
            device.handle().cmd_bind_descriptor_sets(
                self.current_command_list.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.inner.pipeline_layout,
                0,
                &[device.gpu_resources.descriptor_set],
                &[],
            );
        }
    }

    pub fn begin_rendering(&mut self, device: &Device, info: &BeginRenderingInfo) {
        let color_attachments = info
            .color_attachments
            .iter()
            .map(|info| {
                let image_info = device.get_image(info.image);

                vk::RenderingAttachmentInfo::default()
                    .image_view(
                        image_info
                            .view
                            .expect("Image doesnt have color attachment usage applied."),
                    )
                    .load_op(info.load_op.clone().into())
                    .store_op(info.store_op.clone().into())
                    .clear_value(info.clear_value.clone().into())
                    .image_layout(info.layout.into())
            })
            .collect::<Vec<_>>();

        let rendering_info = vk::RenderingInfo::default()
            .render_area(
                vk::Rect2D::default()
                    .offset(vk::Offset2D::default())
                    .extent(info.render_area.into()),
            )
            .color_attachments(&color_attachments)
            .layer_count(1)
            .view_mask(0);

        unsafe {
            device
                .inner()
                .dynamic_rendering_loader
                .cmd_begin_rendering(self.current_command_list.command_buffer, &rendering_info)
        };
    }

    pub fn end_rendering(&mut self, device: &Device) {
        unsafe {
            device
                .inner()
                .dynamic_rendering_loader
                .cmd_end_rendering(self.current_command_list.command_buffer);
        }
    }

    pub fn set_scissor(&mut self, device: &Device, extent: Extent2D) {
        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D::default())
            .extent(extent.into());

        unsafe {
            device.handle().cmd_set_scissor(
                self.current_command_list.command_buffer,
                0,
                &[scissor],
            );
        }
    }

    pub fn set_viewport(&mut self, device: &Device, extent: Extent2D) {
        let viewport = vk::Viewport::default()
            .width(extent.width as f32)
            .height(extent.height as f32)
            .max_depth(1.0);

        unsafe {
            device.handle().cmd_set_viewport(
                self.current_command_list.command_buffer,
                0,
                &[viewport],
            );
        }
    }

    pub fn set_index_buffer(&mut self, device: &Device, buffer: BufferId) {
        let buffer = device.get_buffer(buffer);
        unsafe {
            device.handle().cmd_bind_index_buffer(
                self.current_command_list.command_buffer,
                buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
        }
    }

    pub fn set_vertex_buffer(&mut self, device: &Device, buffer: BufferId) {
        let buffer = device.get_buffer(buffer);
        unsafe {
            device.handle().cmd_bind_vertex_buffers(
                self.current_command_list.command_buffer,
                0,
                &[buffer.handle],
                &[0],
            );
        }
    }

    pub fn draw_indexed(&mut self, device: &Device, index_count: u32) {
        unsafe {
            device.handle().cmd_draw_indexed(
                self.current_command_list.command_buffer,
                index_count,
                1,
                0,
                0,
                0,
            );
        }
    }

    pub fn upload_push_constants<T>(&mut self, device: &Device, pipeline: &dyn Pipeline, data: &T) {
        let data = std::slice::from_ref(data);
        unsafe {
            device.handle().cmd_push_constants(
                self.current_command_list.command_buffer,
                pipeline.inner().pipeline_layout,
                pipeline.shader_stages(),
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

pub struct RenderingAttachment {
    pub image: ImageId,
    pub layout: ImageLayout,
    pub load_op: AttachmentLoadOp,
    pub store_op: AttachmentStoreOp,
    pub clear_value: ClearValue,
}

pub struct BeginRenderingInfo {
    pub render_area: Extent2D,
    pub color_attachments: Vec<RenderingAttachment>,
}

pub struct CopyRegion {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}

impl Into<vk::BufferCopy> for CopyRegion {
    fn into(self) -> vk::BufferCopy {
        vk::BufferCopy::default()
            .size(self.size)
            .src_offset(self.src_offset)
            .dst_offset(self.dst_offset)
    }
}
