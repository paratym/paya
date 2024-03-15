use std::sync::Arc;

use ash::vk::{self, Extent2D, ShaderStageFlags};

use crate::{
    common::{AttachmentLoadOp, AttachmentStoreOp, Format, ImageLayout, PolygonMode, Topology},
    device::{Device, DeviceInner},
    shader::ShaderInfo,
};

pub struct PipelineInner {
    pub(crate) device_dep: Arc<DeviceInner>,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        unsafe {
            self.device_dep.device.destroy_pipeline(self.pipeline, None);
            self.device_dep
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

pub enum RasterVertexAttributeType {
    Float,
    Vec2,
    Vec3,
    Vec4,
}

impl RasterVertexAttributeType {
    pub fn size(&self) -> u32 {
        match self {
            RasterVertexAttributeType::Float => 4,
            RasterVertexAttributeType::Vec2 => 8,
            RasterVertexAttributeType::Vec3 => 12,
            RasterVertexAttributeType::Vec4 => 16,
        }
    }

    pub(crate) fn vk_format(&self) -> vk::Format {
        match self {
            RasterVertexAttributeType::Float => vk::Format::R32_SFLOAT,
            RasterVertexAttributeType::Vec2 => vk::Format::R32G32_SFLOAT,
            RasterVertexAttributeType::Vec3 => vk::Format::R32G32B32_SFLOAT,
            RasterVertexAttributeType::Vec4 => vk::Format::R32G32B32A32_SFLOAT,
        }
    }
}

pub struct RasterPipelineInfo {
    pub vertex_shader: ShaderInfo,
    pub fragment_shader: ShaderInfo,
    pub push_constant_size: u32,

    pub vertex_attributes: Vec<RasterVertexAttributeType>,
    pub polygon_mode: PolygonMode,
    pub topology: Topology,
    pub primitive_restart_enable: bool,
    pub line_width: f32,

    // Only support 1 subpass for now
    pub color_attachments: Vec<Format>,
}

pub struct RasterPipeline {
    pub(crate) inner: PipelineInner,
}

pub struct ComputePipelineInfo {
    pub shader: ShaderInfo,
    pub push_constant_size: u32,
}

pub struct ComputePipeline {
    pub(crate) inner: PipelineInner,
}

pub trait Pipeline {
    fn inner(&self) -> &PipelineInner;
    fn shader_stages(&self) -> ShaderStageFlags;
}

impl Pipeline for RasterPipeline {
    fn inner(&self) -> &PipelineInner {
        &self.inner
    }
    fn shader_stages(&self) -> ShaderStageFlags {
        ShaderStageFlags::ALL_GRAPHICS
    }
}

impl Pipeline for ComputePipeline {
    fn inner(&self) -> &PipelineInner {
        &self.inner
    }

    fn shader_stages(&self) -> ShaderStageFlags {
        ShaderStageFlags::COMPUTE
    }
}
