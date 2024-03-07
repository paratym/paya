use std::sync::Arc;

use ash::vk;

use crate::{
    device::{Device, DeviceInner},
    shader::ShaderInfo,
};

pub struct PipelineInner {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
}

pub struct ComputePipelineInfo {
    pub shader: ShaderInfo,
    pub push_constant_size: u32,
}

pub struct ComputePipeline {
    pub(crate) device_dep: Arc<DeviceInner>,
    pub(crate) inner: PipelineInner,
}

impl ComputePipeline {}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device_dep
                .device
                .destroy_pipeline(self.inner.pipeline, None);
            self.device_dep
                .device
                .destroy_pipeline_layout(self.inner.pipeline_layout, None);
        }
    }
}
