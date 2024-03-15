use ash::vk;
use bitflags::bitflags;

use crate::gpu_resources::{BufferId, ImageId};

#[derive(Debug, Clone, Copy)]
pub enum Format {
    R8G8B8A8Unorm,
    R8G8B8A8Srgb,

    B8G8R8A8Unorm,
    B8G8R8A8Srgb,
}

impl Into<vk::Format> for Format {
    fn into(self) -> vk::Format {
        match self {
            Format::R8G8B8A8Unorm => vk::Format::R8G8B8A8_UNORM,
            Format::R8G8B8A8Srgb => vk::Format::R8G8B8A8_SRGB,
            Format::B8G8R8A8Unorm => vk::Format::B8G8R8A8_UNORM,
            Format::B8G8R8A8Srgb => vk::Format::B8G8R8A8_SRGB,
        }
    }
}

impl From<vk::Format> for Format {
    fn from(format: vk::Format) -> Self {
        match format {
            vk::Format::R8G8B8A8_UNORM => Format::R8G8B8A8Unorm,
            vk::Format::R8G8B8A8_SRGB => Format::R8G8B8A8Srgb,
            vk::Format::B8G8R8A8_UNORM => Format::B8G8R8A8Unorm,
            vk::Format::B8G8R8A8_SRGB => Format::B8G8R8A8Srgb,
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ImageTiling {
    Optimal,
    Linear,
}

impl Into<vk::ImageTiling> for ImageTiling {
    fn into(self) -> vk::ImageTiling {
        match self {
            ImageTiling::Optimal => vk::ImageTiling::OPTIMAL,
            ImageTiling::Linear => vk::ImageTiling::LINEAR,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

impl Extent2D {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }
}

impl Into<vk::Extent2D> for Extent2D {
    fn into(self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }
}

impl From<Extent3D> for Extent2D {
    fn from(value: Extent3D) -> Self {
        Self {
            width: value.width,
            height: value.height,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Extent3D {
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    pub fn depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }
}

impl Into<vk::Extent3D> for Extent3D {
    fn into(self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
        }
    }
}

impl From<Extent2D> for Extent3D {
    fn from(value: Extent2D) -> Self {
        Self {
            width: value.width,
            height: value.height,
            depth: 1,
        }
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct ImageUsageFlags: u32 {
        const TRANSFER_SRC = vk::ImageUsageFlags::TRANSFER_SRC.as_raw();
        const TRANSFER_DST = vk::ImageUsageFlags::TRANSFER_DST.as_raw();
        const SAMPLED = vk::ImageUsageFlags::SAMPLED.as_raw();
        const STORAGE = vk::ImageUsageFlags::STORAGE.as_raw();
        const COLOR_ATTACHMENT = vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw();
        const DEPTH_STENCIL_ATTACHMENT = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT.as_raw();
        const TRANSIENT_ATTACHMENT = vk::ImageUsageFlags::TRANSIENT_ATTACHMENT.as_raw();
        const INPUT_ATTACHMENT = vk::ImageUsageFlags::INPUT_ATTACHMENT.as_raw();
    }
}

impl ImageUsageFlags {
    pub fn needs_view(&self) -> bool {
        self.contains(ImageUsageFlags::SAMPLED)
            || self.contains(ImageUsageFlags::STORAGE)
            || self.contains(ImageUsageFlags::COLOR_ATTACHMENT)
            || self.contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            || self.contains(ImageUsageFlags::INPUT_ATTACHMENT)
    }
}

impl Into<vk::ImageUsageFlags> for ImageUsageFlags {
    fn into(self) -> vk::ImageUsageFlags {
        vk::ImageUsageFlags::from_raw(self.bits())
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct BufferUsageFlags: u32 {
        const UNIFORM = vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw();
        const STORAGE = vk::BufferUsageFlags::STORAGE_BUFFER.as_raw();
        const TRANSFER_SRC = vk::BufferUsageFlags::TRANSFER_SRC.as_raw();
        const TRANSFER_DST = vk::BufferUsageFlags::TRANSFER_DST.as_raw();
        const INDEX = vk::BufferUsageFlags::INDEX_BUFFER.as_raw();
        const VERTEX = vk::BufferUsageFlags::VERTEX_BUFFER.as_raw();
        const INDIRECT = vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw();
    }
}

impl Into<vk::BufferUsageFlags> for BufferUsageFlags {
    fn into(self) -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::from_raw(self.bits())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImageLayout {
    Undefined,
    General,
    ColorAttachmentOptimal,
    DepthStencilAttachmentOptimal,
    DepthStencilReadOnlyOptimal,
    ShaderReadOnlyOptimal,
    TransferSrcOptimal,
    TransferDstOptimal,
    Preinitialized,
    PresentSrc,
}

impl Into<vk::ImageLayout> for ImageLayout {
    fn into(self) -> vk::ImageLayout {
        match self {
            ImageLayout::Undefined => vk::ImageLayout::UNDEFINED,
            ImageLayout::General => vk::ImageLayout::GENERAL,
            ImageLayout::ColorAttachmentOptimal => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ImageLayout::DepthStencilAttachmentOptimal => {
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            }
            ImageLayout::DepthStencilReadOnlyOptimal => {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            }
            ImageLayout::ShaderReadOnlyOptimal => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ImageLayout::TransferSrcOptimal => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ImageLayout::TransferDstOptimal => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageLayout::Preinitialized => vk::ImageLayout::PREINITIALIZED,
            ImageLayout::PresentSrc => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct AccessFlags: u32 {
        const INDIRECT_COMMAND_READ = vk::AccessFlags::INDIRECT_COMMAND_READ.as_raw();
        const INDEX_READ = vk::AccessFlags::INDEX_READ.as_raw();
        const VERTEX_ATTRIBUTE_READ = vk::AccessFlags::VERTEX_ATTRIBUTE_READ.as_raw();
        const UNIFORM_READ = vk::AccessFlags::UNIFORM_READ.as_raw();
        const INPUT_ATTACHMENT_READ = vk::AccessFlags::INPUT_ATTACHMENT_READ.as_raw();
        const SHADER_READ = vk::AccessFlags::SHADER_READ.as_raw();
        const SHADER_WRITE = vk::AccessFlags::SHADER_WRITE.as_raw();
        const COLOR_ATTACHMENT_READ = vk::AccessFlags::COLOR_ATTACHMENT_READ.as_raw();
        const COLOR_ATTACHMENT_WRITE = vk::AccessFlags::COLOR_ATTACHMENT_WRITE.as_raw();
        const DEPTH_STENCIL_ATTACHMENT_READ = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ.as_raw();
        const DEPTH_STENCIL_ATTACHMENT_WRITE = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw();
        const TRANSFER_READ = vk::AccessFlags::TRANSFER_READ.as_raw();
        const TRANSFER_WRITE = vk::AccessFlags::TRANSFER_WRITE.as_raw();
        const HOST_READ = vk::AccessFlags::HOST_READ.as_raw();
        const HOST_WRITE = vk::AccessFlags::HOST_WRITE.as_raw();
        const MEMORY_READ = vk::AccessFlags::MEMORY_READ.as_raw();
        const MEMORY_WRITE = vk::AccessFlags::MEMORY_WRITE.as_raw();
    }
}

impl AccessFlags {
    pub fn vk_stages(&self) -> vk::PipelineStageFlags {
        let mut flags = vk::PipelineStageFlags::empty();

        if self.contains(AccessFlags::INDIRECT_COMMAND_READ) {
            flags |= vk::PipelineStageFlags::DRAW_INDIRECT;
        }

        if self.contains(AccessFlags::INDEX_READ)
            || self.contains(AccessFlags::VERTEX_ATTRIBUTE_READ)
            || self.contains(AccessFlags::UNIFORM_READ)
            || self.contains(AccessFlags::INPUT_ATTACHMENT_READ)
            || self.contains(AccessFlags::SHADER_READ)
            || self.contains(AccessFlags::SHADER_WRITE)
        {
            flags |= vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER
                | vk::PipelineStageFlags::COMPUTE_SHADER;
        }

        if self.contains(AccessFlags::COLOR_ATTACHMENT_READ)
            || self.contains(AccessFlags::COLOR_ATTACHMENT_WRITE)
        {
            flags |= vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        }

        if self.contains(AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
            || self.contains(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
        {
            flags |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
        }

        if self.contains(AccessFlags::TRANSFER_READ) || self.contains(AccessFlags::TRANSFER_WRITE) {
            flags |= vk::PipelineStageFlags::TRANSFER;
        }

        if self.contains(AccessFlags::HOST_READ) || self.contains(AccessFlags::HOST_WRITE) {
            flags |= vk::PipelineStageFlags::HOST;
        }

        if self.contains(AccessFlags::MEMORY_READ) || self.contains(AccessFlags::MEMORY_WRITE) {
            flags |= vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        }

        if flags.is_empty() {
            flags |= vk::PipelineStageFlags::TOP_OF_PIPE;
        }

        flags
    }
}

impl Into<vk::AccessFlags> for AccessFlags {
    fn into(self) -> vk::AccessFlags {
        vk::AccessFlags::from_raw(self.bits())
    }
}

pub struct ImageTransition {
    pub image: ImageId,
    pub src_layout: ImageLayout,
    pub dst_layout: ImageLayout,
    pub src_access: AccessFlags,
    pub dst_access: AccessFlags,
}

pub struct BufferTransition {
    pub buffer: BufferId,
    pub src_access: AccessFlags,
    pub dst_access: AccessFlags,
}

#[derive(Debug, Clone, Copy)]
pub enum PolygonMode {
    Line,
    Fill,
}

impl Into<vk::PolygonMode> for PolygonMode {
    fn into(self) -> vk::PolygonMode {
        match self {
            PolygonMode::Line => vk::PolygonMode::LINE,
            PolygonMode::Fill => vk::PolygonMode::FILL,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AttachmentLoadOp {
    Undefined,
    Clear,
    Store,
}

impl Into<vk::AttachmentLoadOp> for AttachmentLoadOp {
    fn into(self) -> vk::AttachmentLoadOp {
        match self {
            AttachmentLoadOp::Undefined => vk::AttachmentLoadOp::DONT_CARE,
            AttachmentLoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
            AttachmentLoadOp::Store => vk::AttachmentLoadOp::LOAD,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AttachmentStoreOp {
    Undefined,
    Store,
}

impl Into<vk::AttachmentStoreOp> for AttachmentStoreOp {
    fn into(self) -> vk::AttachmentStoreOp {
        match self {
            AttachmentStoreOp::Undefined => vk::AttachmentStoreOp::DONT_CARE,
            AttachmentStoreOp::Store => vk::AttachmentStoreOp::STORE,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ClearValue {
    None,
    Color(f32, f32, f32),
    Depth(f32),
}

impl Into<vk::ClearValue> for ClearValue {
    fn into(self) -> vk::ClearValue {
        match self {
            ClearValue::Color(r, g, b) => vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [r, g, b, 1.0],
                },
            },
            ClearValue::Depth(depth) => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
            },
            ClearValue::None => vk::ClearValue::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Topology {
    TriangleList,
}

impl Into<vk::PrimitiveTopology> for Topology {
    fn into(self) -> vk::PrimitiveTopology {
        match self {
            Topology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        }
    }
}
