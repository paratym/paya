use std::sync::Arc;

use ash::{extensions::khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    common::{Extent2D, Extent3D, Format, ImageUsageFlags},
    device::{Device, DeviceInner, ImageInfo},
    gpu_resources::ImageId,
    sync::{BinarySemaphore, TimelineSemaphore},
};

pub struct SwapchainCreateInfo<'a> {
    pub window_handle: &'a dyn HasWindowHandle,
    pub display_handle: &'a dyn HasDisplayHandle,
    pub preferred_extent: (u32, u32),
    pub image_usage: ImageUsageFlags,
    pub max_frames_in_flight: u32,
}

struct InternalSwapchainKHRCreateInfo {
    surface: vk::SurfaceKHR,
    old_swapchain: Option<vk::SwapchainKHR>,
    preferred_extent: vk::Extent2D,
    image_usage: ImageUsageFlags,
    max_frames_in_flight: u32,
}

pub struct SwapchainInfo {
    pub format: Format,
    pub extent: Extent2D,
    pub image_usage: ImageUsageFlags,
    pub max_frames_in_flight: u32,
}

pub struct Swapchain {
    device_dep: Arc<DeviceInner>,
    swapchain_loader: khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    surface_loader: khr::Surface,
    surface: vk::SurfaceKHR,
    images: Vec<ImageId>,
    info: SwapchainInfo,

    acquire_image_semaphores: Vec<BinarySemaphore>,
    present_image_semaphores: Vec<BinarySemaphore>,
    gpu_timeline_semaphore: TimelineSemaphore,
    cpu_timeline: u64,

    last_aquired_image_index: Option<u32>,
}

impl Swapchain {
    pub(crate) fn new(device: &mut Device, create_info: SwapchainCreateInfo<'_>) -> Self {
        let surface_loader =
            khr::Surface::new(&device.instance().loader, &device.instance().instance);

        let surface = unsafe {
            ash_window::create_surface(
                &device.instance().loader,
                &device.instance().instance,
                create_info.display_handle.display_handle().unwrap(),
                create_info.window_handle.window_handle().unwrap(),
                None,
            )
        }
        .unwrap();

        let swapchain_loader = khr::Swapchain::new(&device.instance().instance, device.handle());

        let (swapchain, images, info) = Self::create_swapchain(
            device.inner(),
            &swapchain_loader,
            InternalSwapchainKHRCreateInfo {
                surface,
                old_swapchain: None,
                preferred_extent: vk::Extent2D {
                    width: create_info.preferred_extent.0,
                    height: create_info.preferred_extent.1,
                },
                image_usage: create_info.image_usage,
                max_frames_in_flight: create_info.max_frames_in_flight,
            },
        );

        let images = images
            .into_iter()
            .map(|image| {
                device.create_swapchain_image(
                    image,
                    &ImageInfo {
                        dimensions: 2,
                        extent: Extent3D::new(info.extent.width, info.extent.height, 1),
                        format: info.format.clone(),
                        usage: info.image_usage,
                    },
                )
            })
            .collect();

        let (acquire_image_semaphores, present_image_semaphores) = (0..create_info
            .max_frames_in_flight)
            .map(|_| (BinarySemaphore::new(device), BinarySemaphore::new(device)))
            .unzip();

        let gpu_timeline_semaphore = TimelineSemaphore::new(device, 0);

        Swapchain {
            device_dep: device.create_dep(),
            swapchain_loader,
            swapchain,
            surface_loader,
            surface,
            images,
            info,
            acquire_image_semaphores,
            present_image_semaphores,
            gpu_timeline_semaphore,
            cpu_timeline: 0,
            last_aquired_image_index: None,
        }
    }

    fn create_swapchain(
        device_inner: &DeviceInner,
        swapchain_loader: &khr::Swapchain,
        info: InternalSwapchainKHRCreateInfo,
    ) -> (vk::SwapchainKHR, Vec<vk::Image>, SwapchainInfo) {
        let surface_loader = khr::Surface::new(
            &device_inner.instance_dep.loader,
            &device_inner.instance_dep.instance,
        );
        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(
                    device_inner.physical_device,
                    info.surface,
                )
                .unwrap()
        };

        let surface_formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(device_inner.physical_device, info.surface)
                .unwrap()
        };

        let surface_present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(
                    device_inner.physical_device,
                    info.surface,
                )
                .unwrap()
        };

        let surface_format = surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&surface_formats[0]);

        let present_mode = surface_present_modes
            .iter()
            .find(|&present_mode| *present_mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(&vk::PresentModeKHR::FIFO);

        let extent = info.preferred_extent;

        let image_count = u32::max(surface_capabilities.min_image_count, 2);

        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(info.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(info.image_usage.into())
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(*present_mode)
            .clipped(true);

        if let Some(old_swapchain) = info.old_swapchain {
            swapchain_create_info = swapchain_create_info.old_swapchain(old_swapchain);
        }

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap()
        };

        if let Some(old_swapchain) = info.old_swapchain {
            unsafe {
                device_inner.device.device_wait_idle().unwrap();
                swapchain_loader.destroy_swapchain(old_swapchain, None);
            }
        }

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };

        (
            swapchain,
            images,
            SwapchainInfo {
                format: Format::from(surface_format.format),
                extent: Extent2D::new(extent.width, extent.height),
                image_usage: info.image_usage,
                max_frames_in_flight: info.max_frames_in_flight,
            },
        )
    }

    pub fn resize(&mut self, device: &mut Device, width: u32, height: u32) {
        let (new_swapchain, images, info) = Self::create_swapchain(
            &self.device_dep,
            &self.swapchain_loader,
            InternalSwapchainKHRCreateInfo {
                surface: self.surface,
                old_swapchain: Some(self.swapchain),
                preferred_extent: vk::Extent2D { width, height },
                image_usage: self.info.image_usage,
                max_frames_in_flight: self.info.max_frames_in_flight,
            },
        );

        self.swapchain = new_swapchain;
        self.images = images
            .into_iter()
            .map(|image| {
                device.create_swapchain_image(
                    image,
                    &ImageInfo {
                        dimensions: 2,
                        extent: Extent3D::new(info.extent.width, info.extent.height, 1),
                        format: info.format.clone(),
                        usage: info.image_usage,
                    },
                )
            })
            .collect();

        self.info.extent = Extent2D::new(width, height);
    }

    pub fn info(&self) -> &SwapchainInfo {
        &self.info
    }

    pub(crate) fn handle(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub(crate) fn loader(&self) -> &khr::Swapchain {
        &self.swapchain_loader
    }

    pub fn acquire_next_image(&mut self) -> Option<ImageId> {
        let gpu_index = unsafe {
            self.device_dep
                .device
                .get_semaphore_counter_value(self.gpu_timeline_semaphore.handle())
        };

        // println!(
        //     "CPU is {} frames ahead of GPU",
        //   self.cpu_timeline - gpu_index.unwrap()
        //);

        let acquire_semaphore = &self.acquire_image_semaphores
            [((self.cpu_timeline + 1) % self.info.max_frames_in_flight as u64) as usize];
        let result = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                5e+9 as u64,
                acquire_semaphore.handle(),
                vk::Fence::null(),
            )
        };
        let result = match result {
            Ok((image_index, _)) => Some(image_index),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => None,
            Err(vk::Result::SUBOPTIMAL_KHR) => None,
            Err(result) => panic!("Failed to acquire next image: {:?}", result),
        };
        self.last_aquired_image_index = result;

        if result.is_some() {
            self.cpu_timeline += 1;
        }
        result.map(|image_index| self.images[image_index as usize])
    }

    pub fn current_acquire_semaphore(&self) -> &BinarySemaphore {
        &self.acquire_image_semaphores
            [(self.cpu_timeline % self.info.max_frames_in_flight as u64) as usize]
    }

    pub fn current_present_semaphore(&self) -> &BinarySemaphore {
        &self.acquire_image_semaphores
            [(self.cpu_timeline % self.info.max_frames_in_flight as u64) as usize]
    }

    pub fn gpu_timeline_semaphore(&self) -> &TimelineSemaphore {
        &self.gpu_timeline_semaphore
    }

    pub fn last_aquired_image_index(&self) -> Option<u32> {
        self.last_aquired_image_index
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        // This is safe because we are the only one who has access to the swapchain
        unsafe {
            self.device_dep.device.device_wait_idle().unwrap();
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
