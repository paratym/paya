use std::sync::Arc;

use ash::vk;

use crate::device::{Device, DeviceInner};

pub struct BinarySemaphore {
    device_dep: Arc<DeviceInner>,
    handle: vk::Semaphore,
}

impl BinarySemaphore {
    pub(crate) fn new(device: &Device) -> Self {
        let create_info = vk::SemaphoreCreateInfo::default();

        let handle = unsafe {
            device
                .inner()
                .device
                .create_semaphore(&create_info, None)
                .expect("Failed to create semaphore")
        };

        BinarySemaphore {
            device_dep: device.create_dep(),
            handle,
        }
    }

    pub fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for BinarySemaphore {
    fn drop(&mut self) {
        unsafe {
            self.device_dep.device.destroy_semaphore(self.handle, None);
        }
    }
}

pub struct TimelineSemaphore {
    device_dep: Arc<DeviceInner>,
    handle: vk::Semaphore,
    value: u64,
}

impl TimelineSemaphore {
    pub(crate) fn new(device: &Device, value: u64) -> Self {
        let mut type_create_info =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);

        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_create_info);

        let handle = unsafe {
            device
                .inner()
                .device
                .create_semaphore(&create_info, None)
                .expect("Failed to create semaphore")
        };

        TimelineSemaphore {
            device_dep: device.create_dep(),
            handle,
            value,
        }
    }

    pub fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.device_dep.device.destroy_semaphore(self.handle, None);
        }
    }
}
