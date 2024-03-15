use std::{ffi::CString, sync::Arc};

use ash::vk;

pub struct InstanceCreateInfo<'a> {
    pub display_handle: Option<&'a dyn raw_window_handle::HasDisplayHandle>,
}

#[derive(Clone)]
pub struct InstanceInner {
    pub(crate) loader: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) debug_utils: ash::extensions::ext::DebugUtils,
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
}

pub struct Instance {
    inner: Arc<InstanceInner>,
}

impl Instance {
    pub fn new(create_info: InstanceCreateInfo<'_>) -> Self {
        let loader = unsafe { ash::Entry::load().unwrap() };
        let app_name = std::ffi::CString::new("Paya").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&app_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let c_instance_extensions = vec![
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::NAME.to_owned(),
        ];

        let c_instance_layers = vec![
            #[cfg(debug_assertions)]
            CString::new("VK_LAYER_KHRONOS_validation").unwrap(),
        ];

        let mut c_ptr_instance_extensions = c_instance_extensions
            .iter()
            .map(|s| s.as_ptr())
            .collect::<Vec<_>>();
        let c_ptr_instance_layers = c_instance_layers
            .iter()
            .map(|s| s.as_ptr())
            .collect::<Vec<_>>();

        if let Some(display_handle) = create_info.display_handle {
            c_ptr_instance_extensions.extend(
                ash_window::enumerate_required_extensions(display_handle.display_handle().unwrap())
                    .unwrap(),
            );
        }

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&c_ptr_instance_extensions)
            .enabled_layer_names(&c_ptr_instance_layers);

        let instance = unsafe { loader.create_instance(&instance_create_info, None).unwrap() };

        let debug_utils = ash::extensions::ext::DebugUtils::new(&loader, &instance);
        let debug_utils_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(Self::debug_utils_callback));

        let debug_utils_messenger =
            unsafe { debug_utils.create_debug_utils_messenger(&debug_utils_create_info, None) }
                .expect("Failed to create debug utils messenger");

        Instance {
            inner: Arc::new(InstanceInner {
                loader,
                instance,
                debug_utils,
                debug_utils_messenger,
            }),
        }
    }

    unsafe extern "system" fn debug_utils_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_data: *mut std::ffi::c_void,
    ) -> vk::Bool32 {
        let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
        let severity = format!("{:?}", message_severity).to_lowercase();
        let ty = format!("{:?}", message_type).to_lowercase();
        println!("[Debug][{}][{}] {:?}", severity, ty, message);
        vk::FALSE
    }

    pub fn create_dep(&self) -> Arc<InstanceInner> {
        self.inner.clone()
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.inner.loader
    }

    pub unsafe fn handle(&self) -> &ash::Instance {
        &self.inner.instance
    }
}

impl Drop for InstanceInner {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}
