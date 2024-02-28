use paya::{
    common::{AccessFlags, ImageLayout, ImageTransition, ImageUsageFlags},
    device::{Device, ImageInfo, PresentInfo, SubmitInfo},
    gpu_resources::{self, GpuResourcePool},
    instance::{Instance, InstanceCreateInfo},
    swapchain::SwapchainCreateInfo,
    task_list::{Task, TaskList},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Window::new(&event_loop).unwrap();

    // Initialize Paya
    let instance = Instance::new(InstanceCreateInfo {
        display_handle: Some(&window),
    });
    let mut device = Device::new(&instance, |device_properties| {
        // Select the first discrete GPU
        let score = match device_properties.device_type {
            paya::device::DeviceType::Discrete => 100,
            _ => 0,
        };

        score
    });
    let mut swapchain = device.create_swapchain(SwapchainCreateInfo {
        window_handle: &window,
        display_handle: &window,
        preferred_extent: (1280, 720),
        image_usage: ImageUsageFlags::TRANSFER_DST,
        max_frames_in_flight: 2,
    });

    event_loop
        .run(|event, window| {
            window.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        window.exit();
                    }
                    WindowEvent::Resized(size) => {
                        swapchain.resize(&mut device, size.width, size.height);
                    }
                    WindowEvent::RedrawRequested => {
                        let Some(image) = swapchain.acquire_next_image() else {
                            return;
                        };

                        let mut recorder = device.create_command_recorder();

                        recorder.pipeline_barrier_image_transition(
                            &device,
                            ImageTransition {
                                image,
                                src_layout: ImageLayout::Undefined,
                                src_access: AccessFlags::empty(),
                                dst_layout: ImageLayout::TransferDstOptimal,
                                dst_access: AccessFlags::TRANSFER_WRITE,
                            },
                        );

                        recorder.clear_color_image(&device, image, 1.0, 0.0, 0.0, 1.0);

                        recorder.pipeline_barrier_image_transition(
                            &device,
                            ImageTransition {
                                image,
                                src_layout: ImageLayout::TransferDstOptimal,
                                src_access: AccessFlags::TRANSFER_WRITE,
                                dst_layout: ImageLayout::PresentSrc,
                                dst_access: AccessFlags::empty(),
                            },
                        );

                        let command_buffer = recorder.finish(&device);

                        device.submit(SubmitInfo {
                            commands: vec![command_buffer],
                            wait_semaphores: vec![swapchain.current_acquire_semaphore()],
                            signal_semaphores: vec![swapchain.current_present_semaphore()],
                            signal_timeline_semaphores: vec![(
                                swapchain.gpu_timeline_semaphore(),
                                device.cpu_frame_index() as u64 + 1,
                            )],
                        });

                        device.present(PresentInfo {
                            swapchain: &swapchain,
                            wait_semaphores: vec![swapchain.current_present_semaphore()],
                        });
                    }
                    _ => {}
                },
                _ => {}
            }
        })
        .expect("Couldn't run event loop.");
}
