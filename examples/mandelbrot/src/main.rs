use std::time::Instant;

use paya::{
    common::{AccessFlags, ImageLayout, ImageTransition, ImageUsageFlags},
    device::{Device, ImageInfo, PresentInfo, SubmitInfo},
    gpu_resources::{self, GpuResourcePool, PackedGpuResourceId},
    instance::{Instance, InstanceCreateInfo},
    pipeline::ComputePipelineInfo,
    shader::{ShaderCompiler, ShaderInfo},
    swapchain::SwapchainCreateInfo,
    task_list::{Task, TaskList},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[repr(C)]
struct PushConstants {
    resolution: (u32, u32),
    backbuffer_id: PackedGpuResourceId,
    time: f32,
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Voxei")
        .build(&event_loop)
        .unwrap();

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
        image_usage: ImageUsageFlags::STORAGE,
        max_frames_in_flight: 2,
    });

    let shader_compiler = ShaderCompiler::new();
    let compute_pipeline = device.create_compute_pipeline(ComputePipelineInfo {
        shader: ShaderInfo {
            byte_code: shader_compiler.load_from_file("shaders/mandelbrot.comp.glsl".to_owned()),
            entry_point: "main".to_owned(),
        },
        push_constant_size: std::mem::size_of::<PushConstants>() as u32,
    });

    let start_time = Instant::now();

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
                    _ => {}
                },
                Event::AboutToWait => {
                    let Some(image) = swapchain.acquire_next_image() else {
                        return;
                    };
                    let image_extent = device.get_image(image).info.extent;
                    println!("extent: {:?}", image_extent);

                    let mut recorder = device.create_command_recorder();

                    recorder.pipeline_barrier_image_transition(
                        &device,
                        ImageTransition {
                            image,
                            src_layout: ImageLayout::Undefined,
                            src_access: AccessFlags::empty(),
                            dst_layout: ImageLayout::General,
                            dst_access: AccessFlags::SHADER_WRITE,
                        },
                    );

                    recorder.bind_compute_pipeline(&device, &compute_pipeline);
                    recorder.upload_push_constants(
                        &device,
                        &compute_pipeline,
                        &PushConstants {
                            backbuffer_id: image.pack(),
                            resolution: (image_extent.width, image_extent.height),
                            time: Instant::now().duration_since(start_time).as_secs_f32(),
                        },
                    );
                    recorder.dispatch(
                        &device,
                        f32::ceil(image_extent.width as f32 / 16.0) as u32,
                        f32::ceil(image_extent.height as f32 / 16.0) as u32,
                        1,
                    );

                    recorder.pipeline_barrier_image_transition(
                        &device,
                        ImageTransition {
                            image,
                            src_layout: ImageLayout::General,
                            src_access: AccessFlags::SHADER_WRITE,
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

                    device.collect_garbage(swapchain.gpu_timeline_semaphore());
                }
                _ => {}
            }
        })
        .expect("Couldn't run event loop.");
}
