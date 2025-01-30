use pollster::FutureExt;

const BYTES_PER_PIXEL: usize = 4;

pub struct OutputTarget {
    pub resolution: [usize; 2],
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
    pub buffer: wgpu::Buffer,
}

pub struct Graphics {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Graphics {
    pub fn new_from_adapter(adapter: &wgpu::Adapter) -> Self {
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .block_on()
            .unwrap();

        Self { device, queue }
    }

    pub fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()
            .unwrap();

        Self::new_from_adapter(&adapter)
    }

    pub fn new_output_target(&self, &[width, height]: &[usize; 2]) -> OutputTarget {
        // Output texture
        let texture_desc = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
            view_formats: &[],
        };
        let output_texture = self.device.create_texture(&texture_desc);
        let output_texture_view = output_texture.create_view(&Default::default());

        // Output buffer
        let output_buffer_size = (BYTES_PER_PIXEL * width * height) as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST
            // this tells wpgu that we want to read this buffer from the cpu
            | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };
        let output_buffer = self.device.create_buffer(&output_buffer_desc);

        OutputTarget {
            resolution: [width, height],
            texture: output_texture,
            texture_view: output_texture_view,
            buffer: output_buffer,
        }
    }

    pub fn save_output(&self, output_target: &OutputTarget) -> Vec<u8> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let [width, height] = output_target.resolution;
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &output_target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_target.buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some((BYTES_PER_PIXEL * width) as u32),
                    rows_per_image: Some(height as u32),
                },
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        // We need to scope the mapping variables so that we can
        // unmap the buffer
        let image_data = {
            let buffer_slice = output_target.buffer.slice(..);

            // NOTE: We have to create the mapping THEN device.poll() before await
            // the future. Otherwise the application will freeze.
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.receive().block_on().unwrap().unwrap();
            buffer_slice.get_mapped_range().to_owned()
        };
        output_target.buffer.unmap();

        image_data
    }
}
