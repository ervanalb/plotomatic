use crate::output_writer::{OutputWriter, OutputWriterError};
use crate::{AnimationDescriptor, Output};
use pollster::FutureExt;
use rayon::prelude::*;
use std::ops::Range;
use std::sync::{mpsc, Arc, Condvar, Mutex};

const BYTES_PER_PIXEL: usize = 4;

pub type LayerIndex = usize;

pub enum LayerResolution {
    MatchCanvas,
    Explicit([usize; 2]),
}

impl LayerResolution {
    pub fn resolve(&self, animation: &AnimationDescriptor) -> [usize; 2] {
        match self {
            LayerResolution::MatchCanvas => animation.resolution,
            LayerResolution::Explicit(r) => r.clone(),
        }
    }
}

pub struct FrameDescriptor {
    pub frame: usize,
    pub t: f64,
}

pub struct LayerDescriptor {
    pub resolution: LayerResolution,
}

pub struct Layer {
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,
}

pub struct FrameRenderer<'a> {
    pub r: &'a Renderer,
    pub layers: Vec<Layer>,
    pub output_texture: wgpu::Texture,
    pub output_texture_view: wgpu::TextureView,
    pub output_buffer: wgpu::Buffer,
}

pub struct Renderer {
    pub animation: AnimationDescriptor,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub layers: Vec<LayerDescriptor>,
}

impl Renderer {
    pub fn new_from_adapter(adapter: &wgpu::Adapter, animation: AnimationDescriptor) -> Self {
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .block_on()
            .unwrap();

        Self {
            animation,
            device,
            queue,
            layers: vec![],
        }
    }

    pub fn new(animation: AnimationDescriptor) -> Self {
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

        Self::new_from_adapter(&adapter, animation)
    }

    pub fn t(&self, i: usize) -> f64 {
        i as f64 / self.animation.fps
    }

    pub fn add_layer(&mut self, layer: LayerDescriptor) -> LayerIndex {
        let ix = self.layers.len();
        self.layers.push(layer);
        ix
    }

    pub fn remove_layer(&mut self, layer_index: LayerIndex) {
        self.layers.remove(layer_index);
    }

    pub fn frame_renderer(&self) -> FrameRenderer {
        let layers = self
            .layers
            .iter()
            .map(|layer_descriptor| {
                let [width, height] = layer_descriptor.resolution.resolve(&self.animation);

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
                let texture = self.device.create_texture(&texture_desc);
                let texture_view = texture.create_view(&Default::default());

                Layer {
                    texture,
                    texture_view,
                }
            })
            .collect();

        // Output texture
        let &[width, height] = &self.animation.resolution;

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

        FrameRenderer {
            r: self,
            layers,
            output_texture,
            output_texture_view,
            output_buffer,
        }
    }

    pub fn from_time(&self, t: f64) -> usize {
        (t * self.animation.fps) as usize
    }

    pub fn from_time_range(&self, range: Range<f64>) -> Range<usize> {
        let [start, end] = [range.start, range.end].map(|t| self.from_time(t));
        Range { start, end }
    }

    pub fn render<TF, FF>(
        &self,
        frame_range: Range<usize>,
        thread_function: TF,
    ) -> Result<Output, OutputWriterError>
    where
        TF: Send + Sync + Fn(&Self) -> FF,
        FF: Fn(usize) -> Vec<u8>,
    {
        // Fire up the output writer
        let mut output_writer = OutputWriter::new(&self.animation)?;

        // Channel with enough capacity to hold an item from each thread
        let (tx, rx) =
            mpsc::sync_channel(std::thread::available_parallelism().map_or(8, |n| n.get()));

        let gate = Arc::new((Mutex::new(0), Condvar::new()));

        rayon::scope(|s| {
            s.spawn(move |_| {
                frame_range
                    .par_bridge()
                    .map_init(
                        || thread_function(&self),
                        |frame_function, i| (i, frame_function(i)),
                    )
                    .for_each_with((tx, gate), |(tx, gate), (i, frame_result)| {
                        let (lock, cond) = &**gate;

                        {
                            let mut guard =
                                cond.wait_while(lock.lock().unwrap(), |v| *v < i).unwrap();
                            let _ = tx.send(frame_result);
                            *guard = i + 1;
                        }

                        cond.notify_all();
                    });
            });

            // Write output
            for image_data in rx {
                output_writer.write(&image_data)?;
            }
            output_writer.close()
        })
    }
}

impl FrameRenderer<'_> {
    pub fn save(&self) -> Vec<u8> {
        let mut encoder = self
            .r
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let &[width, height] = &self.r.animation.resolution;
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.output_buffer,
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

        self.r.queue.submit(Some(encoder.finish()));

        // We need to scope the mapping variables so that we can
        // unmap the buffer
        let image_data = {
            let buffer_slice = self.output_buffer.slice(..);

            // NOTE: We have to create the mapping THEN device.poll() before await
            // the future. Otherwise the application will freeze.
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.r.device.poll(wgpu::Maintain::Wait);
            rx.receive().block_on().unwrap().unwrap();
            buffer_slice.get_mapped_range().to_owned()
        };
        self.output_buffer.unmap();

        image_data
    }
}
