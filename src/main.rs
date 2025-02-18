use ngeom::ops::*;
use ngeom::re2::Vector;
use ngeom_polygon::triangulate::triangulate;
use plotomatic::*;
use std::mem;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderVertex {
    position: [f32; 2],
    color: [f32; 4],
}

fn main() {
    let r = Renderer {
        resolution: [1920, 1080],
        fps: 60.,
        output_target: OutputTargetDescriptor::File("test.mkv".to_string()),
        //output_target: OutputTargetDescriptor::Memory,
    };

    let g = Graphics::new();

    let shader_src = "
        // Vertex shader

        struct VertexInput {
            @location(0) position: vec2<f32>,
            @location(1) color: vec4<f32>,
        };

        struct VertexOutput {
            @builtin(position) clip_position: vec4<f32>,
            @location(0) color: vec4<f32>,
        };

        @vertex
        fn vs_main(
            model: VertexInput,
        ) -> VertexOutput {
            var out: VertexOutput;
            out.color = model.color;
            out.clip_position = vec4<f32>(model.position, 0., 1.);
            return out;
        }

        // Fragment shader

        @fragment
        fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
            return in.color;
        }
    ";

    let shader_module = g.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });

    let render_pipeline_layout = g
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

    let render_pipeline = g
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: mem::size_of::<ShaderVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

    r.render(r.from_time_range((0.)..(1.)), |i, c| {
        let vertices = vec![
            Vector::point([0., 0.]),
            Vector::point([1., 0.]),
            Vector::point([1., 1.]),
            Vector::point([0., 1.]),
            Vector::point([0.4 - 0.2 * r.t(i) as f32, 0.4]),
            Vector::point([0.4 - 0.2 * r.t(i) as f32, 0.6]),
            Vector::point([0.6 - 0.2 * r.t(i) as f32, 0.6]),
            Vector::point([0.6 - 0.2 * r.t(i) as f32, 0.4]),
        ];

        let edges = vec![
            Edge::Line(Line { start: 0, end: 1 }),
            Edge::Arc(Arc {
                start: 1,
                end: 3,
                axis: Vector::point([0., 0.]),
            }),
            Edge::Line(Line { start: 3, end: 0 }),
            // Interior hole
            Edge::Line(Line { start: 4, end: 5 }),
            Edge::Line(Line { start: 5, end: 6 }),
            Edge::Line(Line { start: 6, end: 7 }),
            Edge::Line(Line { start: 7, end: 4 }),
        ];

        let face_boundary_elements = vec![
            (0, FaceBoundaryElement::Edge(0, Dir::Fwd)),
            (0, FaceBoundaryElement::Edge(1, Dir::Fwd)),
            (0, FaceBoundaryElement::Edge(2, Dir::Fwd)),
            // Interior hole
            (0, FaceBoundaryElement::Edge(3, Dir::Fwd)),
            (0, FaceBoundaryElement::Edge(4, Dir::Fwd)),
            (0, FaceBoundaryElement::Edge(5, Dir::Fwd)),
            (0, FaceBoundaryElement::Edge(6, Dir::Fwd)),
        ];

        let geometry = Geometry {
            vertices,
            edges,
            faces: 1,
            face_boundary_elements,
        };

        let Interpolation { points, edges } = geometry.interpolate(0);

        let triangles = triangulate(&points, edges).unwrap();

        let vertices: Vec<_> = points
            .into_iter()
            .map(|vector| {
                let vector = vector.unitized();
                ShaderVertex {
                    position: [vector.x, vector.y],
                    color: [r.t(i) as f32, 1., 1., 1.],
                }
            })
            .collect();

        // Flatten indices and convert to u32
        let indices: Vec<_> = triangles
            .into_iter()
            .flat_map(|[a, b, c]| [a as u32, b as u32, c as u32])
            .collect();

        // WGPU
        let output = c.reuse_or_create("output", || g.new_output_target(&r.resolution));

        let l = vertices.len().next_power_of_two();
        let vertex_buffer = c.reuse_or_create(("vertices", l), || {
            g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Vertex Buffer"),
                size: (l * std::mem::size_of::<ShaderVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let l = indices.len().next_power_of_two();
        let index_buffer = c.reuse_or_create(("indices", l), || {
            g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Index Buffer"),
                size: (l * std::mem::size_of::<[u32; 3]>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        g.queue
            .write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        g.queue
            .write_buffer(&index_buffer, 0, bytemuck::cast_slice(&indices));

        let mut encoder = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output.texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.,
                            g: 0.,
                            b: 0.,
                            a: 0.,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&render_pipeline);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }
        g.queue.submit(Some(encoder.finish()));

        g.save_output(&output)
    })
    .unwrap();
}
