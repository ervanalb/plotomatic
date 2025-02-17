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
            Curve::Line(Line {}),
            Curve::Arc(Arc {
                axis: Vector::point([0., 0.]),
            }),
            Curve::Line(Line {}),
            // Interior hole
            Curve::Line(Line {}),
            Curve::Line(Line {}),
            Curve::Line(Line {}),
            Curve::Line(Line {}),
        ];

        let edge_vertices = vec![
            EdgeVertex {
                edge: 0,
                vertex: 0,
                dir: Dir::Fwd,
            },
            EdgeVertex {
                edge: 0,
                vertex: 1,
                dir: Dir::Rev,
            },
            EdgeVertex {
                edge: 1,
                vertex: 1,
                dir: Dir::Fwd,
            },
            EdgeVertex {
                edge: 1,
                vertex: 3,
                dir: Dir::Rev,
            },
            EdgeVertex {
                edge: 2,
                vertex: 3,
                dir: Dir::Fwd,
            },
            EdgeVertex {
                edge: 2,
                vertex: 0,
                dir: Dir::Rev,
            },
            // Interior hole
            EdgeVertex {
                edge: 3,
                vertex: 4,
                dir: Dir::Fwd,
            },
            EdgeVertex {
                edge: 3,
                vertex: 5,
                dir: Dir::Rev,
            },
            EdgeVertex {
                edge: 4,
                vertex: 5,
                dir: Dir::Fwd,
            },
            EdgeVertex {
                edge: 4,
                vertex: 6,
                dir: Dir::Rev,
            },
            EdgeVertex {
                edge: 5,
                vertex: 6,
                dir: Dir::Fwd,
            },
            EdgeVertex {
                edge: 5,
                vertex: 7,
                dir: Dir::Rev,
            },
            EdgeVertex {
                edge: 6,
                vertex: 7,
                dir: Dir::Fwd,
            },
            EdgeVertex {
                edge: 6,
                vertex: 4,
                dir: Dir::Rev,
            },
        ];

        let face_edges = vec![
            FaceEdge {
                edge: 0,
                dir: Dir::Fwd,
            },
            FaceEdge {
                edge: 1,
                dir: Dir::Fwd,
            },
            FaceEdge {
                edge: 2,
                dir: Dir::Fwd,
            },
            // Interior hole
            FaceEdge {
                edge: 3,
                dir: Dir::Fwd,
            },
            FaceEdge {
                edge: 4,
                dir: Dir::Fwd,
            },
            FaceEdge {
                edge: 5,
                dir: Dir::Fwd,
            },
            FaceEdge {
                edge: 6,
                dir: Dir::Fwd,
            },
        ];

        let geometry = Geometry {
            vertices,
            edges,
            edge_vertices,
            face_edges,
        };

        let Interpolation { points, edges } = geometry.interpolate();

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
