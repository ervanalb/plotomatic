use ngeom::ops::*;
use ngeom::re2;
use ngeom::scalar::*;
use ngeom_polygon::graph::EdgeSet;

pub type Scalar = f32;
pub type Point = re2::Vector<Scalar>;
pub type EdgeIndex = usize;
pub type VertexIndex = usize;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Dir {
    Fwd,
    Rev,
}

pub trait WithDir {
    fn with_dir(self, dir: Dir) -> Self;
}

impl<T> WithDir for (T, T) {
    fn with_dir(self, dir: Dir) -> Self {
        match dir {
            Dir::Fwd => self,
            Dir::Rev => (self.1, self.0),
        }
    }
}

pub struct EdgeVertex {
    pub edge: EdgeIndex,
    pub vertex: VertexIndex,
    pub dir: Dir,
}

pub struct FaceEdge {
    pub edge: EdgeIndex,
    pub dir: Dir,
}

pub struct Geometry {
    pub vertices: Vec<Point>,
    pub edges: Vec<Curve>,

    pub edge_vertices: Vec<EdgeVertex>,
    pub face_edges: Vec<FaceEdge>,
}

#[derive(Clone, Debug)]
pub enum Curve {
    Line(Line),
    Arc(Arc),
}

#[derive(Clone, Debug)]
pub struct Line {}

/*
impl<SPACE: Space, VI: Index> EdgeTrait<SPACE, VI> for Line<VI> {
    fn endpoints(&self) -> Option<(VI, VI)> {
        Some((self.start, self.end))
    }

    fn x(
        &self,
        vertices: &impl VertexCollection<Space = SPACE, Index = VI>,
        t: SPACE::Scalar,
    ) -> SPACE::Vector {
        vertices[self.start] * (SPACE::Scalar::one() - t) + vertices[self.end] * t
    }

    fn interpolate(
        &self,
        vertices: &impl VertexCollection<Space = SPACE, Index = VI>,
        dir: Dir,
        point_stream: &mut impl PointStream<Point = SPACE::Vector>,
    ) {
        point_stream.push(
            vertices[match dir {
                Dir::Fwd => self.start,
                Dir::Rev => self.end,
            }],
        );
    }
}
*/

#[derive(Clone, Debug)]
pub struct Arc {
    pub axis: Point,
}

/*
impl<SPACE: Space, VI: Index> EdgeTrait<SPACE, VI> for Arc<SPACE, VI> {
    fn endpoints(&self) -> Option<(VI, VI)> {
        Some((self.start, self.end))
    }

    fn x(
        &self,
        vertices: &impl VertexCollection<Space = SPACE, Index = VI>,
        t: SPACE::Scalar,
    ) -> SPACE::Vector {
        let start_pt = vertices[self.start];
        start_pt.transform(SPACE::AntiEven::axis_angle(self.axis, self.end_angle * t))
    }

    fn interpolate(
        &self,
        vertices: &impl VertexCollection<Space = SPACE, Index = VI>,
        dir: Dir,
        point_stream: &mut impl PointStream<Point = SPACE::Vector>,
    ) {
        for i in 0..10 {
            // TODO dynamic spacing
            let i = match dir {
                Dir::Fwd => i,
                Dir::Rev => 9 - i,
            };
            let t = SPACE::Scalar::from(i as f32 / 10.);
            point_stream.push(self.x(vertices, t))
        }
    }
}
*/

//#[derive(Clone, Debug)]
//pub struct Circle<SPACE: Space> {
//    pub axis: SPACE::AxisOfRotation,
//    pub pt: SPACE::Vector,
//}

#[derive(Debug)]
pub enum PatchTopologyError {
    Empty,
    Open,
    Branching,
}

pub struct Interpolation {
    pub points: Vec<Point>,
    pub edges: EdgeSet,
}

impl Geometry {
    pub fn endpoints(&self, edge: EdgeIndex) -> Option<(VertexIndex, VertexIndex)> {
        let mut start = None;
        let mut end = None;

        for ev in &self.edge_vertices {
            if ev.edge != edge {
                continue;
            }
            let endpoint = match ev.dir {
                Dir::Fwd => &mut start,
                Dir::Rev => &mut end,
            };
            assert!(endpoint.is_none(), "Found duplicate endpoint for edge");
            *endpoint = Some(ev.vertex);
        }
        match (start, end) {
            (Some(start), Some(end)) => Some((start, end)),
            (None, None) => None,
            _ => panic!("Found just one endpoint (expected 0 or 2)"),
        }
    }

    pub fn interpolate(&self) -> Interpolation {
        // First, interpolate the boundaries of the faces
        // and store their connectivity
        let mut points = self.vertices.clone();
        let mut edges = EdgeSet::new();

        for fe in &self.face_edges {
            match &self.edges[fe.edge] {
                Curve::Line(_) => {
                    let (start, end) = self
                        .endpoints(fe.edge)
                        .unwrap_or_else(|| panic!("Edge {:?} (line) has no endpoints", fe.edge))
                        .with_dir(fe.dir);

                    edges.insert(start, end);
                }
                Curve::Arc(arc) => {
                    let (start, end) = self
                        .endpoints(fe.edge)
                        .unwrap_or_else(|| panic!("Edge {:?} (arc) has no endpoints", fe.edge))
                        .with_dir(fe.dir);

                    let mut prev_pt_i = start;
                    for i in 1..10 {
                        // TODO dynamic spacing
                        let t = i as f32 / 10.;

                        // TODO this shouldn't be hardcoded
                        let end_angle = 0.25 * std::f32::consts::TAU;

                        let start_pt = self.vertices[start];
                        let pt =
                            start_pt.transform(re2::AntiEven::axis_angle(arc.axis, end_angle * t));

                        let pt_i = points.len();
                        points.push(pt);
                        edges.insert(prev_pt_i, pt_i);
                        prev_pt_i = pt_i;
                    }
                    edges.insert(prev_pt_i, end);
                }
            }
        }

        Interpolation { points, edges }
    }
}
