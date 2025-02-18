use ngeom::ops::*;
use ngeom::re2;
use ngeom_polygon::graph::EdgeSet;
use std::collections::BTreeMap;

pub type Scalar = f32;
pub type Point = re2::Vector<Scalar>;

// Indices to embedding
pub type PointIndex = usize;
pub type CurveIndex = usize;

// Indices to topology
pub type VertexIndex = usize;
pub type EdgeIndex = usize;
pub type FaceIndex = usize;

pub const TAU: f32 = std::f32::consts::TAU;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
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

#[derive(Clone, Copy, Debug)]
pub enum FaceBoundaryElement {
    Vertex(VertexIndex),
    Edge(EdgeIndex, Dir),
}

/// A struct representing a single sided vertex (perhaps it could be called a half-vertex.)
// Note: rather than have a separate Vec of Vertex structs,
// we will perform a (purely computational) optimization
// by storing the 0 or 2 vertices for a given edge inside the edge itself.
// The first will be assumed to have positive direction, and the second, negative direction,
// so we can also omit the direction field in this struct.
// Additionally, we will store the twin's edge index, rather than the twin vertex index,
// since not storing vertices in a Vec means there is no such thing as a VertexIndex.
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub point: PointIndex,
    pub twin_edge: Option<EdgeIndex>,
}

// A struct representing a single sided edge (sometimes called a half-edge)
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub curve: CurveIndex,
    pub dir: Dir,

    // Half-edges conventionally store a "next" / "prev" pointer,
    // which can be found inside of `vertices`
    pub vertices: Option<[Vertex; 2]>,
    pub twin: Option<EdgeIndex>,
}

#[derive(Clone, Debug, Default)]
pub struct Geometry {
    // Embedding
    pub points: Vec<Point>,
    pub curves: Vec<Curve>,

    // Topology
    pub edges: Vec<(FaceIndex, Edge)>,
    pub faces: FaceIndex,
}

#[derive(Clone, Debug)]
pub enum Curve {
    SingularPoint(PointIndex),
    Line(Line),
    Arc(Arc),
}

#[derive(Clone, Debug)]
pub struct Line {
    pub start: PointIndex,
    pub end: PointIndex,
}

#[derive(Clone, Debug)]
pub struct Arc {
    pub start: PointIndex,
    pub end: PointIndex,
    pub axis: Point,
}

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

pub struct EdgeWalker<'a> {
    edges: &'a [(FaceIndex, Edge)],
    next: Option<(EdgeIndex, &'a Edge)>,
}

impl<'a> Iterator for EdgeWalker<'a> {
    type Item = (EdgeIndex, &'a Edge);

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.next;

        self.next = self.next.and_then(|(_, edge)| {
            let [_, to_v] = edge.vertices?; // Discard from_v
            let next_i = to_v.twin_edge?;
            let (_, next_e) = &self.edges[next_i]; // Discard face_i
            Some((next_i, next_e))
        });

        result
    }
}

impl Geometry {
    pub fn interpolate(&self) -> Interpolation {
        // Add all points to the interpolation
        // (their indices will be preserved)
        let mut points = self.points.clone();

        // Add all interpolated curve points
        let mut interpolated_curve_endpoints = vec![];
        let mut interpolated_curve_link_fwd = BTreeMap::<usize, usize>::new();
        let mut interpolated_curve_link_rev = BTreeMap::<usize, usize>::new();
        for curve in &self.curves {
            interpolated_curve_endpoints.push(match curve {
                &Curve::SingularPoint(pt) => {
                    interpolated_curve_link_fwd.insert(pt, pt);
                    interpolated_curve_link_rev.insert(pt, pt);
                    (pt, pt)
                }
                &Curve::Line(Line { start, end }) => {
                    interpolated_curve_link_fwd.insert(start, end);
                    interpolated_curve_link_rev.insert(end, start);
                    (start, end)
                }
                &Curve::Arc(Arc { start, end, axis }) => {
                    let start_pt = self.points[start];

                    let end_angle = {
                        let end_pt = self.points[end];
                        let (cos, sin) = axis
                            .join(start_pt)
                            .unitized()
                            .cos_sin_angle_to(axis.join(end_pt).unitized());
                        Scalar::from(sin).atan2(cos).rem_euclid(TAU)
                    };

                    let mut prev_pt_i = start;
                    // TODO dynamic spacing
                    for i in 1..10 {
                        let t = i as f32 / 10.;

                        let pt = start_pt.transform(re2::AntiEven::axis_angle(axis, end_angle * t));

                        let pt_i = points.len();
                        points.push(pt);
                        interpolated_curve_link_fwd.insert(prev_pt_i, pt_i);
                        interpolated_curve_link_rev.insert(pt_i, prev_pt_i);
                        prev_pt_i = pt_i;
                    }
                    interpolated_curve_link_fwd.insert(prev_pt_i, end);
                    interpolated_curve_link_rev.insert(end, prev_pt_i);
                    (start, end)
                }
            })
        }

        // Add all edges
        let mut edges = EdgeSet::new();

        for &(_, Edge { curve, dir, .. }) in &self.edges {
            let (start, _) = interpolated_curve_endpoints[curve].with_dir(dir);
            let link = match dir {
                Dir::Fwd => &interpolated_curve_link_fwd,
                Dir::Rev => &interpolated_curve_link_fwd,
            };
            let mut next = link[&start];
            edges.insert(start, next);
            while next != start {
                let cur = next;
                next = link[&next];
                edges.insert(cur, next);
            }
        }

        Interpolation { points, edges }
    }
}
