use ngeom::ops::*;
use ngeom::re2;
use ngeom_polygon::graph::EdgeSet;

pub type Scalar = f32;
pub type Point = re2::Vector<Scalar>;
pub type EdgeIndex = usize;
pub type VertexIndex = usize;
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

#[derive(Clone, Debug)]
pub struct Geometry {
    pub vertices: Vec<Point>,
    pub edges: Vec<Edge>,
    pub faces: FaceIndex,

    pub face_boundary_elements: Vec<(FaceIndex, FaceBoundaryElement)>,
}

#[derive(Clone, Debug)]
pub enum Edge {
    Line(Line),
    Arc(Arc),
}

#[derive(Clone, Debug)]
pub struct Line {
    pub start: VertexIndex,
    pub end: VertexIndex,
}

#[derive(Clone, Debug)]
pub struct Arc {
    pub start: VertexIndex,
    pub end: VertexIndex,
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

impl Geometry {
    pub fn boundary_elements_for_face(
        &self,
        face: FaceIndex,
    ) -> impl Iterator<Item = FaceBoundaryElement> + use<'_> {
        self.face_boundary_elements
            .iter()
            .filter_map(move |&(fi, be)| (fi == face).then_some(be))
    }

    pub fn interpolate(&self, face: FaceIndex) -> Interpolation {
        // First, interpolate the boundaries of the faces
        // and store their connectivity
        let mut points = vec![];
        let mut new_pt_i = vec![usize::MAX; self.vertices.len()];

        let mut push_or_get_pt = |points: &mut Vec<Point>, pt: VertexIndex| -> usize {
            let existing_i = new_pt_i[pt];
            if existing_i != usize::MAX {
                return existing_i;
            }
            let new_i = points.len();
            points.push(self.vertices[pt]);
            new_pt_i[pt] = new_i;
            new_i
        };

        let mut edges = EdgeSet::new();

        for face_boundary_element in self.boundary_elements_for_face(face) {
            match face_boundary_element {
                FaceBoundaryElement::Vertex(vertex) => {
                    // Singular point--
                    // add a self-edge in the polygon
                    let i = push_or_get_pt(&mut points, vertex);
                    edges.insert(i, i);
                }
                FaceBoundaryElement::Edge(edge, dir) => {
                    match &self.edges[edge] {
                        &Edge::Line(Line { start, end }) => {
                            let (start, end) = (start, end).with_dir(dir);
                            let i = push_or_get_pt(&mut points, start);
                            let j = push_or_get_pt(&mut points, end);
                            edges.insert(i, j);
                        }
                        &Edge::Arc(Arc { start, end, axis }) => {
                            let start_pt = self.vertices[start];
                            let end_pt = self.vertices[end];

                            let mut prev_pt_i = push_or_get_pt(
                                &mut points,
                                match dir {
                                    Dir::Fwd => start,
                                    Dir::Rev => end,
                                },
                            );
                            // TODO dynamic spacing
                            for i in 1..10 {
                                let i = match dir {
                                    Dir::Fwd => i,
                                    Dir::Rev => 10 - i,
                                };
                                let t = i as f32 / 10.;

                                let end_angle = {
                                    let (cos, sin) = axis
                                        .join(start_pt)
                                        .unitized()
                                        .cos_sin_angle_to(axis.join(end_pt).unitized());
                                    Scalar::from(sin).atan2(cos).rem_euclid(TAU)
                                };

                                let pt = start_pt
                                    .transform(re2::AntiEven::axis_angle(axis, end_angle * t));

                                let pt_i = points.len();
                                points.push(pt);
                                edges.insert(prev_pt_i, pt_i);
                                prev_pt_i = pt_i;
                            }
                            let new_end_i = push_or_get_pt(
                                &mut points,
                                match dir {
                                    Dir::Fwd => end,
                                    Dir::Rev => start,
                                },
                            );
                            edges.insert(prev_pt_i, new_end_i);
                        }
                    }
                }
            }
        }

        Interpolation { points, edges }
    }
}
