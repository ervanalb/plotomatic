use ngeom::ops::*;
use ngeom::re2;
use ngeom_polygon::graph::EdgeSet;
use std::collections::{BTreeMap, BTreeSet};

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

impl WithDir for Scalar {
    fn with_dir(self, dir: Dir) -> Self {
        match dir {
            Dir::Fwd => self,
            Dir::Rev => -self,
        }
    }
}

impl WithDir for Point {
    fn with_dir(self, dir: Dir) -> Self {
        match dir {
            Dir::Fwd => self,
            Dir::Rev => -self,
        }
    }
}

// A struct representing a single sided edge (sometimes called a half-edge)
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub curve: CurveIndex,
    pub dir: Dir,
    pub next: EdgeIndex,
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
    Circle(Circle),
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

#[derive(Clone, Debug)]
pub struct Circle {
    pub axis: Point,
    pub radius: Scalar,
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
    start: EdgeIndex,
    next: Option<(EdgeIndex, &'a Edge)>,
}

impl<'a> Iterator for EdgeWalker<'a> {
    type Item = (EdgeIndex, &'a Edge);

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.next;

        self.next = self.next.and_then(|(_, edge)| {
            let next_i = edge.next;
            if next_i == self.start {
                return None;
            }
            let (_, next_e) = &self.edges[next_i]; // Discard face_i
            Some((next_i, next_e))
        });

        result
    }
}

impl Geometry {
    pub fn edges_for_face(&self, face: FaceIndex) -> impl Iterator<Item = EdgeIndex> + use<'_> {
        self.edges
            .iter()
            .enumerate()
            .filter_map(move |(ei, (fi, _))| (*fi == face).then_some(ei))
    }

    pub fn walk_edges(&self, start: EdgeIndex) -> EdgeWalker<'_> {
        let (_, edge) = &self.edges[start];
        EdgeWalker {
            edges: &self.edges,
            start,
            next: Some((start, edge)),
        }
    }

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
                &Curve::Circle(Circle { axis, radius }) => {
                    let start_pt = axis + Point::x_hat() * radius;

                    let start_pt_i = points.len();
                    points.push(start_pt);

                    let mut prev_pt_i = start_pt_i;
                    // TODO dynamic spacing
                    for i in 0..10 {
                        let t = i as f32 / 10.;

                        let pt = start_pt.transform(re2::AntiEven::axis_angle(axis, TAU * t));

                        let pt_i = points.len();
                        points.push(pt);
                        interpolated_curve_link_fwd.insert(prev_pt_i, pt_i);
                        interpolated_curve_link_rev.insert(pt_i, prev_pt_i);
                        prev_pt_i = pt_i;
                    }
                    interpolated_curve_link_fwd.insert(prev_pt_i, start_pt_i);
                    interpolated_curve_link_rev.insert(start_pt_i, prev_pt_i);
                    (start_pt_i, start_pt_i)
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

    pub fn offset(&self, amount: Scalar) -> Geometry {
        let mut new_points = vec![];
        let mut new_curves = vec![];

        for fi in 0..self.faces {
            let mut unvisited: BTreeSet<EdgeIndex> = self.edges_for_face(fi).collect();
            while let Some(&start) = unvisited.first() {
                new_points.clear();
                new_curves.clear();
                for (ei, &Edge { curve, dir, .. }) in self.walk_edges(start) {
                    unvisited.remove(&ei);
                    println!("Walk {:?}", ei);
                    match &self.curves[curve] {
                        &Curve::SingularPoint(pt) => {
                            let radius = amount.with_dir(dir);
                            if radius > 0. {
                                new_curves.push(Curve::Circle(Circle {
                                    axis: self.points[pt].with_dir(dir),
                                    radius,
                                }));
                            }
                        }
                        &Curve::Line(Line { start, end }) => {
                            let (start, end) = (start, end).with_dir(dir);
                            let start_pt = self.points[start];
                            let end_pt = self.points[end];
                            let offset = start_pt.join(end_pt).normal().normalized() * amount;

                            let start_pt_i = new_points.len();
                            new_points.push(start_pt - offset);
                            let end_pt_i = new_points.len();
                            new_points.push(end_pt - offset);

                            new_curves.push(Curve::Line(Line {
                                start: start_pt_i,
                                end: end_pt_i,
                            }));
                        }
                        &Curve::Arc(Arc { start, end, axis }) => {
                            let (start, end) = (start, end).with_dir(dir);
                            let start_pt = self.points[start];
                            let end_pt = self.points[end];

                            let axis = axis.with_dir(dir);
                            let amount = amount.with_dir(dir);
                            let new_radius = (start_pt - axis).bulk_norm() + amount;

                            if new_radius > 0. {
                                // TODO Epsilon
                                let new_start_pt_i = new_points.len();
                                new_points.push(axis + (start_pt - axis) * new_radius);
                                let new_end_pt_i = new_points.len();
                                new_points.push(axis + (end_pt - axis) * new_radius);
                                new_curves.push(Curve::Arc(Arc {
                                    start: new_start_pt_i,
                                    end: new_end_pt_i,
                                    axis,
                                }));
                            }
                        }
                        &Curve::Circle(Circle { axis, radius }) => {
                            let axis = axis.with_dir(dir);
                            let amount = amount.with_dir(dir);
                            let new_radius = radius + amount;

                            if new_radius > 0. {
                                // TODO: Epsilon
                                new_curves.push(Curve::Circle(Circle {
                                    axis,
                                    radius: new_radius,
                                }));
                            }
                        }
                    }
                }
                dbg!(&new_points, &new_curves);
            }
        }
        todo!();
    }
}

#[cfg(test)]
mod test {
    use crate::vector_graphics::*;
    use ngeom::ops::Point;
    use ngeom::re2::Vector;

    #[test]
    fn test_offset() {
        let points = vec![
            Vector::point([0., 0.]),
            Vector::point([1., 0.]),
            Vector::point([1., 1.]),
            Vector::point([0., 1.]),
            Vector::point([0.4, 0.4]),
            Vector::point([0.4, 0.6]),
            Vector::point([0.6, 0.6]),
            Vector::point([0.6, 0.4]),
        ];

        let curves = vec![
            Curve::Line(Line { start: 0, end: 1 }),
            Curve::Arc(Arc {
                start: 1,
                end: 3,
                axis: Vector::point([0., 0.]),
            }),
            Curve::Line(Line { start: 3, end: 0 }),
            // Interior hole
            Curve::Line(Line { start: 4, end: 5 }),
            Curve::Line(Line { start: 5, end: 6 }),
            Curve::Line(Line { start: 6, end: 7 }),
            Curve::Line(Line { start: 7, end: 4 }),
        ];

        let edges = vec![
            (
                0,
                Edge {
                    curve: 0,
                    dir: Dir::Fwd,
                    next: 1,
                },
            ),
            (
                0,
                Edge {
                    curve: 1,
                    dir: Dir::Fwd,
                    next: 2,
                },
            ),
            (
                0,
                Edge {
                    curve: 2,
                    dir: Dir::Fwd,
                    next: 0,
                },
            ),
            // Interior hole
            (
                0,
                Edge {
                    curve: 3,
                    dir: Dir::Fwd,
                    next: 4,
                },
            ),
            (
                0,
                Edge {
                    curve: 4,
                    dir: Dir::Fwd,
                    next: 5,
                },
            ),
            (
                0,
                Edge {
                    curve: 5,
                    dir: Dir::Fwd,
                    next: 6,
                },
            ),
            (
                0,
                Edge {
                    curve: 6,
                    dir: Dir::Fwd,
                    next: 3,
                },
            ),
        ];

        let geometry = Geometry {
            points,
            curves,

            edges,
            faces: 1,
        };

        geometry.offset(0.1);
    }
}
