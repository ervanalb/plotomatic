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

#[derive(Debug)]
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
        #[derive(Debug)]
        struct OffsetEdge {
            element: OffsetElement,
            next: EdgeIndex,
        }

        #[derive(Debug)]
        enum OffsetElement {
            Curve(Curve),
            ConcaveCap(PointIndex, PointIndex),
        }

        #[derive(Debug)]
        struct Endpoint {
            pt_i: PointIndex,
            pt: Point,
            tangent: Point,
        }

        #[derive(Debug)]
        struct State {
            orig_start_pt: Point,
            start_edge_i: EdgeIndex,
            start_endpoint: Endpoint,

            prev_edge_i: EdgeIndex,
            prev_endpoint: Endpoint,
        }

        fn new_cap(from: Endpoint, axis: Point, to: Endpoint) -> OffsetElement {
            println!("Generate cap between {:?} and {:?}", from, to);
            // See if the triangle formed by from - to - axis has positive or negative area
            // (negative indicates a concave corner
            if from.pt.join(to.pt).join(axis) < (0.).into() {
                // TODO epsilon
                // Corner is concave
                OffsetElement::ConcaveCap(from.pt_i, to.pt_i)
            } else {
                // Corner is convex-- join with arc
                OffsetElement::Curve(Curve::Arc(Arc {
                    start: from.pt_i,
                    end: to.pt_i,
                    axis,
                }))
            }
        }

        fn push_open(
            new_edges: &mut Vec<OffsetEdge>,
            state: &mut Option<State>,
            orig_start_pt: Point,
            start_endpoint: Endpoint,
            curve: Curve,
            end_endpoint: Endpoint,
        ) {
            match state {
                Some(state) => {
                    // Allocate cap and edge
                    let cap_i = new_edges.len();
                    let edge_i = cap_i + 1;

                    // Push cap
                    new_edges.push(OffsetEdge {
                        element: new_cap(
                            std::mem::replace(&mut state.prev_endpoint, end_endpoint),
                            orig_start_pt,
                            start_endpoint,
                        ),
                        next: edge_i,
                    });

                    // Connect prev to cap
                    new_edges[state.prev_edge_i].next = cap_i;

                    // Push edge
                    new_edges.push(OffsetEdge {
                        element: OffsetElement::Curve(curve),
                        next: edge_i,
                    });

                    // Set state
                    state.prev_edge_i = edge_i;
                    //state.prev_endpoint already set
                }
                None => {
                    // No prior edge--this is the first

                    // Push edge
                    let edge_i = new_edges.len();
                    new_edges.push(OffsetEdge {
                        element: OffsetElement::Curve(curve),
                        next: edge_i,
                    });

                    // Set state
                    *state = Some(State {
                        orig_start_pt,
                        start_edge_i: edge_i,
                        start_endpoint,
                        prev_edge_i: edge_i,
                        prev_endpoint: end_endpoint,
                    });
                }
            }
        }

        let mut new_points = vec![];
        let mut new_edges = vec![];

        for fi in 0..self.faces {
            let mut unvisited: BTreeSet<EdgeIndex> = self.edges_for_face(fi).collect();
            while let Some(&start) = unvisited.first() {
                let mut state: Option<State> = None;
                for (ei, &Edge { curve, dir, .. }) in self.walk_edges(start) {
                    unvisited.remove(&ei);
                    match &self.curves[curve] {
                        &Curve::SingularPoint(pt) => {
                            let radius = amount.with_dir(dir);
                            if radius > 0. {
                                assert!(state.is_none());
                                let edge_i = new_edges.len();
                                new_edges.push(OffsetEdge {
                                    element: OffsetElement::Curve(Curve::Circle(Circle {
                                        axis: self.points[pt].with_dir(dir),
                                        radius,
                                    })),
                                    next: edge_i,
                                });
                            }
                        }
                        &Curve::Line(Line { start, end }) => {
                            let (start, end) = (start, end).with_dir(dir);
                            let start_pt = self.points[start];
                            let end_pt = self.points[end];
                            let offset = start_pt.join(end_pt).normal().normalized() * amount;

                            let new_start_pt_i = new_points.len();
                            let new_start_pt = start_pt - offset;
                            new_points.push(new_start_pt);
                            let new_end_pt_i = new_points.len();
                            let new_end_pt = end_pt - offset;
                            new_points.push(new_end_pt);

                            let tangent = (new_end_pt - new_start_pt).normalized();

                            let start_endpoint = Endpoint {
                                pt_i: new_start_pt_i,
                                pt: new_start_pt,
                                tangent,
                            };

                            let curve = Curve::Line(Line {
                                start: new_start_pt_i,
                                end: new_end_pt_i,
                            });

                            let end_endpoint = Endpoint {
                                pt_i: new_end_pt_i,
                                pt: new_end_pt,
                                tangent,
                            };

                            push_open(
                                &mut new_edges,
                                &mut state,
                                start_pt,
                                start_endpoint,
                                curve,
                                end_endpoint,
                            );
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
                                let new_start_pt = axis + (start_pt - axis) * new_radius;
                                new_points.push(new_start_pt);
                                let new_end_pt_i = new_points.len();
                                let new_end_pt = axis + (end_pt - axis) * new_radius;
                                new_points.push(new_end_pt);

                                let start_endpoint = Endpoint {
                                    pt_i: new_start_pt_i,
                                    pt: new_start_pt,
                                    tangent: axis.join(new_start_pt).normal().normalized(),
                                };

                                let curve = Curve::Arc(Arc {
                                    start: new_start_pt_i,
                                    end: new_end_pt_i,
                                    axis,
                                });

                                let end_endpoint = Endpoint {
                                    pt_i: new_end_pt_i,
                                    pt: new_end_pt,
                                    tangent: axis.join(new_end_pt).normal().normalized(),
                                };

                                push_open(
                                    &mut new_edges,
                                    &mut state,
                                    start_pt,
                                    start_endpoint,
                                    curve,
                                    end_endpoint,
                                );
                            }
                        }
                        &Curve::Circle(Circle { axis, radius }) => {
                            let axis = axis.with_dir(dir);
                            let amount = amount.with_dir(dir);
                            let new_radius = radius + amount;

                            if new_radius > 0. {
                                // TODO: Epsilon
                                assert!(state.is_none());
                                let edge_i = new_edges.len();
                                new_edges.push(OffsetEdge {
                                    element: OffsetElement::Curve(Curve::Circle(Circle {
                                        axis,
                                        radius: new_radius,
                                    })),
                                    next: edge_i,
                                });
                            }
                        }
                    }
                }

                if let Some(state) = state {
                    // Add final cap
                    // Push cap
                    let cap_i = new_edges.len();
                    new_edges.push(OffsetEdge {
                        element: new_cap(
                            state.prev_endpoint,
                            state.orig_start_pt,
                            state.start_endpoint,
                        ),
                        next: state.start_edge_i,
                    });

                    // Connect prev edge to cap
                    new_edges[state.prev_edge_i].next = cap_i;
                }
            }
        }

        // TODO temporary conversion back to geometry
        let edges = new_edges
            .iter()
            .enumerate()
            .map(|(i, &OffsetEdge { element: _, next })| {
                (
                    0,
                    Edge {
                        curve: i,
                        dir: Dir::Fwd,
                        next,
                    },
                )
            })
            .collect();
        let curves = new_edges
            .into_iter()
            .map(|OffsetEdge { element, next: _ }| match element {
                OffsetElement::ConcaveCap(_, _) => {
                    panic!("Concave corners not yet supported!");
                }
                OffsetElement::Curve(curve) => curve,
            })
            .collect();

        Geometry {
            points: new_points,
            curves,
            edges,
            faces: 1,
        }
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
            /*
            Vector::point([0.4, 0.4]),
            Vector::point([0.4, 0.6]),
            Vector::point([0.6, 0.6]),
            Vector::point([0.6, 0.4]),
            */
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
            /*
            Curve::Line(Line { start: 4, end: 5 }),
            Curve::Line(Line { start: 5, end: 6 }),
            Curve::Line(Line { start: 6, end: 7 }),
            Curve::Line(Line { start: 7, end: 4 }),
            */
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
            /*
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
            */
        ];

        let geometry = Geometry {
            points,
            curves,

            edges,
            faces: 1,
        };

        dbg!(geometry.offset(0.1).interpolate());
        panic!();
    }
}
