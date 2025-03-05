use ngeom::ops::*;
use ngeom::re2;
use ngeom_polygon::graph::EdgeSet;
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;

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

pub const EPSILON: f32 = 1e-5;

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

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub curve: CurveIndex,
    pub dir: Dir,
    pub next: EdgeIndex,
}

#[derive(Debug)]
pub struct Proper;

#[derive(Debug)]
pub struct Improper;

#[derive(Clone, Debug, Default)]
pub struct Geometry<S> {
    // Embedding
    pub points: Vec<Point>,
    pub curves: Vec<Curve>,

    // Topology
    pub edges: Vec<Edge>,

    pub _state: PhantomData<S>,
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

impl Curve {
    pub fn endpoints(&self) -> Option<(PointIndex, PointIndex)> {
        match self {
            &Curve::SingularPoint(_) => None,
            &Curve::Line(Line { start, end }) => Some((start, end)),
            &Curve::Arc(Arc {
                start,
                end,
                axis: _,
            }) => Some((start, end)),
            &Curve::Circle(_) => None,
        }
    }
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
    edges: &'a [Edge],
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
            let next_e = &self.edges[next_i];
            Some((next_i, next_e))
        });

        result
    }
}

fn intersect_segments(p1a: Point, p1b: Point, p2a: Point, p2b: Point) -> Option<Point> {
    let l1 = p1a.join(p1b);
    let l1w = l1.weight_norm();
    assert!(l1w > EPSILON.into());
    let l2 = p2a.join(p2b);
    let l2w = l2.weight_norm();
    assert!(l2w > EPSILON.into());

    let da = l1.join(p2a);
    let db = l1.join(p2b);
    let test1 =
        da > l1w * EPSILON && db < l1w * -EPSILON || db > l1w * EPSILON && da < l1w * -EPSILON;

    let da = l2.join(p1a);
    let db = l2.join(p1b);
    let test2 =
        da > l2w * EPSILON && db < l2w * -EPSILON || db > l2w * EPSILON && da < l2w * -EPSILON;

    (test1 && test2).then(|| l1.meet(l2).unitized())
}

impl<S> Geometry<S> {
    pub fn walk_edges(&self, start: EdgeIndex) -> EdgeWalker<'_> {
        let edge = &self.edges[start];
        EdgeWalker {
            edges: &self.edges,
            start,
            next: Some((start, edge)),
        }
    }
}

impl<S> Geometry<S> {
    // Adjusts the edge topology to handle splitting a curve `a`
    // into two curves `a b`
    fn split_curve_topology(&mut self, a: CurveIndex, b: CurveIndex) {
        // Adjust edge topology
        for ei in 0..self.edges.len() {
            let Edge { curve, dir, next } = self.edges[ei];
            if curve == a && matches!(dir, Dir::Fwd) {
                let eib = self.edges.len();
                self.edges.push(Edge {
                    curve: b,
                    dir: Dir::Fwd,
                    next,
                });
                self.edges[ei].next = eib;
            }
        }

        for prev in 0..self.edges.len() {
            let Edge { next: ei, .. } = self.edges[prev];
            let Edge {
                curve,
                dir,
                next: _,
            } = self.edges[ei];
            if curve == a && matches!(dir, Dir::Rev) {
                let eib = self.edges.len();
                self.edges.push(Edge {
                    curve: b,
                    dir: Dir::Rev,
                    next: ei,
                });
                self.edges[prev].next = eib;
            }
        }
    }

    fn split_all_curves_at_intersections(&mut self) {
        for c1 in 0..self.curves.len() {
            // self.curves will change during iteration--but the outer for-loop won't change.
            // This is OK due to the non-intersection guarantees of the newly pushed curves
            for c2 in (c1 + 1)..self.curves.len() {
                match (&self.curves[c1], &self.curves[c2]) {
                    (
                        &Curve::Line(Line {
                            start: start1,
                            end: end1,
                        }),
                        &Curve::Line(Line {
                            start: start2,
                            end: end2,
                        }),
                    ) => {
                        let start1pt = self.points[start1];
                        let end1pt = self.points[end1];
                        let start2pt = self.points[start2];
                        let end2pt = self.points[end2];

                        if let Some(intersection_pt) =
                            intersect_segments(start1pt, end1pt, start2pt, end2pt)
                        {
                            // Push intersection point
                            let intersection_ix = self.points.len();
                            self.points.push(intersection_pt);

                            // Rewrite existing 2 curves and push new curves
                            self.curves[c1] = Curve::Line(Line {
                                start: start1,
                                end: intersection_ix,
                            });
                            self.curves[c2] = Curve::Line(Line {
                                start: start2,
                                end: intersection_ix,
                            });
                            let c1b = self.curves.len();
                            self.curves.push(Curve::Line(Line {
                                start: intersection_ix,
                                end: end1,
                            }));
                            let c2b = self.curves.len();
                            self.curves.push(Curve::Line(Line {
                                start: intersection_ix,
                                end: end2,
                            }));

                            // Adjust edge topology
                            self.split_curve_topology(c1, c1b);
                            self.split_curve_topology(c2, c2b);
                        }
                    }
                    _ => {} // TODO
                }
            }
        }
    }

    fn recalculate_connectivity(&mut self) {
        let mut unvisited: BTreeSet<EdgeIndex> = (0..self.edges.len()).collect();
        let mut new_edges = vec![];
        let mut edge_loop = vec![];
        while let Some(&start_edge) = unvisited.first() {
            unvisited.remove(&start_edge);
            let edge = self.edges[start_edge];
            let Some((start_pt, mut pt)) = self.curves[edge.curve]
                .endpoints()
                .map(|ep| ep.with_dir(edge.dir))
            else {
                // Push self-loop with no points into the output
                let i = new_edges.len();
                new_edges.push(Edge {
                    curve: edge.curve,
                    dir: edge.dir,
                    next: i,
                });
                continue;
            };
            println!("Start at edge {:?}, pt {:?}", start_edge, start_pt);
            edge_loop.clear();
            edge_loop.push((start_pt, start_edge));

            let cycle_start = loop {
                // See if we have found a loop and are done
                if let Some(cycle_start) =
                    edge_loop.iter().position(|&(start_pt, _)| start_pt == pt)
                {
                    println!("Found cycle (cycle_start: {:?})", cycle_start);
                    break Some(cycle_start);
                }

                // Find & walk next edge
                let Some((next_edge, next_pt)) = unvisited
                    .iter()
                    .filter_map(|&i| {
                        let Edge {
                            curve,
                            dir,
                            next: _,
                        } = self.edges[i];
                        let ep = self.curves[curve].endpoints()?;
                        let (from, to) = ep.with_dir(dir);
                        if from != pt {
                            return None;
                        }
                        Some((i, to))
                    })
                    .min_by_key(|&(_, pt)| pt)
                // TODO take edge with sharpest turn angle
                else {
                    // No cycle found
                    println!("No cycle found! Discarding {:?}", edge_loop);
                    break None;
                };
                println!("Walk next edge {:?} to pt {:?}", next_edge, next_pt);

                unvisited.remove(&next_edge);
                edge_loop.push((pt, next_edge));
                pt = next_pt;
            };
            let Some(cycle_start) = cycle_start else {
                continue;
            };

            // Push cycle
            let start_i = new_edges.len();
            let mut i = start_i;
            for &(_, ei) in &edge_loop[cycle_start..] {
                let Edge {
                    curve,
                    dir,
                    next: _,
                } = self.edges[ei];
                i = new_edges.len();
                new_edges.push(Edge {
                    curve,
                    dir,
                    next: i + 1,
                });
            }
            new_edges[i].next = start_i;
        }

        self.edges = new_edges;
    }
}

impl Geometry<Improper> {
    pub fn extend<S>(&mut self, other: &Geometry<S>) {
        let point_offset = self.points.len();
        self.points.extend(&other.points);

        let curve_offset = self.curves.len();
        self.curves.reserve(other.curves.len());
        for curve in &other.curves {
            let new_curve = match curve {
                &Curve::SingularPoint(ix) => Curve::SingularPoint(point_offset + ix),
                &Curve::Line(Line { start, end }) => Curve::Line(Line {
                    start: point_offset + start,
                    end: point_offset + end,
                }),
                &Curve::Arc(Arc { start, end, axis }) => Curve::Arc(Arc {
                    start: point_offset + start,
                    end: point_offset + end,
                    axis,
                }),
                Curve::Circle(c) => Curve::Circle(c.clone()),
            };
            self.curves.push(new_curve);
        }

        let edge_offset = self.edges.len();
        self.edges.reserve(other.edges.len());
        for &Edge { curve, dir, next } in &other.edges {
            self.edges.push(Edge {
                curve: curve_offset + curve,
                dir,
                next: edge_offset + next,
            });
        }
    }

    pub unsafe fn as_proper_unchecked(self) -> Geometry<Proper> {
        Geometry::<Proper> {
            points: self.points,
            curves: self.curves,
            edges: self.edges,

            _state: Default::default(),
        }
    }

    pub fn clip(mut self) -> Geometry<Proper> {
        // Split all curves at curve-curve intersections
        self.split_all_curves_at_intersections();

        // TODO split all curves at curve-point intersections

        // TODO combine points that are close to each other

        // Recalculate connectivity by walking loops (discarding spurs)
        self.recalculate_connectivity();

        // TODO Remove unfilled areas according to fill rule

        dbg!(self);
        todo!();
    }
}

impl From<Geometry<Proper>> for Geometry<Improper> {
    fn from(value: Geometry<Proper>) -> Geometry<Improper> {
        Geometry::<Improper> {
            points: value.points,
            curves: value.curves,
            edges: value.edges,

            _state: Default::default(),
        }
    }
}

impl Geometry<Proper> {
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

        for &Edge { curve, dir, .. } in &self.edges {
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

    pub fn offset(&self, amount: Scalar) -> Geometry<Improper> {
        #[derive(Debug)]
        struct OffsetEdge {
            curve: Curve,
            next: EdgeIndex,
        }

        #[derive(Debug, Clone)]
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

        fn new_cap(from: &Endpoint, axis: Point, to: &Endpoint) -> Option<Curve> {
            // See if the triangle formed by from - to - axis has positive or negative area
            // (negative indicates a concave corner that does not need to be capped,
            // positive indicates a convex corner that needs a cap)
            if from.pt.join(to.pt).join(axis) < (-EPSILON).into() {
                return None;
            }

            println!("Generate cap between {:?} and {:?}", from, to);

            // Corner is convex-- join with arc
            Some(Curve::Arc(Arc {
                start: from.pt_i,
                end: to.pt_i,
                axis,
            }))
            // TODO other join types (miter, square, bevel)
            // https://www.angusj.com/clipper2/Docs/Units/Clipper/Types/JoinType.htm
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
                    // Push cap

                    if let Some(cap_curve) =
                        new_cap(&state.prev_endpoint, orig_start_pt, &start_endpoint)
                    {
                        // Corner is convex -- cap needed

                        let cap_i = new_edges.len();
                        new_edges.push(OffsetEdge {
                            curve: cap_curve,
                            next: cap_i,
                        });

                        // Connect prev to cap
                        new_edges[state.prev_edge_i].next = cap_i;

                        // Set state
                        state.prev_edge_i = cap_i;
                        state.prev_endpoint = start_endpoint
                    }

                    // Push edge
                    let edge_i = new_edges.len();
                    new_edges.push(OffsetEdge {
                        curve,
                        next: edge_i,
                    });

                    // Connect prev to edge
                    new_edges[state.prev_edge_i].next = edge_i;

                    // Set state
                    state.prev_edge_i = edge_i;
                    state.prev_endpoint = end_endpoint;
                }
                None => {
                    // No prior edge--this is the first

                    // Push edge
                    let edge_i = new_edges.len();
                    new_edges.push(OffsetEdge {
                        curve,
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

        let mut unvisited: BTreeSet<EdgeIndex> = (0..self.edges.len()).collect();
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
                                curve: Curve::Circle(Circle {
                                    axis: self.points[pt].with_dir(dir),
                                    radius,
                                }),
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

                        if new_radius > EPSILON {
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

                        if new_radius > EPSILON {
                            assert!(state.is_none());
                            let edge_i = new_edges.len();
                            new_edges.push(OffsetEdge {
                                curve: Curve::Circle(Circle {
                                    axis,
                                    radius: new_radius,
                                }),
                                next: edge_i,
                            });
                        }
                    }
                }
            }

            if let Some(state) = &mut state {
                // See if we need to add the final cap
                if let Some(cap_curve) = new_cap(
                    &state.prev_endpoint,
                    state.orig_start_pt,
                    &state.start_endpoint,
                ) {
                    // Push cap
                    let cap_i = new_edges.len();
                    new_edges.push(OffsetEdge {
                        curve: cap_curve,
                        next: cap_i,
                    });

                    // Update state
                    state.prev_edge_i = cap_i;
                    //state.prev_endpoint doesn't need assignment since we're done
                }

                // Form a loop
                new_edges[state.prev_edge_i].next = state.start_edge_i;
            }
        }

        let edges = new_edges
            .iter()
            .enumerate()
            .map(|(i, &OffsetEdge { curve: _, next })| Edge {
                curve: i,
                dir: Dir::Fwd,
                next,
            })
            .collect();
        let curves = new_edges
            .into_iter()
            .map(|OffsetEdge { curve, next: _ }| curve)
            .collect();

        Geometry {
            points: new_points,
            curves,
            edges,

            _state: Default::default(),
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
            Edge {
                curve: 0,
                dir: Dir::Fwd,
                next: 1,
            },
            Edge {
                curve: 1,
                dir: Dir::Fwd,
                next: 2,
            },
            Edge {
                curve: 2,
                dir: Dir::Fwd,
                next: 0,
            },
            // Interior hole
            /*
                Edge {
                    curve: 3,
                    dir: Dir::Fwd,
                    next: 4,
                },
                Edge {
                    curve: 4,
                    dir: Dir::Fwd,
                    next: 5,
                },
                Edge {
                    curve: 5,
                    dir: Dir::Fwd,
                    next: 6,
                },
                Edge {
                    curve: 6,
                    dir: Dir::Fwd,
                    next: 3,
                },
            */
        ];

        let geometry = Geometry::<Proper> {
            points,
            curves,
            edges,
            _state: Default::default(),
        };

        dbg!(geometry.offset(0.1));
        panic!();
    }

    #[test]
    fn test_intersect_segments() {
        // Segments intersect
        assert_eq!(
            intersect_segments(
                Vector::point([0., 0.]),
                Vector::point([10., 10.]),
                Vector::point([10., 0.]),
                Vector::point([0., 10.]),
            ),
            Some(Vector::point([5., 5.]))
        );

        // Non-intersecting
        assert!(intersect_segments(
            Vector::point([0., 0.]),
            Vector::point([1., 1.]),
            Vector::point([3., 0.]),
            Vector::point([0., 3.]),
        )
        .is_none());

        // End-to-end
        assert!(intersect_segments(
            Vector::point([0., 0.]),
            Vector::point([1., 1.]),
            Vector::point([1., 1.]),
            Vector::point([2., 0.]),
        )
        .is_none());

        // Parallel overlapping segments
        assert!(intersect_segments(
            Vector::point([0., 0.]),
            Vector::point([4., 4.]),
            Vector::point([1., 1.]),
            Vector::point([3., 3.]),
        )
        .is_none());
    }

    #[test]
    fn test_split_all_curves_at_intersections() {
        let points = vec![
            Vector::point([0., 0.]),
            Vector::point([1., 1.]),
            Vector::point([1., 0.]),
            Vector::point([0., 1.]),
        ];

        let curves = vec![
            Curve::Line(Line { start: 0, end: 1 }),
            Curve::Line(Line { start: 2, end: 3 }),
        ];

        let edges = vec![
            Edge {
                curve: 0,
                dir: Dir::Fwd,
                next: 1,
            },
            Edge {
                curve: 0,
                dir: Dir::Rev,
                next: 0,
            },
            Edge {
                curve: 1,
                dir: Dir::Fwd,
                next: 3,
            },
            Edge {
                curve: 1,
                dir: Dir::Rev,
                next: 2,
            },
        ];

        let mut geometry = Geometry::<Improper> {
            points,
            curves,
            edges,
            _state: Default::default(),
        };

        geometry.split_all_curves_at_intersections();

        // All edges should now be split:
        assert_eq!(
            geometry.edges[geometry.edges[geometry.edges[0].next].next].next,
            1
        );
        assert_eq!(
            geometry.edges[geometry.edges[geometry.edges[2].next].next].next,
            3
        );
    }
    #[test]
    fn test_recalculate_connectivity() {
        let points = vec![
            // Hash symbol #
            Vector::point([0.1, 0.1]), // 0
            Vector::point([0.9, 0.1]), // 1
            Vector::point([0.9, 0.9]), // 2
            Vector::point([0.1, 0.9]), // 3
            Vector::point([0., 0.1]),
            Vector::point([1., 0.1]),
            Vector::point([0.9, 0.]),
            Vector::point([0.9, 1.]),
            Vector::point([1., 0.9]),
            Vector::point([0., 0.9]),
            Vector::point([0.1, 1.]),
            Vector::point([0.1, 0.]),
        ];

        let curves = vec![
            Curve::Line(Line { start: 4, end: 0 }),
            Curve::Line(Line { start: 0, end: 1 }),
            Curve::Line(Line { start: 1, end: 5 }),
            Curve::Line(Line { start: 6, end: 1 }),
            Curve::Line(Line { start: 1, end: 2 }),
            Curve::Line(Line { start: 2, end: 7 }),
            Curve::Line(Line { start: 8, end: 2 }),
            Curve::Line(Line { start: 2, end: 3 }),
            Curve::Line(Line { start: 3, end: 9 }),
            Curve::Line(Line { start: 10, end: 3 }),
            Curve::Line(Line { start: 3, end: 0 }),
            Curve::Line(Line { start: 0, end: 11 }),
        ];

        let edges: Vec<_> = (0..12)
            .map(|i| Edge {
                curve: i,
                dir: Dir::Fwd,
                next: (i + 1) % 12,
            })
            .collect();

        let mut geometry = Geometry::<Improper> {
            points,
            curves,
            edges,
            _state: Default::default(),
        };

        geometry.recalculate_connectivity();

        assert!(geometry.edges.len() == 4);
        for &Edge { curve, .. } in &geometry.edges {
            let Curve::Line(Line { start, end, .. }) = geometry.curves[curve] else {
                panic!();
            };
            assert!(start < 4); // Only the first 4 points should be used
            assert!(end < 4); // Only the first 4 points should be used
        }
    }
}
