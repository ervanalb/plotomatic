use ngeom::scalar::*;
use std::array::from_fn;
use std::ops::Range;
use super_seq_macro::seq;

///////////////////////////////////////////////////////////

/// Compute the binomial coefficient (n choose k)
pub fn binomial(n: usize, k: usize) -> usize {
    let numerator: usize = (1..=k).map(|i| n + 1 - i).product();
    let denominator: usize = (1..=k).product();
    numerator / denominator
}

/*
/// Convert a polynomial of degree N to Bernstein basis
/// c_m: The coefficients of the polynomial in monomial form
/// i.e. the polynomial = sum of c_m * [1, x, x^2]
/// For example, x^2 + x - 6 would be represented as [-6, 1, 1]
fn bernstein_coef<A, O, S: Shape, T: Ring + Rational + std::iter::Sum>(c_m: &A, out: &mut O)
where
    for<'a> &'a A: Into<View<'a, S, [T]>>,
    for<'a> &'a mut O: Into<ViewMut<'a, S, [T]>>,
{
    let c_m = c_m.into();
    let mut out = out.into();

    for (i, out_entry) in (&mut out).into_iter().nd_enumerate() {
        *out_entry = (..=i)
            .nd_iter()
            .map(|j| {
                let num: usize = i
                    .into_iter()
                    .zip(j.into_iter())
                    .map(|(i_n, j_n)| binomial(i_n, j_n))
                    .product();
                let den: usize = S::SHAPE
                    .into_iter()
                    .zip(j.into_iter())
                    .map(|(d_n, j_n)| binomial(d_n - 1, j_n))
                    .product();
                let b = T::from_fraction(num.try_into().unwrap(), den.try_into().unwrap());
                b * c_m[j]
            })
            .sum();
    }
}
*/

/*
fn bernstein_coef2<T: Ring + Rational + std::iter::Sum, const N0: usize, const N1: usize>(
    c_m: &[[T; N1]; N0],
    out: &mut [[T; N1]; N0],
) {
    for (i0, e1) in out.iter_mut().enumerate() {
        for (i1, e) in e1.iter_mut().enumerate() {
            *e = (0..=i0)
                .flat_map(move |j0| {
                    (0..=i1).map(move |j1| {
                        let num = binomial(i0, j0) * binomial(i1, j1);
                        let den = binomial(N0 - 1, j0) * binomial(N1 - 1, j1);
                        let b = T::from_fraction(num.try_into().unwrap(), den.try_into().unwrap());
                        b * c_m[j0][j1]
                    })
                })
                .sum();
        }
    }
}

fn bernstein_coef3<
    T: Ring + Rational + std::iter::Sum,
    const N0: usize,
    const N1: usize,
    const N2: usize,
>(
    c_m: &[[[T; N2]; N1]; N0],
    out: &mut [[[T; N2]; N1]; N0],
) {
    for (i0, e0) in out.iter_mut().enumerate() {
        for (i1, e1) in e0.iter_mut().enumerate() {
            for (i2, e) in e1.iter_mut().enumerate() {
                *e = (0..=i0)
                    .flat_map(move |j0| {
                        (0..=i1).flat_map(move |j1| {
                            (0..=i2).map(move |j2| {
                                let num = binomial(i0, j0) * binomial(i1, j1) * binomial(i2, j2);
                                let den = binomial(N0 - 1, j0)
                                    * binomial(N1 - 1, j1)
                                    * binomial(N2 - 1, j2);
                                let b = T::from_fraction(
                                    num.try_into().unwrap(),
                                    den.try_into().unwrap(),
                                );
                                b * c_m[j0][j1][j2]
                            })
                        })
                    })
                    .sum();
            }
        }
    }
}
*/

macro_rules! ndarray_t {
    ($t:ty; ()) => {
        $t
    };
    ($t:ty; ($n:expr, $($rest:expr,)*)) => {
        [ndarray_t!($t; ($($rest,)*)); $n]
    };
}
macro_rules! ndarray {
    ($x:expr; ()) => {
        $x
    };
    ($x:expr; ($n:expr, $($rest:expr,)*)) => {
        [ndarray!($x; ($($rest,)*)); $n]
    };
}

macro_rules! ndfor {
    ($block:block) => {
        $block
    };

    ($pat:pat in $expr:expr, $($pats:pat in $exprs:expr,)* $block:block) => {
        for $pat in $expr {
            ndfor!($($pats in $exprs,)* $block)
        }
    };
}

macro_rules! ndmap {
    ($block:block) => {
        $block
    };

    ($pat:pat in $expr:expr, $block:block) => {
        $expr.map(move |$pat| $block)
    };

    ($pat:pat in $expr:expr, $($pats:pat in $exprs:expr,)* $block:block) => {
        $expr.flat_map(move |$pat| ndmap!($($pats in $exprs,)* $block))
    };
}
/*
macro_rules! ndmap {
    (() .map(|()| $block:expr )) => {
        $block
    };

    (($first:expr,) .map ( | ($arg:pat,) | $block:expr )) => {
        $first.map(move |$arg|
            $block
        )
    };

    (($first:expr, $($rest:expr,)*) .map ( | ($arg:pat, $($args:pat,)*) | $block:expr )) => {
        $first.flat_map(move |$arg|
            ndmap!(($($rest,)*) .map( |($($args,)*)| $block))
        )
    };
}
*/

/*
fn foo() {
    let mut out = &mut [[[0]]];

    ndfor!((i0, e0) in out.iter_mut().enumerate(), (i1, e1) in e0.iter_mut().enumerate(), (i2, e) in e1.iter_mut().enumerate(), {
        todo!()
    });

    let (i0, i1, i2) = (1, 2, 3);

    //ndmap!(((0..=i0), (0..=i1), (0..=i2),).map(|(j0,j1,j2,)| { todo!(); }));
}
*/

seq!(D in 1..=6 {#(
    seq!(A in 0..D {
        /// Convert a D-variate polynomial of degrees N0..ND to Bernstein basis
        /// c_m: D-dimensional array containing the coefficients of the polynomial in monomial form
        /// i.e. the polynomial = sum of c_m * [1, x, x^2]
        /// For example, x^2 + x - 6 would be represented as [-6, 1, 1]
        pub fn bernstein_coef~D<
            T: Ring + Rational + std::iter::Sum,
            #( const N~A: usize, )*
        >(
            c_m: & ndarray_t![T; (#( N~A, )*) ] ,
        ) -> ndarray_t![T; (#( N~A, )*) ] {
            let mut out = ndarray![T::zero(); (#( N~A, )*) ];
            ndfor!(#(i~A in 0..N~A,)* {
                out #( [i~A] )* = ndmap!( #( j~A in (0..=i~A), )* {
                    let num = 1 #( * binomial(i~A, j~A) )*;
                    let den = 1 #( * binomial(N~A - 1, j~A) )*;
                    let b = T::from_fraction(
                        num.try_into().unwrap(),
                        den.try_into().unwrap(),
                    );
                    b * c_m #( [j~A] )*
                })
                .sum();
            });
            out
        }
    });
)*});

pub fn relative_range<const N: usize, T: Ring + Recip<Output = T>>(
    Range {
        start: base_start,
        end: base_end,
    }: Range<[T; N]>,
    Range {
        start: new_start,
        end: new_end,
    }: Range<[T; N]>,
) -> Range<[T; N]> {
    let base_width_recip: [T; N] = from_fn(|i| (base_end[i] - base_start[i]).recip());
    Range {
        start: from_fn(|i| (new_start[i] - base_start[i]) * base_width_recip[i]),
        end: from_fn(|i| (new_end[i] - base_start[i]) * base_width_recip[i]),
    }
}

pub fn span<T: Ring + Rational + PartialOrd + Recip<Output = T>, const MP1: usize>(
    c: &[T; MP1],
    region: Range<T>,
) -> Option<Range<T>> {
    let m = MP1 - 1;

    // We will compute the span of the X axis
    // that is intersected by the convex hull
    // of the polynomial interpreted as a cubic bezier curve

    // We will do the convex hull computation assuming
    // control points at x=0, 1, 2, ..= M
    // and then, at the end, re-interpret this range relative to the given input region.

    fn cross<T: Ring>([ox, oy]: [T; 2], [ax, ay]: [T; 2], [bx, by]: [T; 2]) -> T {
        (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
    }

    // Monotone convex hull algorithm--the points are already sorted left-to-right.
    let mut lower = [[T::zero(); 2]; MP1];
    let mut lower_sp: usize = 0;
    let mut upper = [[T::zero(); 2]; MP1];
    let mut upper_sp: usize = 0;
    for (x, &y) in c.iter().enumerate() {
        let x = T::from_integer(x.try_into().unwrap());
        while lower_sp >= 2 && cross(lower[lower_sp - 2], lower[lower_sp - 1], [x, y]) <= T::zero()
        {
            lower_sp -= 1;
        }
        lower[lower_sp] = [x, y];
        lower_sp += 1;

        while upper_sp >= 2 && cross(upper[upper_sp - 2], upper[upper_sp - 1], [x, y]) >= T::zero()
        {
            upper_sp -= 1;
        }
        upper[upper_sp] = [x, y];
        upper_sp += 1
    }

    assert!(lower_sp >= 2);
    assert!(upper_sp >= 2);

    pub fn x_intercept<T: Ring + Recip<Output = T>>([x1, y1]: [T; 2], [x2, y2]: [T; 2]) -> T {
        let y_width_recip = (y2 - y1).recip();
        (x1 * y2 - x2 * y1) * y_width_recip
    }

    // Initialize these such that
    // zero or one intersections will result in a backwards (empty) range
    // that will get emitted as None
    let mut x_min: Option<T> = None;
    let mut x_max: Option<T> = None;

    // Find the two segments of the convex hull that intersect the X axis
    let [mut x1, mut y1] = upper[0]; // Check left endcap
    for &[x2, y2] in lower[0..lower_sp]
        .iter()
        .chain(upper[0..upper_sp].iter().rev())
    {
        if y1 >= T::zero() && y2 < T::zero() {
            x_min = Some(x_intercept([x1, y1], [x2, y2]));
        } else if y1 <= T::zero() && y2 > T::zero() {
            x_max = Some(x_intercept([x1, y1], [x2, y2]));
        }
        [x1, y1] = [x2, y2];
    }

    let (x_min, x_max) = match (x_min, x_max) {
        (None, None) => {
            return None;
        }
        (Some(x_min), None) => (x_min, x_min),
        (None, Some(x_max)) => (x_max, x_max),
        (Some(x_min), Some(x_max)) => (x_min, x_max),
    };

    // Re-interpret the range 0..MP1 to region
    let scale = T::from_fraction(1, m.try_into().unwrap()) * (region.end - region.start);
    Some(Range {
        start: x_min * scale + region.start,
        end: x_max * scale + region.start,
    })
}

pub fn decasteljau_split<const N: usize, T: Ring>(
    matrix: &mut [[T; N]; N],
    Range { start: u, end: v }: Range<T>,
) {
    let inv_u = T::one() - u;
    let inv_v = T::one() - v;

    for layer in 1..N {
        // for each "layer"
        let width = N - layer;
        // do the "v" row as a special case
        let row = layer;
        for col in 0..width {
            matrix[row][col] = matrix[row - 1][col] * inv_v + matrix[row - 1][col + 1] * v;
        }
        // do multiple "u" rows
        for row in 0..layer {
            for col in 0..width {
                matrix[row][col] = matrix[row][col] * inv_u + matrix[row][col + 1] * u;
            }
        }
    }
}

seq!(D in 1..=6 {#(
    seq!(A in 0..D {
        pub fn change_domain~D<
            T: Ring + Rational + std::iter::Sum,
            #( const N~A: usize, )*
        >(
            c: &ndarray_t![T; (#( N~A, )*) ] ,
            Range { start, end }: Range<[T; D]>,
        ) -> ndarray_t![T; (#( N~A, )*) ] {
            let mut out = c.clone();
            #(
                // Axis A
                seq!(B in (0..D).collect().filter(|x| x != A) {
                    ndfor!( #( i~B in 0..N~B, )* {
                        let mut decasteljau_matrix = [[T::zero(); N~A]; N~A];
                        // Copy this lane into the 2D matrix for de Casteljau interpolation
                        for i~A in 0..N~A {
                            seq!(C in 0..D {
                                decasteljau_matrix[0][i~A] = out #(#( [ i~C ] )*)#;
                            });
                        }
                        // Compute the new domain along this lane
                        decasteljau_split(&mut decasteljau_matrix, start[A]..end[A]);
                        // Copy the result from the matrix back into the lane
                        for i~A in 0..N~A {
                            #(
                                seq!(C in 0..D {
                                    out #(#( [ i~C ] )*)# = decasteljau_matrix[i~A][0];
                                });
                            )#
                        }
                    });
                });
            )*
            out
        }
    });
)*});

/*
pub fn change_domain<A, B, const N0: usize, const N1: usize, const N2: usize, T: Ring + Rational>(
    c: &A,
    Range { start, end }: Range<[T; 3]>,
    out: &mut B,
) where
    for<'a> &'a A: Into<View<'a, (Const<N0>, Const<N1>. Const<N2>), [T]>>,
    for<'a> &'a mut B: Into<ViewMut<'a, (Const<N0>, Const<N1>, Const<N2>), [T]>>,
{

    let c = c.into();
    let out = out.into();

    //let m = MP1 - 1;

    let flip_start: [T; N] = from_fn(|i| T::one() - start[i]);
    let flip_end: [T; N] = from_fn(|i| T::one() - end[i]);

    for n0 in 0..N0 {
        for n1 in 0..N1 {
        }
    }

    // TODO(optimizations):
    // * lerp_layer1 could be flipped to avoid indexing from the back
    // * the two lerp layers could be done sequentially
    // * the results could then be collected instead of filled in

    // Populate bottom row of the triangle with the existing control points
    let mut lerp_layer1 = c.clone();
    let mut lerp_layer2 = c.clone();
    for i in 0..=m {
        // Output the right-most and left-most entries as new knots of the middle interval
        c_out[m - i] = lerp_layer1[m - i];
        c_out[i] = lerp_layer2[0];

        // Compute the next lerp_layer (i + 1)
        for i2 in 0..(m - i) {
            lerp_layer1[i2] = flip_start * lerp_layer1[i2] + start * lerp_layer1[i2 + 1];
            lerp_layer2[i2] = flip_end * lerp_layer2[i2] + end * lerp_layer2[i2 + 1];
        }
    }

    c_out
}

pub fn bernstein_root_search<
    T: Ring + Rational + PartialOrd + Recip<Output = T>,
    const MP1: usize,
>(
    cs: &[[T; MP1]],
    region: Range<T>,
    tol: T, // How accurately to compute the root
) -> Vec<T> {
    let mut roots = Vec::<T>::new();
    let mut regions_to_process = vec![region];

    //let mut iters = 0;

    while let Some(region) = regions_to_process.pop() {
        //iters += 1;
        if let Some(region_shrunk) = cs
            .iter()
            .map(|c| span(&change_domain(c, region.clone()), region.clone()))
            .reduce(|acc, item| match (acc, item) {
                (
                    Some(Range {
                        start: amin,
                        end: amax,
                    }),
                    Some(Range {
                        start: xmin,
                        end: xmax,
                    }),
                ) => Some(Range {
                    start: if xmin < amin { xmin } else { amin },
                    end: if xmax > amax { xmax } else { amax },
                }),
                (_, None) | (None, _) => None,
            })
            .flatten()
            .filter(|&Range { start, end }| end >= start)
        {
            if region_shrunk.end - region_shrunk.start < tol {
                // Found root
                roots.push(T::one_half() * (region_shrunk.start + region_shrunk.end));
            } else {
                let region_width = region.end - region.start;
                if region_shrunk.start - region.start < tol * region_width
                    && region.end - region_shrunk.end < tol * region_width
                {
                    // Region is not shrinking--multiple roots
                    // Split in half
                    let shrunk_region_split =
                        T::one_half() * (region_shrunk.start + region_shrunk.end);
                    let region_q = Range {
                        start: region_shrunk.start,
                        end: shrunk_region_split,
                    };
                    let region_r = Range {
                        start: shrunk_region_split,
                        end: region_shrunk.end,
                    };
                    regions_to_process.push(region_q);
                    regions_to_process.push(region_r);
                } else {
                    // Region is shrinking--iterate
                    regions_to_process.push(region_shrunk);
                }
            }
        }
    }

    if roots.len() == 0 {
        return roots;
    }

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Remove duplicate roots that are within +/- tol of each other
    let mut first = roots[0];
    let mut last = roots[0];
    let mut roots_dedup = Vec::new();
    for &root in &roots[1..] {
        if root - first < tol + tol {
            // This is a duplicate
            last = root;
        } else {
            // This is not a duplicate; push previous value
            // (or midpoint of previous values if there were duplicates)
            roots_dedup.push(T::one_half() * (first + last));
            first = root;
            last = root;
        }
    }
    // Push final value
    roots_dedup.push(T::one_half() * (first + last));
    roots_dedup
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_bernstein_coef() {
        let c_m = [0., -1., 1.];
        let c_b = bernstein_coef1(&c_m);
        assert_eq!(c_b, [0., -0.5, 0.]);
    }

    #[test]
    pub fn test_relative_range() {
        let r = relative_range([10.]..[50.], [20.]..[40.]);
        assert_eq!(r, [0.25]..[0.75]);
    }

    #[test]
    pub fn test_span_some() {
        let c_b = [1., -1., -1., 1.];
        let r = span(&c_b, (10.)..(16.));
        assert_eq!(r, Some((11.)..(15.)));
    }

    #[test]
    pub fn test_span_none() {
        let c_b = [1., 2., 3., 1.];
        let r = span(&c_b, (10.)..(18.));
        assert_eq!(r, None);
    }

    #[test]
    pub fn test_change_domain() {
        let c = [40., -40., -40., 40.];
        let c = change_domain1(&c, [0.25]..[0.75]);
        assert_eq!(c, [-5., -25., -25., -5.]);
    }

    /*
    #[test]
    pub fn test_root_finding_1() {
        let c_m = [-0.2, 1.4, -1.1];
        let c_b = bernstein_coef(&c_m);
        let mut roots = bernstein_root_search(&[c_b], (0.)..(2.), 1e-5);

        assert!(roots.len() == 2);

        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((roots[0] - 0.16398614).abs() < 1e-5);
        assert!((roots[1] - 1.10874113).abs() < 1e-5);
    }

    #[test]
    pub fn test_root_finding_2() {
        let c_m = [-0.042, 0.55, -1.4, 1.];
        let c_b = bernstein_coef(&c_m);
        let mut roots = bernstein_root_search(&[c_b], (0.)..(1.), 1e-5);

        dbg!(&roots);
        assert!(roots.len() == 3);

        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((roots[0] - 0.1).abs() < 1e-5);
        assert!((roots[1] - 0.6).abs() < 1e-5);
        assert!((roots[2] - 0.7).abs() < 1e-5);
    }

    #[test]
    pub fn test_root_finding_3() {
        let c_m = [0.25, -1., 1.];
        let c_b = bernstein_coef(&c_m);
        let roots = bernstein_root_search(&[c_b], (0.)..(1.), 1e-5);

        dbg!(&roots);
        assert!(roots.len() == 1);

        assert!((roots[0] - 0.5).abs() < 1e-5);
    }
    */
}
