use ngeom::scalar::*;
use std::array::from_fn;
use std::ops::Range;
use super_seq_macro::seq;

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

///////////////////////////////////////////////////////////

/// Compute the binomial coefficient (n choose k)
pub fn binomial(n: usize, k: usize) -> usize {
    let numerator: usize = (1..=k).map(|i| n + 1 - i).product();
    let denominator: usize = (1..=k).product();
    numerator / denominator
}

seq!(D in 1..=6 {#(
    seq!(A in 0..D {
        /// Convert a D-variate polynomial of degrees N0..ND to Bernstein basis
        /// c_m: D-dimensional array containing the coefficients of the polynomial in monomial form
        /// i.e. the polynomial = sum of c_m * [1, x, x^2]
        /// For example, x^2 + x - 6 would be represented as [-6, 1, 1]
        pub fn from_monomial~D<
            T: Ring + Rational,
            #( const N~A: usize, )*
        >(
            c_m: & ndarray_t![T; (#( N~A, )*) ] ,
        ) -> ndarray_t![T; (#( N~A, )*) ] {
            let mut out = ndarray![T::zero(); (#( N~A, )*) ];
            ndfor!(#(i~A in 0..N~A,)* {
                let mut s = T::zero();
                ndfor!( #( j~A in 0..=i~A, )* {
                    let num = 1 #( * binomial(i~A, j~A) )*;
                    let den = 1 #( * binomial(N~A - 1, j~A) )*;
                    let b = T::from_fraction(
                        num.try_into().unwrap(),
                        den.try_into().unwrap(),
                    );
                    s += b * c_m #( [j~A] )*;
                });
                out #( [i~A] )* = s;
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

/// Perform an affine transform on the parameter of a univariate Bernstein polynomial.
/// The output polynomial evaluated at 0..1 will map to the input polynomial on the range u..v.
///
/// The input and output are both taken from a scratchpad matrix,
/// where the input is read from the first row,
/// and the output is placed in the first column.
fn decasteljau_split<const N: usize, T: Ring>(
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
        /// Perform an affine transform on the domain of a D-variate Bernstein polynomial.
        /// The output polynomial evaluated at [0, 0, .., 0]..[1, 1, .., 1]
        /// will map to the input polynomial on the range [u_0, u_1, .., u_D]..[v_0, v_1, .., v_D].
        /// c: D-dimensional array containing the coefficients of the polynomial in Bernstein form
        /// u..v: The portion of the input domain which will be mapped to 0..1 in the output domain
        pub fn change_domain~D<
            T: Ring + Rational,
            #( const N~A: usize, )*
        >(
            c: &ndarray_t![T; (#( N~A, )*) ] ,
            Range { start: u, end: v }: Range<[T; D]>,
        ) -> ndarray_t![T; (#( N~A, )*) ] {
            let mut out = c.clone();
            #(
                // Axis A of D
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
                        decasteljau_split(&mut decasteljau_matrix, u[A]..v[A]);
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
pub fn bernstein_root_search<
    T: Ring + Rational + PartialOrd + Recip<Output = T>,
    const N0: usize,
>(
    cs: &[[T; N0]],
    region: Range<T>,
) -> Vec<T> {
    let mut roots = Vec::<T>::new();
    let mut regions_to_process = vec![region];

    //let mut iters = 0;

    while let Some(region) = regions_to_process.pop() {
        //iters += 1;
        if let Some(region_shrunk) = cs
            .iter()
            .map(|c| span(&change_domain1(c, region.clone()), region.clone()))
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegionCategory {
    NoRoot,
    Unknown,
    OneRoot,
}

seq!(D in 1..=6 {#(
    seq!(A in 0..D {
        pub fn contains_no_roots~D<
            #( const N~A: usize, )*
            T: Ring + PartialOrd + std::fmt::Debug
        >(
            c: &ndarray_t![T; (#( N~A, )*) ] ,
        ) -> bool {
            // Check whether all coefficients are positive or all coefficients are negative
            let all_positive = || {
                ndfor!( #( i~A in 0..N~A, )* {
                    let a = c #( [ i~A ] )* ;
                    if !(a > T::zero()) {
                        return false;
                    }
                });
                true
            };
        
            let all_negative = || {
                ndfor!( #( i~A in 0..N~A, )* {
                    let a = c #( [ i~A ] )* ;
                    if !(a < T::zero()) {
                        return false;
                    }
                });
                true
            };

            all_positive() || all_negative()
        }
    });
)*});

seq!(D in 1..=6 {#(
    seq!(A in 0..D {
        pub fn exactly_one_root~D<
            const N: usize,
            T: Ring + Rational + PartialOrd + Sqrt<Output=T> + std::fmt::Debug
        >(
            cs: &[ndarray_t![T; (#( N, )*) ]; D],
        ) -> bool {

            // TODO better epsilon handling
            let tol = T::from_fraction(1, 10_000);

            // Perform Poincare-Miranda existance check.
            // Due to preconditioning, equation N should be positive
            // on the left side of dimension N
            // and negative on the right side of dimension N.

            #(
                // For each equation / axis A
                let c = cs[A];
                seq!(B in (0..D).collect().filter(|x| x != A) {
                    ndfor!( #( i~B in 0..N, )* {
                        seq!(C in 0..D {
                            let i~A = 0; // Ensure left side is positive
                            if c #(#( [ i~C ] )*)# < T::zero() {
                                return false;
                            }
                            let i~A = N - 1; // Ensure right side is negative
                            if c #(#( [ i~C ] )*)# > T::zero() {
                                return false;
                            }
                        });
                    });
                });
            )*
            // Existance test passed!

            // Perform Elbert-Kim uniqueness test.
            // Due to preconditioning, the bundle of normals of surface N
            // should be clustered around the vector pointing in direction N,
            // and we will use this vector as the central vector.

            #(
                // For each equation A

                // Compute the partial derivatives of c, for all axes B != A,
                // summing up the squares into the first element of the normal bundle ndarray
                seq!(B in 0..D {
                    let mut normal_bundle = ndarray![[T::zero(); 2]; ( #( N, )* ) ];
                });
                seq!(B in (0..D).collect().filter(|x| x != A) {#(
                    // Compute the partial derivative of c with respect to axis B,
                    // with degree elevation to maintain squareness
                    seq!(C in (0..D).collect().filter(|x| x != B) {
                        ndfor!( #( i~C in 0..N, )* {
                            seq!(E in 0..D {
                                // Store the univariate polynomial we are processing in a temporary variable
                                let mut c_u: [T; N] = from_fn(|i~B| c #(#( [i~E] )*)#);

                                in_place_univariate_derivative_and_degree_elevation(&mut c_u);

                                // Store the univariate polynomial into the output
                                for i~B in 0..N {
                                    // Accumulate sum of squares
                                    normal_bundle #(#( [i~E] )*)# [0] += c_u[i~B] * c_u[i~B];
                                }
                            });
                        });
                    });
                )*});

                // Take the square root
                // TODO is this necessary!?
                seq!(B in 0..D {
                    ndfor!( #( i~B in 0..N, )* {
                        normal_bundle #( [i~B] )* [0] = normal_bundle #( [i~B] )* [0].sqrt(); 
                    });
                });

                // Compute the remaining partial derivative of c with respect to axis A
                // and put it in the second element
                seq!(B in (0..D).collect().filter(|x| x != A) {
                    ndfor!( #( i~B in 0..N, )* {
                        seq!(C in 0..D {
                            // Store the univariate polynomial we are processing in a temporary variable
                            let mut c_u: [T; N] = from_fn(|i~A| c #(#( [i~C] )*)#);

                            in_place_univariate_derivative_and_degree_elevation(&mut c_u);

                            // Store the univariate polynomial into the output
                            for i~A in 0..N {
                                // This is the projective dimension--
                                // we expect all of these to be definitively positive
                                if c_u[i~A] < tol {
                                    return false;
                                }
                                normal_bundle #(#( [i~C] )*)# [1] = c_u[i~A];
                            }
                        });
                    });
                });

                // Now that we have the normal bundle [x, w] pairs,
                // find the normal vector with the largest deviation (largest x / w)
                let mut max_normal = [T::zero(), T::one()];
                seq!(B in 0..D {
                    ndfor!( #( i~B in 0..N, )* {
                        let e = normal_bundle #( [i~B] )*;
                        if e[0] * max_normal[1] > max_normal[0] * e[1] {
                            max_normal = e;
                        }
                    });
                });
                let max_normal~A = max_normal;
            )*

            // Now that we have found the max normal deviation in each axis,
            // we will see if any two axes have overlapping normal cones.
            // If the sum of the two axes' max normal deviations is > 90 degrees,
            // then they overlap.

            #(
                seq!(B in A..D {#(
                    // Given a vector [x, w], tan(theta) = x / w.
                    // The "sum of angles" formula for tangent tells us that if tan(a) * tan(b) > 1,
                    // then a + b > 90 degrees.
                    if max_normal~A[0] * max_normal~B[0] > max_normal~A[1] * max_normal~B[1] {
                        // Overlapping tangent cones
                        return false;
                    }
                )*});
            )*

            // Uniqueness test passed!
            true
        }
    });
)*});

fn _decasteljau_eval<const N0: usize, T: Ring>(arr: &[T; N0], [t]: [T; 1]) -> T {
    let inv_t = T::one() - t;
    let mut arr = arr.clone();

    for layer in 1..N0 {
        // for each "layer"
        let width = N0 - layer;
        for i in 0..width {
            arr[i] = arr[i] * inv_t + arr[i + 1] * t;
        }
    }

    arr[0]
}

fn _derivative<const N0: usize, const M0: usize, T: Ring>(c: &[T; N0]) -> [T; M0] {
    const {
        assert!(N0 == M0 + 1);
    }

    let mut out = [T::zero(); M0];

    for i in 0..M0 {
        let n = T::from_integer((N0 - 2).try_into().unwrap());
        out[i] = n * (c[i + 1] - c[i]);
    }

    out
}

fn in_place_univariate_derivative_and_degree_elevation<const N: usize, T: Ring + Rational>(
    c: &mut [T; N],
) {
    // Univariate derivative
    let derivative_degree = T::from_integer((N - 2).try_into().unwrap());
    for i in 0..(N - 1) {
        c[i + 1] = derivative_degree * (c[i + 1] - c[i]);
    }

    // Univariate degree elevation

    // Populate the last slot
    // before N - 2 gets overwritten
    c[N - 1] = c[N - 2];

    // Work backwards
    for i in (1..(N - 1)).rev() {
        let a = T::from_fraction(i.try_into().unwrap(), (N - 1).try_into().unwrap());
        let a_inv = T::from_fraction((N - 1 - i).try_into().unwrap(), (N - 1).try_into().unwrap());

        c[i] = a * c[i - 1] + a_inv * c[i];
    }
}

macro_rules! subdivide_ndpush {
    ($vec:ident $tol:ident $([$($start:ident,)*]..[$($end:ident,)*])*) => {
        $(
            $vec.push([$($start,)*]..[$($end,)*]);
        )*
    };

    ($vec:ident $tol:ident $u0:ident $t0:ident $v0:ident, $($u:ident $t:ident $v:ident,)* $([$($start:ident,)*]..[$($end:ident,)*])*) => {
        if ($u0 - $v0) < $tol && ($u0 - $v0) > -$tol {
            subdivide_ndpush!($vec $tol $($u $t $v,)* $([$($start,)* $t0,]..[$($end,)* $t0,])*);
        } else {
            subdivide_ndpush!($vec $tol $($u $t $v,)* $([$($start,)* $u0,]..[$($end,)* $t0,])* $([$($start,)* $t0,]..[$($end,)* $v0,])* );
        }
    };
}

seq!(D in 1..=6 {#(
    seq!(A in 0..D {
        pub fn root_search~D<
            const N: usize,
            T: Ring + Rational + Sqrt<Output=T> + PartialOrd + Recip<Output = T> + std::fmt::Debug,
        >(
            cs: &[ndarray_t![T; (#( N, )*) ]; D],
            region: Range<[T; D]>,
            tol: T,
        ) -> Vec<[T; D]> {
            let mut roots = Vec::<[T; D]>::new();
            let mut regions_to_process = vec![region];

            while let Some(Range {
                start: [#( u~A, )*],
                end: [#( v~A, )*],
            }) = regions_to_process.pop()
            {
                println!("Process region {:?}..{:?}", [#( u~A, )*], [#( v~A, )*]);
                // See if this region has no width in any dimension
                if true #( && u~A == v~A )* {
                    roots.push([ #( u~A, )* ]);
                    continue;
                }

                // See if we can eliminate this region
                if cs.iter().any(|c| contains_no_roots~D(c)) {
                    // Ignore this region
                    println!("No root!");
                    continue;
                }

                // Precondition
                // TODO
                
                if exactly_one_root~D(cs) {
                    println!("One root!");

                    // TODO don't subdivide once we are down to one root
                    // TODO use Newton-Raphson or box contraction
                    #(
                        let t~A = T::one_half() * (u~A + v~A);
                    )*

                    subdivide_ndpush!(regions_to_process tol #(u~A t~A v~A,)* []..[]);
                } else {
                    // Subdivide
                    println!("Subdivide!");
                    #(
                        let t~A = T::one_half() * (u~A + v~A);
                    )*

                    subdivide_ndpush!(regions_to_process tol #(u~A t~A v~A,)* []..[]);
                }
            }

            roots
        }
    });
)*});

seq!(N in 2..=6 {#(
    pub fn adjugate~N<
        T: Ring + Rational,
    >(
        mat: &[[T; N]; N],
    ) -> [[T; N]; N] {

        seq!(CODE in {
            // Helper functions
            let cartesian_product = |a| {
                let acc = [[]];
                
                loop {
                    let last = a.pop();
                    if last.type_of() == "()" {
                        return acc;
                    }

                    let new_acc = [];
                    for e1 in last {
                        for e2 in acc {
                            new_acc.push([e1] + e2);
                        }
                    }
                    acc = new_acc;
                }
                acc
            };

            let strictly_increasing = |a| {
                let prev = a.pop();
                if prev.type_of() == "()" {
                    return true;
                }
                loop {
                    let e = a.pop();
                    if e.type_of() == "()" {
                        return true;
                    } else if e >= prev {
                        return false;
                    }
                    prev = e;
                }
            };

            let choose = |a, k| {
                let acc = [];
                
                for i in 0..k {
                    acc.push(a);
                }
                cartesian_product.call(acc).filter(strictly_increasing)
            };

            let sub_name = |r, c| {
                let r_str = r.reduce(|acc, x| acc + x, "");
                let c_str = c.reduce(|acc, x| acc + x, "");
                $"r${r_str}_c${c_str}"$
            };

            let code = [];

            // Unpack matrix into local variables
            for row in 0..N {
                for col in 0..N {
                    code.push($"let ${sub_name.call([row], [col])} = mat[${row}][${col}];"$);
                }
            }

            // Compute successively larger determinants, from 2x2 up to (N-1) x (N-1)
            for det_n in 2..N {
                // Compute determinants of size det_n.
                // Due to expansion around the first row,
                // not all rows are needed depending on what det_n we are at
                let row_options = ((N - det_n - 1)..N).collect();
                let col_options = (0..N).collect();

                for rowscols in cartesian_product.call([choose.call(row_options, det_n), choose.call(col_options, det_n)]) {
                    let rows = rowscols[0];
                    let cols = rowscols[1];

                    // Compute the determinant of the sub-matrix
                    // composed of the intersection of these rows & cols
                    // by expanding along the first row (a)
                    // and using a previously-computed smaller determinant (b)
                    let a_row = rows[0];
                    let b_rows = rows.filter(|r| r != a_row);

                    let expr = "";
                    let sign = 1;
                    let first = true;
                    for a_col in cols {
                        if !first {
                            expr += " + ";
                        }
                        let b_cols = cols.filter(|c| c != a_col);
                        if sign < 0 {
                            expr += "-";
                        }
                        expr += $"${sub_name.call([a_row], [a_col])} * ${sub_name.call(b_rows, b_cols)}"$;
                        sign *= -1;
                        first = false;
                    }
                    code.push($"let ${sub_name.call(rows, cols)} = ${expr};"$);
                }
            }

            // Collect the result into a 2D array
            let adjugate_entry = |row, col| {
                let minor_rows = (0..N).collect().filter(|i| i != row);
                let minor_cols = (0..N).collect().filter(|i| i != col);
                let neg = if (row + col) % 2 == 0 { "" } else { "-" };
                neg + sub_name.call(minor_cols, minor_rows) // Note transposition
            };

            let result_expr = "[";
            for row in 0..N {
                result_expr += "[";
                for col in 0..N {
                    result_expr += adjugate_entry.call(row, col) + ",";
                }
                result_expr += "],";
            }
            result_expr += "]";

            code.push($"let result = ${result_expr};"$);

            code
        } {#( CODE )*});
        result
    }
)*});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_from_monomial() {
        let c_m = [0., -1., 1.];
        let c_b = from_monomial1(&c_m);
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

    #[test]
    pub fn test_root_finding_1() {
        let c_m = [-0.2, 1.4, -1.1];
        let c_b = from_monomial1(&c_m);
        let mut roots = root_search1(&[c_b], [0.]..[2.], 1e-5);

        assert!(roots.len() == 2);

        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((roots[0][0] - 0.16398614).abs() < 1e-5);
        assert!((roots[1][0] - 1.10874113).abs() < 1e-5);
    }

    #[test]
    pub fn test_root_finding_2() {
        let c_m = [-0.042, 0.55, -1.4, 1.];
        let c_b = from_monomial1(&c_m);
        let mut roots = root_search1(&[c_b], [0.]..[1.], 1e-5);

        dbg!(&roots);
        assert!(roots.len() == 3);

        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert!((roots[0][0] - 0.1).abs() < 1e-5);
        assert!((roots[1][0] - 0.6).abs() < 1e-5);
        assert!((roots[2][0] - 0.7).abs() < 1e-5);
    }

    #[test]
    pub fn test_root_finding_3() {
        let c_m = [0.25, -1., 1.];
        let c_b = from_monomial1(&c_m);
        let roots = root_search1(&[c_b], [0.]..[1.], 1e-5);

        dbg!(&roots);
        //assert!(roots.len() == 1); // TODO combine equal roots

        assert!((roots[0][0] - 0.5).abs() < 1e-5);
    }

    #[test]
    pub fn categorize_region_2d_one_root() {
        let c1 = [[-10., -5., -15.], [3., 0., -5.], [10., 2., 8.]];
        let c2 = [[-8., -1., 10.], [-20., 0., 5.], [-30., 1., 3.]];

        assert!(!contains_no_roots2(&c1));
        assert!(!contains_no_roots2(&c2));
        assert!(exactly_one_root2(&[c1, c2]));
    }

    #[test]
    pub fn categorize_region_2d_no_roots() {
        let c = [[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]];
        // This polynomial is monotonically increasing in top-to-bottom and left-to-right
        // but it doesn't span zero
        // so it should have no roots

        assert!(contains_no_roots2(&c));
    }

    /*
    #[test]
    pub fn categorize_region_2d_unknown() {
        let c = [[-10., -10., -10.], [-10., 10., -10.], [-10., -10., -10.]];
        // This polynomial is not monotonic
        // and spans 0, so we can't tell how many roots it has

        assert!(!contains_no_roots2(&c1));
        assert!(!contains_no_roots2(&c2));
        assert_eq!(categorize_region2(&c), RegionCategory::Unknown);
    }
    */

    #[test]
    pub fn test_2d_root_finding() {
        let f1 = [[-9. / 16., 0., 1.], [0., 0., 0.], [1., 0., 0.]]; // x^2 + y^2 - 9/16 = 0
        let f2 = [[3. / 4., 0., 1.], [-2., 0., 0.], [1., 0., 0.]]; // x^2 - 2x + y^2 + 3/4 = 0
        let c1 = from_monomial2(&f1);
        let c2 = from_monomial2(&f2);
        let roots = root_search2(&[c1, c2], [-1., -1.]..[1., 1.], 0.001);

        dbg!(&roots);
        panic!();
    }
}
