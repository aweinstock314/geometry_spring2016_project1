#[macro_use] extern crate glium;
extern crate cgmath;
extern crate hyper;
//extern crate nalgebera as na;
extern crate rand;
extern crate serde_json;

use cgmath::{Basis3, Matrix3, Matrix4, Point3, Vector3, rad};
use cgmath::{EuclideanVector, Point, Rotation, Rotation3, SquareMatrix};
use glium::glutin::{Event, ElementState, VirtualKeyCode};
use glium::{DisplayBuild, Surface};
use hyper::client::Client;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::f32::consts::PI;
use std::time::Duration;
use rand::{Isaac64Rng,Rng,SeedableRng};
use rand::distributions::{IndependentSample, Range};
//use na::{Inverse,EignenQR};

const EPSILON: f32 = 0.0001;
const TAU: f32 = 2.0 * PI;

const T_MAX: f32= 32.000084;

#[derive(Copy, Clone, Debug)]
struct Vertex3d { position: [f32; 3], vcolor: [f32; 3] }
implement_vertex!(Vertex3d, position, vcolor);

macro_rules! v3 {
    ($x:expr, $y:expr, $z:expr) => { Vertex3d {
        position: [$x, $y, $z],
        vcolor: [0.0, 1.0, 0.0],
    } }
}

fn mapslice<X: Copy, Y, F: Fn(X) -> Y>(f: F, x: [X; 3]) -> [Y; 3] {
    [f(x[0]), f(x[1]), f(x[2])]
}

fn triangle_in_plane(theta: f32) -> [Vertex3d; 3] {
    let f = |x| theta + (x * TAU);
    let g = |x:f32| v3!(x.cos(), x.sin(), 0.0);
    mapslice(|x| g(f(x)), [0.0, 1.0/3.0, 2.0/3.0])
}

fn triangle_normal_to_vector(v: Vector3<f32>, color: [f32; 3]) -> [Vertex3d; 3] {
    // there are infinitely many perpendicular vectors to a given vector
    // so pick an arbitrary one by crossing a vector with one not-parallel to it
    let q = Vector3::unit_x() + Vector3::unit_y() + Vector3::unit_z();
    let u = (v - q).cross(v);
    let r = Basis3::from_axis_angle(v, rad(TAU/3.0));
    let f = |x: Vector3<f32>| Vertex3d { position: x.into(), vcolor: color };
    let tmp = mapslice(f, [u, r.rotate_vector(u), r.invert().rotate_vector(u)]);
    //println!("{:?}", tmp);
    tmp
}

fn append_path_to_buf<S: SurfaceCurve>(v: &mut Vec<Vertex3d>, r: &S, t0: f32, t1: f32, dt: f32, color: [f32; 3]) {
    // "tube" as a surface patch, from "Elemental Differential Geometry, 2nd Edition" 4.2.7
    let sigma = |s, theta: f32| {
        let radius = 0.1;
        let (tangent, normal, binormal) = frenet_frame(r, s);
        r.r(s) + (normal * theta.cos() + binormal * theta.sin())*radius
    };
    let f = |t, i| { Vertex3d {
        position: sigma(t, TAU * (i as f32) / 3.0).into(),
        vcolor: color,
    } };
    let mut t = t0;
    while t < t1 {
        let (a, b, c) = (f(t, 0), f(t, 1), f(t, 2));
        t += dt;
        let (x, y, z) = (f(t, 0), f(t, 1), f(t, 2));

/*
                x
             --/\
          --  /  \
      a--    /____\
      /\    z    -- y
     /  \      --
    /    \   --
   /______\--
  c        b

*/
        // The base
        v.push(a); v.push(b); v.push(c);

        // Each face as quads
        v.push(a); v.push(x); v.push(b);
        v.push(b); v.push(x); v.push(y);

        v.push(a); v.push(x); v.push(c);
        v.push(c); v.push(x); v.push(z);

        v.push(b); v.push(y); v.push(c);
        v.push(c); v.push(y); v.push(z);

    }
}

fn append_surface_to_buf<F: FnMut(f32, f32) -> [[f32; 3]; 2]>(buf: &mut Vec<Vertex3d>, mut sigma: F, u0: f32, v0: f32, u1: f32, v1: f32, delta: f32) {
    let mut f = |u,v| {
        let res = sigma(u,v);
        Vertex3d {
            position: res[0].into(),
            vcolor: res[1],
        }
    };
    let pushquad = |buf: &mut Vec<Vertex3d>, a, b, c, d| {
        /*
        a---b
        | / |
        |/  |
        d---c
        */
        buf.push(a); buf.push(b); buf.push(d);
        buf.push(b); buf.push(c); buf.push(d);
    };
    let mut v = v0;
    while v < v1 {
        let mut u = u0;
        while u < u1 {
            pushquad(buf, f(u,v), f(u1.min(u+delta), v), f(u1.min(u+delta), v1.min(v+delta)), f(u, v1.min(v+delta)));
            u += delta;
        }
        v += delta;
    }
}

trait SurfaceCurve {
    fn r(&self, t: f32) -> Point3<f32>;
    fn r1(&self, t: f32) -> Point3<f32>;
    fn r2(&self, t: f32) -> Point3<f32>;
}

struct Helix;
impl SurfaceCurve for Helix {
    fn r(&self, t: f32) -> Point3<f32> {
        let t2 = t * TAU;
        [t2.cos(), t2.sin(), t].into()
    }
    fn r1(&self, t: f32) -> Point3<f32> {
        unimplemented!();
    }
    fn r2(&self, t: f32) -> Point3<f32> {
        unimplemented!();
    }
}

struct Ellipse { a: f32, b: f32 }
impl SurfaceCurve for Ellipse {
    fn r(&self, t: f32) -> Point3<f32> {
        [self.a*t.cos(), self.b*t.sin(), 0.0].into()
    }
    fn r1(&self, t: f32) -> Point3<f32> {
        [-self.a*t.sin(), self.b*t.cos(), 0.0].into()
    }
    fn r2(&self, t: f32) -> Point3<f32> {
        [-self.a*t.cos(), -self.b*t.sin(), 0.0].into()
    }
}

struct LineSegment {
    p0: Point3<f32>,
    v: Vector3<f32>,
}

impl SurfaceCurve for LineSegment {
    fn r(&self, t: f32) -> Point3<f32> {
        self.p0 + self.v*t
    }
    fn r1(&self, t: f32) -> Point3<f32> {
        Point3::from_vec(self.v)
    }
    fn r2(&self, t: f32) -> Point3<f32> {
        (1.0, 1.0, 1.0).into() // ugly hack to get axis showing up (doesn't work with 0)
    }
}

fn frenet_frame<C: SurfaceCurve>(c: &C, s: f32) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>) {
    let t = c.r1(s).to_vec().normalize();
    let n = c.r2(s).to_vec().normalize();
    let b = t.cross(n);
    (t, n, b)
}

struct StaticCurve {
    t0: f32,
    dt: f32,
    d0: Vec<Point3<f32>>,
    d1: Vec<Point3<f32>>,
    d2: Vec<Point3<f32>>,
}

impl StaticCurve {
    fn sample<T: Clone>(&self, v: &Vec<T>, t: f32) -> Option<T> {
        let i = ((t - self.t0)/self.dt) as usize;
        v.get(i).map(|p| p.clone())
    }
}

impl SurfaceCurve for StaticCurve {
    fn r(&self, t: f32) -> Point3<f32> {
        self.sample(&self.d0, t).expect(&format!("t={} is out of bounds for StaticCurve::r", t))
    }
    fn r1(&self, t: f32) -> Point3<f32> {
        self.sample(&self.d1, t).expect(&format!("t={} is out of bounds for StaticCurve::r1", t))
    }
    fn r2(&self, t: f32) -> Point3<f32> {
        self.sample(&self.d2, t).expect(&format!("t={} is out of bounds for StaticCurve::r2", t))
    }
}

fn static_curve_over_http(host: &str, curve_id: usize, t0: f32, t1: f32, dt: f32) -> Result<StaticCurve, Box<Error>> {
    let client = Client::new();
    let fetch_nth_derivative = |n: usize| -> Result<Vec<Point3<f32>>, Box<Error>> {
        let jsonblob = try!(client.get(&format!("{}/points?ID={}&order={}&tstart={}&tend={}&dt={}", host, curve_id, n, t0, t1, dt)).send());
        let points: Vec<[f32; 3]> = try!(serde_json::de::from_reader(jsonblob));
        let points: Vec<Point3<f32>> = points.into_iter().map(|p| p.into()).collect();
        Ok(points)
    };
    Ok(StaticCurve {
        t0: t0,
        dt: dt,
        d0: try!(fetch_nth_derivative(0)),
        d1: try!(fetch_nth_derivative(1)),
        d2: try!(fetch_nth_derivative(2)),
    })
}

fn bernstein_polynomial(d: usize, i: usize, x: f32) -> f32 {
    fn factorial(n: usize) -> usize {
        let mut tmp = 1;
        for i in 2..n { tmp *= i; }
        tmp
    }
    fn combination(n: usize, k: usize) -> usize {
        factorial(n) / (factorial(n-k) * factorial(k))
    }
    // B_i^d(t) = \choose{d}{i}(1-t)^{d-i}t^i
    (combination(d, i) as f32) * ((1.0-x).powi((d-i) as i32)) * x.powi(i as i32)
}

struct HeightMapSurface {
    width: usize,
    height: usize,
    spacewidth: f32,
    spaceheight: f32,
    points: HashMap<(usize, usize), f32>,
}

#[derive(Default)]
struct DerivativePoint {
    u: f32, v: f32,
    sigma: f32,
    sigma_u: f32, sigma_v: f32,
    sigma_uu: f32, sigma_uv: f32, sigma_vv: f32,
}

/*impl DerivativePoint {
    fn get_vec(x: f32) -> na::Vector3<f32> {
        na::Vector3::new(self.u, x, self.v)
    }
    fn gaussian_curvature(&self) -> f32 {
        //let e = get_vec(self.sigma_u) * 
        //let w = na::Matrix2(
        0.0
    }
}*/

impl HeightMapSurface {
    fn new_noise(width: usize, height: usize, spacewidth: f32, spaceheight: f32) -> HeightMapSurface {
        let mut rng = Isaac64Rng::from_seed(&[42]);

        let mut points = HashMap::new();
        for i in 0..width {
            for j in 0..height {
                points.insert((i,j), Range::new(0.0, 5.0).ind_sample(&mut rng));
            }
        }

        HeightMapSurface {
            width: width, height: height,
            spacewidth: spacewidth, spaceheight: spaceheight,
            points: points,
        }
    }
    fn extended_discrete_lookup(&self, i: usize, j: usize) -> f32 {
        *self.points.get(&(i,j)).unwrap_or(&0.0)
    }
    fn sample(&self, u: f32, v: f32) -> DerivativePoint {
        if u < 0.0 || u >= self.spacewidth || v < 0.0 || v >= self.spaceheight {
            //return [u, 0.0, v];
            return DerivativePoint {
                u: u, v: v, sigma: 0.0,
                .. Default::default()
            };
        }
        // let linear_interpolate = |alpha, p0, p1| { (1.0 - alpha) * p0 + alpha * p1 };
        let i = ((u / self.spacewidth) * self.width as f32) as usize;
        let j = ((v / self.spaceheight) * self.height as f32) as usize;
        let degree = 3;
        // B(u,v) = \Sigma_{i=0}^r\Sigma_{j=0}^s(b_{i,j}B_i^r(u)B_j^s(v)
        let uprime = u % (self.spacewidth / self.width as f32);
        let vprime = v % (self.spaceheight / self.height as f32);
        //println!("({}, {}) ({}, {}) ({}, {})", i, j, u, v, uprime, vprime);
        let mut result = DerivativePoint {
            u: u, v: v,
            sigma: 0.0,
            sigma_u: 0.0, sigma_v: 0.0,
            sigma_uu: 0.0, sigma_uv: 0.0, sigma_vv: 0.0,
        };
        for di in 0..(degree+1) {
            for dj in 0..(degree+1) {
                let iprime = (i+di).wrapping_sub(degree/2);
                let jprime = (j+dj).wrapping_sub(degree/2);
                //tmp += self.extended_discrete_lookup(iprime, jprime) * bernstein_polynomial(degree, di, uprime) * bernstein_polynomial(degree, dj, vprime);
                let b = self.extended_discrete_lookup(iprime, jprime);
                let (u,v) = (uprime, vprime);
                result.sigma += b * bernstein_polynomial(degree, di, u) * bernstein_polynomial(degree, dj, v);
            }
        }
        let b00 = self.extended_discrete_lookup(i.wrapping_sub(1), j.wrapping_sub(1));
        let b01 = self.extended_discrete_lookup(i.wrapping_sub(1), j                );
        let b02 = self.extended_discrete_lookup(i.wrapping_sub(1), j.wrapping_add(1));

        let b10 = self.extended_discrete_lookup(i                , j.wrapping_sub(1));
        let b11 = self.extended_discrete_lookup(i                , j                );
        let b12 = self.extended_discrete_lookup(i                , j.wrapping_add(1));

        let b20 = self.extended_discrete_lookup(i.wrapping_add(1), j.wrapping_sub(1));
        let b21 = self.extended_discrete_lookup(i.wrapping_add(1), j                );
        let b22 = self.extended_discrete_lookup(i.wrapping_add(1), j.wrapping_add(1));

        { 
            let (u,v) = (uprime, vprime);
// BEGIN code generated by bernstein_polynomials.py
//result.sigma = b00*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b01*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b02*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b10*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b11*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b12*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b20*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b21*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b22*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3));

result.sigma_u = b00*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b01*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b02*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b10*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b11*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b12*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b20*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b21*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3)) + b22*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(v.powi(3) + v.powi(2)*(-3.0*v + 3.0) + 3.0*v*(-v + 1.0).powi(2) + (-v + 1.0).powi(3));
result.sigma_v = b00*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0))*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3)) + b01*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0))*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3)) + b02*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0))*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3)) + b10*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0))*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3)) + b11*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0))*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3)) + b12*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0))*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3)) + b20*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0))*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3)) + b21*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0))*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3)) + b22*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0))*(u.powi(3) + u.powi(2)*(-3.0*u + 3.0) + 3.0*u*(-u + 1.0).powi(2) + (-u + 1.0).powi(3));

result.sigma_uu = 0.0;
result.sigma_uv = b00*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0)) + b01*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0)) + b02*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0)) + b10*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0)) + b11*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0)) + b12*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0)) + b20*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0)) + b21*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0)) + b22*(2.0*u*(-3.0*u + 3.0) + 3.0*u*(2.0*u - 2.0))*(2.0*v*(-3.0*v + 3.0) + 3.0*v*(2.0*v - 2.0));
result.sigma_vv = 0.0;
// END code generated by bernstein_polynomials.py
        }
        //[u, self.points[&(i,j)], v]
        //[u, tmp, v]
        result
    }
}

fn main() {
    let display = glium::glutin::WindowBuilder::new()
        .with_dimensions(640, 480)
        .with_title("Fun with Frenet Frames".to_string())
        .with_depth_buffer(24)
        .build_glium()
        .expect("Failed to initialize window.");

    let draw_params = glium::DrawParameters {
        depth: glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLess,
            write: true,
            .. Default::default()
        },
        .. Default::default()
    };

    let mut t = 0.0;
    let mut delta = 0.001;

    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

    let vertex_shader_src = r#"
        #version 130
        in vec3 position;
        in vec3 vcolor;
        out vec3 fcolor;
        uniform mat4 matrix;
        void main() {
            fcolor = vcolor;
            gl_Position = matrix * vec4(position, 1.0);
        }
    "#;
    let fragment_shader_src = r#"
        #version 130
        in vec3 fcolor;
        out vec4 color;
        void main() {
            color = vec4(fcolor, 1.0);
        }
    "#;
    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();
    let (mut x, mut y, mut z) = (-0.5, -0.5, -1.0);
    let (mut theta, mut phi) = (0.0, 0.0);

    let mut held_keys = HashSet::new();
    let movement_delta = 0.075;
    let rotation_delta = 0.02;

    let mut mutable_buffer = vec![];
    //mutable_buffer.extend_from_slice(&triangle_normal_to_vector(Vector3::unit_x(), [1.0, 0.0, 0.0]));
    //mutable_buffer.extend_from_slice(&triangle_normal_to_vector(Vector3::unit_y(), [0.0, 1.0, 0.0]));
    //mutable_buffer.extend_from_slice(&triangle_normal_to_vector(Vector3::unit_z(), [0.0, 0.0, 1.0]));

    // from "Elemental Differential Geometry, 2nd Edition" 4.1.4
    /*let sphere_patch = |theta: f32, phi: f32| {
        let radius = 3.0;
        let (x,y,z) = (radius*theta.cos()*phi.cos(), radius*theta.cos()*phi.sin(), radius*theta.sin());
        //println!("{}, {}, {}", x, y, z);
        let color = [y.abs() / radius, 0.5, 0.0];
        //let color = [1.0, 0.0, 0.0];
        [[x,y,z],color]
    };
    append_surface_to_buf(&mut mutable_buffer, sphere_patch, -TAU/4.0, 0.0, TAU/4.0, TAU, 0.2);*/

    {
        let size = 10;
        let spacesize = 10.0;
        let fringe = 0.3;
        let delta = 0.1;
        let translation = 2.5;
        let surface = HeightMapSurface::new_noise(size, size, spacesize, spacesize);
        let mut color_rng = rand::thread_rng();
        let surface_patch = |u,v| {
            let pt = surface.sample(u, v);
            let r = Range::new(0.0, 1.0);
            let col = [r.ind_sample(&mut color_rng), r.ind_sample(&mut color_rng), r.ind_sample(&mut color_rng)];
            //let col = [u % (spacesize / size as f32), v % (spacesize / size as f32), 0.5];
            let translated_pt = [pt.u + translation, pt.sigma, pt.v + translation];
            [translated_pt, col]
        };
        append_surface_to_buf(&mut mutable_buffer, surface_patch, -fringe, -fringe, spacesize+fringe, spacesize+fringe, delta);
    }

    // hack due to borrowck issues using a lambda
    let v = (1.0, 0.0, 0.0).into();
    append_path_to_buf(&mut mutable_buffer, &LineSegment { p0: (0.0, 0.0, 0.0).into(), v: v }, 0.0, 1.0, 1.0, v.into());
    let v = (0.0, 1.0, 0.0).into();
    append_path_to_buf(&mut mutable_buffer, &LineSegment { p0: (0.0, 0.0, 0.0).into(), v: v }, 0.0, 1.0, 1.0, v.into());
    let v = (0.0, 0.0, 1.0).into();
    append_path_to_buf(&mut mutable_buffer, &LineSegment { p0: (0.0, 0.0, 0.0).into(), v: v }, 0.0, 1.0, 1.0, v.into());

    //append_path_to_buf(&mut mutable_buffer, &Ellipse { a: 2.0, b: 3.0 }, 0.0, TAU, TAU/20.0, [1.0, 1.0, 1.0]);
    //append_path_to_buf(&mut mutable_buffer, &Ellipse { a: 5.0, b: 4.0 }, 0.0, TAU, TAU/20.0, [1.0, 1.0, 0.0]);
    match static_curve_over_http("http://localhost:1337", 0, 0.0, T_MAX, 0.1) {
        Ok(curve) => append_path_to_buf(&mut mutable_buffer, &curve, 0.0, T_MAX, 0.1, [1.0, 0.0, 1.0]),
        Err(e) => println!("Failed to connect to the python: {}", e),
    }
    let vertex_buffer = glium::VertexBuffer::new(&display, &*mutable_buffer).unwrap();

    'outer: loop {
        //println!("{}, {}, {}", x, y, z);
        //println!("{}, {}", theta, phi);
        for e in display.poll_events() {
            //println!("Got an event: {:?}", e);
            match e {
                Event::Closed => break 'outer,
                Event::KeyboardInput(ElementState::Pressed, _, Some(c)) => { held_keys.insert(c); },
                Event::KeyboardInput(ElementState::Released, _, Some(c)) => { held_keys.remove(&c); },
                _ => (),
            }
        }
        fn app_polar(x: &mut f32, z: &mut f32, m: f32, r: f32) {
            *x += m * r.cos();
            *z += m * r.sin();
        }
        for key in &held_keys {
            match *key {
                VirtualKeyCode::A => app_polar(&mut x, &mut z,  movement_delta, phi),
                VirtualKeyCode::D => app_polar(&mut x, &mut z, -movement_delta, phi),
                VirtualKeyCode::W => app_polar(&mut x, &mut z,  movement_delta, phi + TAU/4.0),
                VirtualKeyCode::S => app_polar(&mut x, &mut z, -movement_delta, phi + TAU/4.0),

                VirtualKeyCode::Q => y += movement_delta,
                VirtualKeyCode::E => y -= movement_delta,

                VirtualKeyCode::I => theta -= rotation_delta,
                VirtualKeyCode::K => theta += rotation_delta,
                VirtualKeyCode::J => phi -= rotation_delta,
                VirtualKeyCode::L => phi += rotation_delta,
                _ => (),
            }
            theta = theta.min(TAU/4.0).max(-TAU/4.0);
        }
        t += delta;
        if t < EPSILON || (t - 1.0).abs() < EPSILON {
            delta = -delta;
        }
        let mut frame = display.draw();
        frame.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);
        let perspective = Matrix4::<f32>::from({
            let (width, height) = frame.get_dimensions();
            cgmath::PerspectiveFov {
                fovy: rad(TAU / 6.0),
                aspect: width as f32 / height as f32,
                near:  0.1,
                far: 1024.0,
            }
        });
        let translation: Matrix4<f32> = Matrix4::from_translation((x, y, z).into());
        let rotation: Matrix4<f32> = (Matrix3::from_angle_x(rad(theta)) * Matrix3::from_angle_y(rad(phi))).into();
        let matrix: [[f32; 4]; 4] = (perspective * rotation * translation).into();
        let uniforms = uniform! { matrix: matrix };
        frame.draw(&vertex_buffer, &indices, &program, &uniforms, &draw_params).unwrap();
        frame.finish().unwrap();

        std::thread::sleep(Duration::from_millis(1));
    }
}
