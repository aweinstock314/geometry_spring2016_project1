#[macro_use] extern crate glium;
extern crate cgmath;
extern crate hyper;
extern crate serde_json;

use cgmath::{Basis3, Matrix3, Matrix4, Point3, Vector3, rad};
use cgmath::{EuclideanVector, Point, Rotation, Rotation3, SquareMatrix};
use glium::glutin::{Event, ElementState, VirtualKeyCode};
use glium::{DisplayBuild, Surface};
use hyper::client::Client;
use std::collections::HashSet;
use std::error::Error;
use std::f32::consts::PI;
use std::time::Duration;

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

fn main() {
    let display = glium::glutin::WindowBuilder::new()
        .with_dimensions(640, 480)
        .with_title("Fun with Frenet Frames".to_string())
        .build_glium()
        .expect("Failed to initialize window.");

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
    let movement_delta = 0.05;
    let rotation_delta = 0.01;

    let mut mutable_buffer = vec![];
    //mutable_buffer.extend_from_slice(&triangle_normal_to_vector(Vector3::unit_x(), [1.0, 0.0, 0.0]));
    //mutable_buffer.extend_from_slice(&triangle_normal_to_vector(Vector3::unit_y(), [0.0, 1.0, 0.0]));
    //mutable_buffer.extend_from_slice(&triangle_normal_to_vector(Vector3::unit_z(), [0.0, 0.0, 1.0]));

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
        frame.clear_color(0.0, 0.0, 0.0, 1.0);
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
        frame.draw(&vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();
        frame.finish().unwrap();

        std::thread::sleep(Duration::from_millis(1));
    }
}
