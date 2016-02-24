#[macro_use]
extern crate glium;

use glium::{DisplayBuild, Surface};
use glium::glutin::Event;
use std::time::Duration;
use std::f32::consts::PI;

const EPSILON: f32 = 0.0001;
const TAU: f32 = 2.0 * PI;

#[derive(Copy, Clone)]
struct Vertex3d { position: [f32; 3] }
implement_vertex!(Vertex3d, position);

macro_rules! v3 {
    ($x:expr, $y:expr, $z:expr) => { Vertex3d { position: [$x, $y, $z] } }
}

fn mapslice<X: Copy, Y, F: Fn(X) -> Y>(f: F, x: [X; 3]) -> [Y; 3] {
    [f(x[0]), f(x[1]), f(x[2])]
}

fn triangle(theta: f32) -> [Vertex3d; 3] {
    let f = |x| theta + (x * TAU);
    let g = |x:f32| v3!(x.cos(), x.sin(), 0.0);
    mapslice(|x| g(f(x)), [0.0, 1.0/3.0, 2.0/3.0])
}

trait SurfaceCurve {
    fn evaluate(&self, t: f32) -> (f32, f32, f32);
    // TODO: derivative?
}

struct Helix;

impl SurfaceCurve for Helix {
    fn evaluate(&self, t: f32) -> (f32, f32, f32) {
        let t2 = t * TAU;
        (t2.cos(), t2.sin(), t)
    }
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

        void main() {
            gl_Position = vec4(position, 1.0);
        }
    "#;
    let fragment_shader_src = r#"
        #version 130

        out vec4 color;

        void main() {
            color = vec4(0.0, 1.0, 1.0, 1.0);
        }
    "#;
    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    'outer: loop {
        for e in display.poll_events() {
            println!("Got an event: {:?}", e);
            match e {
                Event::Closed => break 'outer,
                _ => (),
            }
        }
        t += delta;
        if t < EPSILON || (t - 1.0).abs() < EPSILON {
            delta = -delta;
        }
        /*let (x, y, z) = Helix.evaluate(t);
        let f = |x| { 0.5 * x + 0.5 };
        let (r, g, b) = (f(x), f(y), z);*/
        let (r, g, b) = (t, t, t);
        println!("{}, {}, {}", r, g, b);
        let mut frame = display.draw();
        frame.clear_color(r, g, b, 1.0);
        let vertex_buffer = glium::VertexBuffer::new(&display, &triangle(t*TAU)).unwrap();
        frame.draw(&vertex_buffer, &indices, &program, &glium::uniforms::EmptyUniforms, &Default::default()).unwrap();
        frame.finish().unwrap();

        std::thread::sleep(Duration::from_millis(1));
    }
}
