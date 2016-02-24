extern crate glium;

use glium::{DisplayBuild, Surface};
use glium::glutin::Event;
use std::time::Duration;
use std::f32::consts::PI;

const EPSILON : f32 = 0.0001;
const TAU : f32 = 2.0 * PI;



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
    let mut delta = 0.01;

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
        frame.finish().unwrap();

        std::thread::sleep(Duration::from_millis(10));
    }
}
