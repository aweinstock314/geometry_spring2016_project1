extern crate glium;

fn main() {
    use glium::DisplayBuild;

    let display = glium::glutin::WindowBuilder::new()
        .with_dimensions(640, 480)
        .with_title("Fun with Frenet Frames".to_string())
        .build_glium()
        .expect("Failed to initialize window.");

    for e in display.wait_events() {
        println!("Got an event: {:?}", e);
    }
}
