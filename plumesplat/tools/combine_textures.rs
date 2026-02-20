//! Tool to combine multiple textures into a vertically stacked texture array image.
//! Run with: cargo run --bin combine_textures

use image::{GenericImageView, ImageBuffer, Rgba, RgbaImage};
use std::path::Path;

fn main() {
    let texture_paths = [
        "/tmp/grass/Grass001_1K-PNG_Color.png",
        "/tmp/dirt/Ground054_1K-PNG_Color.png",
        "/tmp/rock/Rock030_1K-PNG_Color.png",
        "/tmp/snow/Snow006_1K-PNG_Color.png",
    ];

    let output_path = "assets/textures/terrain_albedo_array.png";

    // Load first image to get dimensions
    let first = image::open(texture_paths[0]).expect("Failed to load first texture");
    let (width, height) = first.dimensions();
    println!("Texture dimensions: {}x{}", width, height);

    // Create output image (stacked vertically)
    let total_height = height * texture_paths.len() as u32;
    let mut output: RgbaImage = ImageBuffer::new(width, total_height);

    for (i, path) in texture_paths.iter().enumerate() {
        println!("Loading: {}", path);
        let img = image::open(path).expect(&format!("Failed to load {}", path));
        let img = img.resize_exact(width, height, image::imageops::FilterType::Lanczos3);

        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                output.put_pixel(x, i as u32 * height + y, pixel);
            }
        }
    }

    // Create output directory if needed
    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    output.save(output_path).expect("Failed to save output");
    println!("Saved combined texture array to: {}", output_path);
    println!(
        "Array has {} layers of {}x{}",
        texture_paths.len(),
        width,
        height
    );
}
