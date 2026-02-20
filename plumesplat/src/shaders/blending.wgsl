// Material blending utilities for PlumeSplat
//
// Supports blending up to 4 materials per vertex using packed indices and weights.

#define_import_path plumesplat::blending

// Unpack 4 u8 values from a u32
fn unpack_u8x4(packed: u32) -> vec4<u32> {
    return vec4<u32>(
        packed & 0xFFu,
        (packed >> 8u) & 0xFFu,
        (packed >> 16u) & 0xFFu,
        (packed >> 24u) & 0xFFu
    );
}

// Convert packed weights to normalized floats
fn unpack_weights(packed: u32) -> vec4<f32> {
    let raw = unpack_u8x4(packed);
    return vec4<f32>(raw) / 255.0;
}

// Get material indices from packed value
fn unpack_indices(packed: u32) -> vec4<i32> {
    let raw = unpack_u8x4(packed);
    return vec4<i32>(raw);
}

// Height-based blending for more realistic material transitions
// Based on technique from: https://www.gamedeveloper.com/programming/advanced-terrain-texture-splatting
fn height_blend(heights: vec4<f32>, weights: vec4<f32>, sharpness: f32) -> vec4<f32> {
    // Add height to weight
    let height_weighted = heights + weights;

    // Find the maximum
    let max_height = max(max(height_weighted.x, height_weighted.y),
                         max(height_weighted.z, height_weighted.w));

    // Create blend mask
    let blend = max(height_weighted - max_height + sharpness, vec4<f32>(0.0));

    // Normalize
    let sum = blend.x + blend.y + blend.z + blend.w;
    if sum > 0.0 {
        return blend / sum;
    }
    return weights;
}

// Simple linear blend (default when height blending disabled)
fn linear_blend(weights: vec4<f32>) -> vec4<f32> {
    let sum = weights.x + weights.y + weights.z + weights.w;
    if sum > 0.0 {
        return weights / sum;
    }
    return vec4<f32>(1.0, 0.0, 0.0, 0.0);
}

// Blend 4 color samples based on weights
fn blend_colors(
    color0: vec4<f32>,
    color1: vec4<f32>,
    color2: vec4<f32>,
    color3: vec4<f32>,
    weights: vec4<f32>
) -> vec4<f32> {
    return color0 * weights.x + color1 * weights.y + color2 * weights.z + color3 * weights.w;
}

// Blend 4 normal samples based on weights
fn blend_normals(
    normal0: vec3<f32>,
    normal1: vec3<f32>,
    normal2: vec3<f32>,
    normal3: vec3<f32>,
    weights: vec4<f32>
) -> vec3<f32> {
    let blended = normal0 * weights.x + normal1 * weights.y + normal2 * weights.z + normal3 * weights.w;
    return normalize(blended);
}

// Blend ARM (AO/Roughness/Metallic) values
fn blend_arm(
    arm0: vec3<f32>,
    arm1: vec3<f32>,
    arm2: vec3<f32>,
    arm3: vec3<f32>,
    weights: vec4<f32>
) -> vec3<f32> {
    return arm0 * weights.x + arm1 * weights.y + arm2 * weights.z + arm3 * weights.w;
}
