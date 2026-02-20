// Triplanar mapping utilities for PlumeSplat
//
// Based on techniques from:
// - Ben Golus: https://bgolus.medium.com/normal-mapping-for-a-triplanar-shader-10bf39dca05a
// - Inigo Quilez: https://iquilezles.org/articles/biplanar/

#define_import_path plumesplat::triplanar

// Compute triplanar blend weights from world normal
fn triplanar_weights(world_normal: vec3<f32>, sharpness: f32) -> vec3<f32> {
    // Take absolute value of normal for blend weights
    var weights = abs(world_normal);

    // Apply sharpness (power) to create sharper transitions
    weights = pow(weights, vec3<f32>(sharpness));

    // Normalize so weights sum to 1
    let sum = weights.x + weights.y + weights.z;
    return weights / sum;
}

// Sample a 2D array texture using triplanar projection
fn sample_triplanar(
    tex: texture_2d_array<f32>,
    tex_sampler: sampler,
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec4<f32> {
    let weights = triplanar_weights(world_normal, sharpness);
    let scaled_pos = world_pos * uv_scale;

    // Sample from each projection plane
    let sample_x = textureSample(tex, tex_sampler, scaled_pos.zy, layer);
    let sample_y = textureSample(tex, tex_sampler, scaled_pos.xz, layer);
    let sample_z = textureSample(tex, tex_sampler, scaled_pos.xy, layer);

    // Blend based on surface orientation
    return sample_x * weights.x + sample_y * weights.y + sample_z * weights.z;
}

// Sample a normal map using triplanar projection with proper tangent space handling
fn sample_triplanar_normal(
    tex: texture_2d_array<f32>,
    tex_sampler: sampler,
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec3<f32> {
    let weights = triplanar_weights(world_normal, sharpness);
    let scaled_pos = world_pos * uv_scale;

    // Sample normal maps from each plane
    let normal_x = textureSample(tex, tex_sampler, scaled_pos.zy, layer).xyz * 2.0 - 1.0;
    let normal_y = textureSample(tex, tex_sampler, scaled_pos.xz, layer).xyz * 2.0 - 1.0;
    let normal_z = textureSample(tex, tex_sampler, scaled_pos.xy, layer).xyz * 2.0 - 1.0;

    // Swizzle to world space for each projection axis
    // This uses the "Whiteout" blending method
    let sign_x = sign(world_normal.x);
    let sign_y = sign(world_normal.y);
    let sign_z = sign(world_normal.z);

    let world_normal_x = vec3<f32>(normal_x.z * sign_x, normal_x.y, normal_x.x);
    let world_normal_y = vec3<f32>(normal_y.x, normal_y.z * sign_y, normal_y.y);
    let world_normal_z = vec3<f32>(normal_z.x, normal_z.y, normal_z.z * sign_z);

    // Blend normals
    var blended = world_normal_x * weights.x + world_normal_y * weights.y + world_normal_z * weights.z;

    return normalize(blended);
}

// Biplanar mapping (faster than triplanar, uses only 2 texture samples)
fn biplanar_weights(world_normal: vec3<f32>) -> vec2<f32> {
    let n = abs(world_normal);

    // Find the two dominant axes
    if n.x > n.y && n.x > n.z {
        // X is dominant - skip X plane, use Y and Z
        let w = vec2<f32>(n.y, n.z);
        return w / (w.x + w.y);
    } else if n.y > n.z {
        // Y is dominant - skip Y plane, use X and Z
        let w = vec2<f32>(n.x, n.z);
        return w / (w.x + w.y);
    } else {
        // Z is dominant - skip Z plane, use X and Y
        let w = vec2<f32>(n.x, n.y);
        return w / (w.x + w.y);
    }
}
