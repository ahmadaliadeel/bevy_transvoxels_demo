// PlumeSplat - Advanced terrain splatting shader for Bevy
//
// Supports up to 256 materials via texture arrays with per-vertex material selection.
// Uses triplanar mapping for UV-less texturing.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    pbr_types::PbrInput,
    pbr_types::pbr_input_new,
    pbr_functions::apply_pbr_lighting,
    pbr_functions::main_pass_post_lighting_processing,
}
#import bevy_render::view::View

// Bind group for view (group 0)
@group(0) @binding(0) var<uniform> view: View;

// Material bind group (uses MATERIAL_BIND_GROUP)
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var albedo_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var albedo_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var normal_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var normal_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var arm_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var arm_sampler: sampler;

struct PlumeSplatUniforms {
    uv_scale: f32,
    triplanar_sharpness: f32,
    height_blend_sharpness: f32,
    base_roughness: f32,
    base_metallic: f32,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var<uniform> uniforms: PlumeSplatUniforms;

// Vertex input
struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material_indices: u32,
    @location(4) material_weights: u32,
}

// Vertex output / Fragment input
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) material_indices: u32,
    @location(4) material_weights: vec4<f32>,
}

// Unpack 4 u8 values from a u32
fn unpack_u8x4(packed: u32) -> vec4<u32> {
    return vec4<u32>(
        packed & 0xFFu,
        (packed >> 8u) & 0xFFu,
        (packed >> 16u) & 0xFFu,
        (packed >> 24u) & 0xFFu
    );
}

// Compute triplanar blend weights from world normal
fn triplanar_weights(world_normal: vec3<f32>, sharpness: f32) -> vec3<f32> {
    var weights = abs(world_normal);
    weights = pow(weights, vec3<f32>(sharpness));
    let sum = weights.x + weights.y + weights.z;
    return weights / max(sum, 0.001);
}

// Sample texture array using triplanar projection
// Properly handles UV direction based on normal sign to avoid stretching
fn sample_triplanar_albedo(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec4<f32> {
    let weights = triplanar_weights(world_normal, sharpness);
    let scaled_pos = world_pos * uv_scale;

    // Flip UVs based on normal direction to prevent mirroring/stretching
    let flip = sign(world_normal);

    // X-axis projection (YZ plane) - flip Z based on normal.x direction
    let uv_x = vec2<f32>(scaled_pos.z * flip.x, scaled_pos.y);
    // Y-axis projection (XZ plane) - flip X based on normal.y direction
    let uv_y = vec2<f32>(scaled_pos.x * flip.y, scaled_pos.z);
    // Z-axis projection (XY plane) - flip X based on normal.z direction
    let uv_z = vec2<f32>(scaled_pos.x * -flip.z, scaled_pos.y);

    let sample_x = textureSample(albedo_array, albedo_sampler, uv_x, layer);
    let sample_y = textureSample(albedo_array, albedo_sampler, uv_y, layer);
    let sample_z = textureSample(albedo_array, albedo_sampler, uv_z, layer);

    return sample_x * weights.x + sample_y * weights.y + sample_z * weights.z;
}

// Sample normal map with triplanar and proper tangent space
fn sample_triplanar_normal(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec3<f32> {
    let weights = triplanar_weights(world_normal, sharpness);
    let scaled_pos = world_pos * uv_scale;

    // Flip UVs based on normal direction
    let flip = sign(world_normal);
    let uv_x = vec2<f32>(scaled_pos.z * flip.x, scaled_pos.y);
    let uv_y = vec2<f32>(scaled_pos.x * flip.y, scaled_pos.z);
    let uv_z = vec2<f32>(scaled_pos.x * -flip.z, scaled_pos.y);

    // Sample normal maps
    let raw_x = textureSample(normal_array, normal_sampler, uv_x, layer).xyz * 2.0 - 1.0;
    let raw_y = textureSample(normal_array, normal_sampler, uv_y, layer).xyz * 2.0 - 1.0;
    let raw_z = textureSample(normal_array, normal_sampler, uv_z, layer).xyz * 2.0 - 1.0;

    // Convert to world space for each projection (UDN blending)
    let normal_x = vec3<f32>(0.0, raw_x.y, raw_x.x * flip.x) + vec3<f32>(flip.x, 0.0, 0.0);
    let normal_y = vec3<f32>(raw_y.x, 0.0, raw_y.y * flip.y) + vec3<f32>(0.0, flip.y, 0.0);
    let normal_z = vec3<f32>(raw_z.x * -flip.z, raw_z.y, 0.0) + vec3<f32>(0.0, 0.0, flip.z);

    return normalize(normal_x * weights.x + normal_y * weights.y + normal_z * weights.z);
}

// Sample ARM texture with triplanar
fn sample_triplanar_arm(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec3<f32> {
    let weights = triplanar_weights(world_normal, sharpness);
    let scaled_pos = world_pos * uv_scale;

    // Flip UVs based on normal direction
    let flip = sign(world_normal);
    let uv_x = vec2<f32>(scaled_pos.z * flip.x, scaled_pos.y);
    let uv_y = vec2<f32>(scaled_pos.x * flip.y, scaled_pos.z);
    let uv_z = vec2<f32>(scaled_pos.x * -flip.z, scaled_pos.y);

    let sample_x = textureSample(arm_array, arm_sampler, uv_x, layer).rgb;
    let sample_y = textureSample(arm_array, arm_sampler, uv_y, layer).rgb;
    let sample_z = textureSample(arm_array, arm_sampler, uv_z, layer).rgb;

    return sample_x * weights.x + sample_y * weights.y + sample_z * weights.z;
}

// Height-based blending for better transitions
fn height_blend_weights(heights: vec4<f32>, weights: vec4<f32>, sharpness: f32) -> vec4<f32> {
    if sharpness <= 0.0 {
        // Linear blend fallback
        let sum = weights.x + weights.y + weights.z + weights.w;
        if sum > 0.0 {
            return weights / sum;
        }
        return vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }

    let height_weighted = heights + weights;
    let max_h = max(max(height_weighted.x, height_weighted.y),
                    max(height_weighted.z, height_weighted.w));

    var blend = max(height_weighted - max_h + sharpness, vec4<f32>(0.0));
    let sum = blend.x + blend.y + blend.z + blend.w;

    if sum > 0.0 {
        return blend / sum;
    }
    return weights;
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Transform position to world space
    let world_pos = mesh_functions::mesh_position_local_to_world(
        mesh_functions::get_world_from_local(input.instance_index),
        vec4<f32>(input.position, 1.0)
    );

    out.world_position = world_pos.xyz;
    out.clip_position = position_world_to_clip(world_pos.xyz);

    // Transform normal to world space
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        input.normal,
        input.instance_index
    );

    out.uv = input.uv;
    out.material_indices = input.material_indices;

    // Unpack weights and normalize
    let raw_weights = unpack_u8x4(input.material_weights);
    out.material_weights = vec4<f32>(raw_weights) / 255.0;

    return out;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    // Unpack material indices
    let indices = unpack_u8x4(input.material_indices);
    let idx0 = i32(indices.x);
    let idx1 = i32(indices.y);
    let idx2 = i32(indices.z);
    let idx3 = i32(indices.w);

    let world_pos = input.world_position;
    let world_normal = normalize(input.world_normal);
    let uv_scale = uniforms.uv_scale;
    let sharpness = uniforms.triplanar_sharpness;

    // Sample albedo for each material layer
    let albedo0 = sample_triplanar_albedo(world_pos, world_normal, idx0, uv_scale, sharpness);
    let albedo1 = sample_triplanar_albedo(world_pos, world_normal, idx1, uv_scale, sharpness);
    let albedo2 = sample_triplanar_albedo(world_pos, world_normal, idx2, uv_scale, sharpness);
    let albedo3 = sample_triplanar_albedo(world_pos, world_normal, idx3, uv_scale, sharpness);

    // Compute final blend weights (optionally using height blending)
    var weights = input.material_weights;
    if uniforms.height_blend_sharpness > 0.0 {
        // Use albedo alpha as height for blending
        let heights = vec4<f32>(albedo0.a, albedo1.a, albedo2.a, albedo3.a);
        weights = height_blend_weights(heights, weights, uniforms.height_blend_sharpness);
    } else {
        // Normalize weights
        let sum = weights.x + weights.y + weights.z + weights.w;
        if sum > 0.0 {
            weights = weights / sum;
        }
    }

    // Blend albedo
    let final_albedo = albedo0 * weights.x + albedo1 * weights.y +
                       albedo2 * weights.z + albedo3 * weights.w;

    // Sample and blend normals
    let normal0 = sample_triplanar_normal(world_pos, world_normal, idx0, uv_scale, sharpness);
    let normal1 = sample_triplanar_normal(world_pos, world_normal, idx1, uv_scale, sharpness);
    let normal2 = sample_triplanar_normal(world_pos, world_normal, idx2, uv_scale, sharpness);
    let normal3 = sample_triplanar_normal(world_pos, world_normal, idx3, uv_scale, sharpness);
    let final_normal = normalize(normal0 * weights.x + normal1 * weights.y +
                                  normal2 * weights.z + normal3 * weights.w);

    // Sample and blend ARM
    let arm0 = sample_triplanar_arm(world_pos, world_normal, idx0, uv_scale, sharpness);
    let arm1 = sample_triplanar_arm(world_pos, world_normal, idx1, uv_scale, sharpness);
    let arm2 = sample_triplanar_arm(world_pos, world_normal, idx2, uv_scale, sharpness);
    let arm3 = sample_triplanar_arm(world_pos, world_normal, idx3, uv_scale, sharpness);
    let final_arm = arm0 * weights.x + arm1 * weights.y + arm2 * weights.z + arm3 * weights.w;

    // For now, output simple lit result with better visibility
    // TODO: Full PBR integration with Bevy's lighting system
    //let ambient = 0.3;
    //let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.4));
    //let ndotl = max(dot(final_normal, light_dir), 0.0);
    //let diffuse = ndotl * 0.7;

    //let lighting = (ambient + diffuse) * final_albedo.rgb;

    //return vec4<f32>(lighting, 1.0);
}
