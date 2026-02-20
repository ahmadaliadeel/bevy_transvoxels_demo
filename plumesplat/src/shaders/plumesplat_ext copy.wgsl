// PlumeSplat Material Extension Shader
// Extends StandardMaterial to add texture array splatting with triplanar mapping

#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput, FragmentOutput},
    view_transformations::position_world_to_clip,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    mesh_view_bindings::view,
}

// Extension bindings - using bindless/shader_defs compatible syntax
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var albedo_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var albedo_array_sampler: sampler;

struct PlumeSplatSettings {
    uv_scale: f32,
    triplanar_sharpness: f32,
    height_blend_sharpness: f32,
    blend_offset: f32,
    blend_exponent: f32,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> settings: PlumeSplatSettings;

// Normal map array (optional - bound when available)
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var normal_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var normal_array_sampler: sampler;

// PBR packed texture array: R=Metallic, G=Roughness, B=AO, A=Height (optional)
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var pbr_array: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var pbr_array_sampler: sampler;

// Extended vertex output with our custom material data
struct PlumeSplatVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Custom attributes for material splatting
    @location(10) @interpolate(flat) material_indices: u32,
    // Weights unpacked to vec4<f32> for proper interpolation
    @location(11) material_weights: vec4<f32>,
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

// Hash function for stochastic tiling
fn hash2(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

// Stochastic tiling - samples texture at 3 random offsets and blends based on proximity
// This breaks up visible tiling patterns
fn sample_stochastic(
    tex: texture_2d_array<f32>,
    tex_sampler: sampler,
    uv: vec2<f32>,
    layer: i32,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>
) -> vec4<f32> {
    // Get integer tile coordinate
    let tile = floor(uv);
    let f = fract(uv);
    
    // Sample from 3 nearby tiles with random offsets
    var color = vec4<f32>(0.0);
    var total_weight = 0.0;
    
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let offset = vec2<f32>(f32(i), f32(j));
            let tile_pos = tile + offset;
            
            // Random offset for this tile (0-1 range)
            let rand = hash2(tile_pos);
            
            // Offset UV within tile
            let offset_uv = uv + rand;
            
            // Weight based on distance from tile center (smooth falloff)
            let dist = f - offset + 0.5;
            let weight = max(0.0, 1.0 - dot(dist, dist) * 2.0);
            
            if weight > 0.001 {
                color += textureSampleGrad(tex, tex_sampler, offset_uv, layer, ddx_uv, ddy_uv) * weight;
                total_weight += weight;
            }
        }
    }
    
    return color / max(total_weight, 0.001);
}

// Compute triplanar blend weights from world normal
// Uses squared weights for smoother falloff, then applies sharpness
fn triplanar_weights(world_normal: vec3<f32>, sharpness: f32) -> vec3<f32> {
    // Square the absolute normal components for initial weights
    var weights = world_normal * world_normal;
    
    // Apply additional sharpness power if needed
    if sharpness > 1.0 {
        weights = pow(weights, vec3<f32>(sharpness * 0.5));
    }
    
    // Normalize
    let sum = weights.x + weights.y + weights.z;
    return weights / max(sum, 0.0001);
}

// Sample at a single scale with triplanar + stochastic
fn sample_triplanar_single_scale(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec4<f32> {
    let weights = triplanar_weights(world_normal, sharpness);
    let scaled_pos = world_pos * uv_scale;

    // X projection: YZ plane (for surfaces facing +/- X)
    let uv_x = scaled_pos.yz;
    // Y projection: XZ plane (for surfaces facing +/- Y) - main terrain projection
    let uv_y = scaled_pos.xz;
    // Z projection: XY plane (for surfaces facing +/- Z)
    let uv_z = scaled_pos.xy;

    // Compute smooth derivatives from world position (these are continuous across triangles)
    let dpdx_pos = dpdx(scaled_pos);
    let dpdy_pos = dpdy(scaled_pos);
    
    // Derive UV gradients from world position gradients
    let ddx_x = dpdx_pos.yz;
    let ddy_x = dpdy_pos.yz;
    let ddx_y = dpdx_pos.xz;
    let ddy_y = dpdy_pos.xz;
    let ddx_z = dpdx_pos.xy;
    let ddy_z = dpdy_pos.xy;

    // Sample with stochastic tiling to break up repetition patterns
    let sample_x = sample_stochastic(albedo_array, albedo_array_sampler, uv_x, layer, ddx_x, ddy_x);
    let sample_y = sample_stochastic(albedo_array, albedo_array_sampler, uv_y, layer, ddx_y, ddy_y);
    let sample_z = sample_stochastic(albedo_array, albedo_array_sampler, uv_z, layer, ddx_z, ddy_z);

    return sample_x * weights.x + sample_y * weights.y + sample_z * weights.z;
}

// Sample texture array using triplanar projection with multi-scale blending
// Blends 3 different scales to break up visible repetition
fn sample_triplanar(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec4<f32> {
    // Sample at 3 different scales
    let scale1 = uv_scale;        // Base scale
    let scale2 = uv_scale * 0.37; // Medium scale (use prime-ish ratio to avoid alignment)
    let scale3 = uv_scale * 0.11; // Large scale for overall variation
    
    let sample1 = sample_triplanar_single_scale(world_pos, world_normal, layer, scale1, sharpness);
    let sample2 = sample_triplanar_single_scale(world_pos, world_normal, layer, scale2, sharpness);
    let sample3 = sample_triplanar_single_scale(world_pos, world_normal, layer, scale3, sharpness);
    
    // Blend scales together
    // Use overlay-style blending: detail from small scale, color variation from large scales
    let base_color = sample3 * 0.3 + sample2 * 0.35 + sample1 * 0.35;
    
    // Add detail contrast from the fine scale
    let detail = sample1 - vec4<f32>(0.5);
    let final_color = base_color + detail * 0.3;
    
    return clamp(final_color, vec4<f32>(0.0), vec4<f32>(1.0));
}

// Height-based blending for better transitions
fn height_blend(heights: vec4<f32>, weights: vec4<f32>, sharpness: f32) -> vec4<f32> {
    if sharpness <= 0.0 {
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

// Apply blend weight adjustments (offset and exponent) for sharper material transitions
// Inspired by MegaSplat's blend weight controls
fn adjust_blend_weights(weights: vec4<f32>, offset: f32, exponent: f32) -> vec4<f32> {
    // Apply offset - subtract from all weights, reducing influence of weaker materials
    var adjusted = max(weights - vec4<f32>(offset), vec4<f32>(0.0));
    
    // Apply exponent - raise to power for sharper transitions
    // Only apply if exponent > 1.0 (exponent of 1.0 = no change)
    if exponent > 1.0 {
        adjusted = pow(adjusted, vec4<f32>(exponent));
    }
    
    // Renormalize so weights sum to 1.0
    let sum = adjusted.x + adjusted.y + adjusted.z + adjusted.w;
    if sum > 0.0 {
        return adjusted / sum;
    }
    
    // Fallback: if all weights became zero, return original normalized weights
    let orig_sum = weights.x + weights.y + weights.z + weights.w;
    if orig_sum > 0.0 {
        return weights / orig_sum;
    }
    return vec4<f32>(1.0, 0.0, 0.0, 0.0);
}

// Sample normal map with stochastic tiling (similar to albedo sampling)
fn sample_normal_stochastic(
    uv: vec2<f32>,
    layer: i32,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>
) -> vec3<f32> {
    // Get integer tile coordinate
    let tile = floor(uv);
    let f = fract(uv);
    
    var normal = vec3<f32>(0.0);
    var total_weight = 0.0;
    
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let offset = vec2<f32>(f32(i), f32(j));
            let tile_pos = tile + offset;
            let rand = hash2(tile_pos);
            let offset_uv = uv + rand;
            
            let dist = f - offset + 0.5;
            let weight = max(0.0, 1.0 - dot(dist, dist) * 2.0);
            
            if weight > 0.001 {
                // Sample normal map and decode from [0,1] to [-1,1]
                let sampled = textureSampleGrad(normal_array, normal_array_sampler, offset_uv, layer, ddx_uv, ddy_uv).rgb;
                let decoded = sampled * 2.0 - 1.0;
                normal += decoded * weight;
                total_weight += weight;
            }
        }
    }
    
    return normalize(normal / max(total_weight, 0.001));
}

// Sample normal map at single scale with triplanar projection
fn sample_normal_triplanar_single_scale(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec3<f32> {
    let weights = triplanar_weights(world_normal, sharpness);
    let scaled_pos = world_pos * uv_scale;

    let uv_x = scaled_pos.yz;
    let uv_y = scaled_pos.xz;
    let uv_z = scaled_pos.xy;

    let dpdx_pos = dpdx(scaled_pos);
    let dpdy_pos = dpdy(scaled_pos);
    
    let ddx_x = dpdx_pos.yz;
    let ddy_x = dpdy_pos.yz;
    let ddx_y = dpdx_pos.xz;
    let ddy_y = dpdy_pos.xz;
    let ddx_z = dpdx_pos.xy;
    let ddy_z = dpdy_pos.xy;

    // Sample normals for each projection
    let normal_x = sample_normal_stochastic(uv_x, layer, ddx_x, ddy_x);
    let normal_y = sample_normal_stochastic(uv_y, layer, ddx_y, ddy_y);
    let normal_z = sample_normal_stochastic(uv_z, layer, ddx_z, ddy_z);

    // Swizzle normals to world space orientation for each projection
    // X projection (YZ plane): normal is in YZ tangent space
    let world_normal_x = vec3<f32>(normal_x.z, normal_x.x, normal_x.y);
    // Y projection (XZ plane): normal is in XZ tangent space  
    let world_normal_y = vec3<f32>(normal_y.x, normal_y.z, normal_y.y);
    // Z projection (XY plane): normal is in XY tangent space
    let world_normal_z = vec3<f32>(normal_z.x, normal_z.y, normal_z.z);

    // Blend normals based on triplanar weights
    let blended = world_normal_x * weights.x + world_normal_y * weights.y + world_normal_z * weights.z;
    return normalize(blended);
}

// Sample normal map with triplanar projection (single scale for normals - multi-scale causes artifacts)
fn sample_normal_triplanar(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> vec3<f32> {
    // Use single scale for normals to avoid blending artifacts
    return sample_normal_triplanar_single_scale(world_pos, world_normal, layer, uv_scale, sharpness);
}

// Reoriented Normal Mapping (RNM) blend - better than linear blending for normals
fn blend_rnm(n1: vec3<f32>, n2: vec3<f32>) -> vec3<f32> {
    let t = n1 + vec3<f32>(0.0, 0.0, 1.0);
    let u = n2 * vec3<f32>(-1.0, -1.0, 1.0);
    return normalize(t * dot(t, u) - u * t.z);
}

// PBR data structure
struct PbrData {
    metallic: f32,
    roughness: f32,
    ao: f32,
    height: f32,
}

// Sample PBR texture with stochastic tiling
fn sample_pbr_stochastic(
    uv: vec2<f32>,
    layer: i32,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>
) -> PbrData {
    let tile = floor(uv);
    let f = fract(uv);
    
    var pbr = vec4<f32>(0.0);
    var total_weight = 0.0;
    
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let offset = vec2<f32>(f32(i), f32(j));
            let tile_pos = tile + offset;
            let rand = hash2(tile_pos);
            let offset_uv = uv + rand;
            
            let dist = f - offset + 0.5;
            let weight = max(0.0, 1.0 - dot(dist, dist) * 2.0);
            
            if weight > 0.001 {
                let sampled = textureSampleGrad(pbr_array, pbr_array_sampler, offset_uv, layer, ddx_uv, ddy_uv);
                pbr += sampled * weight;
                total_weight += weight;
            }
        }
    }
    
    pbr = pbr / max(total_weight, 0.001);
    
    return PbrData(pbr.r, pbr.g, pbr.b, pbr.a);
}

// Sample PBR at single scale with triplanar projection
fn sample_pbr_triplanar_single_scale(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> PbrData {
    let weights = triplanar_weights(world_normal, sharpness);
    let scaled_pos = world_pos * uv_scale;

    let uv_x = scaled_pos.yz;
    let uv_y = scaled_pos.xz;
    let uv_z = scaled_pos.xy;

    let dpdx_pos = dpdx(scaled_pos);
    let dpdy_pos = dpdy(scaled_pos);
    
    let ddx_x = dpdx_pos.yz;
    let ddy_x = dpdy_pos.yz;
    let ddx_y = dpdx_pos.xz;
    let ddy_y = dpdy_pos.xz;
    let ddx_z = dpdx_pos.xy;
    let ddy_z = dpdy_pos.xy;

    let pbr_x = sample_pbr_stochastic(uv_x, layer, ddx_x, ddy_x);
    let pbr_y = sample_pbr_stochastic(uv_y, layer, ddx_y, ddy_y);
    let pbr_z = sample_pbr_stochastic(uv_z, layer, ddx_z, ddy_z);

    // Blend PBR values based on triplanar weights
    return PbrData(
        pbr_x.metallic * weights.x + pbr_y.metallic * weights.y + pbr_z.metallic * weights.z,
        pbr_x.roughness * weights.x + pbr_y.roughness * weights.y + pbr_z.roughness * weights.z,
        pbr_x.ao * weights.x + pbr_y.ao * weights.y + pbr_z.ao * weights.z,
        pbr_x.height * weights.x + pbr_y.height * weights.y + pbr_z.height * weights.z
    );
}

// Sample PBR textures with triplanar projection
fn sample_pbr_triplanar(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    layer: i32,
    uv_scale: f32,
    sharpness: f32
) -> PbrData {
    return sample_pbr_triplanar_single_scale(world_pos, world_normal, layer, uv_scale, sharpness);
}

@vertex
fn vertex(
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(10) material_indices: u32,
    @location(11) material_weights: u32,
) -> PlumeSplatVertexOutput {
    var out: PlumeSplatVertexOutput;

    let world_from_local = mesh_functions::get_world_from_local(instance_index);
    let world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(position, 1.0));

    out.position = position_world_to_clip(world_position.xyz);
    out.world_position = world_position;
    out.world_normal = mesh_functions::mesh_normal_local_to_world(normal, instance_index);
    out.uv = uv;
    out.material_indices = material_indices;
    
    // Unpack weights to vec4<f32> for proper interpolation across the triangle
    let raw_weights = unpack_u8x4(material_weights);
    out.material_weights = vec4<f32>(raw_weights) / 255.0;

    return out;
}

@fragment
fn fragment(
    in: PlumeSplatVertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Get world position and normal for triplanar
    let world_pos = in.world_position.xyz;
    var geometry_normal = normalize(in.world_normal);

    // Handle back faces
    if !is_front {
        geometry_normal = -geometry_normal;
    }

    // Unpack material indices (flat interpolated - same for whole triangle)
    let indices = unpack_u8x4(in.material_indices);
    let idx0 = i32(indices.x);
    let idx1 = i32(indices.y);
    let idx2 = i32(indices.z);
    let idx3 = i32(indices.w);

    // Weights are already unpacked in vertex shader and properly interpolated
    var weights = in.material_weights;

    // Normalize weights
    let sum = weights.x + weights.y + weights.z + weights.w;
    if sum > 0.0 {
        weights = weights / sum;
    }
    
    // Apply blend weight adjustments (offset and exponent) for sharper transitions
    weights = adjust_blend_weights(weights, settings.blend_offset, settings.blend_exponent);

    // Full terrain rendering with material blending
    let uv_scale = settings.uv_scale;
    let sharpness = settings.triplanar_sharpness;
    
    // Sample each material layer (albedo)
    let albedo0 = sample_triplanar(world_pos, geometry_normal, idx0, uv_scale, sharpness);
    let albedo1 = sample_triplanar(world_pos, geometry_normal, idx1, uv_scale, sharpness);
    let albedo2 = sample_triplanar(world_pos, geometry_normal, idx2, uv_scale, sharpness);
    let albedo3 = sample_triplanar(world_pos, geometry_normal, idx3, uv_scale, sharpness);

    // Optional height-based blending (applied after offset/exponent)
    if settings.height_blend_sharpness > 0.0 {
        let heights = vec4<f32>(albedo0.a, albedo1.a, albedo2.a, albedo3.a);
        weights = height_blend(heights, weights, settings.height_blend_sharpness);
    }

    // Blend final albedo
    let final_albedo = albedo0 * weights.x + albedo1 * weights.y +
                       albedo2 * weights.z + albedo3 * weights.w;
    
    // Sample and blend normal maps
    let normal0 = sample_normal_triplanar(world_pos, geometry_normal, idx0, uv_scale, sharpness);
    let normal1 = sample_normal_triplanar(world_pos, geometry_normal, idx1, uv_scale, sharpness);
    let normal2 = sample_normal_triplanar(world_pos, geometry_normal, idx2, uv_scale, sharpness);
    let normal3 = sample_normal_triplanar(world_pos, geometry_normal, idx3, uv_scale, sharpness);
    
    // Blend normals
    var blended_normal = normal0 * weights.x + normal1 * weights.y +
                         normal2 * weights.z + normal3 * weights.w;
    blended_normal = normalize(blended_normal);
    
    // Combine sampled normal with geometry normal using RNM blending
    let final_normal = blend_rnm(geometry_normal, blended_normal);
    
    // Sample and blend PBR textures
    let pbr0 = sample_pbr_triplanar(world_pos, geometry_normal, idx0, uv_scale, sharpness);
    let pbr1 = sample_pbr_triplanar(world_pos, geometry_normal, idx1, uv_scale, sharpness);
    let pbr2 = sample_pbr_triplanar(world_pos, geometry_normal, idx2, uv_scale, sharpness);
    let pbr3 = sample_pbr_triplanar(world_pos, geometry_normal, idx3, uv_scale, sharpness);
    
    // Blend PBR values
    let final_metallic = pbr0.metallic * weights.x + pbr1.metallic * weights.y +
                         pbr2.metallic * weights.z + pbr3.metallic * weights.w;
    let final_roughness = pbr0.roughness * weights.x + pbr1.roughness * weights.y +
                          pbr2.roughness * weights.z + pbr3.roughness * weights.w;
    let final_ao = pbr0.ao * weights.x + pbr1.ao * weights.y +
                   pbr2.ao * weights.z + pbr3.ao * weights.w;
    
    // PBR Lighting calculation
    // View direction - from fragment to camera (using actual camera position from view bindings)
    let view_dir = normalize(view.world_position - world_pos);
    
    // Light direction and color
    let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.4));
    let light_color = vec3<f32>(1.0, 0.98, 0.95); // Slightly warm sunlight
    
    // Half vector for specular
    let half_vec = normalize(light_dir + view_dir);
    
    // Fresnel (Schlick approximation)
    let f0 = mix(vec3<f32>(0.04), final_albedo.rgb, final_metallic);
    let cos_theta = max(dot(half_vec, view_dir), 0.0);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
    
    // Normal dot products
    let n_dot_l = max(dot(final_normal, light_dir), 0.0);
    let n_dot_v = max(dot(final_normal, view_dir), 0.0);
    let n_dot_h = max(dot(final_normal, half_vec), 0.0);
    
    // Roughness for specular
    let alpha = final_roughness * final_roughness;
    let alpha2 = alpha * alpha;
    
    // GGX Distribution
    let denom = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    let distribution = alpha2 / (3.14159 * denom * denom + 0.0001);
    
    // Geometry (Schlick-GGX)
    let k = (final_roughness + 1.0) * (final_roughness + 1.0) / 8.0;
    let g1_l = n_dot_l / (n_dot_l * (1.0 - k) + k + 0.0001);
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
    let geometry = g1_l * g1_v;
    
    // Cook-Torrance BRDF
    let specular = (distribution * geometry * fresnel) / (4.0 * n_dot_l * n_dot_v + 0.0001);
    
    // Diffuse (Lambertian with energy conservation)
    let kd = (1.0 - fresnel) * (1.0 - final_metallic);
    let diffuse = kd * final_albedo.rgb / 3.14159;
    
    // Final lighting
    let direct_light = (diffuse + specular) * light_color * n_dot_l;
    
    // Ambient with AO
    let ambient = vec3<f32>(0.15, 0.18, 0.22) * final_albedo.rgb * final_ao;
    
    // Combine
    var lit_color = ambient + direct_light;
    
    // Simple tone mapping (Reinhard)
    //lit_color = lit_color / (lit_color + vec3<f32>(1.0));
    
    // Gamma correction (approximate)
    //lit_color = pow(lit_color, vec3<f32>(1.0 / 2.2));
    
   // var out: FragmentOutput;
    //out.color = vec4<f32>(lit_color, 1.0);


     // generate a PbrInput struct from the StandardMaterial bindings
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // we can optionally modify the input before lighting and alpha_discard is applied
    pbr_input.material.base_color.b = pbr_input.material.base_color.r;

    // alpha discard
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    // in deferred mode we can't modify anything after that, as lighting is run in a separate fullscreen shader.
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    // apply lighting
    out.color = apply_pbr_lighting(pbr_input);

    // we can optionally modify the lit color before post-processing is applied
    //out.color = vec4<f32>(vec4<u32>(out.color * f32(my_extended_material.quantize_steps))) / f32(my_extended_material.quantize_steps);

    // apply in-shader post processing (fog, alpha-premultiply, and also tonemapping, debanding if the camera is non-hdr)
    // note this does not include fullscreen postprocessing effects like bloom.
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    // we can optionally modify the final result here
    out.color = out.color * 2.0;
#endif
    return out;
}
