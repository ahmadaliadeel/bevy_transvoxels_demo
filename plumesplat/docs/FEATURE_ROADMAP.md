# PlumeSplat Feature Roadmap

This document outlines features to add to PlumeSplat, inspired by MegaSplat and modern terrain rendering techniques.

## Current Features (Completed)

- [x] Texture array splatting (up to 256 materials)
- [x] Triplanar mapping with configurable sharpness
- [x] Material blending based on vertex weights
- [x] Height/slope-based material assignment
- [x] Runtime mipmap generation
- [x] Stochastic tiling to eliminate seams
- [x] Multi-scale texture blending to reduce repetition
- [x] MSAA support
- [x] Basic diffuse lighting

---

## Phase 1: Core PBR Support

### 1.1 Normal Maps
**Priority:** High | **Difficulty:** Medium | **Impact:** High

Add per-material normal maps for surface detail. This is the single biggest visual improvement we can make.

**Implementation:**
- Add a second texture array for normal maps (`normal_array`)
- Sample normal maps in fragment shader using same triplanar/stochastic approach
- Transform normals from tangent space to world space
- Blend normals across materials (use RNM or UDN blending)

**Files to modify:**
- `src/material.rs` - Add normal array binding
- `src/shaders/plumesplat_ext.wgsl` - Sample and apply normals
- `examples/basic.rs` - Load normal textures

**Resources:**
- Normal blending techniques: https://blog.selfshadow.com/publications/blending-in-detail/

---

### 1.2 Roughness/Metallic Maps (Full PBR)
**Priority:** High | **Difficulty:** Medium | **Impact:** High

Add roughness and metallic maps for physically-based rendering.

**Implementation:**
- Option A: Separate texture arrays for roughness and metallic
- Option B: Pack into single array (R=metallic, G=roughness, B=AO, A=height)
- Integrate with Bevy's PBR lighting (we already have the ExtendedMaterial setup)

**Files to modify:**
- `src/material.rs` - Add PBR texture array bindings
- `src/shaders/plumesplat_ext.wgsl` - Sample PBR textures, output to PBR pipeline

---

### 1.3 Ambient Occlusion Maps
**Priority:** Medium | **Difficulty:** Low | **Impact:** Medium

Per-material ambient occlusion for added depth in crevices.

**Implementation:**
- Can be packed into the alpha channel of albedo, or into a packed PBR texture
- Multiply final color by AO value
- Consider distance-based AO fade (less AO at distance)

---

## Phase 2: Advanced Blending

### 2.1 Height-Based Material Blending
**Priority:** High | **Difficulty:** Medium | **Impact:** High

Currently we blend materials linearly based on vertex weights. Height-based blending uses texture height (from a height map or albedo alpha) to create more natural transitions.

**How it works:**
- Each material has a height value at each pixel
- Materials with higher height "poke through" at transitions
- Creates rocky outcrops poking through grass, etc.

**Implementation:**
```wgsl
fn height_blend(colors: array<vec4<f32>, 4>, heights: array<f32, 4>, weights: vec4<f32>) -> vec4<f32> {
    // Add height to weights
    let h0 = weights.x + heights[0];
    let h1 = weights.y + heights[1];
    let h2 = weights.z + heights[2];
    let h3 = weights.w + heights[3];
    
    // Find max and create blend
    let max_h = max(max(h0, h1), max(h2, h3));
    let blend_range = 0.2; // Adjustable
    
    let b0 = max(h0 - max_h + blend_range, 0.0);
    let b1 = max(h1 - max_h + blend_range, 0.0);
    let b2 = max(h2 - max_h + blend_range, 0.0);
    let b3 = max(h3 - max_h + blend_range, 0.0);
    
    let sum = b0 + b1 + b2 + b3;
    return (colors[0] * b0 + colors[1] * b1 + colors[2] * b2 + colors[3] * b3) / sum;
}
```

**Note:** We already have a basic `height_blend` function but it's not fully utilized.

---

### 2.2 Macro Texturing
**Priority:** Medium | **Difficulty:** Low | **Impact:** Medium

Add large-scale color variation to break up repetition at distance.

**Implementation:**
- Sample a low-frequency noise or macro texture at very large UV scale (0.01x)
- Blend with albedo using overlay or soft light blending
- Fade effect based on distance (more at distance, less up close)

---

### 2.3 Detail Textures
**Priority:** Low | **Difficulty:** Low | **Impact:** Medium

Add high-frequency detail visible only at close range.

**Implementation:**
- Sample detail texture at high UV scale (10x-20x)
- Blend using overlay blending mode
- Fade out based on distance (only visible up close)

---

## Phase 3: Depth and Displacement

### 3.1 Parallax Mapping
**Priority:** Medium | **Difficulty:** Medium-High | **Impact:** High

Create illusion of depth without additional geometry.

**Types (in order of quality/cost):**
1. **Simple Parallax** - Single offset based on height and view angle
2. **Steep Parallax** - Multiple steps along view ray
3. **Parallax Occlusion Mapping (POM)** - Raymarching with binary search

**Implementation:**
```wgsl
fn parallax_offset(uv: vec2<f32>, view_dir: vec3<f32>, height_scale: f32) -> vec2<f32> {
    let height = texture_sample(height_map, uv).r;
    let offset = view_dir.xy / view_dir.z * (height * height_scale);
    return uv - offset;
}
```

**Considerations:**
- Requires height map per material
- Need tangent-space view direction
- Performance cost increases with quality
- Silhouettes still flat (unlike tessellation)

---

### 3.2 Tessellation (GPU Displacement)
**Priority:** Low | **Difficulty:** High | **Impact:** Medium

Actual geometry displacement using GPU tessellation.

**Considerations:**
- Requires tessellation shader stages (not sure of Bevy/wgpu support)
- High performance cost
- Best for close-up terrain
- May not be worth it for Bevy 0.17

---

## Phase 4: Environmental Effects

### 4.1 Wetness/Puddles
**Priority:** Medium | **Difficulty:** Medium | **Impact:** High

Dynamic wet surfaces with puddles in low areas.

**Implementation:**
- Add wetness uniform (0-1, can be driven by weather system)
- Wet surfaces: darker albedo, lower roughness, higher reflectivity
- Puddles: flat water in concave areas (based on AO or height)
- Optional: rain ripples animation

```wgsl
fn apply_wetness(albedo: vec3<f32>, roughness: f32, wetness: f32, ao: f32) -> PBROutput {
    let puddle_mask = saturate((wetness - ao) * 4.0); // Puddles in low AO areas
    
    let wet_albedo = albedo * mix(1.0, 0.7, wetness); // Darken when wet
    let wet_roughness = mix(roughness, 0.1, wetness); // Smoother when wet
    
    // Puddles are very smooth and reflective
    let final_albedo = mix(wet_albedo, vec3(0.02), puddle_mask);
    let final_roughness = mix(wet_roughness, 0.0, puddle_mask);
    
    return PBROutput(final_albedo, final_roughness);
}
```

---

### 4.2 Snow Accumulation
**Priority:** Low | **Difficulty:** Low | **Impact:** Medium

Procedural snow on upward-facing surfaces.

**Implementation:**
- Add snow amount uniform (0-1)
- Apply snow material based on world normal Y component
- Blend with existing material based on slope threshold

```wgsl
fn apply_snow(albedo: vec3<f32>, world_normal: vec3<f32>, snow_amount: f32) -> vec3<f32> {
    let snow_color = vec3(0.95, 0.95, 0.97);
    let slope = 1.0 - world_normal.y; // 0 = flat, 1 = vertical
    let snow_mask = saturate((snow_amount - slope) * 4.0);
    return mix(albedo, snow_color, snow_mask);
}
```

---

### 4.3 Flow Maps (Water/Lava)
**Priority:** Low | **Difficulty:** Medium | **Impact:** Medium

Animated directional flow for water or lava materials.

**Implementation:**
- Flow map texture defines flow direction per pixel
- Animate UV offset over time based on flow direction
- Blend two offset samples to hide discontinuities

---

## Phase 5: Performance Optimization

### 5.1 Distance-Based LOD
**Priority:** High | **Difficulty:** Medium | **Impact:** High (Performance)

Reduce shader complexity at distance.

**Implementation:**
- Near: Full quality (multi-scale, stochastic, all PBR maps)
- Medium: Reduced samples, no stochastic tiling
- Far: Single texture sample, simplified lighting
- Use `dpdx`/`dpdy` or distance to camera to determine LOD

---

### 5.2 Virtual Texturing
**Priority:** Low | **Difficulty:** Very High | **Impact:** High (Memory)

Stream texture data on demand for massive terrains.

**Note:** This is a major undertaking, probably out of scope for now.

---

## Implementation Order

Recommended order based on impact vs effort:

1. **Normal Maps** (Phase 1.1) - Biggest visual bang for buck
2. **Height-Based Blending** (Phase 2.1) - Better material transitions  
3. **Full PBR** (Phase 1.2) - Complete the PBR pipeline
4. **Distance-Based LOD** (Phase 5.1) - Performance optimization
5. **Parallax Mapping** (Phase 3.1) - Depth without geometry
6. **Wetness/Puddles** (Phase 4.1) - Environmental realism
7. **Macro Texturing** (Phase 2.2) - Large-scale variation
8. **Snow Accumulation** (Phase 4.2) - Easy environmental effect
9. **AO Maps** (Phase 1.3) - Subtle depth improvement
10. **Detail Textures** (Phase 2.3) - Close-up detail

---

## Resources

- Normal map blending: https://blog.selfshadow.com/publications/blending-in-detail/
- Parallax mapping: https://catlikecoding.com/unity/tutorials/rendering/part-20/
- PBR theory: https://learnopengl.com/PBR/Theory
- Height blending: https://www.gamedeveloper.com/programming/advanced-terrain-texture-splatting
- Triplanar mapping: https://catlikecoding.com/unity/tutorials/advanced-rendering/triplanar-mapping/

---

## Notes

- Each feature should be optional (compile-time or runtime toggle)
- Consider mobile/low-end GPU support with quality presets
- Test performance impact of each feature
- Document shader permutation count to avoid explosion
