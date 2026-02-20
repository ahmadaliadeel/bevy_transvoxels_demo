//! PlumeSplat material definition and configuration.
//!
//! This module provides the core material types for terrain splatting:
//!
//! - [`PlumeSplatMaterial`]: Type alias for the full extended material
//! - [`PlumeSplatExtension`]: The material extension with texture arrays and settings
//! - [`PlumeSplatSettings`]: Shader uniform configuration
//!
//! # Overview
//!
//! PlumeSplat extends Bevy's [`StandardMaterial`] to add texture array splatting
//! while retaining full PBR lighting support. This is accomplished using Bevy's
//! [`ExtendedMaterial`] system.
//!
//! # Example
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use plumesplat::prelude::*;
//!
//! fn create_terrain_material(
//!     asset_server: &Res<AssetServer>,
//! ) -> PlumeSplatMaterial {
//!     let albedo = asset_server.load("textures/terrain_albedo.png");
//!     let normals = asset_server.load("textures/terrain_normal.png");
//!     let pbr = asset_server.load("textures/terrain_pbr.png");
//!
//!     PlumeSplatMaterial {
//!         base: StandardMaterial {
//!             perceptual_roughness: 0.8,
//!             metallic: 0.0,
//!             ..default()
//!         },
//!         extension: PlumeSplatExtension::new(albedo)
//!             .with_normal_array(normals)
//!             .with_pbr_array(pbr)
//!             .with_uv_scale(2.0)
//!             .with_triplanar_sharpness(4.0)
//!             .with_blend_offset(0.1)
//!             .with_blend_exponent(2.0),
//!     }
//! }
//! ```

use bevy::asset::{Asset, Handle};
use bevy::image::Image;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
    StandardMaterial,
};
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    VertexAttribute, VertexFormat,
};
use bevy::shader::ShaderRef;

use crate::vertex_attributes::{ATTRIBUTE_MATERIAL_INDICES, ATTRIBUTE_MATERIAL_WEIGHTS};

/// Type alias for the complete PlumeSplat material.
///
/// This combines Bevy's [`StandardMaterial`] (for PBR lighting) with
/// [`PlumeSplatExtension`] (for texture array splatting).
///
/// # Usage
///
/// ```rust,no_run
/// use bevy::prelude::*;
/// use plumesplat::prelude::*;
///
/// fn setup(mut materials: ResMut<Assets<PlumeSplatMaterial>>) {
///     let material = PlumeSplatMaterial {
///         base: StandardMaterial::default(),
///         extension: PlumeSplatExtension::default(),
///     };
///     let handle = materials.add(material);
/// }
/// ```
pub type PlumeSplatMaterial = ExtendedMaterial<StandardMaterial, PlumeSplatExtension>;

/// Material extension that adds texture array splatting to StandardMaterial.
///
/// This extension enables blending up to 256 different materials using texture arrays
/// and per-vertex material indices/weights. It supports:
///
/// - **Albedo textures**: Base color with optional height in alpha channel
/// - **Normal maps**: Per-material tangent-space normal maps
/// - **PBR textures**: Packed metallic, roughness, AO, and height maps
/// - **Triplanar mapping**: UV-less texturing for arbitrary geometry
/// - **Stochastic tiling**: Eliminates visible texture repetition
///
/// # Texture Array Format
///
/// All texture arrays should be provided as vertically stacked 2D images that get
/// converted to GPU texture arrays at runtime. Each layer corresponds to one material:
///
/// | Texture | Channel Layout | Description |
/// |---------|----------------|-------------|
/// | `albedo_array` | RGBA | RGB = color, A = height (optional) |
/// | `normal_array` | RGB | Tangent-space normals |
/// | `pbr_array` | RGBA | R = metallic, G = roughness, B = AO, A = height |
///
/// # Vertex Attributes
///
/// Meshes using this material must include custom vertex attributes:
///
/// - [`ATTRIBUTE_MATERIAL_INDICES`](crate::ATTRIBUTE_MATERIAL_INDICES): 4 material indices packed into u32
/// - [`ATTRIBUTE_MATERIAL_WEIGHTS`](crate::ATTRIBUTE_MATERIAL_WEIGHTS): 4 blend weights packed into u32
///
/// See [`MaterialVertex`](crate::MaterialVertex) for helpers to create these.
///
/// # Example
///
/// ```rust,no_run
/// use bevy::prelude::*;
/// use plumesplat::prelude::*;
///
/// fn create_extension(asset_server: &Res<AssetServer>) -> PlumeSplatExtension {
///     let albedo = asset_server.load("terrain_albedo.png");
///
///     PlumeSplatExtension::new(albedo)
///         .with_uv_scale(2.0)           // Texture tiling
///         .with_triplanar_sharpness(4.0) // Edge sharpness
///         .with_blend_offset(0.1)        // Sharper transitions
///         .with_blend_exponent(2.0)      // Even sharper
/// }
/// ```
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone, Default)]
pub struct PlumeSplatExtension {
    /// Albedo (base color) texture array.
    ///
    /// Each layer is a different material's color texture. The alpha channel
    /// can optionally store height values for height-based blending.
    ///
    /// **Required** - This must be set for the material to render correctly.
    #[texture(100, dimension = "2d_array")]
    #[sampler(101, sampler_type = "filtering")]
    pub albedo_array: Handle<Image>,

    /// Normal map texture array.
    ///
    /// Each layer contains tangent-space normal maps for the corresponding
    /// material index. Normals are blended across materials using reoriented
    /// normal mapping (RNM) for correct results.
    ///
    /// **Optional** - If not provided, geometry normals are used.
    #[texture(103, dimension = "2d_array")]
    #[sampler(104, sampler_type = "filtering")]
    pub normal_array: Option<Handle<Image>>,

    /// Packed PBR texture array.
    ///
    /// Each layer contains packed PBR properties for physically-based rendering:
    ///
    /// | Channel | Property | Range | Description |
    /// |---------|----------|-------|-------------|
    /// | R | Metallic | 0.0-1.0 | 0 = dielectric, 1 = metal |
    /// | G | Roughness | 0.0-1.0 | 0 = smooth/glossy, 1 = rough/matte |
    /// | B | Ambient Occlusion | 0.0-1.0 | 0 = fully occluded, 1 = no occlusion |
    /// | A | Height | 0.0-1.0 | Used for height-based blending |
    ///
    /// **Optional** - If not provided, default PBR values from StandardMaterial are used.
    #[texture(105, dimension = "2d_array")]
    #[sampler(106, sampler_type = "filtering")]
    pub pbr_array: Option<Handle<Image>>,

    /// Shader configuration settings.
    ///
    /// Controls UV scaling, blending behavior, and triplanar mapping parameters.
    /// See [`PlumeSplatSettings`] for details on each setting.
    #[uniform(102)]
    pub settings: PlumeSplatSettings,
}

/// Shader uniform settings for PlumeSplat rendering.
///
/// These settings control how textures are sampled and blended in the shader.
/// All settings have sensible defaults but can be tuned for different visual effects.
///
/// # Example
///
/// ```rust
/// use plumesplat::PlumeSplatSettings;
///
/// let settings = PlumeSplatSettings {
///     uv_scale: 2.0,              // 2x texture tiling
///     triplanar_sharpness: 8.0,   // Sharp triplanar edges
///     height_blend_sharpness: 0.5, // Enable height blending
///     blend_offset: 0.1,          // Sharper material transitions
///     blend_exponent: 2.0,        // Squared weights
/// };
/// ```
#[derive(Debug, Clone, Copy, ShaderType)]
pub struct PlumeSplatSettings {
    /// UV scale multiplier for texture sampling.
    ///
    /// Higher values result in more texture tiling (smaller apparent texture size).
    /// Lower values stretch the texture over a larger area.
    ///
    /// - **Default**: 1.0
    /// - **Range**: 0.1 to 10.0+ (no hard limit)
    /// - **Example**: 2.0 = textures tile twice as often
    pub uv_scale: f32,

    /// Triplanar mapping blend sharpness.
    ///
    /// Controls how sharply textures blend at geometry edges when using
    /// triplanar mapping. Higher values create sharper transitions between
    /// the X, Y, and Z projection planes.
    ///
    /// - **Default**: 4.0
    /// - **Range**: 1.0 (soft) to 16.0 (very sharp)
    pub triplanar_sharpness: f32,

    /// Height-based blend sharpness.
    ///
    /// When greater than 0, enables height-based material blending where
    /// materials with higher height values "poke through" at transitions.
    /// This creates more natural-looking material boundaries (e.g., rocks
    /// poking through grass).
    ///
    /// - **Default**: 0.0 (disabled)
    /// - **Range**: 0.0 (disabled) to 2.0+ (sharp height transitions)
    pub height_blend_sharpness: f32,

    /// Blend weight offset for sharper material transitions.
    ///
    /// Subtracts this value from all blend weights before normalization.
    /// This reduces the influence of materials with low weights, creating
    /// sharper boundaries between dominant materials.
    ///
    /// - **Default**: 0.0 (no offset)
    /// - **Range**: 0.0 to 0.5
    /// - **Warning**: Values above 0.5 may cause artifacts if all weights become zero
    pub blend_offset: f32,

    /// Blend weight exponent for sharper material transitions.
    ///
    /// Raises blend weights to this power before normalization. Higher values
    /// emphasize the dominant material and reduce blending at boundaries.
    ///
    /// - **Default**: 1.0 (linear blending)
    /// - **Range**: 1.0 (linear) to 8.0 (very sharp)
    /// - **Example**: 2.0 = quadratic falloff, 3.0 = cubic falloff
    pub blend_exponent: f32,
}

impl Default for PlumeSplatSettings {
    fn default() -> Self {
        Self {
            uv_scale: 1.0,
            triplanar_sharpness: 4.0,
            height_blend_sharpness: 0.0,
            blend_offset: 0.0,
            blend_exponent: 1.0,
        }
    }
}

impl PlumeSplatExtension {
    /// Creates a new PlumeSplat extension with the given albedo texture array.
    ///
    /// # Arguments
    ///
    /// * `albedo_array` - Handle to the albedo texture array (vertically stacked layers)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use bevy::prelude::*;
    /// use plumesplat::PlumeSplatExtension;
    ///
    /// fn setup(asset_server: Res<AssetServer>) {
    ///     let albedo = asset_server.load("terrain_albedo.png");
    ///     let extension = PlumeSplatExtension::new(albedo);
    /// }
    /// ```
    pub fn new(albedo_array: Handle<Image>) -> Self {
        Self {
            albedo_array,
            ..Default::default()
        }
    }

    /// Sets the UV scale for texture sampling.
    ///
    /// Higher values increase texture tiling density.
    ///
    /// # Arguments
    ///
    /// * `scale` - UV multiplier (default: 1.0)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use plumesplat::PlumeSplatExtension;
    /// # let albedo = bevy::asset::Handle::default();
    /// let ext = PlumeSplatExtension::new(albedo)
    ///     .with_uv_scale(2.0);  // 2x tiling
    /// ```
    pub fn with_uv_scale(mut self, scale: f32) -> Self {
        self.settings.uv_scale = scale;
        self
    }

    /// Sets the triplanar mapping blend sharpness.
    ///
    /// Controls how sharply the three projection planes blend at edges.
    ///
    /// # Arguments
    ///
    /// * `sharpness` - Blend exponent (default: 4.0, range: 1.0-16.0)
    pub fn with_triplanar_sharpness(mut self, sharpness: f32) -> Self {
        self.settings.triplanar_sharpness = sharpness;
        self
    }

    /// Enables height-based material blending.
    ///
    /// When enabled, materials with higher height values (from albedo alpha
    /// or PBR texture) will "poke through" at material transitions, creating
    /// more natural-looking boundaries.
    ///
    /// # Arguments
    ///
    /// * `sharpness` - Height blend sharpness (0.0 = disabled, default: 0.0)
    pub fn with_height_blending(mut self, sharpness: f32) -> Self {
        self.settings.height_blend_sharpness = sharpness;
        self
    }

    /// Sets the normal map texture array.
    ///
    /// # Arguments
    ///
    /// * `normal_array` - Handle to normal map texture array
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use bevy::prelude::*;
    /// use plumesplat::PlumeSplatExtension;
    ///
    /// fn setup(asset_server: Res<AssetServer>) {
    ///     let albedo = asset_server.load("terrain_albedo.png");
    ///     let normals = asset_server.load("terrain_normal.png");
    ///
    ///     let ext = PlumeSplatExtension::new(albedo)
    ///         .with_normal_array(normals);
    /// }
    /// ```
    pub fn with_normal_array(mut self, normal_array: Handle<Image>) -> Self {
        self.normal_array = Some(normal_array);
        self
    }

    /// Sets the packed PBR texture array.
    ///
    /// The PBR texture should have channels packed as:
    /// - R: Metallic (0-1)
    /// - G: Roughness (0-1)
    /// - B: Ambient Occlusion (0-1)
    /// - A: Height (0-1, for height-based blending)
    ///
    /// # Arguments
    ///
    /// * `pbr_array` - Handle to packed PBR texture array
    pub fn with_pbr_array(mut self, pbr_array: Handle<Image>) -> Self {
        self.pbr_array = Some(pbr_array);
        self
    }

    /// Sets the blend offset for material weight adjustment.
    ///
    /// Subtracts this value from all weights before normalization, causing
    /// materials with lower weights to have reduced influence. This creates
    /// sharper transitions between materials.
    ///
    /// # Arguments
    ///
    /// * `offset` - Weight offset (clamped to 0.0-0.5, default: 0.0)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use plumesplat::PlumeSplatExtension;
    /// # let albedo = bevy::asset::Handle::default();
    /// // Create sharper material boundaries
    /// let ext = PlumeSplatExtension::new(albedo)
    ///     .with_blend_offset(0.15);
    /// ```
    pub fn with_blend_offset(mut self, offset: f32) -> Self {
        self.settings.blend_offset = offset.clamp(0.0, 0.5);
        self
    }

    /// Sets the blend exponent for material weight adjustment.
    ///
    /// Raises weights to this power before normalization. Higher values
    /// create sharper transitions by emphasizing the dominant material.
    ///
    /// # Arguments
    ///
    /// * `exponent` - Weight exponent (clamped to 1.0-8.0, default: 1.0)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use plumesplat::PlumeSplatExtension;
    /// # let albedo = bevy::asset::Handle::default();
    /// // Use quadratic weight falloff for sharper boundaries
    /// let ext = PlumeSplatExtension::new(albedo)
    ///     .with_blend_exponent(2.0);
    /// ```
    pub fn with_blend_exponent(mut self, exponent: f32) -> Self {
        self.settings.blend_exponent = exponent.clamp(1.0, 8.0);
        self
    }
}

impl MaterialExtension for PlumeSplatExtension {
    fn vertex_shader() -> ShaderRef {
        "embedded://plumesplat/shaders/plumesplat_ext.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://plumesplat/shaders/plumesplat_ext.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Register custom vertex attributes for material indices and weights
        // These are used by the shader to determine which materials to sample
        // and how to blend them at each vertex

        // Material indices: 4 u8 values packed into a single u32
        // Shader location 10 is used to avoid conflicts with Bevy's built-in attributes
        if let Some(index) = layout
            .0
            .attribute_ids()
            .iter()
            .position(|id| *id == ATTRIBUTE_MATERIAL_INDICES.id)
        {
            let attr = &layout.0.layout().attributes[index];
            descriptor.vertex.buffers[0]
                .attributes
                .push(VertexAttribute {
                    format: VertexFormat::Uint32,
                    offset: attr.offset,
                    shader_location: 10,
                });
        }

        // Material weights: 4 u8 values (0-255) packed into a single u32
        // Normalized to 0.0-1.0 in the shader
        // Shader location 11 for blend weights
        if let Some(index) = layout
            .0
            .attribute_ids()
            .iter()
            .position(|id| *id == ATTRIBUTE_MATERIAL_WEIGHTS.id)
        {
            let attr = &layout.0.layout().attributes[index];
            descriptor.vertex.buffers[0]
                .attributes
                .push(VertexAttribute {
                    format: VertexFormat::Uint32,
                    offset: attr.offset,
                    shader_location: 11,
                });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = PlumeSplatSettings::default();
        assert_eq!(settings.uv_scale, 1.0);
        assert_eq!(settings.triplanar_sharpness, 4.0);
        assert_eq!(settings.height_blend_sharpness, 0.0);
        assert_eq!(settings.blend_offset, 0.0);
        assert_eq!(settings.blend_exponent, 1.0);
    }

    #[test]
    fn test_blend_offset_clamping() {
        let ext = PlumeSplatExtension::default()
            .with_blend_offset(1.0); // Above max
        assert_eq!(ext.settings.blend_offset, 0.5);

        let ext = PlumeSplatExtension::default()
            .with_blend_offset(-0.5); // Below min
        assert_eq!(ext.settings.blend_offset, 0.0);
    }

    #[test]
    fn test_blend_exponent_clamping() {
        let ext = PlumeSplatExtension::default()
            .with_blend_exponent(10.0); // Above max
        assert_eq!(ext.settings.blend_exponent, 8.0);

        let ext = PlumeSplatExtension::default()
            .with_blend_exponent(0.5); // Below min
        assert_eq!(ext.settings.blend_exponent, 1.0);
    }

    #[test]
    fn test_builder_chain() {
        let ext = PlumeSplatExtension::default()
            .with_uv_scale(2.0)
            .with_triplanar_sharpness(8.0)
            .with_height_blending(0.5)
            .with_blend_offset(0.2)
            .with_blend_exponent(3.0);

        assert_eq!(ext.settings.uv_scale, 2.0);
        assert_eq!(ext.settings.triplanar_sharpness, 8.0);
        assert_eq!(ext.settings.height_blend_sharpness, 0.5);
        assert_eq!(ext.settings.blend_offset, 0.2);
        assert_eq!(ext.settings.blend_exponent, 3.0);
    }
}
