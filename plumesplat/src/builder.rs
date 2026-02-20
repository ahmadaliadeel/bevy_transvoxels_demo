//! Builder API for creating PlumeSplat materials from individual textures.
//!
//! This module provides an ergonomic way to create terrain materials without
//! manually pre-combining textures into arrays. Simply specify individual
//! textures for each material layer and the plugin handles combination automatically.
//!
//! # Overview
//!
//! The builder API consists of:
//!
//! - [`MaterialLayer`]: Defines textures for a single material (grass, dirt, rock, etc.)
//! - [`SplatMaterialBuilder`]: Builder for configuring materials and settings
//! - [`PendingSplatMaterial`]: Component that triggers automatic texture combination
//!
//! # Example
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use plumesplat::prelude::*;
//!
//! fn setup(
//!     mut commands: Commands,
//!     asset_server: Res<AssetServer>,
//!     mut meshes: ResMut<Assets<Mesh>>,
//! ) {
//!     // Create material layers for each terrain type
//!     let grass = MaterialLayer::new(asset_server.load("grass_albedo.png"))
//!         .with_normal(asset_server.load("grass_normal.png"));
//!
//!     let dirt = MaterialLayer::new(asset_server.load("dirt_albedo.png"))
//!         .with_normal(asset_server.load("dirt_normal.png"));
//!
//!     let rock = MaterialLayer::new(asset_server.load("rock_albedo.png"))
//!         .with_normal(asset_server.load("rock_normal.png"))
//!         .with_pbr(asset_server.load("rock_arm.png"));
//!
//!     // Build the pending material
//!     let pending = SplatMaterialBuilder::new()
//!         .add_layer(grass)
//!         .add_layer(dirt)
//!         .add_layer(rock)
//!         .with_uv_scale(2.0)
//!         .with_triplanar_sharpness(4.0)
//!         .build();
//!
//!     // Spawn mesh with pending material - plugin auto-converts when textures load
//!     commands.spawn((
//!         Mesh3d(meshes.add(create_terrain_mesh())),
//!         pending,
//!     ));
//! }
//! # fn create_terrain_mesh() -> Mesh { todo!() }
//! ```
//!
//! # How It Works
//!
//! 1. You spawn an entity with [`PendingSplatMaterial`] and a [`Mesh3d`]
//! 2. The plugin observes asset load events
//! 3. When all textures for the material are loaded, it:
//!    - Combines individual textures into texture arrays
//!    - Creates the final [`PlumeSplatMaterial`]
//!    - Replaces [`PendingSplatMaterial`] with [`MeshMaterial3d<PlumeSplatMaterial>`]
//! 4. Your mesh renders with the combined material

use std::sync::Arc;

use bevy::asset::Handle;
use bevy::ecs::component::Component;
use bevy::image::Image;
use bevy::pbr::StandardMaterial;

use crate::material::PlumeSplatSettings;

/// A single material layer defining textures for one terrain type.
///
/// Each layer represents a distinct material (e.g., grass, dirt, rock, snow)
/// that can be blended with other layers based on vertex weights.
///
/// # Textures
///
/// - **Albedo** (required): Base color texture, RGB for color, A for optional height
/// - **Normal** (optional): Tangent-space normal map for surface detail
/// - **PBR** (optional): Packed texture with R=Metallic, G=Roughness, B=AO, A=Height
///
/// # Example
///
/// ```rust,no_run
/// use bevy::prelude::*;
/// use plumesplat::MaterialLayer;
///
/// fn create_layers(asset_server: &AssetServer) -> Vec<MaterialLayer> {
///     vec![
///         // Simple layer with just albedo
///         MaterialLayer::new(asset_server.load("grass.png")),
///
///         // Layer with normal map
///         MaterialLayer::new(asset_server.load("dirt_albedo.png"))
///             .with_normal(asset_server.load("dirt_normal.png")),
///
///         // Full PBR layer
///         MaterialLayer::new(asset_server.load("rock_albedo.png"))
///             .with_normal(asset_server.load("rock_normal.png"))
///             .with_pbr(asset_server.load("rock_arm.png")),
///     ]
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MaterialLayer {
    /// Base color texture (required).
    ///
    /// RGB channels contain the diffuse color.
    /// Alpha channel can optionally store height for height-based blending.
    pub albedo: Handle<Image>,

    /// Normal map texture (optional).
    ///
    /// Should be a tangent-space normal map. If not provided,
    /// the geometry normal will be used for this material.
    pub normal: Option<Handle<Image>>,

    /// Packed PBR texture (optional).
    ///
    /// Channel layout:
    /// - R: Metallic (0.0 = dielectric, 1.0 = metal)
    /// - G: Roughness (0.0 = smooth, 1.0 = rough)
    /// - B: Ambient Occlusion (0.0 = occluded, 1.0 = not occluded)
    /// - A: Height (for height-based blending)
    pub pbr: Option<Handle<Image>>,
}

impl MaterialLayer {
    /// Creates a new material layer with the given albedo texture.
    ///
    /// # Arguments
    ///
    /// * `albedo` - Handle to the base color texture
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use bevy::prelude::*;
    /// use plumesplat::MaterialLayer;
    ///
    /// fn setup(asset_server: Res<AssetServer>) {
    ///     let grass = MaterialLayer::new(asset_server.load("grass.png"));
    /// }
    /// ```
    #[must_use]
    pub fn new(albedo: Handle<Image>) -> Self {
        Self {
            albedo,
            normal: None,
            pbr: None,
        }
    }

    /// Adds a normal map to this material layer.
    ///
    /// # Arguments
    ///
    /// * `normal` - Handle to tangent-space normal map texture
    #[must_use]
    pub fn with_normal(mut self, normal: Handle<Image>) -> Self {
        self.normal = Some(normal);
        self
    }

    /// Adds a packed PBR texture to this material layer.
    ///
    /// The texture should have channels packed as:
    /// - R: Metallic
    /// - G: Roughness
    /// - B: Ambient Occlusion
    /// - A: Height (optional, for height-based blending)
    ///
    /// # Arguments
    ///
    /// * `pbr` - Handle to packed PBR texture
    #[must_use]
    pub fn with_pbr(mut self, pbr: Handle<Image>) -> Self {
        self.pbr = Some(pbr);
        self
    }

    /// Returns `true` if this layer has a normal map.
    #[must_use]
    pub fn has_normal(&self) -> bool {
        self.normal.is_some()
    }

    /// Returns `true` if this layer has PBR textures.
    #[must_use]
    pub fn has_pbr(&self) -> bool {
        self.pbr.is_some()
    }

    /// Returns all texture handles for this layer.
    ///
    /// Useful for checking if all textures are loaded.
    pub fn all_handles(&self) -> Vec<&Handle<Image>> {
        let mut handles = vec![&self.albedo];
        if let Some(ref normal) = self.normal {
            handles.push(normal);
        }
        if let Some(ref pbr) = self.pbr {
            handles.push(pbr);
        }
        handles
    }
}

/// Builder for creating [`PendingSplatMaterial`] with a fluent API.
///
/// This builder allows you to configure material layers and settings
/// before spawning the material on an entity.
///
/// # Example
///
/// ```rust,no_run
/// use bevy::prelude::*;
/// use plumesplat::prelude::*;
///
/// fn setup(asset_server: Res<AssetServer>) {
///     let pending = SplatMaterialBuilder::new()
///         .add_layer(MaterialLayer::new(asset_server.load("grass.png")))
///         .add_layer(MaterialLayer::new(asset_server.load("dirt.png")))
///         .add_layer(MaterialLayer::new(asset_server.load("rock.png")))
///         .with_uv_scale(2.0)
///         .with_triplanar_sharpness(4.0)
///         .with_blend_offset(0.1)
///         .build();
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct SplatMaterialBuilder {
    layers: Vec<Arc<MaterialLayer>>,
    settings: PlumeSplatSettings,
    base_material: StandardMaterial,
}

impl SplatMaterialBuilder {
    /// Creates a new empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a material layer to the builder.
    ///
    /// Layers are indexed in order of addition (first layer = index 0).
    /// You can add up to 256 layers.
    ///
    /// # Arguments
    ///
    /// * `layer` - The material layer to add
    ///
    /// # Panics
    ///
    /// Panics if more than 256 layers are added.
    #[must_use]
    pub fn add_layer(mut self, layer: Arc<MaterialLayer>) -> Self {
        assert!(
            self.layers.len() < 256,
            "Cannot add more than 256 material layers"
        );
        self.layers.push(layer);
        self
    }

    /// Adds multiple material layers at once.
    ///
    /// # Arguments
    ///
    /// * `layers` - Iterator of material layers to add
    #[must_use]
    pub fn add_layers(mut self, layers: impl IntoIterator<Item = Arc<MaterialLayer>>) -> Self {
        for layer in layers {
            self = self.add_layer(layer);
        }
        self
    }

    /// Sets the UV scale for texture sampling.
    ///
    /// Higher values increase texture tiling density.
    ///
    /// - **Default**: 1.0
    /// - **Range**: 0.1+
    #[must_use]
    pub fn with_uv_scale(mut self, scale: f32) -> Self {
        self.settings.uv_scale = scale;
        self
    }

    /// Sets the triplanar mapping blend sharpness.
    ///
    /// Controls how sharply the three projection planes blend at edges.
    ///
    /// - **Default**: 4.0
    /// - **Range**: 1.0 (soft) to 16.0 (sharp)
    #[must_use]
    pub fn with_triplanar_sharpness(mut self, sharpness: f32) -> Self {
        self.settings.triplanar_sharpness = sharpness;
        self
    }

    /// Enables height-based material blending.
    ///
    /// When enabled, materials with higher height values "poke through"
    /// at transitions, creating more natural-looking boundaries.
    ///
    /// - **Default**: 0.0 (disabled)
    /// - **Range**: 0.0+ (higher = sharper height transitions)
    #[must_use]
    pub fn with_height_blending(mut self, sharpness: f32) -> Self {
        self.settings.height_blend_sharpness = sharpness;
        self
    }

    /// Sets the blend offset for sharper material transitions.
    ///
    /// Subtracts this value from all weights before normalization,
    /// reducing the influence of materials with low weights.
    ///
    /// - **Default**: 0.0
    /// - **Range**: 0.0 to 0.5
    #[must_use]
    pub fn with_blend_offset(mut self, offset: f32) -> Self {
        self.settings.blend_offset = offset.clamp(0.0, 0.5);
        self
    }

    /// Sets the blend exponent for sharper material transitions.
    ///
    /// Raises weights to this power before normalization.
    ///
    /// - **Default**: 1.0 (linear)
    /// - **Range**: 1.0 to 8.0
    #[must_use]
    pub fn with_blend_exponent(mut self, exponent: f32) -> Self {
        self.settings.blend_exponent = exponent.clamp(1.0, 8.0);
        self
    }

    /// Sets the base [`StandardMaterial`] properties.
    ///
    /// These properties affect the overall material appearance and are
    /// used as defaults when PBR textures aren't provided.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use bevy::prelude::*;
    /// use plumesplat::SplatMaterialBuilder;
    ///
    /// let builder = SplatMaterialBuilder::new()
    ///     .with_base_material(StandardMaterial {
    ///         perceptual_roughness: 0.8,
    ///         reflectance: 0.3,
    ///         ..default()
    ///     });
    /// ```
    #[must_use]
    pub fn with_base_material(mut self, base: StandardMaterial) -> Self {
        self.base_material = base;
        self
    }

    /// Returns the number of layers added so far.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns `true` if any layer has normal maps.
    #[must_use]
    pub fn has_any_normals(&self) -> bool {
        self.layers.iter().any(|layer| layer.has_normal())
    }

    /// Returns `true` if any layer has PBR textures.
    #[must_use]
    pub fn has_any_pbr(&self) -> bool {
        self.layers.iter().any(|layer| layer.has_pbr())
    }

    /// Builds the [`PendingSplatMaterial`] component.
    ///
    /// Spawn this component on an entity with a `Mesh3d` to have the
    /// plugin automatically create the final material when textures load.
    ///
    /// # Panics
    ///
    /// Panics if no layers have been added.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use bevy::prelude::*;
    /// use plumesplat::prelude::*;
    ///
    /// fn setup(
    ///     mut commands: Commands,
    ///     asset_server: Res<AssetServer>,
    ///     mut meshes: ResMut<Assets<Mesh>>,
    /// ) {
    ///     let pending = SplatMaterialBuilder::new()
    ///         .add_layer(MaterialLayer::new(asset_server.load("grass.png")))
    ///         .add_layer(MaterialLayer::new(asset_server.load("dirt.png")))
    ///         .build();
    ///
    ///     commands.spawn((
    ///         Mesh3d(meshes.add(Cuboid::default())),
    ///         pending,
    ///     ));
    /// }
    /// ```
    #[must_use]
    pub fn build(self) -> PendingSplatMaterial {
        assert!(!self.layers.is_empty(), "At least one material layer is required");

        PendingSplatMaterial {
            layers: self.layers,
            settings: self.settings,
            base_material: self.base_material,
        }
    }
}

/// Component marking an entity as having a pending splat material.
///
/// When spawned on an entity with a `Mesh3d`, the plugin will:
///
/// 1. Wait for all referenced textures to load
/// 2. Combine individual textures into texture arrays
/// 3. Create the final `PlumeSplatMaterial`
/// 4. Replace this component with `MeshMaterial3d<PlumeSplatMaterial>`
///
/// # Creating
///
/// Use [`SplatMaterialBuilder`] to create this component:
///
/// ```rust,no_run
/// use bevy::prelude::*;
/// use plumesplat::prelude::*;
///
/// fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
///     let pending = SplatMaterialBuilder::new()
///         .add_layer(MaterialLayer::new(asset_server.load("grass.png")))
///         .add_layer(MaterialLayer::new(asset_server.load("dirt.png")))
///         .build();
///
///     // Spawn with mesh - material will be auto-created when textures load
///     commands.spawn((
///         Mesh3d(Handle::default()),
///         pending,
///     ));
/// }
/// ```
///
/// # Manual Processing
///
/// If you need more control, you can check readiness with [`PendingSplatMaterial::is_ready`]
/// and access the layers directly via the `layers` field.
#[derive(Component, Debug, Clone)]
pub struct PendingSplatMaterial {
    /// Material layers to combine.
    pub layers: Vec<Arc<MaterialLayer>>,
    /// Shader settings.
    pub settings: PlumeSplatSettings,
    /// Base material properties.
    pub base_material: StandardMaterial,
}

impl PendingSplatMaterial {
    /// Returns `true` if all textures for all layers are loaded.
    ///
    /// # Arguments
    ///
    /// * `images` - The image assets to check against
    #[must_use]
    pub fn is_ready(&self, images: &bevy::asset::Assets<Image>) -> bool {
        self.layers
            .iter()
            .flat_map(|layer| layer.all_handles())
            .all(|handle| images.contains(handle.id()))
    }

    /// Returns the total number of texture handles that need to load.
    #[must_use]
    pub fn total_texture_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.all_handles().len())
            .sum()
    }

    /// Returns the number of textures that are currently loaded.
    #[must_use]
    pub fn loaded_texture_count(&self, images: &bevy::asset::Assets<Image>) -> usize {
        self.layers
            .iter()
            .flat_map(|layer| layer.all_handles())
            .filter(|handle| images.contains(handle.id()))
            .count()
    }

    /// Returns the number of layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns `true` if any layer has normal maps.
    #[must_use]
    pub fn has_normals(&self) -> bool {
        self.layers.iter().any(|layer| layer.has_normal())
    }

    /// Returns `true` if any layer has PBR textures.
    #[must_use]
    pub fn has_pbr(&self) -> bool {
        self.layers.iter().any(|layer| layer.has_pbr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::Handle;

    #[test]
    fn test_material_layer_new() {
        let handle = Handle::default();
        let layer = MaterialLayer::new(handle.clone());

        assert!(!layer.has_normal());
        assert!(!layer.has_pbr());
        assert_eq!(layer.all_handles().len(), 1);
    }

    #[test]
    fn test_material_layer_with_all_textures() {
        let albedo = Handle::default();
        let normal = Handle::default();
        let pbr = Handle::default();

        let layer = MaterialLayer::new(albedo)
            .with_normal(normal)
            .with_pbr(pbr);

        assert!(layer.has_normal());
        assert!(layer.has_pbr());
        assert_eq!(layer.all_handles().len(), 3);
    }

    #[test]
    fn test_builder_empty() {
        let builder = SplatMaterialBuilder::new();
        assert_eq!(builder.layer_count(), 0);
        assert!(!builder.has_any_normals());
        assert!(!builder.has_any_pbr());
    }

    #[test]
    fn test_builder_add_layers() {
        let builder = SplatMaterialBuilder::new()
            .add_layer(Arc::new(MaterialLayer::new(Handle::default())))
            .add_layer(Arc::new(MaterialLayer::new(Handle::default())))
            .add_layer(Arc::new(MaterialLayer::new(Handle::default())));

        assert_eq!(builder.layer_count(), 3);
    }

    #[test]
    fn test_builder_settings() {
        let pending = SplatMaterialBuilder::new()
            .add_layer(Arc::new(MaterialLayer::new(Handle::default())))
            .with_uv_scale(2.0)
            .with_triplanar_sharpness(8.0)
            .with_height_blending(0.5)
            .with_blend_offset(0.2)
            .with_blend_exponent(3.0)
            .build();

        assert_eq!(pending.settings.uv_scale, 2.0);
        assert_eq!(pending.settings.triplanar_sharpness, 8.0);
        assert_eq!(pending.settings.height_blend_sharpness, 0.5);
        assert_eq!(pending.settings.blend_offset, 0.2);
        assert_eq!(pending.settings.blend_exponent, 3.0);
    }

    #[test]
    fn test_builder_clamps_values() {
        let pending = SplatMaterialBuilder::new()
            .add_layer(Arc::new(MaterialLayer::new(Handle::default())))
            .with_blend_offset(1.0) // Should clamp to 0.5
            .with_blend_exponent(10.0) // Should clamp to 8.0
            .build();

        assert_eq!(pending.settings.blend_offset, 0.5);
        assert_eq!(pending.settings.blend_exponent, 8.0);
    }

    #[test]
    #[should_panic(expected = "At least one material layer is required")]
    fn test_builder_panics_with_no_layers() {
        let _ = SplatMaterialBuilder::new().build();
    }

    #[test]
    fn test_pending_material_counts() {
        let pending = SplatMaterialBuilder::new()
            .add_layer(Arc::new(
                MaterialLayer::new(Handle::default())
                    .with_normal(Handle::default())
            ))
            .add_layer(Arc::new(
                MaterialLayer::new(Handle::default())
                    .with_normal(Handle::default())
                    .with_pbr(Handle::default())
            ))
            .build();

        assert_eq!(pending.layer_count(), 2);
        assert_eq!(pending.total_texture_count(), 5); // 2 albedo + 2 normal + 1 pbr
        assert!(pending.has_normals());
        assert!(pending.has_pbr());
    }
}
