//! # PlumeSplat
//!
//! Advanced terrain and mesh splatting for [Bevy Engine](https://bevyengine.org) with support
//! for up to 256 materials in a single draw call.
//!
//! PlumeSplat extends Bevy's `StandardMaterial` with texture array splatting, enabling
//! complex multi-material terrains with full PBR lighting support.
//!
//! ## Features
//!
//! - **256 Material Support**: Blend up to 256 materials using texture arrays and per-vertex indices
//! - **Single Draw Call**: All materials rendered efficiently in one draw call
//! - **Full PBR Integration**: Extends `StandardMaterial` for proper lighting, shadows, and reflections
//! - **Triplanar Mapping**: UV-less texturing that works on arbitrary geometry without seams
//! - **Normal Maps**: Per-material normal mapping for detailed surfaces
//! - **Packed PBR Textures**: Support for metallic, roughness, ambient occlusion, and height maps
//! - **Stochastic Tiling**: Eliminates visible texture repetition patterns
//! - **Height-Based Blending**: Natural material transitions based on height maps
//! - **Builder API**: Ergonomic API for creating materials from individual textures
//!
//! ## Quick Start (Builder API - Recommended)
//!
//! The easiest way to use PlumeSplat is with the builder API:
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use plumesplat::prelude::*;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(PlumeSplatPlugin)
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//!
//! fn setup(
//!     mut commands: Commands,
//!     mut meshes: ResMut<Assets<Mesh>>,
//!     asset_server: Res<AssetServer>,
//! ) {
//!     // Define material layers
//!     let grass = MaterialLayer::new(asset_server.load("grass.png"));
//!     let dirt = MaterialLayer::new(asset_server.load("dirt.png"));
//!     let rock = MaterialLayer::new(asset_server.load("rock.png"))
//!         .with_normal(asset_server.load("rock_normal.png"));
//!
//!     // Build the pending material
//!     let pending = SplatMaterialBuilder::new()
//!         .add_layer(grass)
//!         .add_layer(dirt)
//!         .add_layer(rock)
//!         .with_uv_scale(2.0)
//!         .build();
//!
//!     // Spawn with mesh - material auto-creates when textures load
//!     commands.spawn((
//!         Mesh3d(meshes.add(Cuboid::default())),
//!         pending,
//!     ));
//! }
//! ```
//!
//! ## Vertex Attributes
//!
//! Meshes must include custom vertex attributes specifying material indices and blend weights:
//!
//! ```rust
//! use plumesplat::prelude::*;
//!
//! // Create per-vertex material data
//! let vertices = vec![
//!     MaterialVertex::single(0),           // 100% material 0
//!     MaterialVertex::blend2(0, 1, 0.5),   // 50% material 0, 50% material 1
//!     MaterialVertex::blend4([0, 1, 2, 3], [0.4, 0.3, 0.2, 0.1]),
//! ];
//!
//! // Encode for the GPU
//! let (indices, weights) = encode_material_data(&vertices);
//!
//! // Add to your mesh
//! // mesh.insert_attribute(ATTRIBUTE_MATERIAL_INDICES, indices);
//! // mesh.insert_attribute(ATTRIBUTE_MATERIAL_WEIGHTS, weights);
//! ```
//!
//! ## Key Types
//!
//! - [`PlumeSplatPlugin`]: The Bevy plugin that registers shaders and the material
//! - [`PlumeSplatMaterial`]: The extended material type for terrain splatting
//! - [`PlumeSplatExtension`]: Material extension with texture arrays and settings
//! - [`MaterialVertex`]: Per-vertex material data for blending
//! - [`SplatMaterialBuilder`]: Ergonomic builder API for creating materials

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]

mod builder;
mod material;
mod plugin;
mod vertex_attributes;

/// Convenient imports for common PlumeSplat types.
///
/// # Usage
///
/// ```rust
/// use plumesplat::prelude::*;
/// ```
///
/// This imports:
/// - [`PlumeSplatPlugin`] - The Bevy plugin
/// - [`PlumeSplatMaterial`] - Type alias for the extended material
/// - [`PlumeSplatExtension`] - The material extension with texture arrays
/// - [`PlumeSplatSettings`] - Shader uniform settings
/// - [`MaterialVertex`] - Per-vertex material data
/// - [`ATTRIBUTE_MATERIAL_INDICES`] - Vertex attribute for material indices
/// - [`ATTRIBUTE_MATERIAL_WEIGHTS`] - Vertex attribute for blend weights
/// - [`encode_material_data`] - Helper to encode vertex data for the GPU
/// - [`MaterialLayer`] - Single material layer definition
/// - [`SplatMaterialBuilder`] - Builder for creating materials
/// - [`PendingSplatMaterial`] - Component for pending material processing
pub mod prelude {
    pub use crate::builder::{MaterialLayer, PendingSplatMaterial, SplatMaterialBuilder};
    pub use crate::material::{PlumeSplatExtension, PlumeSplatMaterial, PlumeSplatSettings};
    pub use crate::plugin::PlumeSplatPlugin;
    pub use crate::vertex_attributes::{
        encode_material_data, MaterialVertex, ATTRIBUTE_MATERIAL_INDICES,
        ATTRIBUTE_MATERIAL_WEIGHTS,
    };
}

// Re-export all public items at the crate root for convenience
pub use builder::*;
pub use material::*;
pub use plugin::*;
pub use vertex_attributes::*;
