//! Bevy plugin for PlumeSplat terrain splatting.
//!
//! This module provides the [`PlumeSplatPlugin`] which registers all necessary
//! shaders, materials, and systems with Bevy's rendering pipeline.
//!
//! # Usage
//!
//! Simply add the plugin to your Bevy app:
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use plumesplat::PlumeSplatPlugin;
//!
//! App::new()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugins(PlumeSplatPlugin)
//!     .run();
//! ```
//!
//! # What This Plugin Registers
//!
//! - **Embedded Shaders**: All WGSL shader files are embedded into the binary
//! - **Material Plugin**: Registers [`PlumeSplatMaterial`](crate::PlumeSplatMaterial)
//! - **Processing System**: Automatically converts [`PendingSplatMaterial`](crate::PendingSplatMaterial)
//!   into final materials when textures are loaded

use bevy::app::{App, Plugin, Update};
use bevy::asset::{Assets, RenderAssetUsages, embedded_asset};
use bevy::ecs::entity::Entity;
use bevy::ecs::resource::Resource;
use bevy::ecs::system::{Commands, Query, ResMut};
use bevy::image::{Image, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::pbr::{MaterialPlugin, MeshMaterial3d};
use bevy::render::render_resource::{Extent3d, TextureDataOrder, TextureDimension};

use crate::builder::PendingSplatMaterial;
use crate::material::{PlumeSplatExtension, PlumeSplatMaterial};

/// Bevy plugin that enables PlumeSplat terrain splatting.
///
/// This plugin must be added to your Bevy app to use PlumeSplat materials.
/// It handles:
///
/// - Embedding and registering WGSL shader files
/// - Registering the [`PlumeSplatMaterial`](crate::PlumeSplatMaterial) with Bevy's material system
/// - Processing [`PendingSplatMaterial`](crate::PendingSplatMaterial) components into final materials
///
/// # Example
///
/// ```rust,no_run
/// use bevy::prelude::*;
/// use plumesplat::prelude::*;
///
/// fn main() {
///     App::new()
///         .add_plugins(DefaultPlugins)
///         .add_plugins(PlumeSplatPlugin)
///         .add_systems(Startup, setup)
///         .run();
/// }
///
/// fn setup(
///     mut commands: Commands,
///     asset_server: Res<AssetServer>,
/// ) {
///     // Using the builder API (recommended)
///     let pending = SplatMaterialBuilder::new()
///         .add_layer(MaterialLayer::new(asset_server.load("grass.png")))
///         .add_layer(MaterialLayer::new(asset_server.load("dirt.png")))
///         .build();
///
///     commands.spawn((
///         Mesh3d(Handle::default()),
///         pending,
///     ));
///     // Plugin automatically creates the material when textures load!
/// }
/// ```
///
/// # Shader Files
///
/// The plugin embeds the following shader files:
///
/// - `plumesplat.wgsl` - Base shader support
/// - `plumesplat_ext.wgsl` - Main material extension shader
/// - `triplanar.wgsl` - Triplanar mapping implementation
/// - `blending.wgsl` - Material blending utilities
pub struct PlumeSplatPlugin;

type CacheMat =
    MeshMaterial3d<bevy::pbr::ExtendedMaterial<bevy::pbr::StandardMaterial, PlumeSplatExtension>>;
#[derive(Resource, Default)]
pub struct CacheTable(Option<CacheMat>);

impl Plugin for PlumeSplatPlugin {
    fn build(&self, app: &mut App) {
        // Embed the shader files into the binary for self-contained distribution
        embedded_asset!(app, "shaders/plumesplat.wgsl");
        embedded_asset!(app, "shaders/plumesplat_ext.wgsl");
        embedded_asset!(app, "shaders/triplanar.wgsl");
        embedded_asset!(app, "shaders/blending.wgsl");

        app.init_resource::<CacheTable>();

        // Register the extended material type with Bevy's material system
        app.add_plugins(MaterialPlugin::<PlumeSplatMaterial>::default());

        // Add system to process pending materials when textures are loaded
        app.add_systems(Update, process_pending_materials);
    }
}

/// System that processes [`PendingSplatMaterial`] components.
///
/// When all textures for a pending material are loaded, this system:
/// 1. Combines individual textures into texture arrays
/// 2. Creates the final [`PlumeSplatMaterial`]
/// 3. Removes the [`PendingSplatMaterial`] component
/// 4. Adds [`MeshMaterial3d<PlumeSplatMaterial>`] to the entity
fn process_pending_materials(
    mut commands: Commands,
    query: Query<(Entity, &PendingSplatMaterial)>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<PlumeSplatMaterial>>,
    mut cache_table: ResMut<CacheTable>,
) {
    //let mut extension2: Option<PlumeSplatMaterial> = None;

    for (entity, pending) in query.iter() {
        // Check if all textures are loaded
        if !pending.is_ready(&images) {
            continue;
        }
        if let Some(cached) = &cache_table.0 {
            // Cache hit - use the cached material
            commands
                .entity(entity)
                .remove::<PendingSplatMaterial>()
                .insert(cached.clone());
        //continue;
        } else {
            // Cache miss - process the material and store it in the cache
            // (Processing code will go here)

            // Combine textures into arrays
            let (albedo_array, normal_array, pbr_array) =
                combine_textures_into_arrays(pending, &mut images);

            // Create the extension
            let mut extension = PlumeSplatExtension::new(albedo_array);
            extension.settings = pending.settings;

            if let Some(normal) = normal_array {
                extension.normal_array = Some(normal);
            }
            if let Some(pbr) = pbr_array {
                extension.pbr_array = Some(pbr);
            }

            // Create the final material
            let material = PlumeSplatMaterial {
                base: pending.base_material.clone(),
                extension,
            };

            let material_handle: bevy::asset::Handle<
                bevy::pbr::ExtendedMaterial<bevy::pbr::StandardMaterial, PlumeSplatExtension>,
            > = materials.add(material);
            let mat: MeshMaterial3d<
                bevy::pbr::ExtendedMaterial<bevy::pbr::StandardMaterial, PlumeSplatExtension>,
            > = MeshMaterial3d(material_handle);
            cache_table.0 = Some(mat.clone());
            //Update the entity: remove pending, add final material
            commands
                .entity(entity)
                .remove::<PendingSplatMaterial>()
                .insert(mat);
        }
    }
}

/// Combines individual layer textures into texture arrays.
///
/// Returns (albedo_array, optional_normal_array, optional_pbr_array).
fn combine_textures_into_arrays(
    pending: &PendingSplatMaterial,
    images: &mut Assets<Image>,
) -> (
    bevy::asset::Handle<Image>,
    Option<bevy::asset::Handle<Image>>,
    Option<bevy::asset::Handle<Image>>,
) {
    let num_layers = pending.layer_count() as u32;
    let has_normals = pending.has_normals();
    let has_pbr = pending.has_pbr();

    // Get dimensions from first albedo texture
    let first_albedo = images.get(&pending.layers[0].albedo).unwrap();
    let width = first_albedo.texture_descriptor.size.width;
    let height = first_albedo.texture_descriptor.size.height;

    // Create albedo array
    let albedo_array = create_texture_array(
        pending.layers.iter().map(|l| &l.albedo),
        width,
        height,
        num_layers,
        images,
    );

    // Create normal array if any layer has normals
    let normal_array = if has_normals {
        Some(create_texture_array_with_default(
            pending.layers.iter().map(|l| l.normal.as_ref()),
            width,
            height,
            num_layers,
            images,
            [128, 128, 255, 255], // Default normal pointing up
        ))
    } else {
        None
    };

    // Create PBR array if any layer has PBR
    let pbr_array = if has_pbr {
        Some(create_texture_array_with_default(
            pending.layers.iter().map(|l| l.pbr.as_ref()),
            width,
            height,
            num_layers,
            images,
            [0, 128, 255, 128], // Default: metallic=0, roughness=0.5, ao=1.0, height=0.5
        ))
    } else {
        None
    };

    (albedo_array, normal_array, pbr_array)
}

/// Creates a texture array from a sequence of texture handles.
fn create_texture_array<'a>(
    handles: impl Iterator<Item = &'a bevy::asset::Handle<Image>>,
    width: u32,
    height: u32,
    num_layers: u32,
    images: &mut Assets<Image>,
) -> bevy::asset::Handle<Image> {
    let handles: Vec<_> = handles.collect();
    let mip_count = calculate_mip_count(width, height);

    // Collect all layer data with mipmaps
    let mut array_data = Vec::new();

    for handle in &handles {
        let image = images.get(*handle).unwrap();
        let layer_data = image.data.as_ref().unwrap();

        // Generate mipmaps for this layer
        let mipmapped = generate_mipmaps(layer_data, width, height, mip_count);
        array_data.extend(mipmapped);
    }

    // Create the array image
    let array_image = Image {
        data: Some(array_data),
        texture_descriptor: bevy::render::render_resource::TextureDescriptor {
            label: Some("splat_texture_array"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: num_layers,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
            usage: bevy::render::render_resource::TextureUsages::TEXTURE_BINDING
                | bevy::render::render_resource::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            address_mode_w: ImageAddressMode::ClampToEdge,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        }),
        texture_view_descriptor: None,
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        data_order: TextureDataOrder::LayerMajor,
        copy_on_resize: false,
    };

    images.add(array_image)
}

/// Creates a texture array where missing textures are filled with a default color.
fn create_texture_array_with_default<'a>(
    handles: impl Iterator<Item = Option<&'a bevy::asset::Handle<Image>>>,
    width: u32,
    height: u32,
    num_layers: u32,
    images: &mut Assets<Image>,
    default_color: [u8; 4],
) -> bevy::asset::Handle<Image> {
    let handles: Vec<_> = handles.collect();
    let mip_count = calculate_mip_count(width, height);

    // Create default texture data
    let default_data: Vec<u8> = (0..(width * height)).flat_map(|_| default_color).collect();

    // Collect all layer data with mipmaps
    let mut array_data = Vec::new();

    for maybe_handle in &handles {
        let layer_data = if let Some(handle) = maybe_handle {
            let image = images.get(*handle).unwrap();
            image.data.as_ref().unwrap().clone()
        } else {
            default_data.clone()
        };

        // Generate mipmaps for this layer
        let mipmapped = generate_mipmaps(&layer_data, width, height, mip_count);
        array_data.extend(mipmapped);
    }

    // Create the array image
    let array_image = Image {
        data: Some(array_data),
        texture_descriptor: bevy::render::render_resource::TextureDescriptor {
            label: Some("splat_texture_array"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: num_layers,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
            usage: bevy::render::render_resource::TextureUsages::TEXTURE_BINDING
                | bevy::render::render_resource::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            address_mode_w: ImageAddressMode::ClampToEdge,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        }),
        texture_view_descriptor: None,
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        data_order: TextureDataOrder::LayerMajor,
        copy_on_resize: false,
    };

    images.add(array_image)
}

/// Calculates the number of mip levels for a texture.
fn calculate_mip_count(width: u32, height: u32) -> u32 {
    (width.min(height) as f32).log2().floor() as u32 + 1
}

/// Generates mipmaps for a single texture layer.
fn generate_mipmaps(data: &[u8], width: u32, height: u32, mip_count: u32) -> Vec<u8> {
    let mut result = Vec::new();
    let mut current_data = data.to_vec();
    let mut current_width = width;
    let mut current_height = height;

    for mip in 0..mip_count {
        result.extend_from_slice(&current_data);

        if mip < mip_count - 1 {
            let next_width = (current_width / 2).max(1);
            let next_height = (current_height / 2).max(1);
            current_data = downsample_rgba(
                &current_data,
                current_width,
                current_height,
                next_width,
                next_height,
            );
            current_width = next_width;
            current_height = next_height;
        }
    }

    result
}

/// Downsamples an RGBA image using box filtering.
fn downsample_rgba(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_width * dst_height * 4) as usize];

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let sx = dx * 2;
            let sy = dy * 2;

            let mut r = 0u32;
            let mut g = 0u32;
            let mut b = 0u32;
            let mut a = 0u32;
            let mut count = 0u32;

            for oy in 0..2 {
                for ox in 0..2 {
                    let px = (sx + ox).min(src_width - 1);
                    let py = (sy + oy).min(src_height - 1);
                    let idx = ((py * src_width + px) * 4) as usize;

                    r += src[idx] as u32;
                    g += src[idx + 1] as u32;
                    b += src[idx + 2] as u32;
                    a += src[idx + 3] as u32;
                    count += 1;
                }
            }

            let dst_idx = ((dy * dst_width + dx) * 4) as usize;
            dst[dst_idx] = (r / count) as u8;
            dst[dst_idx + 1] = (g / count) as u8;
            dst[dst_idx + 2] = (b / count) as u8;
            dst[dst_idx + 3] = (a / count) as u8;
        }
    }

    dst
}
