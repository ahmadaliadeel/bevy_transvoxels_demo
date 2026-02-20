//! Vertex attributes for per-vertex material splatting.
//!
//! This module provides custom vertex attributes and utilities for encoding material
//! data into mesh vertices. Unlike traditional terrain splatting which uses RGBA
//! textures limited to 4 materials, PlumeSplat uses a two-attribute approach that
//! supports blending any 4 materials from a palette of 256.
//!
//! # Architecture
//!
//! Each vertex stores two packed `u32` values:
//!
//! - **Material Indices** ([`ATTRIBUTE_MATERIAL_INDICES`]): Four material indices (0-255),
//!   specifying which materials from the texture array to sample
//! - **Blend Weights** ([`ATTRIBUTE_MATERIAL_WEIGHTS`]): Four blend weights (0-255),
//!   controlling how much each material contributes (normalized in shader)
//!
//! # Data Layout
//!
//! Both attributes pack four `u8` values into a single `u32`:
//!
//! ```text
//! Bit layout: [byte0] [byte1] [byte2] [byte3]
//!             bits 0-7  8-15   16-23   24-31
//!
//! Packed as: byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)
//! ```
//!
//! # Example
//!
//! ```rust
//! use plumesplat::prelude::*;
//!
//! // Create material data for mesh vertices
//! let vertices = vec![
//!     // Single material (100% grass)
//!     MaterialVertex::single(0),
//!
//!     // Blend grass and dirt 50/50
//!     MaterialVertex::blend2(0, 1, 0.5),
//!
//!     // Blend three materials
//!     MaterialVertex::blend3(0, 1, 2, [0.5, 0.3, 0.2]),
//!
//!     // Blend four materials
//!     MaterialVertex::blend4([0, 1, 2, 3], [0.4, 0.3, 0.2, 0.1]),
//! ];
//!
//! // Encode for the GPU
//! let (indices, weights) = encode_material_data(&vertices);
//!
//! // Add to mesh (indices and weights are Vec<u32>)
//! // mesh.insert_attribute(ATTRIBUTE_MATERIAL_INDICES, indices);
//! // mesh.insert_attribute(ATTRIBUTE_MATERIAL_WEIGHTS, weights);
//! ```

use bevy::mesh::MeshVertexAttribute;
use bevy::render::render_resource::VertexFormat;

/// Custom vertex attribute storing 4 material indices packed into a `u32`.
///
/// Each byte represents one material index (0-255), allowing selection from
/// a texture array with up to 256 layers.
///
/// # Packing Format
///
/// ```text
/// u32 = index0 | (index1 << 8) | (index2 << 16) | (index3 << 24)
/// ```
///
/// # Shader Location
///
/// This attribute uses shader location 10 to avoid conflicts with Bevy's
/// built-in vertex attributes.
///
/// # Example
///
/// ```rust
/// use bevy::prelude::*;
/// use plumesplat::prelude::*;
///
/// fn add_material_attributes(mesh: &mut Mesh, material_data: &[MaterialVertex]) {
///     let (indices, weights) = encode_material_data(material_data);
///     mesh.insert_attribute(ATTRIBUTE_MATERIAL_INDICES, indices);
///     mesh.insert_attribute(ATTRIBUTE_MATERIAL_WEIGHTS, weights);
/// }
/// ```
pub const ATTRIBUTE_MATERIAL_INDICES: MeshVertexAttribute =
    MeshVertexAttribute::new("MaterialIndices", 988540910, VertexFormat::Uint32);

/// Custom vertex attribute storing 4 blend weights packed into a `u32`.
///
/// Each byte represents a weight from 0-255, which is normalized to 0.0-1.0
/// in the shader. Weights should ideally sum to 255 for correct blending,
/// though the shader will normalize them regardless.
///
/// # Packing Format
///
/// ```text
/// u32 = weight0 | (weight1 << 8) | (weight2 << 16) | (weight3 << 24)
/// ```
///
/// # Shader Location
///
/// This attribute uses shader location 11 to avoid conflicts with Bevy's
/// built-in vertex attributes.
pub const ATTRIBUTE_MATERIAL_WEIGHTS: MeshVertexAttribute =
    MeshVertexAttribute::new("MaterialWeights", 988540911, VertexFormat::Uint32);

/// Per-vertex material data for terrain splatting.
///
/// This struct represents the material blending information for a single vertex.
/// It stores up to 4 material indices and their corresponding blend weights.
///
/// # Creating MaterialVertex
///
/// Use the provided constructors for common use cases:
///
/// - [`MaterialVertex::single`] - Single material, no blending
/// - [`MaterialVertex::blend2`] - Blend two materials
/// - [`MaterialVertex::blend3`] - Blend three materials
/// - [`MaterialVertex::blend4`] - Blend four materials (full control)
/// - [`MaterialVertex::new`] - Direct construction with arrays
///
/// # Weight Normalization
///
/// Weights are stored as `u8` values (0-255) and normalized to 0.0-1.0 in the shader.
/// The constructors automatically ensure weights sum to 255 for correct blending.
///
/// # Example
///
/// ```rust
/// use plumesplat::MaterialVertex;
///
/// // Single material (grass only)
/// let grass = MaterialVertex::single(0);
/// assert_eq!(grass.indices[0], 0);
/// assert_eq!(grass.weights[0], 255);
///
/// // 50/50 blend of grass and dirt
/// let transition = MaterialVertex::blend2(0, 1, 0.5);
///
/// // Custom 4-way blend
/// let complex = MaterialVertex::blend4(
///     [0, 1, 2, 3],           // grass, dirt, rock, snow
///     [0.4, 0.3, 0.2, 0.1],   // weights (will be normalized)
/// );
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaterialVertex {
    /// Material indices (0-255 each) referencing layers in the texture array.
    ///
    /// Up to 4 materials can be blended per vertex. Unused slots should be 0.
    pub indices: [u8; 4],

    /// Blend weights for each material (0-255 each).
    ///
    /// Weights are normalized in the shader, so they don't need to sum to
    /// exactly 255, but the constructors ensure this for precision.
    pub weights: [u8; 4],
}

impl MaterialVertex {
    /// Creates a new material vertex with explicit indices and weights.
    ///
    /// For most use cases, prefer the convenience constructors like
    /// [`single`](Self::single), [`blend2`](Self::blend2), etc.
    ///
    /// # Arguments
    ///
    /// * `indices` - Four material indices (0-255 each)
    /// * `weights` - Four blend weights (0-255 each, should sum to ~255)
    ///
    /// # Example
    ///
    /// ```rust
    /// use plumesplat::MaterialVertex;
    ///
    /// let v = MaterialVertex::new([0, 1, 2, 3], [100, 80, 50, 25]);
    /// ```
    pub fn new(indices: [u8; 4], weights: [u8; 4]) -> Self {
        Self { indices, weights }
    }

    /// Creates a material vertex using a single material (no blending).
    ///
    /// This is the most common case for areas that should display only
    /// one material type.
    ///
    /// # Arguments
    ///
    /// * `index` - Material index in the texture array (0-255)
    ///
    /// # Example
    ///
    /// ```rust
    /// use plumesplat::MaterialVertex;
    ///
    /// let grass = MaterialVertex::single(0);  // 100% material 0
    /// let rock = MaterialVertex::single(2);   // 100% material 2
    /// ```
    pub fn single(index: u8) -> Self {
        Self {
            indices: [index, 0, 0, 0],
            weights: [255, 0, 0, 0],
        }
    }

    /// Creates a material vertex blending two materials.
    ///
    /// This is useful for transition zones between two material types.
    ///
    /// # Arguments
    ///
    /// * `index0` - First material index
    /// * `index1` - Second material index
    /// * `weight0` - Weight of first material (0.0-1.0), second material gets `1.0 - weight0`
    ///
    /// # Example
    ///
    /// ```rust
    /// use plumesplat::MaterialVertex;
    ///
    /// // 70% grass, 30% dirt
    /// let transition = MaterialVertex::blend2(0, 1, 0.7);
    ///
    /// // 50/50 blend
    /// let half = MaterialVertex::blend2(0, 1, 0.5);
    /// ```
    pub fn blend2(index0: u8, index1: u8, weight0: f32) -> Self {
        let w0 = (weight0.clamp(0.0, 1.0) * 255.0) as u8;
        let w1 = 255 - w0;
        Self {
            indices: [index0, index1, 0, 0],
            weights: [w0, w1, 0, 0],
        }
    }

    /// Creates a material vertex blending three materials.
    ///
    /// Weights are normalized automatically so they sum to 1.0.
    ///
    /// # Arguments
    ///
    /// * `index0`, `index1`, `index2` - Material indices
    /// * `weights` - Relative weights for each material (will be normalized)
    ///
    /// # Example
    ///
    /// ```rust
    /// use plumesplat::MaterialVertex;
    ///
    /// // Blend grass, dirt, and rock
    /// let v = MaterialVertex::blend3(0, 1, 2, [0.5, 0.3, 0.2]);
    /// ```
    pub fn blend3(index0: u8, index1: u8, index2: u8, weights: [f32; 3]) -> Self {
        let total = weights[0] + weights[1] + weights[2];
        let normalized = if total > 0.0 {
            [weights[0] / total, weights[1] / total, weights[2] / total]
        } else {
            [1.0, 0.0, 0.0]
        };
        let w0 = (normalized[0] * 255.0) as u8;
        let w1 = (normalized[1] * 255.0) as u8;
        // Ensure weights sum to exactly 255 by computing w2 from remainder
        let w2 = 255u8.saturating_sub(w0).saturating_sub(w1);
        Self {
            indices: [index0, index1, index2, 0],
            weights: [w0, w1, w2, 0],
        }
    }

    /// Creates a material vertex blending four materials.
    ///
    /// This provides full control over all 4 material slots. Weights are
    /// normalized automatically so they sum to 1.0.
    ///
    /// # Arguments
    ///
    /// * `indices` - Four material indices (0-255 each)
    /// * `weights` - Relative weights for each material (will be normalized)
    ///
    /// # Example
    ///
    /// ```rust
    /// use plumesplat::MaterialVertex;
    ///
    /// // Blend grass, dirt, rock, and snow based on terrain analysis
    /// let complex_blend = MaterialVertex::blend4(
    ///     [0, 1, 2, 3],
    ///     [0.4, 0.3, 0.2, 0.1],
    /// );
    /// ```
    pub fn blend4(indices: [u8; 4], weights: [f32; 4]) -> Self {
        let total: f32 = weights.iter().sum();
        let normalized: [f32; 4] = if total > 0.0 {
            [
                weights[0] / total,
                weights[1] / total,
                weights[2] / total,
                weights[3] / total,
            ]
        } else {
            [1.0, 0.0, 0.0, 0.0]
        };

        let w0 = (normalized[0] * 255.0) as u8;
        let w1 = (normalized[1] * 255.0) as u8;
        let w2 = (normalized[2] * 255.0) as u8;
        // Ensure weights sum to exactly 255 by computing w3 from remainder
        let w3 = 255u8
            .saturating_sub(w0)
            .saturating_sub(w1)
            .saturating_sub(w2);

        Self {
            indices,
            weights: [w0, w1, w2, w3],
        }
    }

    /// Packs material indices into a `u32` for the vertex attribute.
    ///
    /// # Returns
    ///
    /// A `u32` with indices packed as: `i0 | (i1 << 8) | (i2 << 16) | (i3 << 24)`
    ///
    /// # Example
    ///
    /// ```rust
    /// use plumesplat::MaterialVertex;
    ///
    /// let v = MaterialVertex::new([1, 2, 3, 4], [255, 0, 0, 0]);
    /// let packed = v.packed_indices();
    /// assert_eq!(packed, 0x04030201);
    /// ```
    #[inline]
    pub fn packed_indices(&self) -> u32 {
        pack_u8x4(self.indices)
    }

    /// Packs blend weights into a `u32` for the vertex attribute.
    ///
    /// # Returns
    ///
    /// A `u32` with weights packed as: `w0 | (w1 << 8) | (w2 << 16) | (w3 << 24)`
    #[inline]
    pub fn packed_weights(&self) -> u32 {
        pack_u8x4(self.weights)
    }

    /// Returns the sum of all weights.
    ///
    /// For properly constructed vertices, this should be 255.
    ///
    /// # Example
    ///
    /// ```rust
    /// use plumesplat::MaterialVertex;
    ///
    /// let v = MaterialVertex::blend2(0, 1, 0.5);
    /// assert_eq!(v.weight_sum(), 255);
    /// ```
    #[inline]
    pub fn weight_sum(&self) -> u16 {
        self.weights.iter().map(|&w| w as u16).sum()
    }

    /// Returns the number of active materials (non-zero weights).
    ///
    /// # Example
    ///
    /// ```rust
    /// use plumesplat::MaterialVertex;
    ///
    /// let single = MaterialVertex::single(0);
    /// assert_eq!(single.active_material_count(), 1);
    ///
    /// let blend = MaterialVertex::blend2(0, 1, 0.5);
    /// assert_eq!(blend.active_material_count(), 2);
    /// ```
    #[inline]
    pub fn active_material_count(&self) -> usize {
        self.weights.iter().filter(|&&w| w > 0).count()
    }
}

/// Encodes material data for a slice of vertices into packed `u32` vectors.
///
/// This is the primary function for preparing vertex data for the GPU.
/// The returned vectors can be directly inserted into a Bevy mesh.
///
/// # Arguments
///
/// * `vertices` - Slice of [`MaterialVertex`] data for each mesh vertex
///
/// # Returns
///
/// A tuple of `(indices, weights)` where:
/// - `indices`: `Vec<u32>` of packed material indices
/// - `weights`: `Vec<u32>` of packed blend weights
///
/// # Example
///
/// ```rust
/// use bevy::prelude::*;
/// use plumesplat::prelude::*;
///
/// fn create_mesh_with_materials() -> Mesh {
///     let mut mesh = Mesh::new(
///         bevy::mesh::PrimitiveTopology::TriangleList,
///         bevy::asset::RenderAssetUsages::RENDER_WORLD,
///     );
///
///     // ... add positions, normals, etc ...
///
///     // Create material data for each vertex
///     let material_data = vec![
///         MaterialVertex::single(0),
///         MaterialVertex::blend2(0, 1, 0.5),
///         MaterialVertex::single(1),
///     ];
///
///     // Encode and add to mesh
///     let (indices, weights) = encode_material_data(&material_data);
///     mesh.insert_attribute(ATTRIBUTE_MATERIAL_INDICES, indices);
///     mesh.insert_attribute(ATTRIBUTE_MATERIAL_WEIGHTS, weights);
///
///     mesh
/// }
/// ```
pub fn encode_material_data(vertices: &[MaterialVertex]) -> (Vec<u32>, Vec<u32>) {
    let indices: Vec<u32> = vertices.iter().map(|v| v.packed_indices()).collect();
    let weights: Vec<u32> = vertices.iter().map(|v| v.packed_weights()).collect();
    (indices, weights)
}

/// Packs four `u8` values into a single `u32`.
///
/// # Layout
///
/// ```text
/// Result = values[0] | (values[1] << 8) | (values[2] << 16) | (values[3] << 24)
/// ```
///
/// # Example
///
/// ```rust
/// use plumesplat::pack_u8x4;
///
/// let packed = pack_u8x4([0x12, 0x34, 0x56, 0x78]);
/// assert_eq!(packed, 0x78563412);
/// ```
#[inline]
pub fn pack_u8x4(values: [u8; 4]) -> u32 {
    (values[0] as u32)
        | ((values[1] as u32) << 8)
        | ((values[2] as u32) << 16)
        | ((values[3] as u32) << 24)
}

/// Unpacks a `u32` into four `u8` values.
///
/// This is the inverse of [`pack_u8x4`].
///
/// # Example
///
/// ```rust
/// use plumesplat::{pack_u8x4, unpack_u8x4};
///
/// let original = [0x12, 0x34, 0x56, 0x78];
/// let packed = pack_u8x4(original);
/// let unpacked = unpack_u8x4(packed);
/// assert_eq!(original, unpacked);
/// ```
#[inline]
pub fn unpack_u8x4(packed: u32) -> [u8; 4] {
    [
        (packed & 0xFF) as u8,
        ((packed >> 8) & 0xFF) as u8,
        ((packed >> 16) & 0xFF) as u8,
        ((packed >> 24) & 0xFF) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===================
    // MaterialVertex tests
    // ===================

    #[test]
    fn test_single_material() {
        let v = MaterialVertex::single(42);
        assert_eq!(v.indices[0], 42);
        assert_eq!(v.weights[0], 255);
        assert_eq!(v.indices[1], 0);
        assert_eq!(v.weights[1], 0);
        assert_eq!(v.weight_sum(), 255);
        assert_eq!(v.active_material_count(), 1);
    }

    #[test]
    fn test_single_material_max_index() {
        let v = MaterialVertex::single(255);
        assert_eq!(v.indices[0], 255);
        assert_eq!(v.weights[0], 255);
    }

    #[test]
    fn test_blend2_even() {
        let v = MaterialVertex::blend2(10, 20, 0.5);
        assert_eq!(v.indices[0], 10);
        assert_eq!(v.indices[1], 20);
        assert_eq!(v.weight_sum(), 255);
        assert_eq!(v.active_material_count(), 2);
        // 0.5 * 255 = 127.5, truncates to 127
        assert_eq!(v.weights[0], 127);
        assert_eq!(v.weights[1], 128); // 255 - 127 = 128
    }

    #[test]
    fn test_blend2_extreme_values() {
        // weight0 = 0.0 means 100% second material
        let v = MaterialVertex::blend2(0, 1, 0.0);
        assert_eq!(v.weights[0], 0);
        assert_eq!(v.weights[1], 255);

        // weight0 = 1.0 means 100% first material
        let v = MaterialVertex::blend2(0, 1, 1.0);
        assert_eq!(v.weights[0], 255);
        assert_eq!(v.weights[1], 0);
    }

    #[test]
    fn test_blend2_clamping() {
        // Values outside 0-1 should be clamped
        let v = MaterialVertex::blend2(0, 1, -0.5);
        assert_eq!(v.weights[0], 0);
        assert_eq!(v.weights[1], 255);

        let v = MaterialVertex::blend2(0, 1, 1.5);
        assert_eq!(v.weights[0], 255);
        assert_eq!(v.weights[1], 0);
    }

    #[test]
    fn test_blend3() {
        let v = MaterialVertex::blend3(0, 1, 2, [0.5, 0.3, 0.2]);
        assert_eq!(v.indices, [0, 1, 2, 0]);
        assert_eq!(v.weight_sum(), 255);
        assert_eq!(v.active_material_count(), 3);
    }

    #[test]
    fn test_blend3_zero_weights() {
        // All zero weights should default to 100% first material
        let v = MaterialVertex::blend3(5, 6, 7, [0.0, 0.0, 0.0]);
        assert_eq!(v.weights[0], 255);
        assert_eq!(v.weights[1], 0);
        assert_eq!(v.weights[2], 0);
    }

    #[test]
    fn test_blend4() {
        let v = MaterialVertex::blend4([0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25]);
        assert_eq!(v.indices, [0, 1, 2, 3]);
        assert_eq!(v.weight_sum(), 255);
        assert_eq!(v.active_material_count(), 4);
    }

    #[test]
    fn test_blend4_unequal_weights() {
        let v = MaterialVertex::blend4([0, 1, 2, 3], [0.4, 0.3, 0.2, 0.1]);
        assert_eq!(v.weight_sum(), 255);
        // Verify relative ordering is preserved
        assert!(v.weights[0] > v.weights[1]);
        assert!(v.weights[1] > v.weights[2]);
        assert!(v.weights[2] > v.weights[3]);
    }

    #[test]
    fn test_blend4_zero_weights() {
        let v = MaterialVertex::blend4([0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(v.weights[0], 255);
        assert_eq!(v.weight_sum(), 255);
    }

    #[test]
    fn test_new_constructor() {
        let v = MaterialVertex::new([10, 20, 30, 40], [100, 50, 75, 30]);
        assert_eq!(v.indices, [10, 20, 30, 40]);
        assert_eq!(v.weights, [100, 50, 75, 30]);
    }

    // ===================
    // Packing tests
    // ===================

    #[test]
    fn test_pack_unpack_roundtrip() {
        let values = [1, 2, 3, 4];
        let packed = pack_u8x4(values);
        let unpacked = unpack_u8x4(packed);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_unpack_zeros() {
        let values = [0, 0, 0, 0];
        let packed = pack_u8x4(values);
        assert_eq!(packed, 0);
        assert_eq!(unpack_u8x4(packed), values);
    }

    #[test]
    fn test_pack_unpack_max_values() {
        let values = [255, 255, 255, 255];
        let packed = pack_u8x4(values);
        assert_eq!(packed, 0xFFFFFFFF);
        assert_eq!(unpack_u8x4(packed), values);
    }

    #[test]
    fn test_pack_individual_bytes() {
        // Test that each byte goes to the correct position
        assert_eq!(pack_u8x4([0xFF, 0, 0, 0]), 0x000000FF);
        assert_eq!(pack_u8x4([0, 0xFF, 0, 0]), 0x0000FF00);
        assert_eq!(pack_u8x4([0, 0, 0xFF, 0]), 0x00FF0000);
        assert_eq!(pack_u8x4([0, 0, 0, 0xFF]), 0xFF000000);
    }

    #[test]
    fn test_material_vertex_packing() {
        let v = MaterialVertex::new([10, 20, 30, 40], [100, 50, 75, 30]);
        let packed_indices = v.packed_indices();
        let packed_weights = v.packed_weights();

        assert_eq!(unpack_u8x4(packed_indices), [10, 20, 30, 40]);
        assert_eq!(unpack_u8x4(packed_weights), [100, 50, 75, 30]);
    }

    // ===================
    // encode_material_data tests
    // ===================

    #[test]
    fn test_encode_material_data_empty() {
        let (indices, weights) = encode_material_data(&[]);
        assert!(indices.is_empty());
        assert!(weights.is_empty());
    }

    #[test]
    fn test_encode_material_data_single() {
        let vertices = vec![MaterialVertex::single(5)];
        let (indices, weights) = encode_material_data(&vertices);

        assert_eq!(indices.len(), 1);
        assert_eq!(weights.len(), 1);
        assert_eq!(unpack_u8x4(indices[0]), [5, 0, 0, 0]);
        assert_eq!(unpack_u8x4(weights[0]), [255, 0, 0, 0]);
    }

    #[test]
    fn test_encode_material_data_multiple() {
        let vertices = vec![
            MaterialVertex::single(0),
            MaterialVertex::blend2(1, 2, 0.5),
            MaterialVertex::single(3),
        ];
        let (indices, weights) = encode_material_data(&vertices);

        assert_eq!(indices.len(), 3);
        assert_eq!(weights.len(), 3);

        // First vertex: single material 0
        assert_eq!(unpack_u8x4(indices[0])[0], 0);
        assert_eq!(unpack_u8x4(weights[0])[0], 255);

        // Second vertex: blend of 1 and 2
        assert_eq!(unpack_u8x4(indices[1])[0], 1);
        assert_eq!(unpack_u8x4(indices[1])[1], 2);

        // Third vertex: single material 3
        assert_eq!(unpack_u8x4(indices[2])[0], 3);
    }

    // ===================
    // Edge cases and properties
    // ===================

    #[test]
    fn test_default_material_vertex() {
        let v = MaterialVertex::default();
        assert_eq!(v.indices, [0, 0, 0, 0]);
        assert_eq!(v.weights, [0, 0, 0, 0]);
    }

    #[test]
    fn test_material_vertex_equality() {
        let v1 = MaterialVertex::single(5);
        let v2 = MaterialVertex::single(5);
        let v3 = MaterialVertex::single(6);

        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_material_vertex_clone() {
        let v1 = MaterialVertex::blend2(1, 2, 0.7);
        let v2 = v1.clone();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_all_blend_functions_sum_to_255() {
        // Verify invariant that all constructors produce weights summing to 255
        assert_eq!(MaterialVertex::single(0).weight_sum(), 255);
        assert_eq!(MaterialVertex::blend2(0, 1, 0.3).weight_sum(), 255);
        assert_eq!(MaterialVertex::blend2(0, 1, 0.7).weight_sum(), 255);
        assert_eq!(MaterialVertex::blend3(0, 1, 2, [0.1, 0.2, 0.7]).weight_sum(), 255);
        assert_eq!(MaterialVertex::blend4([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]).weight_sum(), 255);
    }
}
