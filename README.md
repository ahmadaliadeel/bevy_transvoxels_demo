# Bevy Transvoxels

A collection of Bevy-based voxel terrain projects demonstrating the Transvoxel algorithm for seamless Level-of-Detail (LOD) transitions.

## Projects

### [BevyBlocks](bevyblocks/README.md)

A real-time voxel terrain demo that renders a procedurally-generated planet using the **Transvoxel algorithm**. Features include:

- **Transvoxel LOD** – 6-level LOD with crack-free transition cells
- **Procedural planet** – Sphere SDF displaced by ridged FBM noise
- **Async mesh generation** – Chunk meshes computed on the thread pool
- **Multi-material terrain splatting** – 4-layer PBR materials blended by height and slope
- **Triplanar mapping** – Eliminates UV seam artifacts on curved surfaces

### [PlumeSplat](plumesplat/README.md)

A shader library for multi-material terrain splatting with triplanar projection and height-based blending.

---

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- GPU with Vulkan, DX12, or Metal support

### Running BevyBlocks

```bash
cd bevyblocks

# Debug build (faster compile, slower runtime)
cargo run

# Release build (recommended for smooth framerate)
cargo run --release
```

### Controls

| Key | Action |
|-----|--------|
| W/A/S/D | Move forward/left/back/right |
| Mouse | Look around |
| F1 | Toggle wireframe |

---

## Dependencies

- [Bevy 0.18](https://bevyengine.org/) – Game engine
- [transvoxel](https://crates.io/crates/transvoxel) – Transvoxel algorithm implementation
- [fastnoise-lite](https://crates.io/crates/fastnoise-lite) – Noise generation
