# BevyBlocks – Transvoxel LOD Sphere Demo

A real-time voxel terrain demo built with [Bevy 0.18](https://bevyengine.org/) that renders a procedurally-generated planet using the **Transvoxel algorithm** for seamless Level-of-Detail (LOD) transitions. The surface is displaced with FBM noise and textured via a multi-material splatting shader.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Density Field](#density-field)
- [Chunk Grid & Coordinate System](#chunk-grid--coordinate-system)
- [LOD System](#lod-system)
  - [Distance-Based LOD Selection](#distance-based-lod-selection)
  - [Constraint Propagation](#constraint-propagation)
  - [Transition Cells](#transition-cells)
- [Async Mesh Generation Pipeline](#async-mesh-generation-pipeline)
  - [Batch Dispatch](#batch-dispatch)
  - [Atomic Batch Application](#atomic-batch-application)
- [Material System](#material-system)
  - [Height & Slope Blending](#height--slope-blending)
  - [PlumeSplat Integration](#plumesplat-integration)
- [ECS Architecture](#ecs-architecture)
  - [Components](#components)
  - [Resources](#resources)
  - [Systems](#systems)
- [Controls](#controls)
- [Dependencies](#dependencies)
- [Building & Running](#building--running)
- [Project Structure](#project-structure)

---

## Features

- **Transvoxel LOD** – 6-level LOD with crack-free transition cells between chunks of differing resolution
- **Procedural planet** – Sphere SDF displaced by ridged FBM noise (OpenSimplex2, 5 octaves)
- **Async mesh generation** – Chunk meshes computed on Bevy's `AsyncComputeTaskPool`; atomic batch application prevents visual cracks
- **Multi-material terrain splatting** – 4-layer PBR materials (grass, dirt, rock, snow) blended by height and slope via the `plumesplat` shader library
- **Triplanar mapping** – Eliminates UV seam artifacts on steep and curved surfaces
- **FPS camera** – WASD + mouse-look flight controller
- **Wireframe toggle** – Press F1 to visualise the mesh topology

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                        main()                            │
│  App::new() ─► DefaultPlugins + WireframePlugin          │
│                + PlumeSplatPlugin                         │
│  Resources:  ChunkManager, LodMap                        │
│  Startup:    setup()                                     │
│  Update:     fps_camera_system                           │
│              update_lod_system                            │
│              handle_mesh_tasks                            │
│              toggle_wireframe                             │
└──────────────────────────────────────────────────────────┘

Frame Loop:
  ┌─────────────────────┐
  │ fps_camera_system   │  Move camera (WASD + mouse)
  └────────┬────────────┘
           ▼
  ┌─────────────────────┐
  │ update_lod_system   │  Recompute LOD map → enforce constraints
  │                     │  → compute transitions → dispatch dirty
  │                     │    chunks as async tasks
  └────────┬────────────┘
           ▼
  ┌─────────────────────┐
  │ handle_mesh_tasks   │  Poll thread pool → buffer results
  │                     │  → when full batch ready, swap all
  │                     │    meshes atomically
  └─────────────────────┘
```

---

## Density Field

The terrain is defined by a scalar density function evaluated at every grid node:

```
density(x, y, z) = sphere(x, y, z) + fbm(x, y, z) * (NOISE_AMPLITUDE / SPHERE_RADIUS)
```

| Component | Formula | Description |
|-----------|---------|-------------|
| **Sphere SDF** | `1.0 - length(p) / R` | Positive inside the sphere (R = 1000 units) |
| **FBM noise** | Ridged OpenSimplex2, 5 octaves | Adds mountains/ridges; amplitude = 70 units |

The noise generator is initialised once via `LazyLock<FastNoiseLite>` and is thread-safe for parallel chunk extraction.

**Key constants:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `SPHERE_RADIUS` | 1000.0 | Base planet radius |
| `NOISE_AMPLITUDE` | 70.0 | Max surface displacement |
| `NOISE_FREQUENCY` | 0.0025 | Spatial frequency of terrain features |
| `CHUNK_SIZE` | 100.0 | World-space extent of one chunk |

---

## Chunk Grid & Coordinate System

The world is partitioned into a regular 3D grid of cubic chunks, each `100 × 100 × 100` world units. Chunks are addressed by integer coordinates `IVec3` ranging from `(-10, -10, -10)` to `(10, 10, 10)` — a total of **9,261 chunks** (21³).

```
World position = coord * CHUNK_SIZE
Chunk center   = coord * CHUNK_SIZE + CHUNK_SIZE / 2
```

Each chunk maps to exactly one Bevy entity with a `Mesh3d` handle and a `VoxelChunk` marker component.

---

## LOD System

### Distance-Based LOD Selection

Each frame, every chunk is assigned a **desired LOD** (0 = highest detail, 5 = lowest) based on camera distance to the chunk center:

| LOD Level | Subdivisions | Distance Threshold |
|-----------|-------------:|-------------------:|
| 0 | 320 | < 5.0 |
| 1 | 160 | < 21.0 |
| 2 | 80 | < 100.7 |
| 3 | 40 | < 260.4 |
| 4 | 20 | < 320.2 |
| 5 | 10 | ≥ 320.2 |

The subdivision count per LOD is computed as `base * SUBDIVISIONS_MULT` where `SUBDIVISIONS_MULT = 5`, giving a 2× geometric progression from LOD 5 up to LOD 0.

### Constraint Propagation

After computing desired LODs, a **constraint propagation pass** ensures that no two 6-connected neighbours differ by more than 1 LOD level. This is required by the Transvoxel algorithm to guarantee watertight transitions.

The algorithm iterates over all chunk pairs until no further changes are needed:

```
loop {
    for each chunk C with lod L:
        for each 6-connected neighbour N with lod M:
            if L > M + 1  →  set C.lod = M + 1
            if M > L + 1  →  set N.lod = L + 1
    break when stable
}
```

### Transition Cells

Once the LOD map is stable, each chunk computes which of its 6 faces need **transition cells** — special Transvoxel geometry that bridges the resolution gap to a higher-detail neighbour.

A face gets a transition cell when the neighbour in that direction has a *lower* LOD index (i.e., higher resolution). The 6 transition sides map directly to the `TransitionSide` flags from the `transvoxel` crate:

| Direction | TransitionSide flag |
|-----------|-------------------|
| +X | `HighX` |
| −X | `LowX` |
| +Y | `HighY` |
| −Y | `LowY` |
| +Z | `HighZ` |
| −Z | `LowZ` |

---

## Async Mesh Generation Pipeline

Mesh extraction is the most expensive operation and is fully offloaded to Bevy's `AsyncComputeTaskPool`.

### Batch Dispatch

1. `update_lod_system` computes the target LOD map and finds all **dirty chunks** (chunks whose committed LOD or transition sides differ from the new target).
2. A new **batch** is created: every dirty chunk gets a `ComputeChunkMesh` task entity spawned with a `Task<Option<Mesh>>`.
3. No new batch is dispatched while the previous one is in flight.

Each task calls `generate_chunk_mesh(coord, lod, transitions)` which:
- Creates a `transvoxel::Block` with the chunk's world-space origin, size, and subdivision count
- Calls `extract_from_field()` with the density function and transition sides
- Converts the resulting positions, normals, and triangle indices into a Bevy `Mesh`
- Computes per-vertex material blend data (indices + weights)

### Atomic Batch Application

`handle_mesh_tasks` polls all task entities each frame:
- Completed tasks are moved into `ChunkManager::batch_results`
- Stale tasks (from a superseded generation) are discarded
- **Only when every chunk in the batch has completed** are all meshes swapped simultaneously

This atomic swap is critical: partially updating meshes would show cracks at LOD boundaries where one chunk has updated but its neighbour hasn't yet.

```
Generation counter: prevents stale task results from corrupting the current batch.
```

---

## Material System

### Height & Slope Blending

Each vertex receives a 4-material blend computed from its position on the planet:

| Layer | Index | Height Band | Notes |
|-------|------:|-------------|-------|
| Grass | 0 | 0.0 – 0.4 | Reduced on steep slopes |
| Dirt | 1 | 0.2 – 0.6 | Transition material |
| Rock | 2 | 0.4 – 0.9 | Amplified on steep slopes |
| Snow | 3 | 0.65 – 1.0 | Reduced on steep slopes |

**Height** is measured as `|length(position) - SPHERE_RADIUS|`, normalised against a max height of 40 units.

**Slope** is derived from the angle between the vertex normal and the radial direction (`acos(dot(normalize(p), n))`), scaled by 0.2.

Blending uses hermite `smoothstep` with configurable fade zones for each band, then slope modulates the weights (rocky surfaces on cliffs, less grass/snow on steep terrain).

The final blend is packed into two `u32` vertex attributes:
- **`ATTRIBUTE_MATERIAL_INDICES`** – 4 material layer indices packed into one `u32` (8 bits each)
- **`ATTRIBUTE_MATERIAL_WEIGHTS`** – 4 blend weights packed into one `u32` (8 bits each, 0–255 range)

### PlumeSplat Integration

The `plumesplat` crate (local dependency) provides:

- **`PlumeSplatPlugin`** – Registers the custom material and asset processing pipeline
- **`MaterialLayer`** – Defines a single texture layer (diffuse + optional normal map)
- **`SplatMaterialBuilder`** – Constructs multi-layer splatted materials with configurable:
  - UV scale (0.9)
  - Triplanar sharpness (1.0)
  - Blend offset and exponent for height blending between layers
- **Custom vertex attributes** – `ATTRIBUTE_MATERIAL_INDICES` and `ATTRIBUTE_MATERIAL_WEIGHTS` consumed by the splatting shader

Textures sourced from 1K PNG texture sets:

| Layer | Diffuse | Normal |
|-------|---------|--------|
| Grass | `Grass004_1K-PNG_Color.png` | `Grass004_1K-PNG_NormalGL.png` |
| Dirt | `Ground054_1K-PNG_Color.png` | `Ground054_1K-PNG_NormalGL.png` |
| Rock | `Rock051_1K-PNG_Color.png` | `Rock051_1K-PNG_NormalGL.png` |
| Snow | `snow_diff.png` | — |

---

## ECS Architecture

### Components

| Component | Fields | Purpose |
|-----------|--------|---------|
| `VoxelChunk` | `coord: IVec3` | Marks a chunk entity; stores its grid coordinate |
| `FpsCameraController` | `yaw, pitch, speed, sensitivity` | FPS-style flight camera state |
| `ComputeChunkMesh` | `coord, generation, task` | Async task entity for mesh generation |

### Resources

| Resource | Key Fields | Purpose |
|----------|------------|---------|
| `ChunkManager` | `entities`, `committed_lod`, `committed_transitions`, `generation`, `batch_*` | Tracks all chunk entities, the currently displayed LOD state, and the in-flight batch |
| `LodMap` | `lods: HashMap<IVec3, usize>` | Stores the latest computed LOD map for external queries |

### Systems

| System | Schedule | Description |
|--------|----------|-------------|
| `setup` | `Startup` | Spawns camera, light, chunk entities; dispatches initial mesh batch |
| `fps_camera_system` | `Update` | WASD movement, mouse look, cursor grab |
| `update_lod_system` | `Update` | Recomputes LODs, enforces constraints, dispatches dirty chunks |
| `handle_mesh_tasks` | `Update` | Polls async tasks, applies completed batch atomically |
| `toggle_wireframe` | `Update` | F1 toggles global wireframe rendering |

---

## Controls

| Key | Action |
|-----|--------|
| **W / A / S / D** | Move forward / left / backward / right |
| **Q / E** | Move down / up |
| **Shift** | 3× speed boost |
| **Mouse** | Look around (cursor is grabbed) |
| **F1** | Toggle wireframe overlay |

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| [bevy](https://crates.io/crates/bevy) | 0.18 | Game engine — rendering, ECS, asset pipeline, async tasks |
| [transvoxel](https://crates.io/crates/transvoxel) | 2.0.0 | Transvoxel isosurface extraction with transition cells |
| [fastnoise-lite](https://crates.io/crates/fastnoise-lite) | 1.1 | FastNoise Lite — FBM / ridged noise generation |
| [flagset](https://crates.io/crates/flagset) | 0.4 | Bitflag sets for transition side combinations |
| [plumesplat](../plumesplat) | local | Multi-material terrain splatting shader (triplanar, height-blended) |

---

## Building & Running

```bash
# Debug (slower to build, faster iteration)
cargo run

# Release (optimised — recommended for smooth framerate)
cargo run --release
```

Requires a GPU with Vulkan, DX12, or Metal support (Bevy's `wgpu` backend).

---

## Project Structure

```
bevyblocks/
├── Cargo.toml                        # Crate manifest & dependencies
├── README.md                         # This file
├── assets/
│   └── textures/
│       ├── terrain_normal_array.ktx2  # (Packed normal array — unused currently)
│       └── raw/                       # Source PBR texture sets (1K PNG)
│           ├── Grass004_1K-PNG_*      # Grass layer textures
│           ├── Ground054_1K-PNG_*     # Dirt/ground layer textures
│           ├── Rock051_1K-PNG_*       # Rock layer textures
│           └── snow_diff.png          # Snow diffuse texture
└── src/
    └── main.rs                        # Single-file application (~705 lines)
```

---

## How It Works — End to End

1. **Startup**: The `setup` system spawns a camera at `(0, 0, 1450)` looking at the origin, a directional light, and 9,261 chunk entities with empty placeholder meshes. It computes the initial LOD map, enforces constraints, and dispatches the first batch of async mesh tasks.

2. **Each frame**:
   - The camera moves based on player input.
   - `update_lod_system` recalculates LODs for all chunks based on the new camera position, propagates constraints, identifies dirty chunks, and (if no batch is in-flight) dispatches a new batch of mesh tasks to the thread pool.
   - `handle_mesh_tasks` polls completed tasks. Once every task in the batch finishes, it atomically swaps all mesh handles and visibility flags — ensuring the visible state is always a consistent, crack-free LOD snapshot.

3. **Mesh extraction** (on worker threads): For each dirty chunk, `generate_chunk_mesh` builds a `transvoxel::Block`, runs `extract_from_field` with the sphere+noise density function and the required transition sides, then converts the output to a Bevy `Mesh` with per-vertex material splatting data.

4. **Rendering**: Bevy's PBR pipeline renders chunks using the `plumesplat` material, which samples 4 texture layers per fragment using triplanar projection and blends them according to the packed per-vertex weights.
